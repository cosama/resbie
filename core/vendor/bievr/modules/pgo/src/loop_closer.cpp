#include "bievr_pgo/loop_closer.h"

#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/registration/icp.h>

#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/GaussNewtonOptimizer.h>
#include <gtsam/nonlinear/GncOptimizer.h>
#include <gtsam/nonlinear/GncParams.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/PriorFactor.h>

#include <cmath>
#include <sstream>
#include <string>

#include "bievr_lio/log.h"

namespace bievr {
namespace {

using gtsam::symbol_shorthand::B;  // IMU bias
using gtsam::symbol_shorthand::V;  // velocity, world frame
using gtsam::symbol_shorthand::X;  // pose

// Conditioning only, never information: a velocity no IMU factor reaches.
constexpr double kLooseVelocitySigma = 10.0;  // m/s
// Roll and pitch are left to gravity: effectively unconstrained by the prior.
constexpr double kFreeRotationVariance = 1.0;  // rad^2
// GTSAM's integration noise (position from velocity); negligible by design.
constexpr double kIntegrationVariance = 1e-8;
constexpr double kGravity = 9.81;  // m/s^2, the front end's

gtsam::Pose3 toPose3(const Transform& t) {
  return gtsam::Pose3(gtsam::Rot3(t.linear()), gtsam::Point3(t.translation()));
}

Transform toTransform(const gtsam::Pose3& p) {
  Eigen::Isometry3d iso = Eigen::Isometry3d::Identity();
  iso.linear() = p.rotation().matrix();
  iso.translation() = p.translation();
  return Transform(iso);
}

Eigen::Matrix4f toMatrix4f(const gtsam::Pose3& p) { return p.matrix().cast<float>(); }

double deg2rad(double deg) { return deg * M_PI / 180.0; }

// resbie: the body frame's heading about world z (world z is up, gravity-aligned).
double headingOf(const Eigen::Matrix3d& R) { return std::atan2(R(1, 0), R(0, 0)); }

// resbie: a pose's gravity-aligned frame: same position and heading, level.
// Scan Context assumes a level cloud with z up; the body frame can be tilted or
// mounted upside down (H420's imu_link is rolled by pi).
gtsam::Pose3 levelFrame(const gtsam::Pose3& pose) {
  return gtsam::Pose3(gtsam::Rot3::Yaw(headingOf(pose.rotation().matrix())), pose.translation());
}

// x y z rx ry rz (rotation vector), for the loop log lines.
std::string poseLog(const gtsam::Pose3& p) {
  const gtsam::Vector3 r = gtsam::Rot3::Logmap(p.rotation());
  std::ostringstream s;
  s.precision(9);
  s << p.x() << ' ' << p.y() << ' ' << p.z() << ' ' << r.x() << ' ' << r.y() << ' ' << r.z();
  return s.str();
}

}  // namespace

LoopCloser::LoopCloser(Config config)
    : config_(std::move(config)),
      keyframe_rad_gap_(deg2rad(config_.keyframe_deg_gap)),
      scan_context_(config_.scan_context) {
  gtsam::ISAM2Params params;
  params.relinearizeThreshold = config_.isam_relinearize_threshold;
  params.relinearizeSkip = config_.isam_relinearize_skip;
  isam_ = std::make_unique<gtsam::ISAM2>(params);

  initNoiseModels();
}

void LoopCloser::initNoiseModels() {
  // gtsam::Pose3's tangent space is [rotation, translation].
  gtsam::Vector6 odom_variance;
  odom_variance << config_.odom_noise_rotation, config_.odom_noise_rotation,
      config_.odom_noise_rotation, config_.odom_noise_translation,
      config_.odom_noise_translation, config_.odom_noise_translation;
  odom_noise_ = gtsam::noiseModel::Diagonal::Variances(odom_variance);

  gtsam::Vector6 loop_variance;
  loop_variance.setConstant(config_.loop_noise_score);
  loop_noise_ = gtsam::noiseModel::Robust::Create(
      gtsam::noiseModel::mEstimator::Cauchy::Create(config_.loop_noise_cauchy_c),
      gtsam::noiseModel::Diagonal::Variances(loop_variance));

  gtsam::Vector6 gnc_loop_variance;
  gnc_loop_variance << config_.loop_noise_rotation, config_.loop_noise_rotation,
      config_.loop_noise_rotation, config_.loop_noise_translation,
      config_.loop_noise_translation, config_.loop_noise_translation;
  loop_noise_gnc_ = gtsam::noiseModel::Diagonal::Variances(gnc_loop_variance);
}

bool LoopCloser::addFrame(uint64_t stamp, const Transform& T_W_I, const PointsView& cloud_body) {
  if (cloud_body.rows() == 0) return false;

  // Gate first: it is a pose delta only, and it saves copying the clouds of
  // frames that will not become keyframes.
  bool is_keyframe = !has_previous_frame_;
  if (has_previous_frame_) {
    const Eigen::Isometry3d delta = previous_pose_.inverse() * T_W_I;
    translation_accumulated_ += delta.translation().norm();
    rotation_accumulated_ += Eigen::AngleAxisd(delta.linear()).angle();
    is_keyframe = translation_accumulated_ > config_.keyframe_meter_gap ||
                  rotation_accumulated_ > keyframe_rad_gap_;
  }
  Eigen::Vector3d velocity = Eigen::Vector3d::Zero();
  if (has_previous_frame_) {
    velocity = (T_W_I.translation() - previous_pose_.translation()) /
               (1e-9 * double(stamp - previous_stamp_));
  }
  previous_pose_ = T_W_I;
  previous_stamp_ = stamp;
  has_previous_frame_ = true;
  if (!is_keyframe) return false;

  translation_accumulated_ = 0.0;
  rotation_accumulated_ = 0.0;

  Cloud::Ptr cloud(new Cloud());
  cloud->resize(cloud_body.rows());
  for (Eigen::Index i = 0; i < cloud_body.rows(); ++i) {
    cloud->points[i] = PointT(static_cast<float>(cloud_body(i, 0)), static_cast<float>(cloud_body(i, 1)),
                              static_cast<float>(cloud_body(i, 2)));
  }

  // Keyframe, loop detection, ICP, iSAM2, in that order. detectLoop() queries
  // the newest keyframe, so each keyframe is queried exactly once.
  integrateKeyframe(stamp, T_W_I, velocity, cloud);
  const ScanContext::Match match = scan_context_.detectLoop();
  if (match.index >= 0) {
    const int current = static_cast<int>(scan_context_.size()) - 1;
    const LoopCandidate candidate{match.index, current};
    LOG(I, "[resbie::pgo] Loop candidate found: from kf=" << candidate.from << " to kf=" << candidate.to
           << ", sc_dist=" << match.distance
           << ", yaw_diff=" << (match.yaw_diff_rad * 180.0 / M_PI) << " deg");
    if (const auto constraint = computeLoopConstraint(candidate)) {
      graph_.add(gtsam::BetweenFactor<gtsam::Pose3>(X(candidate.from), X(candidate.to),
                                                    *constraint, loop_noise_));
      loop_edges_.push_back({candidate.from, candidate.to, *constraint});
      ++num_loops_;
      LOG(I, "[resbie::pgo] >>> Loop ACCEPTED between kf=" << candidate.from << " and kf="
             << candidate.to << " (total accepted: " << num_loops_ << ")");
      // What the loop asserts (from^-1 to) next to what the graph held when it
      // came, so a log alone tells true loops from false ones afterwards.
      const gtsam::Pose3 held =
          keyframe_poses_.at(candidate.from).between(keyframe_poses_.at(candidate.to));
      LOG(I, "[resbie::pgo]     loop stamps " << keyframe_stamps_.at(candidate.from) << ' '
             << keyframe_stamps_.at(candidate.to) << " constraint " << poseLog(*constraint)
             << " graph " << poseLog(held));
    } else {
      ++num_rejected_;
      LOG(I, "[resbie::pgo] --- Loop REJECTED between kf=" << candidate.from << " and kf="
             << candidate.to << " (total rejected: " << num_rejected_ << ")");
    }
  }
  optimize();
  return true;
}

void LoopCloser::integrateKeyframe(uint64_t stamp, const Transform& T_W_I,
                                   const Eigen::Vector3d& odom_velocity, const Cloud::Ptr& cloud) {
  Cloud::Ptr downsampled(new Cloud());
  pcl::VoxelGrid<PointT> filter;
  filter.setLeafSize(config_.keyframe_filter_size, config_.keyframe_filter_size,
                     config_.keyframe_filter_size);
  filter.setInputCloud(cloud);
  filter.filter(*downsampled);

  const gtsam::Pose3 pose = toPose3(T_W_I);

  const int index = static_cast<int>(keyframe_clouds_.size());
  gtsam::Pose3 odom_delta;  // previous keyframe -> this one, by the odometry
  gtsam::Pose3 initial = pose;
  if (index > 0) {
    odom_delta = keyframe_odom_poses_.back().between(pose);
    initial = keyframe_poses_.back() * odom_delta;  // continue from the corrected estimate
  }
  keyframe_clouds_.push_back(downsampled);
  keyframe_odom_poses_.push_back(pose);
  keyframe_poses_.push_back(initial);
  keyframe_stamps_.push_back(stamp);

  // Scan Context gets the downsampled keyframe cloud in the keyframe's level
  // frame (resbie: not the body frame, which may be tilted or upside down). Level by the corrected estimate, whose roll
  // and pitch the graph's gravity factors keep honest, as the loop check does.
  const Eigen::Matrix3d R_level_body =
      levelFrame(initial).rotation().matrix().transpose() * initial.rotation().matrix();
  Eigen::Matrix<double, 3, Eigen::Dynamic> sc_cloud(3, downsampled->size());
  for (size_t i = 0; i < downsampled->size(); ++i) {
    const auto& p = downsampled->points[i];
    sc_cloud.col(i) = R_level_body * Eigen::Vector3d(p.x, p.y, p.z);
  }
  scan_context_.add(sc_cloud);

  // Odometry velocity, rotated like the correction of the pose it belongs to.
  const Eigen::Vector3d velocity =
      initial.rotation().matrix() * pose.rotation().matrix().transpose() * odom_velocity;
  if (!graph_initialized_) {
    // Gauge: fix yaw and position. Roll and pitch stay free for gravity, so the
    // prior's rotation covariance is loose about the world x and y axes (Pose3
    // perturbs rotation in the body frame: world = R * body).
    const Eigen::Matrix3d R = pose.rotation().matrix();
    const double tight = config_.prior_noise_score, relaxed = kFreeRotationVariance;
    gtsam::Matrix6 cov = gtsam::Matrix6::Zero();
    cov.topLeftCorner<3, 3>() = R.transpose() * Eigen::Vector3d(relaxed, relaxed, tight).asDiagonal() * R;
    cov.bottomRightCorner<3, 3>() = tight * Eigen::Matrix3d::Identity();
    const auto prior_x =
        gtsam::PriorFactor<gtsam::Pose3>(X(0), pose, gtsam::noiseModel::Gaussian::Covariance(cov));
    const auto prior_v = gtsam::PriorFactor<gtsam::Vector3>(
        V(0), velocity, gtsam::noiseModel::Isotropic::Sigma(3, kLooseVelocitySigma));
    gtsam::Vector6 bias_std;
    bias_std << config_.acc_bias_init_std, config_.gyro_bias_init_std;
    const auto prior_b = gtsam::PriorFactor<gtsam::imuBias::ConstantBias>(
        B(0), bias_, gtsam::noiseModel::Diagonal::Sigmas(bias_std));
    graph_.add(prior_x);
    graph_.add(prior_v);
    graph_.add(prior_b);
    base_graph_.add(prior_x);
    base_graph_.add(prior_v);
    base_graph_.add(prior_b);
    graph_initialized_ = true;
  } else {
    addMotionFactors(index, odom_delta, stamp, velocity);
  }

  initial_estimate_.insert(X(index), initial);
  initial_estimate_.insert(V(index), velocity);
  initial_estimate_.insert(B(index), bias_);
  base_initial_estimate_.insert(X(index), pose);
  base_initial_estimate_.insert(V(index), odom_velocity);
  base_initial_estimate_.insert(B(index), gtsam::imuBias::ConstantBias());
  // Samples up to this keyframe now belong to the factors ending at it.
  while (!imu_.empty() && imu_.front().stamp <= stamp) imu_.pop_front();
}

void LoopCloser::addImu(uint64_t stamp, const Eigen::Vector3d& acc, const Eigen::Vector3d& gyro) {
  // Samples at or before the newest keyframe belong to factors already built.
  if (!keyframe_stamps_.empty() && stamp <= keyframe_stamps_.back()) return;
  imu_.push_back({stamp, acc, gyro});
}

// GTSAM wants continuous noise densities: a per-sample standard deviation
// sigma at sample interval dt is sigma^2 * dt. Gravity points down world z.
boost::shared_ptr<gtsam::PreintegrationCombinedParams> LoopCloser::makeImuParams(double dt) const {
  auto p = gtsam::PreintegrationCombinedParams::MakeSharedU(kGravity);
  p->accelerometerCovariance = config_.acc_noise_std.cwiseAbs2().asDiagonal() * dt;
  p->gyroscopeCovariance = config_.gyro_noise_std.cwiseAbs2().asDiagonal() * dt;
  p->integrationCovariance = kIntegrationVariance * gtsam::I_3x3;
  p->biasAccCovariance = config_.acc_bias_walk * config_.acc_bias_walk * gtsam::I_3x3;
  p->biasOmegaCovariance = config_.gyro_bias_walk * config_.gyro_bias_walk * gtsam::I_3x3;
  // resbie: GTSAM (<= 4.2) adds biasAccOmegaInt to the measurement noise at every step, and
  // its default I_6x6 made the factor ~100x weaker than the IMU (1 rad, 5.7 m/s after 1 s).
  // The bias uncertainty is the B states' prior and random walk, so nothing goes here.
  p->biasAccOmegaInt = gtsam::Z_6x6;
  return p;
}

void LoopCloser::addMotionFactors(int index, const gtsam::Pose3& odom_delta, uint64_t stamp,
                                  const Eigen::Vector3d& velocity) {
  const auto odom_factor =
      gtsam::BetweenFactor<gtsam::Pose3>(X(index - 1), X(index), odom_delta, odom_noise_);
  graph_.add(odom_factor);
  base_graph_.add(odom_factor);

  // The noise densities need the IMU rate, so two samples.
  if (!imu_params_ && imu_.size() >= 2) {
    imu_params_ = makeImuParams(1e-9 * double(imu_.back().stamp - imu_.front().stamp) /
                                double(imu_.size() - 1));
  }
  // Each sample covers the time since the previous one (the first: since the
  // previous keyframe); the last also covers the rest up to this keyframe, so
  // the factors tile the timeline.
  const ImuSample* last = nullptr;
  for (const ImuSample& s : imu_) {
    if (s.stamp > stamp) break;
    last = &s;
  }
  if (imu_params_ && last) {
    gtsam::PreintegratedCombinedMeasurements pim(imu_params_, bias_);
    uint64_t t = keyframe_stamps_[index - 1];
    for (const ImuSample& s : imu_) {
      if (s.stamp > stamp) break;
      pim.integrateMeasurement(s.acc, s.gyro, 1e-9 * double(s.stamp - t));
      t = s.stamp;
    }
    if (t < stamp) pim.integrateMeasurement(last->acc, last->gyro, 1e-9 * double(stamp - t));
    const auto imu_factor =
        gtsam::CombinedImuFactor(X(index - 1), V(index - 1), X(index), V(index), B(index - 1),
                                 B(index), pim);
    graph_.add(imu_factor);
    base_graph_.add(imu_factor);
  } else {
    // No IMU between them (a gap in the IMU stream, or no rate yet): the bias
    // still drifts by its random walk over the interval, and the velocity is
    // left loose.
    const double sqrt_dt = std::sqrt(1e-9 * double(stamp - keyframe_stamps_[index - 1]));
    gtsam::Vector6 walk;
    walk << gtsam::Vector3::Constant(config_.acc_bias_walk * sqrt_dt),
        gtsam::Vector3::Constant(config_.gyro_bias_walk * sqrt_dt);
    const auto bias_factor = gtsam::BetweenFactor<gtsam::imuBias::ConstantBias>(
        B(index - 1), B(index), gtsam::imuBias::ConstantBias(),
        gtsam::noiseModel::Diagonal::Sigmas(walk.cwiseMax(1e-9)));  // walk 0: near-rigid, not singular
    const auto vel_factor = gtsam::PriorFactor<gtsam::Vector3>(
        V(index), velocity, gtsam::noiseModel::Isotropic::Sigma(3, kLooseVelocitySigma));
    graph_.add(bias_factor);
    graph_.add(vel_factor);
    base_graph_.add(bias_factor);
    base_graph_.add(vel_factor);
  }
}

LoopCloser::Cloud::Ptr LoopCloser::buildSubmap(int key, int span) const {
  // Each cloud goes at its own pose, so the submap is geometrically coherent and
  // ICP's output is a world-frame correction rather than a relative transform.
  Cloud::Ptr submap(new Cloud());
  const int count = static_cast<int>(keyframe_clouds_.size());
  for (int i = -span; i <= span; ++i) {
    const int near = key + i;
    if (near < 0 || near >= count) continue;
    Cloud transformed;
    pcl::transformPointCloud(*keyframe_clouds_[near], transformed, toMatrix4f(keyframe_poses_[near]));
    *submap += transformed;
  }
  if (submap->empty()) return submap;

  Cloud::Ptr downsampled(new Cloud());
  pcl::VoxelGrid<PointT> filter;
  filter.setLeafSize(config_.icp_filter_size, config_.icp_filter_size, config_.icp_filter_size);
  filter.setInputCloud(submap);
  filter.filter(*downsampled);
  return downsampled;
}

std::optional<gtsam::Pose3> LoopCloser::computeLoopConstraint(const LoopCandidate& candidate) {
  Cloud::Ptr source = buildSubmap(candidate.to, 0);
  Cloud::Ptr target = buildSubmap(candidate.from, config_.history_keyframe_search_num);
  if (source->empty() || target->empty()) return std::nullopt;

  const gtsam::Pose3 pose_curr = keyframe_poses_.at(candidate.to);
  const gtsam::Pose3 pose_loop = keyframe_poses_.at(candidate.from);

  pcl::IterativeClosestPoint<PointT, PointT> icp;
  icp.setMaxCorrespondenceDistance(config_.icp_max_correspondence_distance);
  icp.setMaximumIterations(config_.icp_max_iterations);
  icp.setTransformationEpsilon(config_.icp_transformation_epsilon);
  icp.setEuclideanFitnessEpsilon(config_.icp_euclidean_fitness_epsilon);
  icp.setRANSACIterations(config_.icp_ransac_iterations);
  icp.setInputSource(source);
  icp.setInputTarget(target);

  // Both clouds are in the world frame, so identity is the start.
  Cloud aligned;
  icp.align(aligned);
  if (!icp.hasConverged()) {
    LOG(I, "[resbie::pgo]   Loop rejected: ICP did not converge.");
    return std::nullopt;
  }
  const double best_fitness = icp.getFitnessScore();
  LOG(D, "[resbie::pgo]   ICP converged: fitness=" << best_fitness
         << " (threshold=" << config_.loop_fitness_score_threshold << ")");
  const Eigen::Matrix4f best_transformation = icp.getFinalTransformation();

  // getFinalTransformation() includes the initial guess, and corrects the current
  // keyframe in the world frame: P_corrected = T_icp * P_curr. BetweenFactor
  // wants P_loop^-1 * P_corrected.
  const gtsam::Pose3 correction(best_transformation.cast<double>());
  const gtsam::Pose3 asserted = pose_loop.between(correction * pose_curr);
  if (best_fitness > config_.loop_fitness_score_threshold) {
    LOG(I, "[resbie::pgo]   Loop rejected: fitness " << best_fitness << " > threshold "
           << config_.loop_fitness_score_threshold);
    logRejected(candidate, asserted, best_fitness);
    return std::nullopt;
  }
  return asserted;
}

void LoopCloser::logRejected(const LoopCandidate& candidate, const gtsam::Pose3& asserted,
                             double fitness) const {
  const gtsam::Pose3 held =
      keyframe_poses_.at(candidate.from).between(keyframe_poses_.at(candidate.to));
  LOG(I, "[resbie::pgo]     rejected stamps " << keyframe_stamps_.at(candidate.from) << ' '
         << keyframe_stamps_.at(candidate.to) << " fitness " << fitness << " constraint "
         << poseLog(asserted) << " graph " << poseLog(held));
}

void LoopCloser::optimize() {
  if (!graph_initialized_ || (graph_.empty() && initial_estimate_.empty())) return;
  isam_->update(graph_, initial_estimate_);
  isam_->update();
  graph_.resize(0);
  initial_estimate_.clear();
  readEstimate(isam_->calculateEstimate());
}

void LoopCloser::finish() {
  if (!graph_initialized_) return;
  optimize();
  if (loop_edges_.empty()) {
    readEstimate(
        gtsam::LevenbergMarquardtOptimizer(base_graph_, isam_->calculateEstimate()).optimize());
    return;
  }

  gtsam::GaussNewtonParams base_gn_params;
  base_gn_params.maxIterations = 3;
  const gtsam::Values inlier_init =
      gtsam::GaussNewtonOptimizer(base_graph_, base_initial_estimate_, base_gn_params).optimize();

  gtsam::NonlinearFactorGraph gnc_graph = base_graph_;
  using IndexVector = gtsam::GncParams<gtsam::GaussNewtonParams>::IndexVector;
  IndexVector known_inliers(base_graph_.size());
  for (size_t i = 0; i < base_graph_.size(); ++i) known_inliers[i] = i;

  for (const LoopEdge& edge : loop_edges_) {
    gnc_graph.add(
        gtsam::BetweenFactor<gtsam::Pose3>(X(edge.from), X(edge.to), edge.constraint, loop_noise_gnc_));
  }

  gtsam::GaussNewtonParams gn_params;
  gn_params.maxIterations = 10;
  gtsam::GncParams<gtsam::GaussNewtonParams> gnc_params(gn_params);
  gnc_params.setKnownInliers(known_inliers);
  gnc_params.setRelativeCostTol(-1.0);
  gtsam::GncOptimizer<gtsam::GncParams<gtsam::GaussNewtonParams>> gnc(gnc_graph, inlier_init,
                                                                      gnc_params);
  gnc.setInlierCostThresholdsAtProbability(0.99);
  gtsam::Vector barc_sq = gnc.getInlierCostThresholds();
  // GTSAM 4.2's GncOptimizer::initializeMu() loops over all factors in nfg_
  // (ignoring knownInliers), so base-graph thresholds must be set high to
  // prevent IMU/odometry residuals from inflating mu.
  for (size_t i = 0; i < base_graph_.size(); ++i) barc_sq[i] = 1e30;
  gnc.setInlierCostThresholds(barc_sq);
  const gtsam::Values gnc_result = gnc.optimize();
  const gtsam::Vector weights = gnc.getWeights();
  gtsam::NonlinearFactorGraph final_graph = base_graph_;
  size_t gnc_kept = 0, gnc_dropped = 0;
  for (size_t k = 0; k < loop_edges_.size(); ++k) {
    if (weights[base_graph_.size() + k] > 0.5) {
      ++gnc_kept;
      const LoopEdge& edge = loop_edges_[k];
      final_graph.add(gtsam::BetweenFactor<gtsam::Pose3>(X(edge.from), X(edge.to), edge.constraint,
                                                         loop_noise_gnc_));
    } else {
      ++gnc_dropped;
    }
  }
  num_loops_ = gnc_kept;
  num_rejected_ += gnc_dropped;
  LOG(I, "[resbie::pgo] GNC finish: kept=" << gnc_kept << " dropped=" << gnc_dropped
         << " total_loops=" << loop_edges_.size());
  const gtsam::Values& lm_init = (gnc_kept > 0) ? gnc_result : inlier_init;
  readEstimate(gtsam::LevenbergMarquardtOptimizer(final_graph, lm_init).optimize());
}

void LoopCloser::readEstimate(const gtsam::Values& estimate) {
  for (size_t i = 0; i < keyframe_poses_.size(); ++i) {
    if (estimate.exists(X(i))) keyframe_poses_[i] = estimate.at<gtsam::Pose3>(X(i));
  }
  if (!keyframe_poses_.empty() && estimate.exists(B(keyframe_poses_.size() - 1))) {
    bias_ = estimate.at<gtsam::imuBias::ConstantBias>(B(keyframe_poses_.size() - 1));
  }
}

std::vector<Transform> LoopCloser::keyframePoses() const {
  std::vector<Transform> out;
  out.reserve(keyframe_poses_.size());
  for (const gtsam::Pose3& pose : keyframe_poses_) out.push_back(toTransform(pose));
  return out;
}

LoopCloser::Stats LoopCloser::stats() const {
  Stats stats;
  stats.num_keyframes = keyframe_poses_.size();
  stats.num_loops = num_loops_;
  stats.num_rejected = num_rejected_;
  return stats;
}

}  // namespace bievr

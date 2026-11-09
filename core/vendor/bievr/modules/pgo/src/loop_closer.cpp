#include "bievr_pgo/loop_closer.h"

#include <pcl/filters/voxel_grid.h>

#include <Eigen/Eigenvalues>
#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/GaussNewtonOptimizer.h>
#include <gtsam/nonlinear/GncOptimizer.h>
#include <gtsam/nonlinear/GncParams.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/PriorFactor.h>

#include <small_gicp/ann/kdtree.hpp>
#include <small_gicp/factors/gicp_factor.hpp>
#include <small_gicp/points/point_cloud.hpp>
#include <small_gicp/registration/reduction.hpp>
#include <small_gicp/registration/registration.hpp>
#include <small_gicp/util/downsampling.hpp>
#include <small_gicp/util/normal_estimation.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>
#include <string>

#include "BTC.h"
#include "bievr_lio/log.h"

namespace bievr {

// resbie: local odometry submap window and GICP loop verification.
struct LoopCloser::PreparedSubmap {
  small_gicp::PointCloud::Ptr coarse;
  small_gicp::KdTree<small_gicp::PointCloud>::Ptr coarse_tree;
  small_gicp::PointCloud::Ptr fine;
  small_gicp::KdTree<small_gicp::PointCloud>::Ptr fine_tree;
};

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

// Local odometry submap window and GICP loop verification constants.
constexpr int kSubmapKfRadius = 5;            // [-5, +5] keyframes around anchor keyframe
constexpr double kSubmapMaxDistM = 20.0;      // max relative odometry distance [m]
constexpr int kGicpNeighbors = 20;            // k-NN for covariance/normal estimation
constexpr int kGicpCoarseIterations = 30;     // GICP max iterations for coarse stage
constexpr int kGicpMaxIterations = 50;        // GICP max iterations for fine stage
constexpr double kGicpRotEps = 1e-3;          // fine stage rotation convergence [rad]
constexpr double kGicpTransEps = 1e-3;        // fine stage translation convergence [m]
constexpr double kMinLoopOverlap = 0.60;      // minimum fine-submap inlier fraction
constexpr double kMinHessian3dEig = 500.0;    // minimum smallest translational eigenvalue
constexpr double kMinHessianPlaneEig = 800.0; // minimum middle translational eigenvalue

gtsam::Pose3 toPose3(const Transform& t) {
  return gtsam::Pose3(gtsam::Rot3(t.linear()), gtsam::Point3(t.translation()));
}

Transform toTransform(const gtsam::Pose3& p) {
  Eigen::Isometry3d iso = Eigen::Isometry3d::Identity();
  iso.linear() = p.rotation().matrix();
  iso.translation() = p.translation();
  return Transform(iso);
}

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

struct FitMetrics {
  double fitness = std::numeric_limits<double>::infinity();
  double overlap = 0.0;
};

FitMetrics evaluateFit(const small_gicp::PointCloud& target_fine,
                       const small_gicp::PointCloud& source_fine,
                       const small_gicp::KdTree<small_gicp::PointCloud>& target_fine_tree,
                       const Eigen::Isometry3d& T_target_source,
                       double max_radius) {
  FitMetrics m;
  if (source_fine.empty() || target_fine.empty()) return m;
  const double max_sq = max_radius * max_radius;
  size_t inliers = 0;
  double sum_sq = 0.0;
  for (size_t i = 0; i < source_fine.size(); ++i) {
    const Eigen::Vector4d p_trans = T_target_source * source_fine.point(i);
    size_t idx = 0;
    double sq_dist = 0.0;
    if (target_fine_tree.nearest_neighbor_search(p_trans, &idx, &sq_dist) > 0 &&
        sq_dist <= max_sq) {
      ++inliers;
      const Eigen::Vector3d diff = (p_trans - target_fine.point(idx)).head<3>();
      const Eigen::Vector3d n_tgt = target_fine.normal(idx).head<3>();
      const double d_plane = (n_tgt.squaredNorm() > 0.5) ? n_tgt.dot(diff) : diff.norm();
      sum_sq += d_plane * d_plane;
    }
  }
  m.overlap = static_cast<double>(inliers) / static_cast<double>(source_fine.size());
  if (inliers > 0) {
    m.fitness = sum_sq / static_cast<double>(inliers);
  }
  return m;
}

small_gicp::PointCloud buildLocalSubmap(const std::vector<LoopCloser::Cloud::Ptr>& keyframe_clouds,
                                        const std::vector<gtsam::Pose3>& keyframe_odom_poses,
                                        int anchor_key, int span) {
  // Build submap in anchor_key's body frame using local relative odometry so
  // old loop closures never distort the local submap geometry.
  std::vector<Eigen::Vector3d> pts;
  const int count = static_cast<int>(keyframe_clouds.size());
  if (anchor_key < 0 || anchor_key >= count) return small_gicp::PointCloud();
  const gtsam::Pose3& anchor_odom = keyframe_odom_poses[anchor_key];
  const double max_dist_sq = kSubmapMaxDistM * kSubmapMaxDistM;
  for (int i = -span; i <= span; ++i) {
    const int near = anchor_key + i;
    if (near < 0 || near >= count) continue;
    const gtsam::Pose3 rel_odom =
        (near == anchor_key) ? gtsam::Pose3::Identity()
                             : anchor_odom.between(keyframe_odom_poses[near]);
    if (rel_odom.translation().squaredNorm() > max_dist_sq) continue;
    const Eigen::Matrix3d R = rel_odom.rotation().matrix();
    const Eigen::Vector3d t = rel_odom.translation();
    const auto& cloud = *keyframe_clouds[near];
    pts.reserve(pts.size() + cloud.size());
    for (const auto& pt : cloud.points) {
      pts.push_back(R * Eigen::Vector3d(pt.x, pt.y, pt.z) + t);
    }
  }
  return small_gicp::PointCloud(pts);
}

ConfigSetting makeBtcConfig(const LoopCloser::Config& config) {
  ConfigSetting btc;
  const bool outdoor = (config.scan_context.max_radius >= 95.0);
  if (!outdoor) {
    btc.useful_corner_num_ = 100;
    btc.plane_detection_thre_ = 0.01f;
    btc.plane_merge_normal_thre_ = 0.1f;
    btc.plane_merge_dis_thre_ = 0.3f;
    btc.voxel_size_ = 1.0f;
    btc.voxel_init_num_ = 10;
    btc.proj_plane_num_ = 2;
    btc.proj_image_resolution_ = 0.5f;
    btc.proj_image_high_inc_ = 0.1f;
    btc.proj_dis_min_ = -1.0f;
    btc.proj_dis_max_ = 5.0f;
    btc.summary_min_thre_ = 6.0f;
    btc.line_filter_enable_ = 0;
    btc.touch_filter_enable_ = 0;
    btc.descriptor_near_num_ = 15.0f;
    btc.descriptor_min_len_ = 1.5f;
    btc.descriptor_max_len_ = 30.0f;
    btc.non_max_suppression_radius_ = 2.0f;
    btc.std_side_resolution_ = 0.2f;
    btc.skip_near_num_ = config.scan_context.num_exclude_recent;
    btc.candidate_num_ = 30;
    btc.rough_dis_threshold_ = 0.015f;
    btc.similarity_threshold_ = 0.45f;
    btc.normal_threshold_ = 0.2f;
    btc.dis_threshold_ = 0.5f;
    btc.icp_threshold_ = 0.15f;
  } else {
    btc.useful_corner_num_ = 200;
    btc.plane_detection_thre_ = 0.01f;
    btc.plane_merge_normal_thre_ = 0.1f;
    btc.plane_merge_dis_thre_ = 0.3f;
    btc.voxel_size_ = 2.0f;
    btc.voxel_init_num_ = 10;
    btc.proj_plane_num_ = 1;
    btc.proj_image_resolution_ = 0.5f;
    btc.proj_image_high_inc_ = 0.2f;
    btc.proj_dis_min_ = -1.0f;
    btc.proj_dis_max_ = 10.0f;
    btc.summary_min_thre_ = 6.0f;
    btc.line_filter_enable_ = 0;
    btc.touch_filter_enable_ = 0;
    btc.descriptor_near_num_ = 15.0f;
    btc.descriptor_min_len_ = 2.0f;
    btc.descriptor_max_len_ = 50.0f;
    btc.non_max_suppression_radius_ = 3.0f;
    btc.std_side_resolution_ = 0.2f;
    btc.skip_near_num_ = config.scan_context.num_exclude_recent;
    btc.candidate_num_ = 50;
    btc.rough_dis_threshold_ = 0.015f;
    btc.similarity_threshold_ = 0.40f;
    btc.normal_threshold_ = 0.2f;
    btc.dis_threshold_ = 0.5f;
    btc.icp_threshold_ = 0.15f;
  }
  return btc;
}

}  // namespace

LoopCloser::LoopCloser(Config config)
    : config_(std::move(config)),
      keyframe_rad_gap_(deg2rad(config_.keyframe_deg_gap)) {
  ConfigSetting btc_cfg = makeBtcConfig(config_);
  std_manager_ = std::make_unique<STDescManager>(btc_cfg);
  gtsam::ISAM2Params params;
  params.relinearizeThreshold = config_.isam_relinearize_threshold;
  params.relinearizeSkip = config_.isam_relinearize_skip;
  isam_ = std::make_unique<gtsam::ISAM2>(params);

  initNoiseModels();
}

LoopCloser::~LoopCloser() = default;

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

  // Keyframe, BTC loop detection, ICP, iSAM2, in that order.
  integrateKeyframe(stamp, T_W_I, velocity, cloud);

  const int current = static_cast<int>(keyframe_clouds_.size()) - 1;
  const small_gicp::PointCloud raw_btc_submap =
      buildLocalSubmap(btc_keyframe_clouds_, keyframe_odom_poses_, current, kSubmapKfRadius);
  const Eigen::Matrix3d& R_level_curr = keyframe_level_rotations_.at(current);
  pcl::PointCloud<pcl::PointXYZI>::Ptr plbtc(new pcl::PointCloud<pcl::PointXYZI>());
  plbtc->points.resize(raw_btc_submap.size());
  for (size_t i = 0; i < raw_btc_submap.size(); ++i) {
    const Eigen::Vector3d p_level = R_level_curr * raw_btc_submap.point(i).head<3>();
    pcl::PointXYZI pt;
    pt.x = static_cast<float>(p_level.x());
    pt.y = static_cast<float>(p_level.y());
    pt.z = static_cast<float>(p_level.z());
    pt.intensity = 0.0f;
    plbtc->points[i] = pt;
  }
  plbtc->width = static_cast<uint32_t>(plbtc->points.size());
  plbtc->height = 1;
  plbtc->is_dense = true;

  std::vector<STD> stds_vec;
  std_manager_->GenerateSTDescs(plbtc, stds_vec, current);
  std::pair<int, double> search_result(-1, 0.0);
  std::pair<Eigen::Vector3d, Eigen::Matrix3d> loop_transform(
      Eigen::Vector3d::Zero(), Eigen::Matrix3d::Identity());
  std::vector<std::pair<STD, STD>> loop_std_pair;
  std_manager_->SearchLoop(stds_vec, search_result, loop_transform, loop_std_pair,
                           std_manager_->plane_cloud_vec_.back());
  std_manager_->AddSTDescs(stds_vec);

  if (!std_manager_->last_verified_candidates_.empty()) {
    int accepted_from = -1;
    int primary_tries = 0;
    bool added_early_bridge = false;
    for (const auto& vc : std_manager_->last_verified_candidates_) {
      const bool is_early_bridge = (accepted_from >= 0) && (vc.frame_id + 200 <= accepted_from);
      if (accepted_from >= 0 && (!is_early_bridge || added_early_bridge)) {
        continue;
      }
      if (accepted_from < 0 && primary_tries >= 3) {
        continue;
      }
      ++primary_tries;
      ++num_candidates_;
      const LoopCandidate candidate{vc.frame_id, current};
      const Eigen::Matrix3d& R_level_cand = keyframe_level_rotations_.at(candidate.from);
      Eigen::Isometry3d btc_seed = Eigen::Isometry3d::Identity();
      btc_seed.linear() = R_level_cand.transpose() * vc.transform.second * R_level_curr;
      btc_seed.translation() = R_level_cand.transpose() * vc.transform.first;
      const double yaw_diff_rad = headingOf(vc.transform.second);
      LOG(I, "[resbie::pgo] Loop candidate found: from kf=" << candidate.from << " to kf=" << candidate.to
             << ", btc_score=" << vc.score
             << ", sc_dist=" << (1.0 - vc.score)
             << ", yaw_diff=" << (yaw_diff_rad * 180.0 / M_PI) << " deg");
      if (const auto constraint = computeLoopConstraint(candidate, btc_seed)) {
        graph_.add(gtsam::BetweenFactor<gtsam::Pose3>(X(candidate.from), X(candidate.to),
                                                      *constraint, loop_noise_));
        loop_edges_.push_back({candidate.from, candidate.to, *constraint});
        ++num_loops_;
        LOG(I, "[resbie::pgo] >>> Loop ACCEPTED between kf=" << candidate.from << " and kf="
               << candidate.to << " (total accepted: " << num_loops_ << ")");
        const gtsam::Pose3 held =
            keyframe_poses_.at(candidate.from).between(keyframe_poses_.at(candidate.to));
        LOG(I, "[resbie::pgo]     loop stamps " << keyframe_stamps_.at(candidate.from) << ' '
               << keyframe_stamps_.at(candidate.to) << " constraint " << poseLog(*constraint)
               << " graph " << poseLog(held));
        if (accepted_from < 0) {
          accepted_from = candidate.from;
        } else {
          added_early_bridge = true;
        }
      } else {
        ++num_rejected_;
        LOG(I, "[resbie::pgo] --- Loop REJECTED between kf=" << candidate.from << " and kf="
               << candidate.to << " (total rejected: " << num_rejected_ << ")");
      }
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

  // BTC requires finer point density (Voxel-SLAM uses voxel_size / 10 = 0.10 m)
  // so 1 m plane voxels exceed voxel_init_num=10 and 0.10 m vertical bins are populated.
  Cloud::Ptr btc_downsampled(new Cloud());
  const float btc_leaf = static_cast<float>(std::min(config_.keyframe_filter_size, 0.10));
  pcl::VoxelGrid<PointT> btc_filter;
  btc_filter.setLeafSize(btc_leaf, btc_leaf, btc_leaf);
  btc_filter.setInputCloud(cloud);
  btc_filter.filter(*btc_downsampled);

  const gtsam::Pose3 pose = toPose3(T_W_I);

  const int index = static_cast<int>(keyframe_clouds_.size());
  gtsam::Pose3 odom_delta;  // previous keyframe -> this one, by the odometry
  gtsam::Pose3 initial = pose;
  if (index > 0) {
    odom_delta = keyframe_odom_poses_.back().between(pose);
    initial = keyframe_poses_.back() * odom_delta;  // continue from the corrected estimate
  }
  keyframe_clouds_.push_back(downsampled);
  btc_keyframe_clouds_.push_back(btc_downsampled);
  keyframe_odom_poses_.push_back(pose);
  keyframe_poses_.push_back(initial);
  keyframe_stamps_.push_back(stamp);
  prepared_submaps_.push_back(nullptr);

  // Store the gravity-levelled rotation for BTC descriptor extraction and seed conversion.
  const Eigen::Matrix3d R_level_body =
      levelFrame(pose).rotation().matrix().transpose() * pose.rotation().matrix();
  keyframe_level_rotations_.push_back(R_level_body);

  const double step_dist = (index > 0) ? odom_delta.translation().norm() : 0.0;
  const double cum_dist =
      keyframe_travel_dist_.empty() ? 0.0 : (keyframe_travel_dist_.back() + step_dist);
  keyframe_travel_dist_.push_back(cum_dist);

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

std::shared_ptr<LoopCloser::PreparedSubmap> LoopCloser::getOrPrepareSubmap(
    int anchor_key) const {
  const int count = static_cast<int>(keyframe_clouds_.size());
  if (anchor_key < 0 || anchor_key >= count) return nullptr;
  const bool complete = (anchor_key + kSubmapKfRadius < count);
  if (complete && prepared_submaps_[anchor_key]) {
    return prepared_submaps_[anchor_key];
  }
  const double v = std::max(0.05, config_.icp_filter_size);
  const small_gicp::PointCloud raw =
      buildLocalSubmap(keyframe_clouds_, keyframe_odom_poses_, anchor_key, kSubmapKfRadius);
  auto prep = std::make_shared<PreparedSubmap>();
  prep->coarse = small_gicp::voxelgrid_sampling(raw, 2.0 * v);
  prep->fine = small_gicp::voxelgrid_sampling(raw, v);
  if (!prep->coarse->empty()) {
    prep->coarse_tree =
        std::make_shared<small_gicp::KdTree<small_gicp::PointCloud>>(prep->coarse);
    small_gicp::estimate_covariances(*prep->coarse, *prep->coarse_tree, kGicpNeighbors);
  }
  if (!prep->fine->empty()) {
    prep->fine_tree =
        std::make_shared<small_gicp::KdTree<small_gicp::PointCloud>>(prep->fine);
    small_gicp::estimate_normals_covariances(*prep->fine, *prep->fine_tree, kGicpNeighbors);
  }
  if (complete) {
    prepared_submaps_[anchor_key] = prep;
  }
  return prep;
}

std::optional<gtsam::Pose3> LoopCloser::computeLoopConstraint(
    const LoopCandidate& candidate, const Eigen::Isometry3d& btc_seed) {
  const auto source_submap = getOrPrepareSubmap(candidate.to);
  const auto target_submap = getOrPrepareSubmap(candidate.from);
  if (!source_submap || !target_submap || !source_submap->coarse || !source_submap->fine ||
      !target_submap->coarse || !target_submap->fine ||
      source_submap->coarse->size() <= 10 || source_submap->fine->size() <= 10 ||
      target_submap->coarse->size() <= 10 || target_submap->fine->size() <= 10) {
    ++num_rejected_submap_;
    LOG(I, "[resbie::pgo]   Loop rejected: submap empty or too small.");
    return std::nullopt;
  }

  const gtsam::Pose3& pose_loop = keyframe_poses_.at(candidate.from);
  const gtsam::Pose3& pose_curr = keyframe_poses_.at(candidate.to);
  const Eigen::Isometry3d graph_seed(pose_loop.between(pose_curr).matrix());

  const double v = std::max(0.05, config_.icp_filter_size);
  const double coarse_max_corr = 6.0 * v;
  const double fine_max_corr = 3.0 * v;

  auto run_align = [&](const Eigen::Isometry3d& T_init) {
    small_gicp::Registration<small_gicp::GICPFactor, small_gicp::SerialReduction> reg_coarse;
    reg_coarse.rejector.max_dist_sq = coarse_max_corr * coarse_max_corr;
    reg_coarse.optimizer.max_iterations = kGicpCoarseIterations;
    const small_gicp::RegistrationResult res_coarse = reg_coarse.align(
        *target_submap->coarse, *source_submap->coarse, *target_submap->coarse_tree, T_init);

    const Eigen::Isometry3d fine_init =
        res_coarse.converged ? res_coarse.T_target_source : T_init;

    small_gicp::Registration<small_gicp::GICPFactor, small_gicp::SerialReduction> reg_fine;
    reg_fine.rejector.max_dist_sq = fine_max_corr * fine_max_corr;
    reg_fine.optimizer.max_iterations = kGicpMaxIterations;
    reg_fine.criteria.rotation_eps = kGicpRotEps;
    reg_fine.criteria.translation_eps = kGicpTransEps;
    const small_gicp::RegistrationResult res_fine = reg_fine.align(
        *target_submap->fine, *source_submap->fine, *target_submap->fine_tree, fine_init);

    FitMetrics fit;
    if (res_fine.converged) {
      fit = evaluateFit(*target_submap->fine, *source_submap->fine, *target_submap->fine_tree,
                        res_fine.T_target_source, 2.0 * v);
    }
    return std::make_pair(res_fine, fit);
  };

  const double span =
      std::max(0.0, keyframe_travel_dist_.at(candidate.to) -
                        keyframe_travel_dist_.at(candidate.from));

  const char* seed_used = "btc";
  auto [res_fine, fit] = run_align(btc_seed);
  const double max_btc_drift = (fit.overlap >= 0.70 && fit.fitness <= 0.05)
                                   ? std::max(20.0, 0.15 * span)
                                   : std::max(3.0, 0.06 * span);
  bool btc_drift_ok = false;
  if (res_fine.converged) {
    const gtsam::Pose3 btc_asserted(res_fine.T_target_source.matrix());
    const double btc_graph_drift =
        ((pose_loop * btc_asserted).translation() - pose_curr.translation()).norm();
    const double btc_odom_drift =
        ((keyframe_odom_poses_.at(candidate.from) * btc_asserted).translation() -
         keyframe_odom_poses_.at(candidate.to).translation())
            .norm();
    btc_drift_ok = (btc_graph_drift <= max_btc_drift) && (btc_odom_drift <= max_btc_drift);
  }
  const bool btc_ok = res_fine.converged && btc_drift_ok && (fit.overlap >= kMinLoopOverlap) &&
                      std::isfinite(fit.fitness) &&
                      (fit.fitness <= config_.loop_fitness_score_threshold);
  if (!btc_ok) {
    auto [res_graph, fit_graph] = run_align(graph_seed);
    const bool graph_ok = res_graph.converged && (fit_graph.overlap >= kMinLoopOverlap) &&
                          std::isfinite(fit_graph.fitness) &&
                          (fit_graph.fitness <= config_.loop_fitness_score_threshold);
    if (!btc_drift_ok || graph_ok || (!res_fine.converged && res_graph.converged) ||
        (res_graph.converged && fit_graph.overlap > fit.overlap)) {
      res_fine = res_graph;
      fit = fit_graph;
      seed_used = "graph";
    }
  }

  if (!res_fine.converged) {
    ++num_rejected_icp_;
    LOG(I, "[resbie::pgo]   Loop rejected: ICP did not converge (seed=" << seed_used << ").");
    return std::nullopt;
  }

  const Eigen::Isometry3d& T_final = res_fine.T_target_source;
  const gtsam::Pose3 asserted(T_final.matrix());
  if (fit.overlap < kMinLoopOverlap) {
    ++num_rejected_overlap_;
    LOG(I, "[resbie::pgo]   Loop rejected: overlap " << fit.overlap << " < " << kMinLoopOverlap
           << " (seed=" << seed_used << ")");
    logRejected(candidate, asserted, fit.fitness);
    return std::nullopt;
  }
  if (!std::isfinite(fit.fitness) || fit.fitness > config_.loop_fitness_score_threshold) {
    ++num_rejected_fitness_;
    LOG(I, "[resbie::pgo]   Loop rejected: fitness " << fit.fitness << " > threshold "
           << config_.loop_fitness_score_threshold << " (seed=" << seed_used << ")");
    logRejected(candidate, asserted, fit.fitness);
    return std::nullopt;
  }

  const Eigen::Matrix3d H_tt = 0.5 * (res_fine.H.bottomRightCorner<3, 3>() +
                                      res_fine.H.bottomRightCorner<3, 3>().transpose());
  const Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es_tt(H_tt, Eigen::EigenvaluesOnly);
  const Eigen::Vector3d ev_tt = es_tt.eigenvalues();
  const bool is_degenerate =
      !(ev_tt(2) > 0.0) ||
      (ev_tt(0) < kMinHessian3dEig) ||
      (ev_tt(1) < kMinHessianPlaneEig);
  if (is_degenerate) {
    ++num_rejected_hessian_;
    LOG(I, "[resbie::pgo]   Loop rejected: degenerate Hessian eigenvalues [" << ev_tt(0) << ", "
           << ev_tt(1) << ", " << ev_tt(2) << "] (seed=" << seed_used << ")");
    logRejected(candidate, asserted, fit.fitness);
    return std::nullopt;
  }

  // Where the loop would move the current keyframe, with the loop keyframe held: the part of the
  // graph's estimate the loop contradicts. World z is gravity-aligned.
  const gtsam::Point3 shift = (pose_loop * asserted).translation() - pose_curr.translation();
  if (std::abs(shift.z()) > config_.loop_max_correction_z) {
    ++num_rejected_z_;
    LOG(I, "[resbie::pgo]   Loop rejected: vertical correction " << shift.z() << " m > "
           << config_.loop_max_correction_z << " m (seed=" << seed_used << ")");
    logRejected(candidate, asserted, fit.fitness);
    return std::nullopt;
  }
  LOG(I, "[resbie::pgo]   GICP verified (seed=" << seed_used << " overlap=" << fit.overlap
         << " fitness=" << fit.fitness << ")");
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
    num_loops_ = 0;
    num_rejected_gnc_ = 0;
    num_rejected_ = num_rejected_submap_ + num_rejected_icp_ + num_rejected_overlap_ +
                    num_rejected_fitness_ + num_rejected_hessian_ + num_rejected_z_;
    LOG(I, "[resbie::pgo] Loop summary: candidates=" << num_candidates_
           << " kept=" << num_loops_ << " rejected=" << num_rejected_
           << " (submap=" << num_rejected_submap_ << " icp=" << num_rejected_icp_
           << " overlap=" << num_rejected_overlap_ << " fitness=" << num_rejected_fitness_
           << " hessian=" << num_rejected_hessian_ << " z=" << num_rejected_z_
           << " gnc=" << num_rejected_gnc_ << ")");
    readEstimate(
        gtsam::LevenbergMarquardtOptimizer(base_graph_, isam_->calculateEstimate()).optimize());
    return;
  }

  // resbie: Graduated Non-Convexity (GNC) over loop candidates with base
  // graph (priors, odometry, CombinedImuFactor) marked as known inliers.
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
      const LoopEdge& edge = loop_edges_[k];
      LOG(I, "[resbie::pgo] GNC dropped outlier loop edge kf=" << edge.from << " -> kf=" << edge.to);
    }
  }
  num_loops_ = gnc_kept;
  num_rejected_gnc_ = gnc_dropped;
  num_rejected_ = num_rejected_submap_ + num_rejected_icp_ + num_rejected_overlap_ +
                  num_rejected_fitness_ + num_rejected_hessian_ + num_rejected_z_ +
                  num_rejected_gnc_;
  LOG(I, "[resbie::pgo] GNC finish: kept=" << gnc_kept << " dropped=" << gnc_dropped
         << " total_loops=" << loop_edges_.size());
  LOG(I, "[resbie::pgo] Loop summary: candidates=" << num_candidates_
         << " kept=" << num_loops_ << " rejected=" << num_rejected_
         << " (submap=" << num_rejected_submap_ << " icp=" << num_rejected_icp_
         << " overlap=" << num_rejected_overlap_ << " fitness=" << num_rejected_fitness_
         << " hessian=" << num_rejected_hessian_ << " z=" << num_rejected_z_
         << " gnc=" << num_rejected_gnc_ << ")");
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
  stats.num_candidates = num_candidates_;
  stats.num_loops = num_loops_;
  stats.num_rejected = num_rejected_;
  stats.num_rejected_submap = num_rejected_submap_;
  stats.num_rejected_icp = num_rejected_icp_;
  stats.num_rejected_overlap = num_rejected_overlap_;
  stats.num_rejected_fitness = num_rejected_fitness_;
  stats.num_rejected_hessian = num_rejected_hessian_;
  stats.num_rejected_z = num_rejected_z_;
  stats.num_rejected_gnc = num_rejected_gnc_;
  return stats;
}

}  // namespace bievr

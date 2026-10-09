#include "bievr_pgo/loop_closer.h"

#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/registration/icp.h>

#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/PriorFactor.h>

#include <cmath>
#include <iostream>
#include <limits>

#include "bievr_lio/utils.h"

namespace bievr {
namespace {

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
  gtsam::Vector6 prior_variance;
  prior_variance.setConstant(config_.prior_noise_score);
  prior_noise_ = gtsam::noiseModel::Diagonal::Variances(prior_variance);

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
}

void LoopCloser::addFrame(uint64_t stamp, const Transform& T_W_I, const Pointcloud& cloud_body) {
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
  previous_pose_ = T_W_I;
  has_previous_frame_ = true;
  if (!is_keyframe) return;

  translation_accumulated_ = 0.0;
  rotation_accumulated_ = 0.0;

  PendingFrame frame;
  frame.stamp = stamp;
  frame.pose = T_W_I;
  frame.cloud = Cloud::Ptr(new Cloud());
  const auto& data = cloud_body.data();
  frame.cloud->resize(data.cols());
  for (Eigen::Index i = 0; i < data.cols(); ++i) {
    frame.cloud->points[i] = PointT(static_cast<float>(data(0, i)), static_cast<float>(data(1, i)),
                                    static_cast<float>(data(2, i)));
  }

  // BIEVR's keyframe, loop-detect, ICP and iSAM2 workers, one pass each, in
  // that order, for this keyframe. detectLoop() queries the newest keyframe,
  // so each keyframe is queried exactly once.
  integrateKeyframe(std::move(frame));
  const ScanContext::Match match = scan_context_.detectLoop();
  if (match.index >= 0) {
    const int current = static_cast<int>(scan_context_.size()) - 1;
    const LoopCandidate candidate{match.index, current, match.yaw_diff_rad};
    if (const auto constraint = computeLoopConstraint(candidate)) {
      graph_.add(gtsam::BetweenFactor<gtsam::Pose3>(candidate.from, candidate.to, *constraint,
                                                    loop_noise_));
      ++num_loops_;
    } else {
      ++num_rejected_;
    }
  }
  optimize();
}

void LoopCloser::integrateKeyframe(PendingFrame frame) {
  Cloud::Ptr downsampled(new Cloud());
  pcl::VoxelGrid<PointT> filter;
  filter.setLeafSize(config_.keyframe_filter_size, config_.keyframe_filter_size,
                     config_.keyframe_filter_size);
  filter.setInputCloud(frame.cloud);
  filter.filter(*downsampled);

  const gtsam::Pose3 pose = toPose3(frame.pose);

  const int index = static_cast<int>(keyframe_clouds_.size());
  gtsam::Pose3 previous_odom_pose;
  if (index > 0) previous_odom_pose = keyframe_odom_poses_.back();
  keyframe_clouds_.push_back(downsampled);
  keyframe_odom_poses_.push_back(pose);
  keyframe_poses_.push_back(pose);
  keyframe_stamps_.push_back(frame.stamp);

  // Scan Context wants the body-frame cloud, at the same resolution the
  // original pipeline fed it.
  Eigen::Matrix<double, 3, Eigen::Dynamic> sc_cloud(3, downsampled->size());
  for (size_t i = 0; i < downsampled->size(); ++i) {
    const auto& p = downsampled->points[i];
    sc_cloud.col(i) << p.x, p.y, p.z;
  }
  scan_context_.add(sc_cloud);

  if (!graph_initialized_) {
    graph_.add(gtsam::PriorFactor<gtsam::Pose3>(0, pose, prior_noise_));
    initial_estimate_.insert(0, pose);
    graph_initialized_ = true;
  } else {
    graph_.add(gtsam::BetweenFactor<gtsam::Pose3>(index - 1, index,
                                                  previous_odom_pose.between(pose), odom_noise_));
    initial_estimate_.insert(index, pose);
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

  // Both clouds are in the world frame, so identity is the right start unless the
  // revisit runs in the opposite direction. Then Scan Context's yaw is lifted to
  // the world frame as P_loop * Rz(+-yaw) * P_curr^-1; the sign is not measured,
  // so both are tried. That best-of-two also weakens the fitness rejection below,
  // which is why it is off by default.
  std::vector<Eigen::Matrix4f> guesses;
  const double yaw = candidate.yaw_diff_rad;
  if (config_.use_sc_yaw_guess &&
      std::abs(yaw) >= std::max(deg2rad(config_.sc_yaw_guess_min_deg), 1e-6)) {
    for (const double sign : {-1.0, 1.0}) {
      const gtsam::Pose3 rotation(gtsam::Rot3::Yaw(sign * yaw), gtsam::Point3(0, 0, 0));
      guesses.push_back(toMatrix4f(pose_loop * rotation * pose_curr.inverse()));
    }
  } else {
    guesses.push_back(Eigen::Matrix4f::Identity());
  }

  bool converged = false;
  double best_fitness = std::numeric_limits<double>::max();
  Eigen::Matrix4f best_transformation = Eigen::Matrix4f::Identity();
  Cloud aligned;
  for (const auto& guess : guesses) {
    icp.align(aligned, guess);
    if (!icp.hasConverged()) continue;
    const double fitness = icp.getFitnessScore();
    if (fitness < best_fitness) {
      converged = true;
      best_fitness = fitness;
      best_transformation = icp.getFinalTransformation();
    }
  }

  if (!converged || best_fitness > config_.loop_fitness_score_threshold) return std::nullopt;

  // getFinalTransformation() includes the initial guess, and corrects the current
  // keyframe in the world frame: P_corrected = T_icp * P_curr. BetweenFactor
  // wants P_loop^-1 * P_corrected.
  const gtsam::Pose3 correction(best_transformation.cast<double>());
  return pose_loop.between(correction * pose_curr);
}

void LoopCloser::optimize() {
  if (!graph_initialized_ || (graph_.empty() && initial_estimate_.empty())) return;
  isam_->update(graph_, initial_estimate_);
  isam_->update();
  graph_.resize(0);
  initial_estimate_.clear();
  const gtsam::Values estimate = isam_->calculateEstimate();
  for (size_t i = 0; i < keyframe_poses_.size() && i < estimate.size(); ++i) {
    keyframe_poses_[i] = estimate.at<gtsam::Pose3>(i);
  }
}

std::vector<LoopCloser::Keyframe> LoopCloser::keyframes() const {
  std::vector<Keyframe> out(keyframe_poses_.size());
  for (size_t i = 0; i < keyframe_poses_.size(); ++i) {
    out[i].stamp = keyframe_stamps_[i];
    out[i].pose = toTransform(keyframe_poses_[i]);
  }
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

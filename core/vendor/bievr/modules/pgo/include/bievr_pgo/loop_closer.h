#ifndef BIEVR_PGO_LOOP_CLOSER_H_
#define BIEVR_PGO_LOOP_CLOSER_H_

// Scan Context loop closure + GTSAM pose graph, running downstream of the
// odometry. addFrame() runs keyframing, loop detection, ICP and iSAM2 inline,
// so a replay is deterministic.
// resbie: the graph also holds the IMU. Consecutive keyframes are linked by a
// CombinedImuFactor (preintegrated raw IMU, bias random walk) with gravity
// along world z, so roll and pitch are measured by gravity instead of being
// whatever the odometry drifted to; the first keyframe's prior fixes only
// yaw and position.

#include <cmath>
#include <deque>
#include <memory>
#include <optional>
#include <vector>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <gtsam/geometry/Pose3.h>
#include <gtsam/navigation/CombinedImuFactor.h>
#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>

#include "bievr_lio/common.h"
#include "bievr_scancontext/scan_context.h"

namespace bievr {

class LoopCloser {
 public:
  using PointT = pcl::PointXYZ;
  using Cloud = pcl::PointCloud<PointT>;
  // resbie: N x (3 or more) points, x, y, z first, read in place.
  using PointsView =
      Eigen::Ref<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;

  struct Config {
    // Keyframing: a frame is kept once either threshold is exceeded.
    double keyframe_meter_gap = 1.0;
    double keyframe_deg_gap = 15.0;

    double keyframe_filter_size = 0.4;  // voxel leaf of the stored keyframe cloud [m]
    double icp_filter_size = 0.4;       // voxel leaf of the loop-closure submap [m]

    ScanContext::Config scan_context;

    int history_keyframe_search_num = 25;  // [-N, N] keyframes stacked into the target submap
    double icp_max_correspondence_distance = 150.0;
    int icp_max_iterations = 100;
    double icp_transformation_epsilon = 1e-6;
    double icp_euclidean_fitness_epsilon = 1e-6;
    int icp_ransac_iterations = 0;
    double loop_fitness_score_threshold = 0.3;  // reject loops scoring above this
    // Reject a loop whose ICP result would move the current keyframe vertically by more than this [m].
    // A match to the wrong floor or stairwell flight needs a floor height (>= 2-3 m); true loops need
    // < 0.8 m on 99% of the handheld and drone sets. A constant on purpose: a bound that widened with
    // distance would let a one-floor-up return through. Platforms whose odometry sinks further between
    // loops (e.g. a vehicle) need a larger value.
    double loop_max_correction_z = 1.5;

    double prior_noise_score = 1e-12;
    double odom_noise_rotation = 1e-6;
    double odom_noise_translation = 1e-4;
    double loop_noise_score = 0.5;
    double loop_noise_cauchy_c = 1.0;

    double isam_relinearize_threshold = 0.01;
    int isam_relinearize_skip = 1;

    // IMU: per-sample noise, bias random walk per sqrt(s), initial bias
    // uncertainty (defaults: the front end's).
    Eigen::Vector3d acc_noise_std = Eigen::Vector3d::Constant(1.0);
    Eigen::Vector3d gyro_noise_std = Eigen::Vector3d::Constant(std::sqrt(0.1));
    double acc_bias_walk = 0.001;
    double gyro_bias_walk = 0.0001;
    Eigen::Vector3d acc_bias_init_std = Eigen::Vector3d::Constant(std::sqrt(0.2));
    Eigen::Vector3d gyro_bias_init_std = Eigen::Vector3d::Constant(std::sqrt(0.2));
  };

  struct Stats {
    size_t num_keyframes = 0;
    size_t num_loops = 0;      // loop factors accepted
    size_t num_rejected = 0;   // candidates that failed the fitness test
  };

  explicit LoopCloser(Config config);

  LoopCloser(const LoopCloser&) = delete;
  LoopCloser& operator=(const LoopCloser&) = delete;

  // Keyframe gating, then (for a keyframe) every stage inline. True if the
  // frame became a keyframe. Stamps strictly increase.
  bool addFrame(uint64_t stamp, const Transform& T_W_I, const PointsView& cloud_body);
  // One IMU sample in the IMU frame (acc in m/s^2). Stamps strictly increase;
  // a sample at or before the newest keyframe is dropped (its factor is built).
  void addImu(uint64_t stamp, const Eigen::Vector3d& acc, const Eigen::Vector3d& gyro);
  // End of input: one batch optimization of the whole graph.
  void finish();

  // The keyframes' current (optimized) poses, in keyframe order.
  std::vector<Transform> keyframePoses() const;
  Stats stats() const;

 private:
  struct ImuSample {
    uint64_t stamp;
    Eigen::Vector3d acc, gyro;
  };

  struct LoopCandidate {
    int from;  // older keyframe
    int to;    // newer keyframe
  };

  // Stores keyframe `cloud` (body frame) and adds its graph factors; `velocity`
  // is the odometry's, in the world frame.
  void integrateKeyframe(uint64_t stamp, const Transform& pose, const Eigen::Vector3d& velocity,
                         const Cloud::Ptr& cloud);
  // World-frame submap of keyframes within +-span of `key`, each at its own pose.
  Cloud::Ptr buildSubmap(int key, int span) const;
  // Relative constraint from ICP, or nullopt if the fitness or vertical-correction test rejects it.
  std::optional<gtsam::Pose3> computeLoopConstraint(const LoopCandidate& candidate);
  // Logs what a rejected (converged) ICP result asserted next to what the graph held.
  void logRejected(const LoopCandidate& candidate, const gtsam::Pose3& asserted, double fitness) const;
  void optimize();
  void initNoiseModels();
  // Adds the factors linking keyframe `index` to the previous one: odometry,
  // and the IMU preintegrated between their stamps.
  void addMotionFactors(int index, const gtsam::Pose3& odom_delta, uint64_t stamp,
                        const Eigen::Vector3d& velocity);
  void readEstimate(const gtsam::Values& estimate);
  boost::shared_ptr<gtsam::PreintegrationCombinedParams> makeImuParams(double dt) const;

  Config config_;
  double keyframe_rad_gap_;

  // Keyframe gating state.
  bool has_previous_frame_ = false;
  Transform previous_pose_;
  uint64_t previous_stamp_ = 0;
  double translation_accumulated_ = 0.0;
  double rotation_accumulated_ = 0.0;

  std::vector<Cloud::Ptr> keyframe_clouds_;
  std::vector<gtsam::Pose3> keyframe_odom_poses_;
  std::vector<gtsam::Pose3> keyframe_poses_;  // optimized
  std::vector<uint64_t> keyframe_stamps_;

  ScanContext scan_context_;

  gtsam::NonlinearFactorGraph graph_;
  gtsam::Values initial_estimate_;
  std::unique_ptr<gtsam::ISAM2> isam_;
  bool graph_initialized_ = false;

  gtsam::SharedNoiseModel odom_noise_;
  gtsam::SharedNoiseModel loop_noise_;

  std::deque<ImuSample> imu_;  // samples after the newest keyframe
  boost::shared_ptr<gtsam::PreintegrationCombinedParams> imu_params_;  // set at the 2nd keyframe
  gtsam::imuBias::ConstantBias bias_;  // newest keyframe's bias estimate

  size_t num_loops_ = 0;
  size_t num_rejected_ = 0;
};

}  // namespace bievr

#endif  // BIEVR_PGO_LOOP_CLOSER_H_

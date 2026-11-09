#pragma once

// The front end's configuration (the back end's is bievr_pgo/config_loader.h),
// one YAML schema organized by function, standard deviations in SI units
// throughout. Later documents override earlier ones per leaf. Defaults live
// here and only here (the Python package reads them from the binding).
//
//   calibration: translation, rotation          T_I_L, LiDAR -> IMU
//   lidar:  min_range_m, max_range_m, time_offset_s
//   imu:    acc_noise_std, gyro_noise_std,
//           acc_bias_init_std, gyro_bias_init_std, acc_bias_walk, gyro_bias_walk,
//           gravity_init_std_deg
//   spline: knot_hz, iterations, init_std, pos_noise, ort_noise,
//           new_pos_noise, new_ort_noise
//   points: downsample_m, informed_voxels, noise_std_m, huber_delta_m
//   map:    pixel_size_m, voxel_size_m, normal_tolerance_deg, max_size,
//           init_min_voxels
//
// Only what changes the results: how a run executes (threads, which points
// come out) are the front end's constructor arguments.

#include <cmath>
#include <iterator>
#include <string>
#include <vector>

#include "Association.h"
#include "bievr_lio/bievr_map.h"
#include "bievr_lio/config_loader.h"

namespace resbie {

struct Config {
  // calibration: T_I_L, LiDAR -> IMU (rotation row-major).
  bievr::Transform T_I_L = bievr::Transform::Identity();

  // lidar
  double min_range = 0.5;
  double max_range = 100.0;
  double time_offset = 0.0;  // s, added to every point time

  // imu (input in m/s^2 and rad/s). Noise as per-sample standard deviations;
  // biases drift as a random walk (sigma per sqrt(s)).
  Eigen::Vector3d acc_noise_std = Eigen::Vector3d::Constant(1.0);           // m/s^2
  Eigen::Vector3d gyro_noise_std = Eigen::Vector3d::Constant(std::sqrt(0.1));  // rad/s
  Eigen::Vector3d acc_bias_init_std = Eigen::Vector3d::Constant(std::sqrt(0.2));   // m/s^2
  Eigen::Vector3d gyro_bias_init_std = Eigen::Vector3d::Constant(std::sqrt(0.2));  // rad/s
  double acc_bias_walk = 0.001;    // m/s^2/sqrt(s)
  double gyro_bias_walk = 0.0001;  // rad/s/sqrt(s)
  // Gravity direction in the map frame: a constant state, learnt from the
  // start (prior std about the start's mean specific force). 0 fixes it there.
  double gravity_init_std_deg = 180.0 / M_PI;  // 1 rad: uninformative

  // spline: RESPLE's cubic B-spline and its iterated EKF. The motion model
  // adds, per knot, a random step of std noise * dt to every active control
  // point (new_*: the newest one), so the noises are velocity-like. The
  // orientation control points are quaternion logs (half the rotation angle),
  // so the *_ort_noise rates are half the rotation rate's std.
  int knot_hz = 100;
  int iterations = 4;  // filter updates per knot, each re-associating
  double init_std = std::sqrt(0.02);                   // initial control points, per knot interval
  double pos_noise = 0.1 * std::sqrt(0.5);             // m/s
  double ort_noise = 0.1 * std::sqrt(0.5);             // rad/s
  double new_pos_noise = 0.1;                          // m/s
  double new_ort_noise = 0.1;                          // rad/s

  // points: which LiDAR points the filter measures, and how it weighs them.
  double downsample = 0.25;         // m, voxel grid before informed sampling
  size_t informed_voxels = 300;     // most informative map voxels kept (all their points)
  double point_noise_std = 0.1;     // m, against the map surface
  MapAssociationConfig association; // huber_delta_m

  // map (BIEVR bump-image map)
  bievr::BIEVRMap::Config map{};
  size_t init_min_voxels = 100;  // distinct map voxels the seeding sweep must cover
};

// Every key loadConfig reads ("section.key", or top-level). Anything else in
// the YAML is an error.
inline constexpr const char* kKeys[] = {
    "calibration.translation", "calibration.rotation",
    "lidar.min_range_m", "lidar.max_range_m", "lidar.time_offset_s",
    "imu.acc_noise_std", "imu.gyro_noise_std", "imu.acc_bias_init_std",
    "imu.gyro_bias_init_std", "imu.acc_bias_walk", "imu.gyro_bias_walk", "imu.gravity_init_std_deg",
    "spline.knot_hz", "spline.iterations", "spline.init_std", "spline.pos_noise",
    "spline.ort_noise", "spline.new_pos_noise", "spline.new_ort_noise",
    "points.downsample_m", "points.informed_voxels", "points.noise_std_m",
    "points.huber_delta_m",
    "map.pixel_size_m", "map.voxel_size_m", "map.normal_tolerance_deg",
    "map.max_size", "map.init_min_voxels",
};

// `documents`: YAML texts, later ones winning per key. False (logged) on an
// unknown key, a missing calibration or a value out of range.
inline bool loadConfig(const std::vector<std::string>& documents, Config& c) {
  namespace ci = bievr::config_internal;
  using ci::Min;
  ci::MergedYaml yaml;
  // Unknown keys are errors: a misspelled or outdated key would otherwise
  // silently fall back to its default.
  if (!yaml.load(documents) || !yaml.onlyKeys(std::begin(kKeys), std::end(kKeys))) return false;

  std::vector<double> translation, rotation;
  bool ok = ci::read(yaml, "calibration.translation", translation) &&
            ci::read(yaml, "calibration.rotation", rotation) &&
            ci::extrinsicFromVectors(translation, rotation, "calibration (T_I_L)", c.T_I_L);

  ok &= ci::read(yaml, "lidar.min_range_m", c.min_range, Min::kZero);
  ok &= ci::read(yaml, "lidar.max_range_m", c.max_range, Min::kPositive);
  ok &= ci::read(yaml, "lidar.time_offset_s", c.time_offset);
  if (!(c.max_range > c.min_range)) {
    LOG(E, "Config error: 'lidar.max_range_m' must be > 'lidar.min_range_m'.");
    ok = false;
  }

  ok &= ci::read(yaml, "imu.acc_noise_std", c.acc_noise_std, Min::kPositive);
  ok &= ci::read(yaml, "imu.gyro_noise_std", c.gyro_noise_std, Min::kPositive);
  ok &= ci::read(yaml, "imu.acc_bias_init_std", c.acc_bias_init_std, Min::kPositive);
  ok &= ci::read(yaml, "imu.gyro_bias_init_std", c.gyro_bias_init_std, Min::kPositive);
  ok &= ci::read(yaml, "imu.acc_bias_walk", c.acc_bias_walk, Min::kZero);
  ok &= ci::read(yaml, "imu.gyro_bias_walk", c.gyro_bias_walk, Min::kZero);
  ok &= ci::read(yaml, "imu.gravity_init_std_deg", c.gravity_init_std_deg, Min::kZero);

  ok &= ci::read(yaml, "spline.knot_hz", c.knot_hz, Min::kPositive);
  ok &= ci::read(yaml, "spline.iterations", c.iterations, Min::kPositive);
  ok &= ci::read(yaml, "spline.init_std", c.init_std, Min::kPositive);
  ok &= ci::read(yaml, "spline.pos_noise", c.pos_noise, Min::kPositive);
  ok &= ci::read(yaml, "spline.ort_noise", c.ort_noise, Min::kPositive);
  ok &= ci::read(yaml, "spline.new_pos_noise", c.new_pos_noise, Min::kPositive);
  ok &= ci::read(yaml, "spline.new_ort_noise", c.new_ort_noise, Min::kPositive);

  ok &= ci::read(yaml, "points.downsample_m", c.downsample, Min::kPositive);
  ok &= ci::read(yaml, "points.informed_voxels", c.informed_voxels, Min::kPositive);
  ok &= ci::read(yaml, "points.noise_std_m", c.point_noise_std, Min::kPositive);
  ok &= ci::read(yaml, "points.huber_delta_m", c.association.huber_delta, Min::kPositive);

  ok &= ci::read(yaml, "map.pixel_size_m", c.map.px_size, Min::kPositive);
  ok &= ci::read(yaml, "map.voxel_size_m", c.map.voxel_size, Min::kPositive);
  ok &= ci::read(yaml, "map.normal_tolerance_deg", c.map.norm_tol_deg, Min::kPositive);
  ok &= ci::read(yaml, "map.max_size", c.map.max_size, Min::kPositive);
  ok &= ci::read(yaml, "map.init_min_voxels", c.init_min_voxels, Min::kPositive);
  return ok;
}

}  // namespace resbie

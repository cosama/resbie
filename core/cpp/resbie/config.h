#pragma once

// resbie configuration, one YAML schema organized by function, standard
// deviations in SI units throughout. Later files override earlier ones per
// leaf. Defaults live here and only here (the Python package reads them from
// the binding).
//
//   calibration: translation, rotation          T_I_L, LiDAR -> IMU
//   lidar:  min_range_m, max_range_m, time_offset_s
//   imu:    normalized, acc_noise_std, gyro_noise_std,
//           acc_bias_init_std, gyro_bias_init_std, acc_bias_walk, gyro_bias_walk
//   spline: knot_hz, iterations, init_std, pos_noise, ort_noise,
//           new_pos_noise, new_ort_noise
//   points: downsample_m, informed_voxels, per_update, noise_std_m,
//           huber_delta_m, gate_sigma
//   map:    pixel_size_m, voxel_size_m, normal_tolerance_deg, smooth,
//           weighted, max_size, init_min_voxels
//   loop_closure: enable, max_speed_mps, BIEVR's keyframe / Scan Context /
//           ICP / pose-graph keys (bievr_pgo/config_loader.h)
//   max_num_threads

#include <cmath>
#include <string>
#include <vector>

#include <algorithm>
#include <iterator>

#include "Association.h"
#include "bievr_lio/bievr_map.h"
#include "bievr_lio/config_loader.h"
#include "bievr_pgo/config_loader.h"

namespace resbie {

struct Config {
  // calibration: T_I_L, LiDAR -> IMU (rotation row-major).
  bievr::Transform T_I_L = bievr::Transform::Identity();

  // lidar
  double min_range = 0.5;
  double max_range = 100.0;
  double time_offset = 0.0;  // s, added to every point time

  // imu. Noise as per-sample standard deviations; biases drift as a random
  // walk (sigma per sqrt(s)).
  double imu_normalized = -1.0;  // < 0 autodetect, 0 m/s^2, > 0 in g
  Eigen::Vector3d acc_noise_std = Eigen::Vector3d::Constant(1.0);           // m/s^2
  Eigen::Vector3d gyro_noise_std = Eigen::Vector3d::Constant(std::sqrt(0.1));  // rad/s
  Eigen::Vector3d acc_bias_init_std = Eigen::Vector3d::Constant(std::sqrt(0.2));   // m/s^2
  Eigen::Vector3d gyro_bias_init_std = Eigen::Vector3d::Constant(std::sqrt(0.2));  // rad/s
  double acc_bias_walk = 0.001;    // m/s^2/sqrt(s)
  double gyro_bias_walk = 0.0001;  // rad/s/sqrt(s)

  // spline: RESPLE's cubic B-spline and its iterated EKF. The motion model
  // adds, per knot, a random step of std noise * dt to every active control
  // point (new_*: the newest one), so the noises are velocity-like.
  int knot_hz = 100;
  int iterations = 3;
  double init_std = std::sqrt(0.02);                   // initial control points, per knot interval
  double pos_noise = 0.1 * std::sqrt(0.5);             // m/s
  double ort_noise = 0.1 * std::sqrt(0.5);             // rad/s
  double new_pos_noise = 0.1;                          // m/s
  double new_ort_noise = 0.1;                          // rad/s

  // points: which LiDAR points the filter measures, and how it weighs them.
  double downsample = 0.25;         // m, voxel grid before informed sampling
  size_t informed_voxels = 300;     // most informative map voxels kept (all their points)
  int per_update = 100;             // points per filter update (never past the next knot)
  double point_noise_std = 0.1;     // m, against the map surface
  MapAssociationConfig association; // huber_delta_m
  double gate_sigma = 3.0;          // Mahalanobis gate on the predicted residual

  // map (BIEVR bump-image map)
  bievr::BIEVRMap::Config map = [] {
    bievr::BIEVRMap::Config m;
    m.px_size = 0.05;
    m.voxel_size = 0.5;
    m.norm_tol_deg = 3.0;
    m.max_size = 1500000;
    m.smooth = true;
    m.weighted = true;
    return m;
  }();
  size_t init_min_voxels = 100;  // distinct map voxels the seeding sweep must cover

  // Loop closer guard: a sweep faster than this never becomes a keyframe
  // (a diverged run otherwise stalls the synchronous closer).
  double loop_closure_max_speed_mps = 30.0;

  int max_num_threads = 0;  // 0: all cores
  std::vector<std::string> yaml_paths;
};

// Every key loadConfig reads ("section.key", or top-level). Together with
// bievr::kLoopClosureKeys, anything else in the YAML is an error.
inline constexpr const char* kKeys[] = {
    "calibration.translation", "calibration.rotation",
    "lidar.min_range_m", "lidar.max_range_m", "lidar.time_offset_s",
    "imu.normalized", "imu.acc_noise_std", "imu.gyro_noise_std", "imu.acc_bias_init_std",
    "imu.gyro_bias_init_std", "imu.acc_bias_walk", "imu.gyro_bias_walk",
    "spline.knot_hz", "spline.iterations", "spline.init_std", "spline.pos_noise",
    "spline.ort_noise", "spline.new_pos_noise", "spline.new_ort_noise",
    "points.downsample_m", "points.informed_voxels", "points.per_update", "points.noise_std_m",
    "points.huber_delta_m", "points.gate_sigma",
    "map.pixel_size_m", "map.voxel_size_m", "map.normal_tolerance_deg", "map.smooth",
    "map.weighted", "map.max_size", "map.init_min_voxels",
    "loop_closure.max_speed_mps",
    "max_num_threads",
};

namespace detail {

// A number (all axes) or three numbers (x, y, z).
inline bool getVector3(const bievr::config_internal::MergedYaml& yaml, const std::string& section,
                       const std::string& key, Eigen::Vector3d& out) {
  const YAML::Node node = yaml.node(section, key);
  if (!node || node.IsNull()) return true;
  try {
    if (node.IsScalar()) {
      out.setConstant(node.as<double>());
    } else if (node.IsSequence() && node.size() == 3) {
      out = Eigen::Vector3d(node[0].as<double>(), node[1].as<double>(), node[2].as<double>());
    } else {
      throw YAML::Exception(YAML::Mark::null_mark(), "expected a number or 3 numbers");
    }
  } catch (const std::exception& e) {
    LOG(E, "Config error: '" << section << "." << key << "': " << e.what());
    return false;
  }
  if ((out.array() < 0).any()) {
    LOG(E, "Config error: '" << section << "." << key << "' must be >= 0.");
    return false;
  }
  return true;
}

inline bool getNonNegative(const bievr::config_internal::MergedYaml& yaml, const std::string& section,
                           const std::string& key, double& out) {
  out = yaml.get<double>(section, key, out);
  if (out < 0) {
    LOG(E, "Config error: '" << section << "." << key << "' must be >= 0.");
    return false;
  }
  return true;
}

}  // namespace detail

inline bool loadConfig(const std::vector<std::string>& yaml_paths, Config& c) {
  namespace ci = bievr::config_internal;
  ci::MergedYaml yaml;
  for (const std::string& path : yaml_paths) {
    if (path.empty()) continue;
    try {
      yaml.add(YAML::LoadFile(path));
    } catch (const std::exception& e) {
      LOG(E, "Failed to load YAML config '" << path << "': " << e.what());
      return false;
    }
    c.yaml_paths.push_back(path);
  }

  // Unknown keys are errors: a misspelled or outdated key would otherwise
  // silently fall back to its default.
  {
    std::vector<std::string> known(std::begin(kKeys), std::end(kKeys));
    for (const char* k : bievr::kLoopClosureKeys) known.push_back(std::string("loop_closure.") + k);
    bool ok = true;
    for (const std::string& key : yaml.keys()) {
      if (std::find(known.begin(), known.end(), key) == known.end()) {
        LOG(E, "Config error: unknown key '" << key << "'.");
        ok = false;
      }
    }
    if (!ok) return false;
  }

  // --- calibration ---
  const std::vector<double> t_vec =
      yaml.get<std::vector<double>>("calibration", "translation", {});
  const std::vector<double> R_vec = yaml.get<std::vector<double>>("calibration", "rotation", {});
  if (!ci::extrinsicFromVectors(t_vec, R_vec, "calibration (T_I_L)", c.T_I_L)) return false;

  // --- lidar ---
  c.min_range = yaml.get<double>("lidar", "min_range_m", c.min_range);
  c.max_range = yaml.get<double>("lidar", "max_range_m", c.max_range);
  c.time_offset = yaml.get<double>("lidar", "time_offset_s", c.time_offset);

  // --- imu ---
  c.imu_normalized = yaml.get<double>("imu", "normalized", c.imu_normalized);
  if (!detail::getVector3(yaml, "imu", "acc_noise_std", c.acc_noise_std) ||
      !detail::getVector3(yaml, "imu", "gyro_noise_std", c.gyro_noise_std) ||
      !detail::getVector3(yaml, "imu", "acc_bias_init_std", c.acc_bias_init_std) ||
      !detail::getVector3(yaml, "imu", "gyro_bias_init_std", c.gyro_bias_init_std) ||
      !detail::getNonNegative(yaml, "imu", "acc_bias_walk", c.acc_bias_walk) ||
      !detail::getNonNegative(yaml, "imu", "gyro_bias_walk", c.gyro_bias_walk)) {
    return false;
  }

  // --- spline ---
  if (!ci::getPositive(yaml, "spline", "knot_hz", c.knot_hz, c.knot_hz) ||
      !ci::getPositive(yaml, "spline", "iterations", c.iterations, c.iterations) ||
      !ci::getPositive(yaml, "spline", "init_std", c.init_std, c.init_std) ||
      !ci::getPositive(yaml, "spline", "pos_noise", c.pos_noise, c.pos_noise) ||
      !ci::getPositive(yaml, "spline", "ort_noise", c.ort_noise, c.ort_noise) ||
      !ci::getPositive(yaml, "spline", "new_pos_noise", c.new_pos_noise, c.new_pos_noise) ||
      !ci::getPositive(yaml, "spline", "new_ort_noise", c.new_ort_noise, c.new_ort_noise)) {
    return false;
  }
  if (1000000000 % c.knot_hz != 0) {
    LOG(W, "spline.knot_hz " << c.knot_hz << " does not divide 1 s; knot spacing is rounded.");
  }

  // --- points ---
  int informed = static_cast<int>(c.informed_voxels);
  if (!ci::getPositive(yaml, "points", "downsample_m", c.downsample, c.downsample) ||
      !ci::getPositive(yaml, "points", "informed_voxels", informed, informed) ||
      !ci::getPositive(yaml, "points", "per_update", c.per_update, c.per_update) ||
      !ci::getPositive(yaml, "points", "noise_std_m", c.point_noise_std, c.point_noise_std) ||
      !ci::getPositive(yaml, "points", "huber_delta_m", c.association.huber_delta,
                       c.association.huber_delta) ||
      !ci::getPositive(yaml, "points", "gate_sigma", c.gate_sigma, c.gate_sigma)) {
    return false;
  }
  c.informed_voxels = static_cast<size_t>(informed);

  // --- map ---
  int max_size = static_cast<int>(c.map.max_size);
  if (!ci::getPositive(yaml, "map", "pixel_size_m", c.map.px_size, c.map.px_size) ||
      !ci::getPositive(yaml, "map", "voxel_size_m", c.map.voxel_size, c.map.voxel_size) ||
      !ci::getPositive(yaml, "map", "normal_tolerance_deg", c.map.norm_tol_deg,
                       c.map.norm_tol_deg) ||
      !ci::getPositive(yaml, "map", "max_size", max_size, max_size)) {
    return false;
  }
  c.map.max_size = static_cast<size_t>(max_size);
  c.map.smooth = yaml.get<bool>("map", "smooth", c.map.smooth);
  c.map.weighted = yaml.get<bool>("map", "weighted", c.map.weighted);
  int min_init = static_cast<int>(c.init_min_voxels);
  if (!ci::getPositive(yaml, "map", "init_min_voxels", min_init, min_init)) return false;
  c.init_min_voxels = static_cast<size_t>(min_init);

  // --- loop_closure (the rest is read by bievr_pgo/config_loader.h) ---
  if (!ci::getPositive(yaml, "loop_closure", "max_speed_mps", c.loop_closure_max_speed_mps,
                       c.loop_closure_max_speed_mps)) {
    return false;
  }

  c.max_num_threads = yaml.getTopLevel<int>("max_num_threads", c.max_num_threads);
  if (c.max_num_threads < 0) {
    LOG(E, "Config error: 'max_num_threads' must be >= 0.");
    return false;
  }
  return true;
}

}  // namespace resbie

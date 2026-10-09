#pragma once

// resbie configuration: BIEVR's sectioned YAML (calibration, lidar, map,
// preprocess, optimization, imu, loop_closure, read with BIEVR's own helpers
// and key names) plus a `resple:` section holding RESPLE's estimator
// parameters under RESPLE's own names. Later files override earlier ones per
// leaf, as in BIEVR.

#include <cmath>
#include <string>
#include <vector>

#include "Association.h"
#include "bievr_lio/bievr_map.h"
#include "bievr_lio/config_loader.h"

namespace resbie {

struct Config {
  // calibration: T_I_L, LiDAR -> IMU (BIEVR convention, rotation row-major).
  bievr::Transform T_I_L = bievr::Transform::Identity();

  // lidar
  double min_range = 0.5;
  double max_range = 100.0;
  double time_offset = 0.0;  // s, added to every point time (RESPLE lidar_time_offset)

  // map (BIEVR bump-image map)
  bievr::BIEVRMap::Config map;
  size_t min_voxels_for_init = 100;  // distinct map voxels the seeding sweep must cover

  // preprocess (BIEVR point selection)
  double downsample_resolution = 0.1;
  size_t informed_sample_count = 300;

  // optimization (BIEVR residual)
  MapAssociationConfig association;

  // imu
  double imu_normalized = -1.0;  // < 0 autodetect, 0 m/s^2, > 0 in g (BIEVR imu.normalized)

  // resple (spline filter)
  int knot_hz = 100;
  double cov_P0 = 0.02;
  double cov_RCP_pos_old = 0.5;
  double cov_RCP_ort_old = 0.5;
  double cov_RCP_pos_new = 1.0;
  double cov_RCP_ort_new = 1.0;
  double std_sys_pos = 0.1;
  double std_sys_ort = 0.1;
  Eigen::Vector3d cov_acc = Eigen::Vector3d::Constant(1.0);
  Eigen::Vector3d cov_gyro = Eigen::Vector3d::Constant(0.1);
  Eigen::Vector3d cov_ba = Eigen::Vector3d::Constant(0.2);
  Eigen::Vector3d cov_bg = Eigen::Vector3d::Constant(0.2);
  int n_iter = 3;
  int num_points_upd = 100;
  double w_pt = 0.01;      // point measurement variance, m^2
  // Bias random walk, sigma per sqrt(s). RESPLE has none (biases only ever
  // get more certain, so a bias corrupted in a hard area stays corrupted).
  double bias_rw_acc = 0.001;   // m/s^2/sqrt(s)
  double bias_rw_gyro = 0.0001; // rad/s/sqrt(s)
  // Mahalanobis point gate zp^2 <= gate_chi2 (HPH' + R): 9 = 3 sigma.
  double gate_chi2 = 9.0;

  // Loop closer guard: a sweep faster than this never becomes a keyframe
  // (a diverged run otherwise stalls the synchronous closer).
  double loop_closure_max_speed_mps = 30.0;

  int max_num_threads = 0;  // 0: all cores
  std::vector<std::string> yaml_paths;
};

namespace detail {

inline bool getVector3(const bievr::config_internal::MergedYaml& yaml, const std::string& section,
                       const std::string& key, Eigen::Vector3d& out) {
  const std::vector<double> d(out.data(), out.data() + 3);
  const std::vector<double> v = yaml.get<std::vector<double>>(section, key, d);
  if (v.size() != 3) {
    LOG(E, "Config error: '" << section << "." << key << "' must have 3 elements.");
    return false;
  }
  out = Eigen::Vector3d(v[0], v[1], v[2]);
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

  // --- calibration ---
  const std::vector<double> t_vec =
      yaml.get<std::vector<double>>("calibration", "translation", {});
  const std::vector<double> R_vec = yaml.get<std::vector<double>>("calibration", "rotation", {});
  if (!ci::extrinsicFromVectors(t_vec, R_vec, "calibration (T_I_L)", c.T_I_L)) return false;

  // --- lidar ---
  c.min_range = yaml.get<double>("lidar", "min_range_m", c.min_range);
  c.max_range = yaml.get<double>("lidar", "max_range_m", c.max_range);
  c.time_offset = yaml.get<double>("lidar", "time_offset_s", c.time_offset);

  // --- map ---
  int max_size = static_cast<int>(c.map.max_size);
  if (!ci::getPositive(yaml, "map", "pixel_size_m", 0.05, c.map.px_size) ||
      !ci::getPositive(yaml, "map", "voxel_size_m", 0.5, c.map.voxel_size) ||
      !ci::getPositive(yaml, "map", "normal_tolerance_deg", 3., c.map.norm_tol_deg) ||
      !ci::getPositive(yaml, "map", "max_size", 1500000, max_size)) {
    return false;
  }
  c.map.max_size = static_cast<size_t>(max_size);
  c.map.smooth = yaml.get<bool>("map", "smooth", true);
  c.map.weighted = yaml.get<bool>("map", "weighted", true);
  int min_init = static_cast<int>(c.min_voxels_for_init);
  if (!ci::getPositive(yaml, "map", "min_voxels_for_init", min_init, min_init)) return false;
  c.min_voxels_for_init = static_cast<size_t>(min_init);

  // --- preprocess ---
  if (!ci::getPositive(yaml, "preprocess", "downsample_resolution_m", c.downsample_resolution,
                       c.downsample_resolution)) {
    return false;
  }
  int n_informed = static_cast<int>(c.informed_sample_count);
  if (!ci::getPositive(yaml, "preprocess", "informed_sample_count", n_informed, n_informed)) {
    return false;
  }
  c.informed_sample_count = static_cast<size_t>(n_informed);

  // --- optimization ---
  if (!ci::getPositive(yaml, "optimization", "huber_delta", c.association.huber_delta,
                       c.association.huber_delta)) {
    return false;
  }

  // --- imu ---
  c.imu_normalized = yaml.get<double>("imu", "normalized", c.imu_normalized);

  // --- resple ---
  if (!ci::getPositive(yaml, "resple", "knot_hz", c.knot_hz, c.knot_hz) ||
      !ci::getPositive(yaml, "resple", "cov_P0", c.cov_P0, c.cov_P0) ||
      !ci::getPositive(yaml, "resple", "cov_RCP_pos_old", c.cov_RCP_pos_old, c.cov_RCP_pos_old) ||
      !ci::getPositive(yaml, "resple", "cov_RCP_ort_old", c.cov_RCP_ort_old, c.cov_RCP_ort_old) ||
      !ci::getPositive(yaml, "resple", "cov_RCP_pos_new", c.cov_RCP_pos_new, c.cov_RCP_pos_new) ||
      !ci::getPositive(yaml, "resple", "cov_RCP_ort_new", c.cov_RCP_ort_new, c.cov_RCP_ort_new) ||
      !ci::getPositive(yaml, "resple", "std_sys_pos", c.std_sys_pos, c.std_sys_pos) ||
      !ci::getPositive(yaml, "resple", "std_sys_ort", c.std_sys_ort, c.std_sys_ort) ||
      !ci::getPositive(yaml, "resple", "n_iter", c.n_iter, c.n_iter) ||
      !ci::getPositive(yaml, "resple", "num_points_upd", c.num_points_upd, c.num_points_upd) ||
      !ci::getPositive(yaml, "resple", "w_pt", c.w_pt, c.w_pt)) {
    return false;
  }
  if (!detail::getVector3(yaml, "resple", "cov_acc", c.cov_acc) ||
      !detail::getVector3(yaml, "resple", "cov_gyro", c.cov_gyro) ||
      !detail::getVector3(yaml, "resple", "cov_ba", c.cov_ba) ||
      !detail::getVector3(yaml, "resple", "cov_bg", c.cov_bg)) {
    return false;
  }
  c.bias_rw_acc = yaml.get<double>("resple", "bias_rw_acc", c.bias_rw_acc);
  c.bias_rw_gyro = yaml.get<double>("resple", "bias_rw_gyro", c.bias_rw_gyro);
  if (c.bias_rw_acc < 0 || c.bias_rw_gyro < 0) {
    LOG(E, "Config error: resple.bias_rw_* must be >= 0.");
    return false;
  }
  if (!ci::getPositive(yaml, "resple", "gate_chi2", c.gate_chi2, c.gate_chi2) ||
      !ci::getPositive(yaml, "loop_closure", "max_speed_mps", c.loop_closure_max_speed_mps,
                       c.loop_closure_max_speed_mps)) {
    return false;
  }

  if (1000000000 % c.knot_hz != 0) {
    LOG(W, "resple.knot_hz " << c.knot_hz << " does not divide 1 s; knot spacing is rounded.");
  }

  c.max_num_threads = yaml.getTopLevel<int>("max_num_threads", c.max_num_threads);
  if (c.max_num_threads < 0) {
    LOG(E, "Config error: 'max_num_threads' must be >= 0.");
    return false;
  }
  return true;
}

}  // namespace resbie

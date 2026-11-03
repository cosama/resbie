#ifndef BIEVR_PGO_CONFIG_LOADER_H_
#define BIEVR_PGO_CONFIG_LOADER_H_

// resbie: the back end's configuration, its own YAML documents with flat keys
// (the front end's is resbie/config.h). Only the keys the synchronous closer
// uses (no publishing, GPS, worker rates or map bundle). Header-only;
// consumers need yaml-cpp on their include path.

#include <bievr_lio/config_loader.h>

#include <iterator>
#include <string>
#include <vector>

#include "bievr_pgo/loop_closer.h"

namespace bievr {

// Every key loadLoopClosureConfig reads; anything else is an error.
inline constexpr const char* kLoopClosureKeys[] = {
    "keyframe_meter_gap",
    "keyframe_deg_gap",
    "keyframe_filter_size",
    "icp_filter_size",
    "sc_num_rings",
    "sc_num_sectors",
    "sc_max_radius",
    "sc_lidar_height",
    "sc_num_exclude_recent",
    "sc_num_candidates",
    "sc_search_ratio",
    "sc_dist_threshold",
    "sc_tree_making_period",
    "history_keyframe_search_num",
    "icp_max_correspondence_distance",
    "icp_max_iterations",
    "icp_transformation_epsilon",
    "icp_euclidean_fitness_epsilon",
    "icp_ransac_iterations",
    "loop_fitness_score_threshold",
    "use_sc_yaw_guess",
    "sc_yaw_guess_min_deg",
    "loop_max_correction_z",
    "prior_noise_score",
    "odom_noise_rotation",
    "odom_noise_translation",
    "loop_noise_score",
    "loop_noise_cauchy_c",
    "isam_relinearize_threshold",
    "isam_relinearize_skip",
    "acc_noise_std",
    "gyro_noise_std",
    "acc_bias_walk",
    "gyro_bias_walk",
    "acc_bias_init_std",
    "gyro_bias_init_std",
};

// `documents`: YAML texts, later ones winning per key. False (logged) on an
// unknown key or a value out of range.
inline bool loadLoopClosureConfig(const std::vector<std::string>& documents,
                                  LoopCloser::Config& c) {
  namespace ci = config_internal;
  using ci::Min;
  ci::MergedYaml yaml;
  if (!yaml.load(documents) ||
      !yaml.onlyKeys(std::begin(kLoopClosureKeys), std::end(kLoopClosureKeys))) {
    return false;
  }
  bool ok = true;
  ok &= ci::read(yaml, "keyframe_meter_gap", c.keyframe_meter_gap, Min::kZero);
  ok &= ci::read(yaml, "keyframe_deg_gap", c.keyframe_deg_gap, Min::kZero);
  ok &= ci::read(yaml, "keyframe_filter_size", c.keyframe_filter_size, Min::kPositive);
  ok &= ci::read(yaml, "icp_filter_size", c.icp_filter_size, Min::kPositive);

  auto& sc = c.scan_context;
  ok &= ci::read(yaml, "sc_num_rings", sc.num_rings, Min::kPositive);
  ok &= ci::read(yaml, "sc_num_sectors", sc.num_sectors, Min::kPositive);
  ok &= ci::read(yaml, "sc_max_radius", sc.max_radius, Min::kPositive);
  ok &= ci::read(yaml, "sc_lidar_height", sc.lidar_height);
  ok &= ci::read(yaml, "sc_num_exclude_recent", sc.num_exclude_recent, Min::kZero);
  ok &= ci::read(yaml, "sc_num_candidates", sc.num_candidates, Min::kPositive);
  ok &= ci::read(yaml, "sc_search_ratio", sc.search_ratio, Min::kPositive);
  ok &= ci::read(yaml, "sc_dist_threshold", sc.dist_threshold, Min::kPositive);
  ok &= ci::read(yaml, "sc_tree_making_period", sc.tree_making_period, Min::kPositive);

  ok &= ci::read(yaml, "history_keyframe_search_num", c.history_keyframe_search_num, Min::kZero);
  ok &= ci::read(yaml, "icp_max_correspondence_distance", c.icp_max_correspondence_distance, Min::kPositive);
  ok &= ci::read(yaml, "icp_max_iterations", c.icp_max_iterations, Min::kPositive);
  ok &= ci::read(yaml, "icp_transformation_epsilon", c.icp_transformation_epsilon, Min::kZero);
  ok &= ci::read(yaml, "icp_euclidean_fitness_epsilon", c.icp_euclidean_fitness_epsilon, Min::kZero);
  ok &= ci::read(yaml, "icp_ransac_iterations", c.icp_ransac_iterations, Min::kZero);
  ok &= ci::read(yaml, "loop_fitness_score_threshold", c.loop_fitness_score_threshold, Min::kPositive);
  ok &= ci::read(yaml, "use_sc_yaw_guess", c.use_sc_yaw_guess);
  ok &= ci::read(yaml, "sc_yaw_guess_min_deg", c.sc_yaw_guess_min_deg, Min::kZero);
  ok &= ci::read(yaml, "loop_max_correction_z", c.loop_max_correction_z, Min::kPositive);

  ok &= ci::read(yaml, "prior_noise_score", c.prior_noise_score, Min::kPositive);
  ok &= ci::read(yaml, "odom_noise_rotation", c.odom_noise_rotation, Min::kPositive);
  ok &= ci::read(yaml, "odom_noise_translation", c.odom_noise_translation, Min::kPositive);
  ok &= ci::read(yaml, "loop_noise_score", c.loop_noise_score, Min::kPositive);
  ok &= ci::read(yaml, "loop_noise_cauchy_c", c.loop_noise_cauchy_c, Min::kPositive);

  ok &= ci::read(yaml, "isam_relinearize_threshold", c.isam_relinearize_threshold, Min::kZero);
  ok &= ci::read(yaml, "isam_relinearize_skip", c.isam_relinearize_skip, Min::kPositive);

  // The graph's IMU model, per-sample stds: one number (every axis) or three.
  ok &= ci::read(yaml, "acc_noise_std", c.acc_noise_std, Min::kPositive);
  ok &= ci::read(yaml, "gyro_noise_std", c.gyro_noise_std, Min::kPositive);
  ok &= ci::read(yaml, "acc_bias_init_std", c.acc_bias_init_std, Min::kPositive);
  ok &= ci::read(yaml, "gyro_bias_init_std", c.gyro_bias_init_std, Min::kPositive);
  ok &= ci::read(yaml, "acc_bias_walk", c.acc_bias_walk, Min::kZero);
  ok &= ci::read(yaml, "gyro_bias_walk", c.gyro_bias_walk, Min::kZero);
  return ok;
}

}  // namespace bievr

#endif  // BIEVR_PGO_CONFIG_LOADER_H_

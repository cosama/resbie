#ifndef BIEVR_PGO_CONFIG_LOADER_H_
#define BIEVR_PGO_CONFIG_LOADER_H_

// Parses the `loop_closure:` section of the same YAML files bievr_lio loads, so
// the module configures itself without the core knowing it exists. Header-only;
// consumers need yaml-cpp on their include path (as with bievr_lio's loader).

#include <bievr_lio/config_loader.h>

#include <string>
#include <vector>

#include "bievr_pgo/loop_closer.h"

namespace bievr {

// resbie: only the keys the synchronous closer uses (no publishing, GPS,
// worker rates or map bundle).
struct LoopClosureConfig {
  bool enable = true;  // resbie: the graph (gravity, loops) runs by default
  LoopCloser::Config closer;
};

// resbie: every loop_closure key the loader below reads (for strict
// validation of the YAML; keep in step with the loader).
inline constexpr const char* kLoopClosureKeys[] = {
    "enable",
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
    "prior_noise_score",
    "odom_noise_rotation",
    "odom_noise_translation",
    "loop_noise_score",
    "loop_noise_cauchy_c",
    "isam_relinearize_threshold",
    "isam_relinearize_skip",
};

inline bool loadLoopClosureConfig(const std::vector<std::string>& yaml_paths,
                                  LoopClosureConfig& config) {
  config_internal::MergedYaml yaml;
  for (const std::string& path : yaml_paths) {
    if (path.empty()) continue;
    try {
      yaml.add(YAML::LoadFile(path));
    } catch (const std::exception& e) {
      LOG(E, "Failed to load YAML config '" << path << "': " << e.what());
      return false;
    }
  }

  const std::string s = "loop_closure";
  auto& c = config.closer;

  config.enable = yaml.get<bool>(s, "enable", config.enable);
  if (!config.enable) return true;

  c.keyframe_meter_gap = yaml.get<double>(s, "keyframe_meter_gap", c.keyframe_meter_gap);
  c.keyframe_deg_gap = yaml.get<double>(s, "keyframe_deg_gap", c.keyframe_deg_gap);
  c.keyframe_filter_size = yaml.get<double>(s, "keyframe_filter_size", c.keyframe_filter_size);
  c.icp_filter_size = yaml.get<double>(s, "icp_filter_size", c.icp_filter_size);

  auto& sc = c.scan_context;
  sc.num_rings = yaml.get<int>(s, "sc_num_rings", sc.num_rings);
  sc.num_sectors = yaml.get<int>(s, "sc_num_sectors", sc.num_sectors);
  sc.max_radius = yaml.get<double>(s, "sc_max_radius", sc.max_radius);
  sc.lidar_height = yaml.get<double>(s, "sc_lidar_height", sc.lidar_height);
  sc.num_exclude_recent = yaml.get<int>(s, "sc_num_exclude_recent", sc.num_exclude_recent);
  sc.num_candidates = yaml.get<int>(s, "sc_num_candidates", sc.num_candidates);
  sc.search_ratio = yaml.get<double>(s, "sc_search_ratio", sc.search_ratio);
  sc.dist_threshold = yaml.get<double>(s, "sc_dist_threshold", sc.dist_threshold);
  sc.tree_making_period = yaml.get<int>(s, "sc_tree_making_period", sc.tree_making_period);

  c.history_keyframe_search_num =
      yaml.get<int>(s, "history_keyframe_search_num", c.history_keyframe_search_num);
  c.icp_max_correspondence_distance =
      yaml.get<double>(s, "icp_max_correspondence_distance", c.icp_max_correspondence_distance);
  c.icp_max_iterations = yaml.get<int>(s, "icp_max_iterations", c.icp_max_iterations);
  c.icp_transformation_epsilon =
      yaml.get<double>(s, "icp_transformation_epsilon", c.icp_transformation_epsilon);
  c.icp_euclidean_fitness_epsilon =
      yaml.get<double>(s, "icp_euclidean_fitness_epsilon", c.icp_euclidean_fitness_epsilon);
  c.icp_ransac_iterations = yaml.get<int>(s, "icp_ransac_iterations", c.icp_ransac_iterations);
  c.loop_fitness_score_threshold =
      yaml.get<double>(s, "loop_fitness_score_threshold", c.loop_fitness_score_threshold);
  c.use_sc_yaw_guess = yaml.get<bool>(s, "use_sc_yaw_guess", c.use_sc_yaw_guess);
  c.sc_yaw_guess_min_deg = yaml.get<double>(s, "sc_yaw_guess_min_deg", c.sc_yaw_guess_min_deg);

  c.prior_noise_score = yaml.get<double>(s, "prior_noise_score", c.prior_noise_score);
  c.odom_noise_rotation = yaml.get<double>(s, "odom_noise_rotation", c.odom_noise_rotation);
  c.odom_noise_translation =
      yaml.get<double>(s, "odom_noise_translation", c.odom_noise_translation);
  c.loop_noise_score = yaml.get<double>(s, "loop_noise_score", c.loop_noise_score);
  c.loop_noise_cauchy_c = yaml.get<double>(s, "loop_noise_cauchy_c", c.loop_noise_cauchy_c);

  c.isam_relinearize_threshold =
      yaml.get<double>(s, "isam_relinearize_threshold", c.isam_relinearize_threshold);
  c.isam_relinearize_skip = yaml.get<int>(s, "isam_relinearize_skip", c.isam_relinearize_skip);
  return true;
}

}  // namespace bievr

#endif  // BIEVR_PGO_CONFIG_LOADER_H_

#ifndef BIEVR_LIO_CONFIG_LOADER_H_
#define BIEVR_LIO_CONFIG_LOADER_H_

// resbie: only upstream's YAML helpers remain (MergedYaml and the checked
// getters). The odometry Config they used to fill is gone with the pipeline;
// resbie's own loader (cpp/config.h) and the loop-closure loader use these.

#include <bievr_lio/common.h>
#include <bievr_lio/log++.h>
#include <yaml-cpp/yaml.h>

#include <string>
#include <vector>

namespace bievr {

namespace config_internal {

// A read-only view over one or more YAML documents. Lookups address a
// `section.key` pair and scan the documents in reverse, so the last file that
// defines that exact leaf wins (sections shared across files merge per key).
class MergedYaml {
 public:
  void add(const YAML::Node& node) { nodes_.push_back(node); }

  template <typename T>
  T get(const std::string& section, const std::string& key, const T& default_value) const {
    for (auto it = nodes_.rbegin(); it != nodes_.rend(); ++it) {
      const YAML::Node& root = *it;
      if (root[section] && root[section][key]) {
        return root[section][key].as<T>();
      }
    }
    return default_value;
  }

  // Same as get(), but for a key that lives at the document root (no section).
  template <typename T>
  T getTopLevel(const std::string& key, const T& default_value) const {
    for (auto it = nodes_.rbegin(); it != nodes_.rend(); ++it) {
      const YAML::Node& root = *it;
      if (root[key]) {
        return root[key].as<T>();
      }
    }
    return default_value;
  }

 private:
  std::vector<YAML::Node> nodes_;
};

// Builds a Transform from a section holding `translation` (3) and `rotation` (9)
// vectors. Extrinsics are mandatory: a missing section yields empty vectors and
// an incomplete section yields wrong-sized ones, both of which are hard errors
// (no silent fallback to identity). Returns false and leaves `out` untouched on
// failure.
inline bool extrinsicFromVectors(const std::vector<double>& t_vec, const std::vector<double>& R_vec,
                                 const std::string& label, Transform& out) {
  if (t_vec.size() != 3) {
    LOG(E, "Config error: " << label << " translation must have 3 elements, got " << t_vec.size()
                            << ". Extrinsics must be provided completely.");
    return false;
  }
  if (R_vec.size() != 9) {
    LOG(E, "Config error: " << label << " rotation must have 9 elements, got " << R_vec.size()
                            << ". Extrinsics must be provided completely.");
    return false;
  }
  const V3 t(t_vec[0], t_vec[1], t_vec[2]);
  Rotation R;
  R << R_vec[0], R_vec[1], R_vec[2], R_vec[3], R_vec[4], R_vec[5], R_vec[6], R_vec[7], R_vec[8];
  out = Transform(R, t);
  return true;
}

// Reads a scalar that is allowed to be absent (the positive `default_value` is
// used then) but, when present, must be positive. A non-positive value is a hard
// error: reports it and returns false. On success writes the value to `out`.
template <typename T>
bool getPositive(const MergedYaml& yaml, const std::string& section, const std::string& key,
                 const T& default_value, T& out) {
  out = yaml.get<T>(section, key, default_value);
  if (out <= T(0)) {
    LOG(E, "Config error: '" << section << "." << key << "' must be positive, got " << out << ".");
    return false;
  }
  return true;
}

}  // namespace config_internal
}  // namespace bievr

#endif  // BIEVR_LIO_CONFIG_LOADER_H_

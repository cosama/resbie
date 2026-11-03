#ifndef BIEVR_LIO_CONFIG_LOADER_H_
#define BIEVR_LIO_CONFIG_LOADER_H_

// resbie: only upstream's YAML helpers remain, reworked: MergedYaml (one or
// more documents, later ones winning per key), read() (the one checked getter)
// and extrinsicFromVectors(). The odometry Config they used to fill is gone with
// the pipeline; resbie's two loaders use these (cpp/resbie/config.h for the
// front end, bievr_pgo/config_loader.h for the back end).

#include <bievr_lio/common.h>
#include "bievr_lio/log.h"
#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <string>
#include <type_traits>
#include <vector>

namespace bievr {

namespace config_internal {

// A read-only view over one or more YAML documents. Lookups address a
// `section.key` pair and scan the documents in reverse, so the last file that
// defines that exact leaf wins (sections shared across files merge per key).
class MergedYaml {
 public:
  // resbie: parses YAML texts, later ones winning. False (logged) on a parse error.
  bool load(const std::vector<std::string>& documents) {
    for (size_t i = 0; i < documents.size(); ++i) {
      try {
        nodes_.push_back(YAML::Load(documents[i]));
      } catch (const std::exception& e) {
        LOG(E, "Config error: document " << i << " is not valid YAML: " << e.what());
        return false;
      }
    }
    return true;
  }

  // resbie: true if every key the documents set is in [begin, end)
  // ("section.key" or top-level); logs each one that is not.
  template <typename It>
  bool onlyKeys(It begin, It end) const {
    bool ok = true;
    for (const std::string& key : keys()) {
      if (std::find_if(begin, end, [&](const char* k) { return key == k; }) == end) {
        LOG(E, "Config error: unknown key '" << key << "'.");
        ok = false;
      }
    }
    return ok;
  }

  // resbie: the winning node for `key` ("section.key" or top-level),
  // undefined if no document sets it.
  YAML::Node node(const std::string& key) const {
    const size_t dot = key.find('.');
    for (auto it = nodes_.rbegin(); it != nodes_.rend(); ++it) {
      if (dot == std::string::npos) {
        if ((*it)[key]) return (*it)[key];
      } else {
        const YAML::Node section = (*it)[key.substr(0, dot)];
        if (section && section[key.substr(dot + 1)]) return section[key.substr(dot + 1)];
      }
    }
    return YAML::Node(YAML::NodeType::Undefined);  // Node() is a defined null
  }

  // resbie: every key the documents set ("section.key" or top-level).
  std::vector<std::string> keys() const {
    std::vector<std::string> out;
    for (const YAML::Node& root : nodes_) {
      if (!root.IsMap()) continue;
      for (const auto& top : root) {
        const std::string name = top.first.as<std::string>();
        if (top.second.IsMap()) {
          for (const auto& leaf : top.second) out.push_back(name + "." + leaf.first.as<std::string>());
        } else {
          out.push_back(name);
        }
      }
    }
    return out;
  }

 private:
  std::vector<YAML::Node> nodes_;
};

// Builds a Transform from `translation` (3) and `rotation` (9, row-major).
// Extrinsics are mandatory: missing or incomplete vectors are hard errors (no
// silent fallback to identity). Returns false and leaves `out` untouched on
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
  out = Transform();
  out.linear() << R_vec[0], R_vec[1], R_vec[2], R_vec[3], R_vec[4], R_vec[5], R_vec[6], R_vec[7],
      R_vec[8];
  out.translation() = t;
  return true;
}

// resbie: a value's lower bound, checked by read().
enum class Min { kNone, kZero, kPositive };

// resbie: reads `key` ("section.key" or top-level) into `value` if a document
// sets it, keeping `value` (the default) otherwise. False (logged), `value`
// untouched, if it does not convert or is below `min`. An Eigen::Vector3d takes
// one number (every axis) or three; an unsigned integer is at least 0.
template <typename T>
bool read(const MergedYaml& yaml, const std::string& key, T& value, Min min = Min::kNone) {
  const YAML::Node node = yaml.node(key);
  if (!node) return true;
  constexpr bool kUnsigned = std::is_unsigned_v<T> && !std::is_same_v<T, bool>;
  if (kUnsigned && min == Min::kNone) min = Min::kZero;
  const auto inRange = [min](double v) {
    return min == Min::kNone || (min == Min::kZero ? v >= 0.0 : v > 0.0);
  };
  T v = value;
  bool in_range = true;
  try {
    if constexpr (std::is_same_v<T, Eigen::Vector3d>) {
      if (node.IsScalar()) {
        v.setConstant(node.as<double>());
      } else if (node.IsSequence() && node.size() == 3) {
        v = Eigen::Vector3d(node[0].as<double>(), node[1].as<double>(), node[2].as<double>());
      } else {
        throw YAML::Exception(node.Mark(), "expected a number or 3 numbers");
      }
      in_range = inRange(v.x()) && inRange(v.y()) && inRange(v.z());
    } else if constexpr (kUnsigned) {
      const long long x = node.as<long long>();  // as<unsigned> wraps negatives
      in_range = inRange(double(x));
      v = static_cast<T>(x);
    } else if constexpr (std::is_arithmetic_v<T> && !std::is_same_v<T, bool>) {
      v = node.as<T>();
      in_range = inRange(double(v));
    } else {
      v = node.as<T>();
    }
  } catch (const YAML::Exception& e) {
    LOG(E, "Config error: '" << key << "': " << e.what());
    return false;
  }
  if (!in_range) {
    LOG(E, "Config error: '" << key << "' must be " << (min == Min::kZero ? ">= 0." : "> 0."));
    return false;
  }
  value = v;
  return true;
}

}  // namespace config_internal
}  // namespace bievr

#endif  // BIEVR_LIO_CONFIG_LOADER_H_

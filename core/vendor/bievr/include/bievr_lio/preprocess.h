#ifndef BIEVR_LIO_PREPROCESS_H_
#define BIEVR_LIO_PREPROCESS_H_

#include "bievr_lio/bievr_map.h"
#include "bievr_lio/common.h"

namespace bievr {

struct PreprocessConfig {
  size_t n_samples = 1000;
  bool informed_sampling = true;
  double min_range = 0.5;              // meters
  double max_range = 100.0;            // meters
  double downsample_resolution = 0.1;  // meters
};

// resbie: both samplers return indices into their input, so the per-point
// times and intensities of a sweep survive sampling.
void voxelDownsample(const Pointcloud& points_raw, std::vector<size_t>& selected_indices,
                     double voxel_size);

// Informed sampling on points already placed in the world (BIEVR placed the
// whole sweep with one predicted pose; resbie places each point by the spline).
void sampleInformed(const BIEVRMap& map, const Pointcloud& points_w, std::vector<size_t>& selected,
                    size_t n_samples);

template <typename T>
concept HasStamp = requires(T t) {
  t.stamp;
  t.end_stamp;
};

template <typename PointcloudT>
void filterMinMaxRange(const PointcloudT& points_raw, PointcloudT& points_filtered,
                       const double min_range = 0.0,
                       const double max_range = std::numeric_limits<double>::max()) {
  bool filter_min = (min_range > 0.0);
  bool filter_max = (max_range < std::numeric_limits<double>::max());

  if (!filter_min && !filter_max) {
    points_filtered = points_raw;
    return;
  }

  // When a filter is disabled its bound is a no-op: min_range defaults to 0
  // (range_squared >= 0 always), and max_range defaults to double max whose
  // square overflows to +inf (range_squared <= inf always).
  const double min_range_squared = min_range * min_range;
  const double max_range_squared = max_range * max_range;

  size_t n_points = points_raw.size();
  std::vector<char> point_selected(n_points, 0);
  tbb::parallel_for(
      tbb::blocked_range<size_t>(0, n_points), [&](const tbb::blocked_range<size_t>& r) {
        for (size_t i = r.begin(); i != r.end(); ++i) {
          const auto& point = points_raw[i];
          double range_squared =
              point.x() * point.x() + point.y() * point.y() + point.z() * point.z();
          if (range_squared >= min_range_squared && range_squared <= max_range_squared) {
            point_selected[i] = 1;
          }
        }
      });

  // Extract selected point indices
  std::vector<size_t> selected_indices;
  selected_indices.reserve(n_points);
  for (size_t i = 0; i < n_points; ++i) {
    if (point_selected[i] == 1) {
      selected_indices.push_back(i);
    }
  }

  points_filtered.resize(selected_indices.size());
  tbb::parallel_for(tbb::blocked_range<size_t>(0, selected_indices.size()),
                    [&](const tbb::blocked_range<size_t>& r) {
                      for (size_t i = r.begin(); i != r.end(); ++i) {
                        points_filtered[i] = points_raw[selected_indices[i]];
                      }
                    });

  if constexpr (HasStamp<PointcloudT>) {
    points_filtered.stamp = points_raw.stamp;
    points_filtered.end_stamp = points_raw.end_stamp;
  }
}

}  // namespace bievr
#endif  // BIEVR_LIO_PREPROCESS_H_
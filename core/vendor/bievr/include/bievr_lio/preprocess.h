#ifndef BIEVR_LIO_PREPROCESS_H_
#define BIEVR_LIO_PREPROCESS_H_

#include <vector>

#include "bievr_lio/bievr_map.h"

namespace bievr {

// resbie: both samplers return indices into their input, so the per-point
// times and intensities of a sweep survive sampling.
void voxelDownsample(const Pointcloud& points_raw, std::vector<size_t>& selected_indices,
                     double voxel_size);

// Informed sampling on points already placed in the world (BIEVR placed the
// whole sweep with one predicted pose; resbie places each point by the spline).
void sampleInformed(const BIEVRMap& map, const Pointcloud& points_w, std::vector<size_t>& selected,
                    size_t n_samples);

}  // namespace bievr
#endif  // BIEVR_LIO_PREPROCESS_H_
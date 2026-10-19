#include <algorithm>

#include "bievr_lio/preprocess.h"

namespace bievr {

namespace {

struct VoxelHashIdx {
  size_t hash;
  size_t idx;
};

struct VoxelScore {
  double score;
  size_t hash;
  size_t idx;
  uint32_t unique_id;
};

static constexpr size_t kEmptyHash = std::numeric_limits<size_t>::max();

struct DownsampleEntry {
  size_t hash;
  double dist;
};

struct DownsampleSlot {
  size_t hash{kEmptyHash};
  double dist{0.0};
  uint32_t idx{0};
};

struct DownsampleScratch {
  std::vector<DownsampleEntry> entries;
  std::vector<DownsampleSlot> table;
  std::vector<uint32_t> dirty_slots;
  std::vector<VoxelHashIdx> winners;
  uint32_t bits{0};
};

}  // namespace

void voxelDownsample(const Pointcloud& points_raw, std::vector<size_t>& selected_indices,
                     double voxel_size) {
  const size_t n = points_raw.size();
  selected_indices.clear();
  if (n == 0) return;

  thread_local DownsampleScratch scratch;
  if (scratch.entries.size() < n) scratch.entries.resize(n);
  if (scratch.bits < 15) {
    scratch.bits = 15;
    scratch.table.assign(size_t{1} << scratch.bits, DownsampleSlot{});
  } else {
    for (uint32_t p : scratch.dirty_slots) {
      scratch.table[p].hash = kEmptyHash;
    }
  }
  scratch.dirty_slots.clear();
  scratch.dirty_slots.reserve(n);

  DownsampleEntry* entries_ptr = scratch.entries.data();
  tbb::parallel_for(tbb::blocked_range<size_t>(0, n), [&](const tbb::blocked_range<size_t>& r) {
    for (size_t idx = r.begin(); idx != r.end(); ++idx) {
      const Eigen::Vector3d pt = points_raw[idx];
      const Eigen::Vector3i voxel = (pt / voxel_size).array().floor().cast<int>();
      const Eigen::Vector3d voxel_center =
          (voxel.cast<double>() + Eigen::Vector3d::Constant(0.5)) * voxel_size;
      const double dist = (pt - voxel_center).squaredNorm();
      size_t h = hashIndexVoxel(voxel.matrix());
      if (h == kEmptyHash) --h;
      entries_ptr[idx] = {h, dist};
    }
  });

  size_t cap = size_t{1} << scratch.bits;
  size_t mask = cap - 1;
  uint32_t shift = 64 - scratch.bits;

  for (size_t idx = 0; idx < n; ++idx) {
    const size_t hash = entries_ptr[idx].hash;
    const double dist = entries_ptr[idx].dist;
    size_t pos = ((hash ^ (hash >> 32)) * 0x9E3779B97F4A7C15ULL) >> shift;
    while (true) {
      DownsampleSlot& s = scratch.table[pos];
      if (s.hash == kEmptyHash) {
        s.hash = hash;
        s.dist = dist;
        s.idx = static_cast<uint32_t>(idx);
        scratch.dirty_slots.push_back(static_cast<uint32_t>(pos));
        if (__builtin_expect(scratch.dirty_slots.size() * 2 >= cap, 0)) {
          std::vector<DownsampleSlot> active;
          active.reserve(scratch.dirty_slots.size());
          for (uint32_t p : scratch.dirty_slots) active.push_back(scratch.table[p]);
          scratch.dirty_slots.clear();
          ++scratch.bits;
          cap = size_t{1} << scratch.bits;
          scratch.table.assign(cap, DownsampleSlot{});
          mask = cap - 1;
          shift = 64 - scratch.bits;
          for (const auto& item : active) {
            size_t npos = ((item.hash ^ (item.hash >> 32)) * 0x9E3779B97F4A7C15ULL) >> shift;
            while (scratch.table[npos].hash != kEmptyHash) npos = (npos + 1) & mask;
            scratch.table[npos] = item;
            scratch.dirty_slots.push_back(static_cast<uint32_t>(npos));
          }
        }
        break;
      }
      if (s.hash == hash) {
        if (dist < s.dist) {
          s.dist = dist;
          s.idx = static_cast<uint32_t>(idx);
        }
        break;
      }
      pos = (pos + 1) & mask;
    }
  }

  const size_t num_unique = scratch.dirty_slots.size();
  scratch.winners.resize(num_unique);
  for (size_t i = 0; i < num_unique; ++i) {
    DownsampleSlot& s = scratch.table[scratch.dirty_slots[i]];
    scratch.winners[i] = {s.hash, s.idx};
    s.hash = kEmptyHash;
  }
  scratch.dirty_slots.clear();

  // Sort unique voxel winners by hash to preserve exact deterministic output ordering.
  tbb::parallel_sort(scratch.winners.begin(), scratch.winners.end(),
                     [](const VoxelHashIdx& a, const VoxelHashIdx& b) { return a.hash < b.hash; });

  selected_indices.resize(num_unique);
  for (size_t i = 0; i < num_unique; ++i) {
    selected_indices[i] = scratch.winners[i].idx;
  }
}

void sampleInformed(const BIEVRMap& map, const Pointcloud& points_w, std::vector<size_t>& selected,
                    size_t n_samples) {
  selected.clear();
  if (points_w.size() <= n_samples) {
    selected.resize(points_w.size());
    for (size_t i = 0; i < selected.size(); ++i) selected[i] = i;
    return;
  }
  const Pointcloud& points_raw = points_w;

  // Lookup hash for each point based on its transformed position from the registration prior.
  std::vector<VoxelHashIdx> voxel_entries(points_raw.size());
  tbb::parallel_for(tbb::blocked_range<size_t>(0, points_raw.size()),
                    [&](const tbb::blocked_range<size_t>& r) {
                      for (size_t idx = r.begin(); idx != r.end(); ++idx) {
                        voxel_entries[idx] = {map.hashIndex(points_raw[idx]), idx};
                      }
                    });

  tbb::parallel_sort(voxel_entries.begin(), voxel_entries.end(),
                     [](const VoxelHashIdx& a, const VoxelHashIdx& b) {
                       // Add tie to ensure deterministic order for points in the same voxel, even
                       // if they have the same hash
                       return std::tie(a.hash, a.idx) < std::tie(b.hash, b.idx);
                     });

  // Extract contiguous run boundaries of unique voxel hashes
  std::vector<uint32_t> run_start;
  run_start.reserve(voxel_entries.size() + 1);
  for (size_t i = 0; i < voxel_entries.size(); ++i) {
    if (i == 0 || voxel_entries[i].hash != voxel_entries[i - 1].hash) {
      run_start.push_back(static_cast<uint32_t>(i));
    }
  }
  const size_t num_unique = run_start.size();
  run_start.push_back(static_cast<uint32_t>(voxel_entries.size()));

  std::vector<VoxelScore> voxel_scores(num_unique);
  tbb::parallel_for(tbb::blocked_range<size_t>(0, num_unique),
                    [&](const tbb::blocked_range<size_t>& r) {
                      for (size_t u = r.begin(); u != r.end(); ++u) {
                        const uint32_t start = run_start[u];
                        const bool has_extras = (run_start[u + 1] - start) > 1;
                        const auto& entry = voxel_entries[start];
                        double score = 0.0;
                        if (has_extras) {
                          auto voxel = map.getVoxel(entry.hash);
                          if (voxel) score = voxel->mean_img_dist_;
                        }
                        voxel_scores[u] = {score, entry.hash, entry.idx, static_cast<uint32_t>(u)};
                      }
                    });

  // Sort by score, descending. resbie: hash tie-break, so equal scores (all
  // the unobserved voxels score 0) do not order by TBB's scheduling.
  tbb::parallel_sort(voxel_scores.begin(), voxel_scores.end(),
                     [](const VoxelScore& a, const VoxelScore& b) {
                       if (a.score != b.score) return a.score > b.score;
                       return a.hash < b.hash;
                     });

  const size_t n_select = std::min(n_samples, voxel_scores.size());
  std::vector<uint8_t> is_informed(num_unique, 0);
  for (size_t idx = 0; idx < n_select; ++idx) {
    is_informed[voxel_scores[idx].unique_id] = 1;
  }

  // Keep all points in informed voxels, then one point per remaining voxel.
  selected.reserve(points_raw.size());
  for (size_t u = 0; u < num_unique; ++u) {
    if (is_informed[u]) {
      const uint32_t start = run_start[u];
      const uint32_t end = run_start[u + 1];
      for (uint32_t k = start; k < end; ++k) {
        selected.push_back(voxel_entries[k].idx);
      }
    }
  }
  for (size_t idx = n_select; idx < voxel_scores.size(); ++idx) {
    selected.push_back(voxel_scores[idx].idx);
  }
}

}  // namespace bievr

#ifndef BIEVR_LIO_BIEVR_MAP_H_
#define BIEVR_LIO_BIEVR_MAP_H_

#include <tbb/parallel_for.h>
#include <tbb/parallel_sort.h>

#include <cstdint>
#include <limits>
#include <vector>

#include "bievr_lio/common.h"

namespace bievr {

struct Voxel {
  bool observed_{false};  // We consider a voxel observed if we have seen enough points inside it
  uint32_t slot_pos_{UINT32_MAX};
  uint32_t lru_prev_{UINT32_MAX};
  uint32_t lru_next_{UINT32_MAX};
  double mean_img_dist_{0.0};
  V3 mean_ = Eigen::Vector3d::Zero();
  size_t num_points_{0};
  size_t hash_{0};
  Transform T_C_W_ = Transform::Identity();
  Eigen::MatrixXf bump_smoothed_;
  Eigen::MatrixXf bump_weights_;
  Eigen::MatrixXf bump_img_;
  Transform T_O_W_ = Transform::Identity();
  M3 outer_sum_ = Eigen::Matrix3d::Zero();
  V3 sum_ = Eigen::Vector3d::Zero();
  // Raw points accumulated while the voxel is not yet observed (no normal yet, so they cannot be
  // projected into a bump image). Cleared once the voxel becomes observed.
  std::vector<Eigen::Vector4d> pending_points_;
};

class BIEVRMap {
 public:
  struct Config {
    size_t max_size{10000000};  // maximum number of voxels to store
    double voxel_size{0.5};     // voxel side length
    double px_size{0.05};       // size of bump image pixels
    bool weighted = false;      // use range weighted update for bump image
    bool smooth = false;        // apply gaussian smoothing to bump image
    double norm_tol_deg{3.0};   // if normal changes more than this, reproject bump image
  };

  explicit BIEVRMap(Config config);

  bool integratePoints(const Pointcloud& input_cloud, const std::vector<double>* ranges = nullptr);

  inline size_t hashIndex(const Point& point) const {
    Eigen::Vector3i voxel_idx = (point * inv_voxel_size_).array().floor().cast<int>();

    return hashIndexVoxel(voxel_idx);
  }
  size_t size() const { return active_voxels_; }
  Eigen::Vector3i getVoxelIdx(const Point& point) const;

  inline const Voxel* getVoxel(const size_t hash_idx) const {
    size_t pos = ((hash_idx ^ (hash_idx >> 32)) * 0x9E3779B97F4A7C15ULL) >> shift_;
    while (true) {
      const Slot& s = table_[pos];
      if (s.idx == UINT32_MAX) return nullptr;
      if (s.key == hash_idx) {
        return s.observed ? &voxels_[s.idx] : nullptr;
      }
      pos = (pos + 1) & mask_;
    }
  }

  inline const Voxel* nearestNeighborVoxel(const Point& point, size_t& result) const {
    double min_dist = std::numeric_limits<double>::max();
    const Voxel* best_voxel = nullptr;
    for (size_t k = 1; k < neighbor_offsets_.size(); ++k) {
      Eigen::Vector3d nearby_point = point + neighbor_offsets_[k];
      size_t hash_idx = hashIndex(nearby_point);
      const Voxel* voxel = getVoxel(hash_idx);
      if (!voxel) continue;
      double sqdistance = (point - voxel->mean_).squaredNorm();
      if (sqdistance > min_dist) continue;
      min_dist = sqdistance;
      result = hash_idx;
      best_voxel = voxel;
    }
    return best_voxel;
  }

  bool nearestVoxel(const Point& point, size_t& result) const;

  const double& voxel_size = config_.voxel_size;
  const double& pixel_size = config_.px_size;
  const double& inv_px_size = inv_px_size_;

 private:
  struct ImageBounds {
    int width;
    int height;
    double u_min;
    double v_min;
  };

  struct Slot {
    size_t key{0};
    uint32_t idx{UINT32_MAX};
    uint32_t observed{0};
  };

  void rehash(uint32_t new_bits);
  uint32_t findOrInsert(size_t key);
  void lruPushFront(uint32_t idx);
  void lruRemove(uint32_t idx);
  void lruMoveToFront(uint32_t idx);
  void evictLRU();

  // Voxel update / bump-image pipeline, in the order integratePoints invokes them.
  bool updateNormal(Voxel& voxel);

  bool updateBumpImage(const std::vector<Eigen::Vector4d>& points, Voxel& voxel,
                       bool normal_change);

  ImageBounds computeImageSize(const Voxel& voxel, const Point& reference_point) const;

  void reprojectImage(Voxel& voxel, const ImageBounds& bounds, Eigen::MatrixXi& changed,
                      int& r_min, int& r_max, int& c_min, int& c_max);

  void integratePoints(const std::vector<Eigen::Vector4d>& points, Voxel& voxel,
                       Eigen::MatrixXi& changed, int& r_min, int& r_max, int& c_min, int& c_max);

  void dilateMask(const Eigen::MatrixXi& changed, const Eigen::MatrixXf& weights,
                  Eigen::MatrixXi& changed_dilated, int r_min, int r_max, int c_min, int c_max);

  void maskedGaussianSmooth(const Eigen::MatrixXf& image, const Eigen::MatrixXf& weights,
                            const Eigen::MatrixXi& changed, Eigen::MatrixXf& image_smooth,
                            int r_min, int r_max, int c_min, int c_max);

  void computeScore(Voxel& voxel);

  // Geometry helper.
  Eigen::Vector3d getVoxelOrigin(const Point& point) const;

  Eigen::MatrixXf gauss_kernel_;
  Config config_;
  double inv_voxel_size_{1.0};
  double inv_px_size_{0.05};            // inv size of bump image
  double norm_tol_rad_{3.0};            // if normal changes more than this, update bump image
  static constexpr int kNMinValid = 4;  // minimum points for a valid voxel

  uint32_t bits_{16};
  uint32_t shift_{48};
  size_t mask_{65535};
  size_t active_voxels_{0};
  uint32_t lru_head_{UINT32_MAX};
  uint32_t lru_tail_{UINT32_MAX};
  std::vector<Slot> table_;
  std::vector<Voxel> voxels_;
  std::vector<uint32_t> free_voxel_indices_;
  std::vector<std::pair<size_t, Eigen::Vector4d>> hashed_points_;
  std::vector<size_t> hash_change_indices_;
  std::vector<uint32_t> voxel_slots_;

  Eigen::MatrixXd corner_offsets_;
  std::vector<Point> neighbor_offsets_;
};

using VoxelPtr = std::shared_ptr<Voxel>;
using ConstVoxelPtr = std::shared_ptr<const Voxel>;

}  // namespace bievr
#endif  // BIEVR_LIO_BIEVR_MAP_H_
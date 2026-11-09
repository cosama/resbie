#ifndef BIEVR_LIO_BIEVR_MAP_H_
#define BIEVR_LIO_BIEVR_MAP_H_

#include <cmath>
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
    size_t max_size{1500000};   // maximum number of voxels to store
    double voxel_size{0.5};     // voxel side length
    double px_size{0.05};       // size of bump image pixels
    double norm_tol_deg{3.0};   // if normal changes more than this, reproject bump image
  };

  explicit BIEVRMap(Config config);

  // `ranges`: each point's distance from the sensor; nearer points weigh more
  // in the bump image (capped).
  void integratePoints(const Pointcloud& input_cloud, const std::vector<double>& ranges);

  inline size_t hashIndex(const Point& point) const {
    Eigen::Vector3i voxel_idx = (point * inv_voxel_size_).array().floor().cast<int>();

    return hashIndexVoxel(voxel_idx);
  }
  size_t size() const { return active_voxels_; }

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

// Combined bilinear sample + central-difference gradient.
// Shares floor/bounds/fraction work and reuses the 4 inner corner reads.
// Returns false if the center 2x2 stencil is out of bounds or has no valid corners.
// Gradients are set to 0 when their 4x2 stencil is out of bounds or has no valid corners.
inline bool sampleValueAndGradient(const Voxel* voxel, const double x, const double y,
                                   double& value, double& dIdx, double& dIdy) {
  const int x0 = std::floor(x);
  const int y0 = std::floor(y);
  const int x1 = x0 + 1;
  const int y1 = y0 + 1;

  const int max_x = voxel->bump_smoothed_.cols() - 1;
  const int max_y = voxel->bump_smoothed_.rows() - 1;
  if (x0 < 0 || y0 < 0 || x1 > max_x || y1 > max_y) return false;

  const auto& M = voxel->bump_smoothed_;
  const auto& V = voxel->bump_weights_;

  const double dx = x - x0;
  const double dy = y - y0;
  const double dx1 = 1.0 - dx;
  const double dy1 = 1.0 - dy;

  const int v00 = V(y0, x0) > 0, v01 = V(y0, x1) > 0;
  const int v10 = V(y1, x0) > 0, v11 = V(y1, x1) > 0;
  {
    const double w0 = dx1 * dy1 * v00;
    const double w1 = dx * dy1 * v01;
    const double w2 = dx1 * dy * v10;
    const double w3 = dx * dy * v11;
    const double ws = w0 + w1 + w2 + w3;
    if (ws == 0.0) return false;
    value = (w0 * M(y0, x0) + w1 * M(y0, x1) + w2 * M(y1, x0) + w3 * M(y1, x1)) / ws;
  }

  dIdx = 0.0;
  dIdy = 0.0;

  if (x0 >= 1 && x1 + 1 <= max_x) {
    const int xm = x0 - 1;
    const int xp = x1 + 1;

    const int am0 = V(y0, xm) > 0, am2 = V(y1, xm) > 0;
    const double wm0 = dx1 * dy1 * am0;
    const double wm1 = dx * dy1 * v00;
    const double wm2 = dx1 * dy * am2;
    const double wm3 = dx * dy * v10;
    const double wms = wm0 + wm1 + wm2 + wm3;

    const int ap1 = V(y0, xp) > 0, ap3 = V(y1, xp) > 0;
    const double wp0 = dx1 * dy1 * v01;
    const double wp1 = dx * dy1 * ap1;
    const double wp2 = dx1 * dy * v11;
    const double wp3 = dx * dy * ap3;
    const double wps = wp0 + wp1 + wp2 + wp3;

    if (wms > 0.0 && wps > 0.0) {
      const double vm =
          (wm0 * M(y0, xm) + wm1 * M(y0, x0) + wm2 * M(y1, xm) + wm3 * M(y1, x0)) / wms;
      const double vp =
          (wp0 * M(y0, x1) + wp1 * M(y0, xp) + wp2 * M(y1, x1) + wp3 * M(y1, xp)) / wps;
      dIdx = 0.5 * (vp - vm);
    }
  }

  if (y0 >= 1 && y1 + 1 <= max_y) {
    const int ym = y0 - 1;
    const int yp = y1 + 1;

    const int am0 = V(ym, x0) > 0, am1 = V(ym, x1) > 0;
    const double wm0 = dx1 * dy1 * am0;
    const double wm1 = dx * dy1 * am1;
    const double wm2 = dx1 * dy * v00;
    const double wm3 = dx * dy * v01;
    const double wms = wm0 + wm1 + wm2 + wm3;

    const int ap2 = V(yp, x0) > 0, ap3 = V(yp, x1) > 0;
    const double wp0 = dx1 * dy1 * v10;
    const double wp1 = dx * dy1 * v11;
    const double wp2 = dx1 * dy * ap2;
    const double wp3 = dx * dy * ap3;
    const double wps = wp0 + wp1 + wp2 + wp3;

    if (wms > 0.0 && wps > 0.0) {
      const double vm =
          (wm0 * M(ym, x0) + wm1 * M(ym, x1) + wm2 * M(y0, x0) + wm3 * M(y0, x1)) / wms;
      const double vp =
          (wp0 * M(y1, x0) + wp1 * M(y1, x1) + wp2 * M(yp, x0) + wp3 * M(yp, x1)) / wps;
      dIdy = 0.5 * (vp - vm);
    }
  }

  return true;
}

}  // namespace bievr
#endif  // BIEVR_LIO_BIEVR_MAP_H_
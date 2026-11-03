#include "bievr_lio/bievr_map.h"

#include <tbb/parallel_for.h>
#include <tbb/parallel_sort.h>

#include <Eigen/Eigenvalues>
#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>

#include "bievr_lio/log.h"

namespace bievr {

namespace {

// Normalised (2 radius + 1)^2 Gaussian kernel.
Eigen::MatrixXf buildGaussianKernel(int radius, float sigma) {
  int size = 2 * radius + 1;
  Eigen::MatrixXf kernel(size, size);
  float sum = 0.0f;
  for (int y = -radius; y <= radius; ++y) {
    for (int x = -radius; x <= radius; ++x) {
      float value = std::exp(-(x * x + y * y) / (2 * sigma * sigma));
      kernel(y + radius, x + radius) = value;
      sum += value;
    }
  }
  kernel /= sum;
  return kernel;
}

// OpenCV's sigma for a kernel of this radius.
float computeSigmaFromRadius(int radius) {
  int ksize = 2 * radius + 1;
  return 0.3f * ((ksize - 1) * 0.5f - 1) + 0.8f;
}

// Offsets to the 7-connected neighbors of a voxel: the voxel itself plus +-voxel_size along each
// axis. Order is {0}, +x, -x, +y, -y, +z, -z.
std::vector<Point> getNeighborOffsets(double voxel_size) {
  std::vector<Point> offsets;
  offsets.reserve(7);
  offsets.push_back(Point::Zero());
  for (int axis = 0; axis < 3; ++axis) {
    for (double sign : {1.0, -1.0}) {
      Point offset = Point::Zero();
      offset(axis) = sign * voxel_size;
      offsets.push_back(offset);
    }
  }
  return offsets;
}

}  // namespace

BIEVRMap::BIEVRMap(Config config) : config_(config) {
  // Precompute the offsets to the 8 corners of a voxel for quick access when projecting the voxel
  // into the image
  corner_offsets_ = Eigen::MatrixXd(3, 8);
  int count = 0;
  for (int dx : {0, 1}) {
    for (int dy : {0, 1}) {
      for (int dz : {0, 1}) {
        Eigen::Vector3d corner(dx, dy, dz);
        corner_offsets_.col(count) = corner;
        count++;
      }
    }
  }
  corner_offsets_ *= config_.voxel_size;

  // Store offsets for quick access when searching for neighboring voxels.
  neighbor_offsets_ = getNeighborOffsets(config_.voxel_size);

  // Precompute Gaussian kernel for smoothing
  const int radius = 1;
  float sigma = computeSigmaFromRadius(radius);
  gauss_kernel_ = buildGaussianKernel(radius, sigma);

  // For quick access
  norm_tol_rad_ = M_PI * config_.norm_tol_deg / 180.0;
  inv_voxel_size_ = 1.0 / config_.voxel_size;
  inv_px_size_ = 1.0 / config_.px_size;

  bits_ = 16;
  shift_ = 64 - bits_;
  mask_ = (1ULL << bits_) - 1;
  table_.assign(1ULL << bits_, Slot{});
  voxels_.reserve(32768);
}

void BIEVRMap::lruPushFront(uint32_t idx) {
  Voxel& v = voxels_[idx];
  v.lru_prev_ = UINT32_MAX;
  v.lru_next_ = lru_head_;
  if (lru_head_ != UINT32_MAX) {
    voxels_[lru_head_].lru_prev_ = idx;
  } else {
    lru_tail_ = idx;
  }
  lru_head_ = idx;
}

void BIEVRMap::lruRemove(uint32_t idx) {
  Voxel& v = voxels_[idx];
  if (v.lru_prev_ != UINT32_MAX) {
    voxels_[v.lru_prev_].lru_next_ = v.lru_next_;
  } else {
    lru_head_ = v.lru_next_;
  }
  if (v.lru_next_ != UINT32_MAX) {
    voxels_[v.lru_next_].lru_prev_ = v.lru_prev_;
  } else {
    lru_tail_ = v.lru_prev_;
  }
  v.lru_prev_ = UINT32_MAX;
  v.lru_next_ = UINT32_MAX;
}

void BIEVRMap::lruMoveToFront(uint32_t idx) {
  if (lru_head_ == idx) return;
  lruRemove(idx);
  lruPushFront(idx);
}

void BIEVRMap::evictLRU() {
  if (lru_tail_ == UINT32_MAX) return;
  const uint32_t victim_idx = lru_tail_;
  lruRemove(victim_idx);

  size_t hole = voxels_[victim_idx].slot_pos_;
  // Backward-shift deletion (Knuth Algorithm R) to preserve linear probing invariants.
  size_t scan = (hole + 1) & mask_;
  while (table_[scan].idx != UINT32_MAX) {
    const size_t home =
        ((table_[scan].key ^ (table_[scan].key >> 32)) * 0x9E3779B97F4A7C15ULL) >> shift_;
    if (((hole - home) & mask_) < ((scan - home) & mask_)) {
      table_[hole] = table_[scan];
      voxels_[table_[hole].idx].slot_pos_ = static_cast<uint32_t>(hole);
      hole = scan;
    }
    scan = (scan + 1) & mask_;
  }
  table_[hole] = Slot{};

  voxels_[victim_idx] = Voxel{};
  free_voxel_indices_.push_back(victim_idx);
  --active_voxels_;
}

void BIEVRMap::rehash(uint32_t new_bits) {
  bits_ = new_bits;
  shift_ = 64 - bits_;
  mask_ = (1ULL << bits_) - 1;
  table_.assign(1ULL << bits_, Slot{});
  for (uint32_t idx = 0; idx < voxels_.size(); ++idx) {
    if (voxels_[idx].slot_pos_ == UINT32_MAX) continue;
    const size_t key = voxels_[idx].hash_;
    size_t pos = ((key ^ (key >> 32)) * 0x9E3779B97F4A7C15ULL) >> shift_;
    while (table_[pos].idx != UINT32_MAX) {
      pos = (pos + 1) & mask_;
    }
    table_[pos].key = key;
    table_[pos].idx = idx;
    table_[pos].observed = voxels_[idx].observed_ ? 1u : 0u;
    voxels_[idx].slot_pos_ = static_cast<uint32_t>(pos);
  }
}

uint32_t BIEVRMap::findOrInsert(size_t key) {
  if ((active_voxels_ + 1) * 2 >= table_.size()) {
    rehash(bits_ + 1);
  }
  size_t pos = ((key ^ (key >> 32)) * 0x9E3779B97F4A7C15ULL) >> shift_;
  for (;;) {
    Slot& s = table_[pos];
    if (s.idx == UINT32_MAX) {
      uint32_t idx;
      if (!free_voxel_indices_.empty()) {
        idx = free_voxel_indices_.back();
        free_voxel_indices_.pop_back();
      } else {
        idx = static_cast<uint32_t>(voxels_.size());
        voxels_.emplace_back();
      }
      s.key = key;
      s.idx = idx;
      s.observed = 0;
      Voxel& v = voxels_[idx];
      v.hash_ = key;
      v.slot_pos_ = static_cast<uint32_t>(pos);
      v.pending_points_.reserve(8);
      ++active_voxels_;
      lruPushFront(idx);
      return idx;
    }
    if (s.key == key) {
      return s.idx;
    }
    pos = (pos + 1) & mask_;
  }
}

void BIEVRMap::integratePoints(const Pointcloud& cloud, const std::vector<double>& ranges) {
  // Each entry is {voxel hash, point (xyz, plus range in w)}.
  hashed_points_.resize(cloud.size());

  // Calculate Hash indices for each point
  tbb::parallel_for(tbb::blocked_range<size_t>(0, cloud.size()),
                    [&](const tbb::blocked_range<size_t>& r) {
                      for (size_t i = r.begin(); i != r.end(); ++i) {
                        hashed_points_[i].second.head(3) = cloud[i];
                        hashed_points_[i].second(3) = ranges[i];
                        hashed_points_[i].first = hashIndex(hashed_points_[i].second.head(3));
                      }
                    });

  // Group points by voxel; tie-break on x for a deterministic order within each voxel.
  tbb::parallel_sort(hashed_points_.begin(), hashed_points_.end(),
                     [](const auto& a, const auto& b) {
                       if (a.first != b.first) return a.first < b.first;
                       return a.second(0) < b.second(0);
                     });

  // Extract unique hash indices and resolve/insert their voxel slots up front so the parallel loop
  // indexes directly into voxels_ without probing the hash table.
  hash_change_indices_.clear();
  voxel_slots_.clear();
  if (!hashed_points_.empty()) {
    size_t prev_hash = 0;
    for (size_t i = 0; i < hashed_points_.size(); ++i) {
      size_t current_hash = hashed_points_[i].first;
      if (i == 0 || current_hash != prev_hash) {
        hash_change_indices_.push_back(i);
        voxel_slots_.push_back(findOrInsert(current_hash));
        prev_hash = current_hash;
      }
    }
  }

  tbb::parallel_for(
      tbb::blocked_range<size_t>(0, hash_change_indices_.size()),
      [&](const tbb::blocked_range<size_t>& r) {
        std::vector<Eigen::Vector4d> voxel_points;
        for (size_t i = r.begin(); i != r.end(); ++i) {
          int start_idx = hash_change_indices_[i];
          int end_idx = (i + 1 < hash_change_indices_.size()) ? hash_change_indices_[i + 1]
                                                              : hashed_points_.size();

          Voxel& voxel = voxels_[voxel_slots_[i]];
          voxel_points.clear();
          voxel_points.reserve(end_idx - start_idx);
          for (size_t j = start_idx; j < end_idx; ++j) {
            voxel.sum_ += hashed_points_[j].second.head(3);
            voxel.num_points_++;
            voxel.outer_sum_ +=
                hashed_points_[j].second.head(3) * hashed_points_[j].second.head(3).transpose();
            voxel_points.emplace_back(hashed_points_[j].second);
          }

          bool valid_before = voxel.observed_;
          bool normal_change = updateNormal(voxel);

          auto& pending = voxel.pending_points_;
          if (!valid_before && voxel.observed_) {
            // Just became observed: fold the points accumulated while unobserved into this scan's
            // batch so they get projected once, then release the buffer.
            table_[voxel.slot_pos_].observed = 1u;
            voxel_points.insert(voxel_points.end(), pending.begin(), pending.end());
            pending.clear();
            pending.shrink_to_fit();
          } else if (!voxel.observed_) {
            // Still unobserved: keep accumulating raw points until a normal exists.
            pending.insert(pending.end(), voxel_points.begin(), voxel_points.end());
          }

          updateBumpImage(voxel_points, voxel, normal_change);
        }
      });

  // Touched voxels move to the LRU front in hash order.
  for (uint32_t idx : voxel_slots_) {
    lruMoveToFront(idx);
  }

  // If max_size exceeded, remove least recently used voxels
  int removal_counter = 0;
  while (active_voxels_ > config_.max_size && lru_tail_ != UINT32_MAX) {
    evictLRU();
    removal_counter++;
  }

  if (removal_counter > 0) {
    LOG(D, "BIEVRMap exceeded max_size. Removed " << removal_counter << " old voxels.");
  }
}

bool BIEVRMap::updateNormal(Voxel& voxel) {
  if (voxel.num_points_ < kNMinValid) return false;

  Point mean = voxel.sum_ / static_cast<double>(voxel.num_points_);
  voxel.mean_ = mean;
  M3 covariance = (voxel.outer_sum_ - voxel.sum_ * mean.transpose()) /
                  (static_cast<double>(voxel.num_points_) - 1.0);

  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(covariance);

  // Smallest eigenvalue -> normal
  Eigen::Vector3d normal = solver.eigenvectors().col(0);  //

  bool update_normal = false;
  if (voxel.observed_) {
    // The previous normal is the Z axis of the old frame
    const Eigen::Vector3d& prev_normal = voxel.T_O_W_.linear().row(2);
    double angle = std::acos(std::abs(prev_normal.dot(normal)));
    if (angle > norm_tol_rad_) {
      update_normal = true;
    }
  } else {
    update_normal = true;
  }
  if (update_normal) {
    // Get the normal vector (Z axis of the new frame)
    Eigen::Vector3d z_axis = normal;
    Eigen::Vector3d x_axis = z_axis.unitOrthogonal();
    Eigen::Vector3d y_axis = z_axis.cross(x_axis).normalized();

    // Rotation matrix to align with plane
    Eigen::Matrix3d R_align;
    R_align.col(0) = x_axis;
    R_align.col(1) = y_axis;
    R_align.col(2) = z_axis;
    Transform T_W_O;
    T_W_O.linear() = R_align;
    T_W_O.translation() = mean;
    voxel.T_O_W_ = T_W_O.inverse();
  }

  voxel.observed_ = true;

  return update_normal;
}

bool BIEVRMap::updateBumpImage(const std::vector<Eigen::Vector4d>& points, Voxel& voxel,
                               bool normal_change) {
  if (!voxel.observed_) {
    return false;
  }

  const int rows_init = voxel.bump_img_.rows();
  const int cols_init = voxel.bump_img_.cols();
  int r0 = rows_init, r1 = -1, c0 = cols_init, c1 = -1;
  Eigen::MatrixXi changed;
  if (normal_change) {
    // If the normal has changed, we reproject the surface from the old image into the new image
    ImageBounds bounds = computeImageSize(voxel, points[0].head<3>());
    r0 = bounds.height;
    c0 = bounds.width;
    reprojectImage(voxel, bounds, changed, r0, r1, c0, c1);
  } else {
    changed = Eigen::MatrixXi::Zero(rows_init, cols_init);
  }

  // Update the pixel values and weights based on the new points
  integratePoints(points, voxel, changed, r0, r1, c0, c1);

  if (r1 < r0) {
    // No pixel was touched (all points fell out of bounds).
    computeScore(voxel);
    return true;
  }

  Eigen::MatrixXi changed_dilated;
  int dr0 = r0, dr1 = r1, dc0 = c0, dc1 = c1;
  if (!normal_change) {
    // For the smoothing, we also need to consider pixels that are adjacent to the changed pixels
    changed_dilated = Eigen::MatrixXi::Zero(voxel.bump_img_.rows(), voxel.bump_img_.cols());
    dilateMask(changed, voxel.bump_weights_, changed_dilated, r0, r1, c0, c1);
    dr0 = std::max(0, r0 - 1);
    dr1 = std::min<int>(voxel.bump_img_.rows() - 1, r1 + 1);
    dc0 = std::max(0, c0 - 1);
    dc1 = std::min<int>(voxel.bump_img_.cols() - 1, c1 + 1);
  }
  const Eigen::MatrixXi& mask = normal_change ? changed : changed_dilated;

  maskedGaussianSmooth(voxel.bump_img_, voxel.bump_weights_, mask, voxel.bump_smoothed_, dr0, dr1,
                       dc0, dc1);

  computeScore(voxel);

  return true;
}

BIEVRMap::ImageBounds BIEVRMap::computeImageSize(const Voxel& voxel,
                                                 const Eigen::Vector3d& reference_point) const {
  Eigen::Vector3d p_origin = getVoxelOrigin(reference_point);
  Eigen::MatrixXd voxel_corners = corner_offsets_.colwise() + p_origin;
  // Project voxel corners on voxel plane
  Eigen::MatrixXd uv_convers = voxel.T_O_W_.matrix().block<2, 3>(0, 0) * voxel_corners +
                               voxel.T_O_W_.matrix().block<2, 1>(0, 3).replicate(1, 8);
  // Calculate minimum required image size to cover all corners
  ImageBounds bounds;
  bounds.u_min = uv_convers.row(0).minCoeff();
  double u_max = uv_convers.row(0).maxCoeff();
  bounds.v_min = uv_convers.row(1).minCoeff();
  double v_max = uv_convers.row(1).maxCoeff();
  bounds.width = static_cast<int>(std::ceil((u_max - bounds.u_min) * inv_px_size_) + 1);
  bounds.height = static_cast<int>(std::ceil((v_max - bounds.v_min) * inv_px_size_) + 1);
  return bounds;
}

void BIEVRMap::reprojectImage(Voxel& voxel, const ImageBounds& bounds, Eigen::MatrixXi& changed,
                              int& r0, int& r1, int& c0, int& c1) {
  Eigen::MatrixXf bump_original = voxel.bump_img_;
  Eigen::MatrixXf weights_original = voxel.bump_weights_;
  Transform T_W_C_o = voxel.T_C_W_.inverse();
  voxel.bump_img_.resize(bounds.height, bounds.width);
  voxel.bump_smoothed_.resize(bounds.height, bounds.width);
  voxel.bump_weights_.resize(bounds.height, bounds.width);
  changed.resize(bounds.height, bounds.width);
  voxel.bump_img_.setZero();
  voxel.bump_smoothed_.setZero();
  voxel.bump_weights_.setZero();
  changed.setZero();
  Point p_o_planar(bounds.u_min, bounds.v_min, 0.0);
  Point p_w_o = voxel.T_O_W_.inverse() * p_o_planar;
  Transform T_W_C;
  T_W_C.matrix().topLeftCorner<3, 3>() = voxel.T_O_W_.linear().transpose();
  T_W_C.matrix().topRightCorner<3, 1>() = p_w_o;
  voxel.T_C_W_ = T_W_C.inverse();

  Transform T_C1_C0 = voxel.T_C_W_ * T_W_C_o;
  for (int i = 0; i < bump_original.rows(); ++i) {
    for (int j = 0; j < bump_original.cols(); ++j) {
      if (weights_original(i, j) == 0) continue;
      // Lift the point to 3D using the original bump value, then transform it into the new camera
      // frame and project it
      Point p_O = T_C1_C0 * Point(j * config_.px_size, i * config_.px_size, bump_original(i, j));
      int x = static_cast<int>(std::round(p_O(0) * inv_px_size_));
      int y = static_cast<int>(std::round(p_O(1) * inv_px_size_));

      if (x < 0 || x >= voxel.bump_img_.cols() || y < 0 || y >= voxel.bump_img_.rows()) {
        continue;
      }

      voxel.bump_img_(y, x) = p_O(2);
      voxel.bump_weights_(y, x) = weights_original(i, j);
      changed(y, x) = 1;
      r0 = std::min(r0, y);
      r1 = std::max(r1, y);
      c0 = std::min(c0, x);
      c1 = std::max(c1, x);
    }
  }
}

void BIEVRMap::integratePoints(const std::vector<Eigen::Vector4d>& points, Voxel& voxel,
                               Eigen::MatrixXi& changed, int& r0, int& r1, int& c0, int& c1) {
  for (const auto& p : points) {
    Point p_O = voxel.T_C_W_.linear() * p.head(3) + voxel.T_C_W_.translation();

    // resbie: bounds check (upstream writes unchecked). A non-finite point,
    // or a far point whose voxel hash collides with this voxel, otherwise
    // writes outside the image (segfault in 20 s artificial blackouts).
    const double xf = std::round(p_O(0) * inv_px_size_);
    const double yf = std::round(p_O(1) * inv_px_size_);
    if (!(xf >= 0 && yf >= 0 && xf < voxel.bump_img_.cols() && yf < voxel.bump_img_.rows())) {
      continue;
    }
    const int x = static_cast<int>(xf);
    const int y = static_cast<int>(yf);

    double mean_old = voxel.bump_img_(y, x);
    double weight = voxel.bump_weights_(y, x);
    // Limit weighting so points close to the sensor don't get too powerful
    double weight_new = std::min(0.5, 1. / p(3));
    voxel.bump_weights_(y, x) += weight_new;
    voxel.bump_img_(y, x) = (mean_old * weight + weight_new * p_O(2)) / voxel.bump_weights_(y, x);
    changed(y, x) = 1;
    r0 = std::min(r0, y);
    r1 = std::max(r1, y);
    c0 = std::min(c0, x);
    c1 = std::max(c1, x);
  }
}

void BIEVRMap::dilateMask(const Eigen::MatrixXi& changed, const Eigen::MatrixXf& weights,
                          Eigen::MatrixXi& changed_dilated, int r0, int r1, int c0, int c1) {
  const int rows = changed.rows();
  const int cols = changed.cols();
  for (int i = r0; i <= r1; ++i) {
    for (int j = c0; j <= c1; ++j) {
      if (changed(i, j) != 1) continue;
      for (int k = -1; k < 2; k++) {
        int i_n = i + k;
        if (i_n < 0 || i_n >= rows) continue;
        for (int l = -1; l < 2; l++) {
          int j_n = j + l;
          if (j_n < 0 || j_n >= cols) continue;
          if (weights(i_n, j_n) <= 0.) continue;
          changed_dilated(i_n, j_n) = 1;
        }
      }
    }
  }
}

// Apply Gaussian smoothing only over valid pixels
void BIEVRMap::maskedGaussianSmooth(const Eigen::MatrixXf& image, const Eigen::MatrixXf& weights,
                                    const Eigen::MatrixXi& changed, Eigen::MatrixXf& image_smooth,
                                    int r0, int r1, int c0, int c1) {
  int rows = image.rows();
  int cols = image.cols();
  int radius = gauss_kernel_.rows() / 2;
  for (int y = r0; y <= r1; ++y) {
    for (int x = c0; x <= c1; ++x) {
      // Skip pixels not marked
      if (!changed(y, x)) {
        continue;
      }
      float weightedSum = 0.0f;
      float weightSum = 0.0f;

      for (int ky = -radius; ky <= radius; ++ky) {
        for (int kx = -radius; kx <= radius; ++kx) {
          int yy = y + ky;
          int xx = x + kx;

          if (xx < 0 || yy < 0 || xx >= cols || yy >= rows) continue;

          if (weights(yy, xx) > 0.) {
            float w = gauss_kernel_(ky + radius, kx + radius);
            weightedSum += image(yy, xx) * w;
            weightSum += w;
          }
        }
      }

      if (weightSum > 0.0f) {
        image_smooth(y, x) = weightedSum / weightSum;
      } else {
        image_smooth(y, x) = 0.0f;
      }
    }
  }
}

void BIEVRMap::computeScore(Voxel& voxel) {
  int total_count = 0;
  double sum_img_dist = 0.0;

  for (int i = 0; i < voxel.bump_img_.rows(); ++i) {
    for (int j = 0; j < voxel.bump_img_.cols(); ++j) {
      if (voxel.bump_weights_(i, j) > 0) {
        total_count++;
        double val = static_cast<double>(voxel.bump_smoothed_(i, j));
        sum_img_dist += std::abs(val);
      }
    }
  }

  voxel.mean_img_dist_ = sum_img_dist / total_count;

  // Discourage voxels with few observed pixels as they have lower probability of giving successful
  // correspondences
  if (total_count < 5) {
    voxel.mean_img_dist_ = 0;
  }
}

Eigen::Vector3d BIEVRMap::getVoxelOrigin(const Eigen::Vector3d& point) const {
  int hx = static_cast<int>(std::floor(point.x() * inv_voxel_size_));
  int hy = static_cast<int>(std::floor(point.y() * inv_voxel_size_));
  int hz = static_cast<int>(std::floor(point.z() * inv_voxel_size_));
  return Eigen::Vector3d(hx * config_.voxel_size, hy * config_.voxel_size, hz * config_.voxel_size);
}

}  // namespace bievr

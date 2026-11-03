#ifndef BIEVR_LIO_COMMON_H_
#define BIEVR_LIO_COMMON_H_

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <cstdint>

namespace bievr {

using Quaternion = Eigen::Quaternion<double>;
using Point = Eigen::Vector3d;
using V3 = Eigen::Vector3d;
using M3 = Eigen::Matrix3d;

// A 3xN cloud of points, one per column.
class Pointcloud {
 public:
  using Data = Eigen::Matrix<double, 3, Eigen::Dynamic>;

  bool empty() const { return !size(); }
  size_t size() const { return data_.cols(); }
  void resize(size_t n_points) { data_.conservativeResize(3, n_points); }

  Data& data() { return data_; }
  const Data& data() const { return data_; }

  Data::ColXpr operator[](Eigen::Index i) { return data_.col(i); }
  Data::ConstColXpr operator[](Eigen::Index i) const { return data_.col(i); }

 private:
  Data data_;
};

class Transform : public Eigen::Isometry3d {
 public:
  Transform() : Eigen::Isometry3d(Eigen::Isometry3d::Identity()) {}
  Transform(const Quaternion& rot, const Point& trans) : Eigen::Isometry3d(rot) {
    translation() = trans;
  }
  Transform(const Eigen::Isometry3d& iso) : Eigen::Isometry3d(iso) {}
  Quaternion quaternion() const { return Quaternion(linear()); }
};

inline Pointcloud operator*(const Transform& T, Pointcloud P) {
  P.data() = T * P.data().colwise().homogeneous();
  return P;
}

inline uint64_t ExpandBits(int32_t v) {
  uint64_t x = static_cast<uint64_t>(v + (1 << 20)) & 0x1fffff;
  x = (x | (x << 32)) & 0x1f00000000ffffULL;
  x = (x | (x << 16)) & 0x1f0000ff0000ffULL;
  x = (x | (x << 8)) & 0x100f00f00f00f00fULL;
  x = (x | (x << 4)) & 0x10c30c30c30c30c3ULL;
  x = (x | (x << 2)) & 0x1249249249249249ULL;
  return x;
}

inline size_t hashIndexVoxel(const Eigen::Vector3i& voxel) {
  uint64_t morton =
      ExpandBits(voxel.x()) | (ExpandBits(voxel.y()) << 1) | (ExpandBits(voxel.z()) << 2);
  return static_cast<std::size_t>(morton);
}

}  // namespace bievr

#endif  // BIEVR_LIO_COMMON_H_

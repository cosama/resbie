#pragma once

// resbie: RESPLE's common_utils without ROS. Parameter reading, message
// helpers, the per-driver point types and the ikd-tree plane fit are gone;
// PointData carries a BIEVR map linearization instead of nearest neighbours.

#include <eigen3/Eigen/Dense>
#include <cmath>
#include <omp.h>

#include "utils/eigen_utils.hpp"

inline int NUM_OF_THREAD = 5;

struct ImuData {
    int64_t time_ns;
    Eigen::Vector3d gyro;
    Eigen::Vector3d accel;
    Eigen::Matrix<double, 6, 24> H;
    Eigen::Matrix<double, 6, 1> imu_itp;
    ImuData(){}
    ImuData(const int64_t s, const Eigen::Vector3d& w, const Eigen::Vector3d& a)
      : time_ns(s), gyro(w), accel(a) {}

    ImuData(const ImuData& other) : time_ns(other.time_ns), gyro(other.gyro), accel(other.accel),
    H(other.H), imu_itp(other.imu_itp) {
    }
    ImuData& operator=(const ImuData& other) = default;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

class CommonUtils
{
public:
    static Eigen::Vector3d R2ypr(const Eigen::Matrix3d &R)
    {
        Eigen::Vector3d n = R.col(0);
        Eigen::Vector3d o = R.col(1);
        Eigen::Vector3d a = R.col(2);

        Eigen::Vector3d ypr(3);
        double y = atan2(n(1), n(0));
        double p = atan2(-n(2), n(0) * cos(y) + n(1) * sin(y));
        double r = atan2(a(0) * sin(y) - a(1) * cos(y), -o(0) * sin(y) + o(1) * cos(y));
        ypr(0) = y;
        ypr(1) = p;
        ypr(2) = r;

        return ypr / M_PI * 180.0;
    }

    template <typename Derived>
    static Eigen::Matrix<typename Derived::Scalar, 3, 3> ypr2R(const Eigen::MatrixBase<Derived> &ypr)
    {
        typedef typename Derived::Scalar Scalar_t;

        Scalar_t y = ypr(0) / 180.0 * M_PI;
        Scalar_t p = ypr(1) / 180.0 * M_PI;
        Scalar_t r = ypr(2) / 180.0 * M_PI;

        Eigen::Matrix<Scalar_t, 3, 3> Rz;
        Rz << cos(y), -sin(y), 0,
                sin(y), cos(y), 0,
                0, 0, 1;

        Eigen::Matrix<Scalar_t, 3, 3> Ry;
        Ry << cos(p), 0., sin(p),
                0., 1., 0.,
                -sin(p), 0., cos(p);

        Eigen::Matrix<Scalar_t, 3, 3> Rx;
        Rx << 1., 0., 0.,
                0., cos(r), -sin(r),
                0., sin(r), cos(r);

        return Rz * Ry * Rx;
    }

    static Eigen::Matrix3d g2R(const Eigen::Vector3d &g)
    {
        Eigen::Matrix3d R0;
        Eigen::Vector3d ng1 = g.normalized();
        Eigen::Vector3d ng2{0, 0, 1.0};
        R0 = Eigen::Quaterniond::FromTwoVectors(ng1, ng2).toRotationMatrix();
        double yaw = CommonUtils::R2ypr(R0).x();
        R0 = CommonUtils::ypr2R(Eigen::Vector3d{-yaw, 0, 0}) * R0;
        return R0;
    }
};

// One LiDAR point as a filter measurement. Points arrive already in the IMU
// (body) frame, so the spline pose maps them to the world directly.
//
// resbie: the measurement is BIEVR's bump-image residual linearized at the
// point's world position: r(p_w) ~ normvec . p_w + dist, where normvec is the
// residual gradient (not unit length) and dist the matching offset. RESPLE's
// own point-to-plane model is the special case of a unit normal. var_scale is
// the Huber IRLS factor on var_pt for that linearization.
struct PointData {
    int64_t time_ns = 0;
    Eigen::Vector3d pt_b = Eigen::Vector3d::Zero();
    Eigen::Vector3d pt_w = Eigen::Vector3d::Zero();
    Eigen::Vector3d normvec = Eigen::Vector3d::Zero();
    bool if_valid = false;
    double dist = 0;
    double zp = 0;
    Eigen::Matrix<double, 1, 24> H = Eigen::Matrix<double, 1, 24>::Zero();
    double var_pt = 0.01;
    double var_scale = 1.0;

    PointData() {}
    PointData(const Eigen::Vector3d& p_b, int64_t t_ns, double w_pt)
        : time_ns(t_ns), pt_b(p_b), var_pt(w_pt) {}

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

struct Parameters {
    Eigen::Vector3d cov_acc;
    Eigen::Vector3d cov_gyro;
    Eigen::Vector3d gravity;
    double nn_thresh;
    double coeff_cov;

    Parameters() {}

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

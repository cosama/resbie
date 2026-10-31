#pragma once

#include <eigen3/Eigen/Geometry>
#include <eigen3/Eigen/Dense>
#include <boost/math/special_functions/sinc.hpp>
#include <cmath>

class Quater
{
public:
    static void Qleft(const Eigen::Quaterniond &q, Eigen::Matrix4d &ans)
    {
        double q1 = q.w();
        double q2 = q.x();
        double q3 = q.y();
        double q4 = q.z();
        ans << q1, -q2, -q3, -q4,
                q2,  q1, -q4,  q3,
                q3,  q4,  q1, -q2,
                q4, -q3,  q2,  q1;
    }

    static void Qright(const Eigen::Quaterniond &q, Eigen::Matrix4d &ans)
    {
        double q1 = q.w();
        double q2 = q.x();
        double q3 = q.y();
        double q4 = q.z();
        ans << q1, -q2, -q3, -q4,
                q2,  q1,  q4, -q3,
                q3, -q4,  q1,  q2,
                q4,  q3, -q2,  q1;
    }

    static void exp(const Eigen::Vector3d &v, Eigen::Quaterniond& q)
    {
        double v_norm = v.norm();
        q.w() = std::cos(v_norm);
        q.vec() = boost::math::sinc_pi(v_norm) * v;
    }

    template <typename Derived>
    static Eigen::Quaternion<typename Derived::Scalar> positify(const Eigen::QuaternionBase<Derived> &q)
    {
        Eigen::Quaternion<typename Derived::Scalar> p(q.w(), q.x(), q.y(), q.z());
        return q.w() >= (typename Derived::Scalar)(0.0) ? p : Eigen::Quaternion<typename Derived::Scalar>(-q.w(), -q.x(), -q.y(), -q.z());
    }

    static void dexp(const Eigen::Vector3d& v, Eigen::Quaterniond& q, Eigen::Matrix<double,4,3>& J)
    {
        double v_norm = v.norm();
        if (v_norm == 0.0) {
            J.row(0).setZero();
            J.bottomRows<3>().setIdentity();
            q.setIdentity();
            return;
        }
        double sinc = boost::math::sinc_pi(v_norm);
        q.w() = std::cos(v_norm);
        q.vec() = sinc * v;
        double v1 = v(0);
        double v2 = v(1);
        double v3 = v(2);
        double tmp = (q.w() - sinc) / (v_norm * v_norm);
        double tmp_v1 = tmp * v1;
        double tmp_v2 = tmp * v2;
        double tmp_v11 = tmp_v1 * v1 + sinc;
        double tmp_v12 = tmp_v1 * v2;
        double tmp_v13 = tmp_v1 * v3;
        double tmp_v22 = tmp_v2 * v2 + sinc;
        double tmp_v23 = tmp_v2 * v3;
        double tmp_v33 = tmp * v3 * v3 + sinc;
        J << -v1 * sinc, -v2 * sinc, -v3 * sinc,
                tmp_v11, tmp_v12, tmp_v13,
                tmp_v12, tmp_v22, tmp_v23,
                tmp_v13, tmp_v23, tmp_v33;
    }

    static void drot(const Eigen::Vector3d& v,const Eigen::Quaterniond& q, Eigen::Matrix<double, 3, 4> &J)
    {
        double qw = q.w();
        double qx = q.x();
        double qy = q.y();
        double qz = q.z();
        double v1 = v(0);
        double v2 = v(1);
        double v3 = v(2);
        Eigen::Vector3d vec;
        vec << (v1 * qw + v2 * qz - v3 * qy) * 2,
                (v2 * qw - v1 * qz + v3 * qx) * 2,
                (v3 * qw + v1 * qy - v2 * qx) * 2;
        double tmp = (v1 * qx + v2 * qy + v3 * qz) * 2;
        J << vec(0), tmp, -vec(2), vec(1),
                vec(1), vec(2), tmp, -vec(0),
                vec(2), -vec(1), vec(0), tmp;
    }

    static void drotInv(const Eigen::Vector3d& v,const Eigen::Quaterniond& q, Eigen::Matrix<double, 3, 4> &J)
    {
        double qw = q.w();
        double qx = q.x();
        double qy = q.y();
        double qz = q.z();
        double v1 = v(0);
        double v2 = v(1);
        double v3 = v(2);
        Eigen::Vector3d vec;
        vec << (v1 * qw - v2 * qz + v3 * qy) * 2,
                (v2 * qw + v1 * qz - v3 * qx) * 2,
                (v3 * qw - v1 * qy + v2 * qx) * 2;
        double tmp = (v1 * qx + v2 * qy + v3 * qz)*2;
        J << vec(0), tmp, vec(2), -vec(1),
                vec(1), -vec(2), tmp, vec(0),
                vec(2), vec(1), -vec(0), tmp;
    }
};
// resbie: SO(3) Jacobian of the rotation-vector exponential (full angle phi,
// Exp(phi + e) ~ Exp(phi) Exp(Jr(phi) e) ~ Exp(Jl(phi) e) Exp(phi)).
struct SO3
{
    static Eigen::Matrix3d skew(const Eigen::Vector3d& v)
    {
        Eigen::Matrix3d m;
        m << 0, -v.z(), v.y(),
             v.z(), 0, -v.x(),
             -v.y(), v.x(), 0;
        return m;
    }

    // Jr(phi) = I - a [phi]x + b [phi]x^2, a = (1 - cos t)/t^2, b = (t - sin t)/t^3.
    static Eigen::Matrix3d rightJacobian(const Eigen::Vector3d& phi)
    {
        const double t2 = phi.squaredNorm();
        double a, b;
        if (t2 < 1e-8) {
            a = 0.5 - t2 / 24.0;
            b = 1.0 / 6.0 - t2 / 120.0;
        } else {
            const double t = std::sqrt(t2);
            a = (1.0 - std::cos(t)) / t2;
            b = (t - std::sin(t)) / (t2 * t);
        }
        const Eigen::Matrix3d K = skew(phi);
        return Eigen::Matrix3d::Identity() - a * K + b * K * K;
    }
};

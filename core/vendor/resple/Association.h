#pragma once

// resbie: RESPLE's data association against BIEVR's bump-image map instead
// of an ikd-tree. Each point is placed by the spline at its own timestamp and
// linearized on its voxel's oriented height image exactly as BIEVR's
// LsqRegistration::linearize does, then on the spline's control points with
// RESPLE's Jacobians (upstream's Estimator::prepLiDAR, which placed every point
// a second time), once per filter iteration.

#include <eigen3/Eigen/Geometry>
#include <eigen3/Eigen/Dense>
#include "utils/common_utils.h"
#include "SplineState.h"
#include "bievr_lio/bievr_map.h"

struct MapAssociationConfig {
    double huber_delta = 0.1;   // BIEVR optimization.huber_delta, m
};

class Association
{
public:
    static Eigen::Vector3d pointBodyToWorld(int64_t t_ns, const SplineState* spline, const Eigen::Vector3d& p_b)
    {
        Eigen::Quaterniond q;
        Eigen::Vector3d pos = spline->itpPosition(t_ns);
        spline->itpQuaternion(t_ns, &q);
        return q * p_b + pos;
    }

    // Residual of a world point against the map, its gradient d r / d p_w, and
    // the voxel it was measured in. False when no observed voxel with a valid
    // pixel covers the point.
    static bool mapResidual(const bievr::BIEVRMap& map, const Eigen::Vector3d& p_w, double& r, Eigen::Vector3d& grad,
                            size_t& voxel_hash)
    {
        size_t hash = map.hashIndex(p_w);
        const bievr::Voxel* voxel = map.getVoxel(hash);
        if (!voxel) {
            voxel = map.nearestNeighborVoxel(p_w, hash);
            if (!voxel) return false;
        }
        const double inv_size = map.inv_px_size;
        const bievr::Transform& T_C_W = voxel->T_C_W_;
        const Eigen::Vector3d p_o = T_C_W * p_w;
        double I = 0.0, dIdx = 0.0, dIdy = 0.0;
        if (!bievr::sampleValueAndGradient(voxel, p_o.x() * inv_size, p_o.y() * inv_size, I, dIdx, dIdy)) {
            return false;
        }
        const Eigen::Vector3d dr_dpo(-dIdx * inv_size, -dIdy * inv_size, 1.0);
        r = p_o.z() - I;
        grad = T_C_W.linear().transpose() * dr_dpo;
        voxel_hash = hash;
        return true;
    }

    // Associates and linearizes every point at the spline's current estimate:
    // residual zp, its Jacobian H on the active control points, and the Huber
    // factor. Points are all in the active window (Odometry collects them so).
    static void linearize(int& effect_num_k, const SplineState& spline, const bievr::BIEVRMap& map,
                          const MapAssociationConfig& cfg, Eigen::aligned_deque<PointData>& pt_meas)
    {
        const int num_pt = pt_meas.size();
        const int rcp_start = spline.numKnots() - 4;
        #pragma omp parallel for num_threads(NUM_OF_THREAD) schedule(dynamic)
        for (int i = 0; i < num_pt; i++) {
            PointData& pt_data = pt_meas[i];
            Eigen::Quaterniond q_itp;
            Jacobian43 J_ortdel;
            Jacobian J_pos;
            spline.itpQuaternion(pt_data.time_ns, &q_itp, nullptr, &J_ortdel);
            const Eigen::Vector3d p_itp = spline.itpPosition(pt_data.time_ns, &J_pos);
            pt_data.pt_w = q_itp * pt_data.pt_b + p_itp;
            double r;
            Eigen::Vector3d grad;
            pt_data.if_valid = mapResidual(map, pt_data.pt_w, r, grad, pt_data.voxel);
            if (!pt_data.if_valid) continue;
            pt_data.normvec = grad;
            pt_data.zp = r;
            const double abs_r = std::abs(r);
            pt_data.var_scale = abs_r <= cfg.huber_delta ? 1.0 : abs_r / cfg.huber_delta;

            Eigen::Matrix<double, 3, 4> drot;
            Quater::drotInv(pt_data.pt_b, q_itp, drot);
            const Eigen::Matrix<double, 1, 4> tmp = grad.transpose() * drot;
            pt_data.H.setZero();
            for (int k = 0; k < (int) J_pos.d_val_d_knot.size(); k++) {
                const int j = (int) J_pos.start_idx + k - rcp_start;
                if (j >= 0) {
                    pt_data.H.block<1, 3>(0, j*6) = grad.transpose() * J_pos.d_val_d_knot[k];
                    pt_data.H.block<1, 3>(0, j*6 + 3) = tmp * J_ortdel.d_val_d_knot[k];
                }
            }
        }
        for (int i = 0; i < num_pt; i++) {
            if (pt_meas[i].if_valid) effect_num_k++;
        }
    }
};

#pragma once

// resbie: RESPLE's data association against BIEVR's bump-image map instead
// of an ikd-tree. Each point is placed by the spline at its own timestamp and
// linearized on its voxel's oriented height image exactly as BIEVR's
// LsqRegistration::linearize does; the result is handed to the filter in
// RESPLE's plane form (normvec, dist), so Estimator's Jacobians stay unchanged.

#include <eigen3/Eigen/Geometry>
#include <eigen3/Eigen/Dense>
#include "utils/common_utils.h"
#include "SplineState.h"
#include "bievr_lio/bievr_map.h"
#include "bievr_lio/ls_optimizer.h"

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

    // Residual of a world point against the map and its gradient d r / d p_w.
    // False when no observed voxel with a valid pixel covers the point.
    static bool mapResidual(const bievr::BIEVRMap& map, const MapAssociationConfig& cfg,
                            const Eigen::Vector3d& p_w, double& r, Eigen::Vector3d& grad)
    {
        size_t hash = map.hashIndex(p_w);
        const bievr::Voxel* voxel = map.getVoxel(hash);
        if (!voxel) {
            if (!map.nearestVoxel(p_w, hash)) return false;
            voxel = map.getVoxel(hash);
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
        return true;
    }

    static void findCorresp(int& effect_num_k, const SplineState* spline, const bievr::BIEVRMap& map,
                            const MapAssociationConfig& cfg, Eigen::aligned_deque<PointData>& pt_meas)
    {
        int num_pt = pt_meas.size();
        #pragma omp parallel for num_threads(NUM_OF_THREAD) schedule(dynamic)
        for (int i = 0; i < num_pt; i++) {
            PointData& pt_data = pt_meas[i];
            pt_data.if_valid = false;
            if (!(spline->numKnots() == 4) && !(pt_data.time_ns <= spline->maxTimeNs() && pt_data.time_ns >= spline->maxTimeNs() - 4*spline->getKnotTimeIntervalNs())) {
                continue;
            }
            pt_data.pt_w = pointBodyToWorld(pt_data.time_ns, spline, pt_data.pt_b);
            double r;
            Eigen::Vector3d grad;
            if (!mapResidual(map, cfg, pt_data.pt_w, r, grad)) continue;
            pt_data.if_valid = true;
            pt_data.normvec = grad;
            pt_data.dist = r - grad.dot(pt_data.pt_w);
            const double abs_r = std::abs(r);
            pt_data.var_scale = abs_r <= cfg.huber_delta ? 1.0 : abs_r / cfg.huber_delta;
        }
        for (int i = 0; i < num_pt; i++) {
            if (pt_meas[i].if_valid) effect_num_k++;
        }
    }
};

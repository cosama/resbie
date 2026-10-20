#pragma once

#include "SplineState.h"
#include "Association.h"

template<int XSIZE>
class Estimator
{
  public:
    static const int CP_SIZE = 24;
    static const int BA_OFFSET = 24;
    static const int BG_OFFSET = 27;  
    int n_iter = 1;

    Estimator() {};

    void setState(int64_t dt_ns, int64_t start_t_ns, const Eigen::Vector3d& t0, const Eigen::Quaterniond& q0, 
        const Eigen::Matrix<double, XSIZE, XSIZE>& Q, const Eigen::Matrix<double, XSIZE, XSIZE>& P)
    {
        spl.init(dt_ns, 0, start_t_ns, 0, t0, q0);
        for (int i = 0; i < 4; i++) {
            spl.addOneStateKnot(t0, Eigen::Vector3d::Zero());
        }        
        cov_sys = Q;
        cov_rcp = P;
        a_mat = Eigen::Matrix<double, XSIZE, XSIZE>::Zero();
        Eigen::Matrix<double, 6, 6> matblock = Eigen::Matrix<double, 6, 6>::Zero();
        matblock.topLeftCorner<3, 3>().setIdentity();
        matblock.bottomRightCorner<3, 3>().setIdentity();

        a_mat.block(0, 6, 6, 6) = matblock;
        a_mat.block(6, 12, 6, 6) = matblock;
        a_mat.block(12, 18, 6, 6) = matblock;
        a_mat.block(18, 0, 3, 3) = - Eigen::Matrix3d::Identity();
        a_mat.block(18, 12, 3, 3) = 2 * Eigen::Matrix3d::Identity();
        a_mat.block(21, 9, 3, 3) = Eigen::Matrix3d::Identity();   
        if constexpr (XSIZE == 30) {
            a_mat.block(BA_OFFSET, BA_OFFSET, 3, 3) = Eigen::Matrix3d::Identity();
            a_mat.block(BG_OFFSET, BG_OFFSET, 3, 3) = Eigen::Matrix3d::Identity();    
        }
    }

    Eigen::Matrix<double, XSIZE, 1> getState()
    {
        Eigen::Matrix<double, CP_SIZE, 1> cps_win = spl.getRCPs();
        Eigen::Matrix<double, XSIZE, 1> state;
        if constexpr (XSIZE == 24) {
            state = cps_win;
        } else {
            state << cps_win, ba, bg;
        }
        return state;
    }  

    void updateIEKFLiDARInertial(Eigen::aligned_deque<PointData>& pt_meas, const bievr::BIEVRMap& map, const MapAssociationConfig& assoc,
        Eigen::aligned_deque<ImuData>& imu_meas, const Eigen::Vector3d& g, const Eigen::Vector3d& cov_acc, const Eigen::Vector3d& cov_gyro)
    {
        const Eigen::Matrix<double, XSIZE, XSIZE> cov_prop = cov_rcp;
        Eigen::Matrix<double, XSIZE, 1> rcp_prop = getState();
        bool converged = true;
        int num_tot_eff = 0;
        int t = 0;
        point_nis.clear();
        imu_samples = imu_meas.size();
        acc_gated = 0;
        bool updated = false;
        bool reassociated = false;
        for (int i = 0; i < max_iter; i++) {
            record_nis_ = (i == 0);
            Eigen::Matrix<double, XSIZE, 1> rcpi = getState();
            if (converged) {
                num_tot_eff = 0;
                Association::findCorresp(num_tot_eff, &spl, map, assoc, pt_meas);
                reassociated = true;
            } else {
                reassociated = false;
            }
            if (num_tot_eff > 0 && imu_meas.empty()) {
                updateLiDAR(pt_meas, num_tot_eff, rcp_prop, cov_prop, reassociated);
            } else if (num_tot_eff > 0) {
                updateLiDARInertial(pt_meas, imu_meas, num_tot_eff, rcp_prop, cov_prop, g, cov_acc, cov_gyro, reassociated);
            } else if (!imu_meas.empty()) {
                // resbie: no matched point -> IMU-only update. Upstream broke
                // out here, so a LiDAR gap also dropped every IMU sample in it.
                updateLiDARInertial(pt_meas, imu_meas, 0, rcp_prop, cov_prop, g, cov_acc, cov_gyro, reassociated);
            } else {
                // resbie: re-association found nothing to measure. Keep the
                // last iteration's estimate with its posterior covariance
                // (upstream kept the prior covariance for an updated state).
                if (updated) cov_rcp = 0.5*(P_post_ + P_post_.transpose());
                break;
            }
            updated = true;
            converged = true;
            Eigen::Matrix<double, XSIZE, 1> state_af = getState();
            if ((state_af - rcpi).norm() > eps) {
                converged = false;
            } else {
                t++;
            }
            if(!t && i == max_iter - 2) {
                converged = true;
            }
            if ((t > n_iter) || (i == max_iter - 1)) {
                // resbie: the last update's posterior covariance (see update());
                // upstream's (I - KH) P lost positive definiteness on the first
                // dense update after a long gap (huge P, precise points).
                cov_rcp = 0.5*(P_post_ + P_post_.transpose());
                break;
            }              
        }
    }

    void propRCP(int64_t t)
    {
        // resbie: cov_sys is the per-knot process noise; only add it when a
        // new knot is actually appended, never on a no-op call with maxTimeNs() >= t.
        if (spl.maxTimeNs() >= t) return;
        while (spl.maxTimeNs() < t) {
            Eigen::Matrix<double, 24, 1> cps_win = spl.getRCPs();
            Eigen::Matrix<double, 6, 1> cp_prop_pos = 2*cps_win.block<6, 1>(12, 0) - cps_win.block<6, 1>(0, 0);
            Eigen::Vector3d delta = cps_win.segment<3>(9);
            spl.addOneStateKnot(cp_prop_pos.head<3>(), delta);
            cov_rcp = a_mat * cov_rcp * a_mat.transpose() + cov_sys;
        }
    }

    SplineState* getSpline() {
        return &spl;
    }

    // resbie: Mahalanobis point gate, zp^2 <= gate_chi2 * (HPH' + R):
    // narrow when the filter is certain, wide when it is not. (RESPLE's gate
    // accepted |zp| < nn_thresh, or anything while HPH' + R < coeff_cov * R,
    // i.e. it widened only when certain.)
    double gate_chi2 = 9.0;
    bool gateAccepts(double zp, double lid_cov) const { return zp * zp <= gate_chi2 * lid_cov; }

    // resbie: consistency check. Normalized innovation squared of each
    // matched point in the last update's first iteration (prior, before the
    // update pulls it in); ~1 on average if HPH' + R is honest.
    std::vector<double> point_nis;
    bool record_nis_ = false;
    // resbie: IMU samples in the last update's first iteration, and how many
    // of their acceleration axes the 10 m/s^2 residual gate dropped.
    size_t imu_samples = 0;
    size_t acc_gated = 0;
    // resbie: matched points the Mahalanobis gate rejected in the last iteration.
    size_t points_gated = 0;

    // resbie: filter covariance (control points, then biases).
    const Eigen::Matrix<double, XSIZE, XSIZE>& covariance() const { return cov_rcp; }

    // resbie: bias estimates for status reporting.
    Eigen::Vector3d biasAcc() const { return ba; }
    Eigen::Vector3d biasGyro() const { return bg; }

  private:
    using MatX = Eigen::Matrix<double, XSIZE, XSIZE>;

    SplineState spl;
    Eigen::Matrix<double, XSIZE, XSIZE> cov_rcp;  
    Eigen::Matrix<double, XSIZE, XSIZE> cov_sys; 
    Eigen::Matrix<double, XSIZE, XSIZE> a_mat;   
    Eigen::Vector3d bg = Eigen::Vector3d::Zero();
    Eigen::Vector3d ba = Eigen::Vector3d::Zero();     
    Eigen::Matrix<double, XSIZE, XSIZE> KH;
    Eigen::Matrix<double, XSIZE, XSIZE> P_post_;
    Eigen::Matrix<double, Eigen::Dynamic, 24> H_24_buf_;
    Eigen::Matrix<double, 24, Eigen::Dynamic> HT_R_inv_24_buf_;
    Eigen::Matrix<double, XSIZE, Eigen::Dynamic> K_buf_;
    Eigen::Matrix<double, Eigen::Dynamic, 1> innv_buf_;
    Eigen::Matrix<double, Eigen::Dynamic, 1> R_inv_buf_;
    std::vector<int> imu_row_offsets_;
    int max_iter = 5;
    double eps = 0.1;

    void ensureMeasBuffers(int n)
    {
        if (H_24_buf_.rows() < n) {
            H_24_buf_.resize(n, 24);
            HT_R_inv_24_buf_.resize(24, n);
            K_buf_.resize(XSIZE, n);
            innv_buf_.resize(n, 1);
            R_inv_buf_.resize(n, 1);
        }
    }

    // resbie: square-root information form, S_inv = L (I + L' H'R^-1H L)^-1 L'
    // with P_prop = L L' (pivoted LDLT). Never inverts P_prop: its control
    // points are strongly correlated and a long gap makes it huge, so an
    // explicit inverse loses digits even when LLT reports success.
    void solvePosteriorCovariance(const MatX& P_prop, const MatX& HT_R_inv_H, MatX& S_inv) const
    {
        const MatX I_X = MatX::Identity();
        const Eigen::LDLT<MatX> ldlt_P(P_prop);
        const MatX L_p =
            ldlt_P.transpositionsP().transpose() *
            MatX(ldlt_P.matrixL()) *
            ldlt_P.vectorD().cwiseMax(0.0).cwiseSqrt().asDiagonal();
        const MatX M = I_X + L_p.transpose() * HT_R_inv_H * L_p;
        S_inv.noalias() = L_p * M.llt().solve(L_p.transpose());
        S_inv = 0.5 * (S_inv + S_inv.transpose()).eval();
    }

    void prepIMU(ImuData& imu_data, const Eigen::Vector3d& g)
    {
        Eigen::Quaterniond q_itp;
        Eigen::Vector3d rot_vel;
        Jacobian43 J_ortdel;
        Jacobian J_line_acc;
        Jacobian33 J_gyro;
        spl.itpQuaternion(imu_data.time_ns, &q_itp, &rot_vel, &J_ortdel, &J_gyro);
        Eigen::Vector3d a_w_no_g = spl.itpPosition<2>(imu_data.time_ns, &J_line_acc);
        Eigen::Vector3d a_w = a_w_no_g + g;
        Eigen::Matrix3d RT = q_itp.inverse().toRotationMatrix();   
        Eigen::Matrix<double, 3, 4> drot;
        Quater::drot(a_w, q_itp, drot);
        Eigen::Matrix<double, 6, 24> Hi = Eigen::Matrix<double, 6, 24>::Zero();
        int recur_st_id = spl.numKnots() - 4;
        for (int i = 0; i < (int) J_line_acc.d_val_d_knot.size(); i++) {
            int j = J_line_acc.start_idx + i - recur_st_id;
            if (j >= 0) {
                Hi.block(0, j*6, 3, 3) = RT * J_line_acc.d_val_d_knot[i];
                Hi.block(0, j*6 + 3, 3, 3) = drot * J_ortdel.d_val_d_knot[i];
                Hi.block(3, j*6 + 3, 3, 3) = J_gyro.d_val_d_knot[i];                
            }
        }       
        imu_data.imu_itp.head<3>() = RT * a_w + ba;
        imu_data.imu_itp.tail<3>() = rot_vel + bg;
        imu_data.H = Hi;
    }    

    void prepLiDAR(PointData& pt_data) const    
    {
        if (pt_data.if_valid) { 
            Eigen::Matrix<double, 1, 24> Hi = Eigen::Matrix<double, 1, 24>::Zero();
            Eigen::Quaterniond q_itp;
            Jacobian43 J_ortdel;
            Jacobian J_pos;
            spl.itpQuaternion(pt_data.time_ns, &q_itp, nullptr, &J_ortdel);
            Eigen::Vector3d p_itp = spl.itpPosition(pt_data.time_ns, &J_pos);
            // resbie: points are already in the body frame.
            Eigen::Vector3d pt_w = q_itp * pt_data.pt_b + p_itp;
            pt_data.zp = pt_data.normvec.dot(pt_w) + pt_data.dist;

            Eigen::Matrix<double, 3, 4> drot;
            Quater::drotInv(pt_data.pt_b, q_itp, drot);
            Eigen::Matrix<double, 1, 4> tmp = pt_data.normvec.transpose() * drot;
            int RCP_st_id = spl.numKnots() - 4;
            for (int i = 0; i < (int) J_pos.d_val_d_knot.size(); i++) {
                int j = (int) J_pos.start_idx + i - RCP_st_id;
                if (j >= 0) {
                    Hi.block(0, j*6, 1, 3) = pt_data.normvec.transpose() * J_pos.d_val_d_knot[i];
                    Hi.block(0, j*6 + 3, 1, 3) = tmp * J_ortdel.d_val_d_knot[i];
                }
            }  
            pt_data.H = Hi;
        }
    } 

    void updateState(const Eigen::Matrix<double, XSIZE, 1>& xupd)
    {
        Eigen::Matrix<double, CP_SIZE, 1> cp_win = xupd.segment(0, CP_SIZE);
        spl.updateRCPs(cp_win);
        if constexpr (XSIZE == 30) {
            ba = xupd.segment(BA_OFFSET, 3);
            bg = xupd.segment(BG_OFFSET, 3);   
        }
    }        

    bool updateLiDAR(Eigen::aligned_deque<PointData>& pt_meas, int num_valid, const Eigen::Matrix<double, XSIZE, 1>& x_prop, 
        const Eigen::Matrix<double, XSIZE, XSIZE>& P_prop, bool reassociated = false)
    {
        size_t num_pt = pt_meas.size();
        if (!reassociated) {
            #pragma omp parallel for num_threads(NUM_OF_THREAD) schedule(dynamic)
            for (size_t i = 0; i < num_pt; i++) {
                PointData& pt_data = pt_meas[i];
                prepLiDAR(pt_data);
            }
        }
        if (num_valid <= 0) {
            P_post_ = P_prop;
            KH.setZero();
            return false;
        }
        const Eigen::Matrix<double, 24, 24> cov = cov_rcp.template topLeftCorner<24, 24>();
        ensureMeasBuffers(num_valid);
        auto H_24 = H_24_buf_.topRows(num_valid);
        auto innv = innv_buf_.head(num_valid);
        auto R_inv = R_inv_buf_.head(num_valid);
        auto HT_R_inv_24 = HT_R_inv_24_buf_.leftCols(num_valid);
        auto K = K_buf_.leftCols(num_valid);
        int idx_offset = 0;
        points_gated = 0;
        for(size_t i = 0; i < num_pt; i++) {
            const PointData& pt_data = pt_meas[i];
            if (pt_data.if_valid) {
                // resbie: var_scale is the Huber weight of this linearization.
                const double var_pt = pt_data.var_pt * pt_data.var_scale;
                double lid_cov = (pt_data.H * cov * pt_data.H.transpose())(0, 0) + var_pt;
                if (record_nis_) point_nis.push_back(pt_data.zp * pt_data.zp / lid_cov);
                if (gateAccepts(pt_data.zp, lid_cov)) {
                    innv(idx_offset) = - pt_data.zp;
                    H_24.row(idx_offset) = pt_data.H;
                } else {
                    innv(idx_offset) = 0.0;
                    H_24.row(idx_offset).setZero();
                    ++points_gated;
                }
                R_inv(idx_offset) = 1.0 / var_pt;
                idx_offset++;
            }
        }        
        HT_R_inv_24.noalias() = (H_24.transpose().array().rowwise() * R_inv.transpose().array()).matrix();
        Eigen::Matrix<double, XSIZE, XSIZE> HT_R_inv_H = Eigen::Matrix<double, XSIZE, XSIZE>::Zero();
        HT_R_inv_H.template topLeftCorner<24, 24>().noalias() = HT_R_inv_24 * H_24;

        Eigen::Matrix<double, XSIZE, XSIZE> S_inv;
        solvePosteriorCovariance(P_prop, HT_R_inv_H, S_inv);

        K.noalias() = S_inv.template leftCols<24>() * HT_R_inv_24;
        KH.noalias() = S_inv * HT_R_inv_H;
        P_post_ = S_inv;
        Eigen::Matrix<double, XSIZE, 1> delta_cur = (getState() - x_prop);
        Eigen::Matrix<double, XSIZE, 1> deltax = KH * delta_cur + K * innv - delta_cur;
        Eigen::Matrix<double, XSIZE, 1> RCPs_post;
        RCPs_post.noalias() = getState() + deltax;
        updateState(RCPs_post);
        return true;
    }           

    void updateLiDARInertial(Eigen::aligned_deque<PointData>& pt_meas, Eigen::aligned_deque<ImuData>& imu_meas, int num_valid, const Eigen::Matrix<double, XSIZE, 1>& x_prop, 
        const Eigen::Matrix<double, XSIZE, XSIZE>& P_prop, const Eigen::Vector3d& g, const Eigen::Vector3d& cov_acc, const Eigen::Vector3d& cov_gyro,
        bool reassociated = false)
    {
        Eigen::Matrix<double, 6, 1> cov_imu_inv =  Eigen::Matrix<double, 6, 1>(1/cov_acc[0], 1/cov_acc[1], 1/cov_acc[2], 1/cov_gyro[0], 1/cov_gyro[1], 1/cov_gyro[2]);
        if (!reassociated) {
            #pragma omp parallel for num_threads(NUM_OF_THREAD) schedule(dynamic)
            for (size_t i = 0; i < pt_meas.size(); i++) {
                PointData& pt_data = pt_meas[i];
                prepLiDAR(pt_data); 
            }
        }
        #pragma omp parallel for num_threads(NUM_OF_THREAD) 
        for (size_t i = 0; i < imu_meas.size(); i++) {
            prepIMU(imu_meas[i], g);
        }                
        points_gated = 0;
        int dim_meas = 6*imu_meas.size() + num_valid;
        if (dim_meas <= 0) {
            P_post_ = P_prop;
            KH.setZero();
            return;
        }
        const Eigen::Matrix<double, 24, 24> cov = cov_rcp.template topLeftCorner<24, 24>();
        ensureMeasBuffers(dim_meas);
        auto H_24 = H_24_buf_.topRows(dim_meas);
        auto innv = innv_buf_.head(dim_meas);
        auto R_inv = R_inv_buf_.head(dim_meas);
        auto HT_R_inv_24 = HT_R_inv_24_buf_.leftCols(dim_meas);
        auto K = K_buf_.leftCols(dim_meas);
        imu_row_offsets_.resize(imu_meas.size());
        int idx_offset = 0;
        size_t id_imu = 0;
        size_t id_pt = 0;
        for (size_t j = 0; j < imu_meas.size() + pt_meas.size(); j++) {
            if ((id_pt < pt_meas.size() && id_imu < imu_meas.size() && pt_meas[id_pt].time_ns < imu_meas[id_imu].time_ns) ||
                (id_pt < pt_meas.size() && id_imu >= imu_meas.size())) {
                    PointData& pt_data = pt_meas[id_pt];
                    if (pt_data.if_valid) {
                        // resbie: var_scale is the Huber weight of this linearization.
                        const double var_pt = pt_data.var_pt * pt_data.var_scale;
                        double lid_cov = (pt_data.H * cov * pt_data.H.transpose())(0, 0) + var_pt;
                        if (record_nis_) point_nis.push_back(pt_data.zp * pt_data.zp / lid_cov);
                        if (gateAccepts(pt_data.zp, lid_cov)) {
                            innv(idx_offset) = - pt_data.zp;
                            H_24.row(idx_offset) = pt_data.H;
                        } else {
                            innv(idx_offset) = 0.0;
                            H_24.row(idx_offset).setZero();
                            ++points_gated;
                        }
                        R_inv(idx_offset) = 1.0 / var_pt;
                        idx_offset++;
                    }
                    id_pt++;
            } else if ((id_pt < pt_meas.size() && id_imu < imu_meas.size() && pt_meas[id_pt].time_ns >= imu_meas[id_imu].time_ns) ||
                        (id_pt >= pt_meas.size() && id_imu < imu_meas.size())) {
                    const ImuData& imu_data = imu_meas[id_imu];
                    Eigen::Matrix<double, 6, 1> imu_itp = imu_data.imu_itp;
                    Eigen::Matrix<double, 6, 24> Hi = imu_data.H;
                    Eigen::Matrix<double, 6, 1> w_imu = cov_imu_inv;

                    Eigen::Matrix<double, 6, 1> imu;
                    imu.head<3>() = imu_data.accel;
                    imu.tail<3>() = imu_data.gyro;
                    for (int i = 0; i < 3; i++) {
                        if (abs(imu(i) - imu_itp(i)) > 10.0) {
                            if (record_nis_) ++acc_gated;
                            imu(i) = 0;
                            imu_itp(i) = 0;
                            Hi.row(i).setZero();
                            w_imu(i) = 0.0;
                        } 
                        if (abs(imu(i+3) - imu_itp(i+3)) > 5.0) {
                            imu(i+3) = 0;
                            imu_itp(i+3) = 0;
                            Hi.row(i+3).setZero();
                            w_imu(i+3) = 0.0;
                        }                     
                    }
                    imu_row_offsets_[id_imu] = idx_offset;
                    innv.segment<6>(idx_offset) = imu - imu_itp;
                    H_24.block<6, 24>(idx_offset, 0) = Hi;
                    R_inv.segment<6>(idx_offset) = w_imu;  
                    idx_offset += 6;
                    id_imu++;
            }
        }        
        HT_R_inv_24.noalias() = (H_24.transpose().array().rowwise() * R_inv.transpose().array()).matrix();
        Eigen::Matrix<double, XSIZE, XSIZE> HT_R_inv_H = Eigen::Matrix<double, XSIZE, XSIZE>::Zero();
        HT_R_inv_H.template topLeftCorner<24, 24>().noalias() = HT_R_inv_24 * H_24;
        if constexpr (XSIZE == 30) {
            for (size_t m = 0; m < imu_meas.size(); m++) {
                const int row = imu_row_offsets_[m];
                for (int k = 0; k < 6; k++) {
                    HT_R_inv_H.template block<24, 1>(0, 24 + k) += HT_R_inv_24.col(row + k);
                    HT_R_inv_H(24 + k, 24 + k) += R_inv(row + k);
                }
            }
            HT_R_inv_H.template block<6, 24>(24, 0) = HT_R_inv_H.template block<24, 6>(0, 24).transpose();
        }

        Eigen::Matrix<double, XSIZE, XSIZE> S_inv;
        solvePosteriorCovariance(P_prop, HT_R_inv_H, S_inv);

        K.noalias() = S_inv.template leftCols<24>() * HT_R_inv_24;
        if constexpr (XSIZE == 30) {
            for (size_t m = 0; m < imu_meas.size(); m++) {
                const int row = imu_row_offsets_[m];
                for (int k = 0; k < 6; k++) {
                    K.col(row + k).noalias() += S_inv.col(24 + k) * R_inv(row + k);
                }
            }
        }
        KH.noalias() = S_inv * HT_R_inv_H;
        P_post_ = S_inv;
        Eigen::Matrix<double, XSIZE, 1> delta_cur = (getState() - x_prop);
        Eigen::Matrix<double, XSIZE, 1> deltax = KH * delta_cur + K * innv - delta_cur;
        Eigen::Matrix<double, XSIZE, 1> RCPs_post;
        RCPs_post.noalias() = getState() + deltax;
        updateState(RCPs_post);
    }
};

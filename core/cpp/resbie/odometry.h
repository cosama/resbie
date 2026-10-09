#pragma once

// resbie odometry: RESPLE's continuous-time B-spline filter (tightly coupled
// IMU + LiDAR iterated EKF over the spline's control points) measuring LiDAR
// points against BIEVR's bump-image map instead of an ikd-tree.
//
// This is RESPLE.cpp's processData loop made synchronous: every push runs all
// processing the new data allows before returning, so results depend only on
// the pushed data and its order. The parts that changed:
//
// * Sweeps enter the point buffer one at a time, only once the filter has used
//   up the previous one. At that moment the spline reaches the new sweep's
//   start, which is what BIEVR's informed sampling needs to place it in the map.
// * The map gets a sweep once the spline over the sweep can no longer change
//   (it left the 4-knot active window). Each point is placed with the spline
//   pose at its own timestamp, then the whole sweep goes into BIEVR's map.
// * Initialization: gravity from the first 15 IMU samples, as RESPLE; the
//   first sweep with enough points seeds the map at the initial pose (RESPLE
//   seeds with the first 100 ms of points, BIEVR with its first sweep).
// * LiDAR gaps: RESPLE advances the spline only when points arrive and then
//   jumps the gap on its constant-velocity model, discarding the gap's IMU.
//   Here the spline steps knot by knot through any stretch the arrived sweeps
//   show to be point-free, with IMU-only updates.
// * Diagnostics: per-sweep LiDAR information and consistency (SweepDiagnostics).

#include <omp.h>

#include <Eigen/Eigenvalues>
#include <algorithm>
#include <cstdint>
#include <deque>
#include <functional>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "Estimator.h"
#include "bievr_lio/bievr_map.h"
#include "bievr_lio/preprocess.h"
#include "resbie/config.h"

namespace resbie {

// Per-sweep filter diagnostics, accumulated over the updates whose batch ends
// inside the sweep.
struct SweepDiagnostics {
  size_t points_measured = 0;   // points handed to the filter
  size_t points_effective = 0;  // of those, matched to the map
  size_t updates = 0;           // filter updates
  size_t imu_only_updates = 0;  // updates without a matched point
  Eigen::Matrix<double, 6, 6> info = Eigen::Matrix<double, 6, 6>::Zero();  // [rot, trans]
  double pos_sigma = 0.0;       // newest control point position sigma at the last update, m
  double info_rotation = 0.0;    // smallest eigenvalue, rotation block (1/rad^2)
  double info_translation = 0.0; // smallest eigenvalue, translation block (1/m^2)
  double nis_sum = 0.0;          // sum of point zp^2 / (HPH' + R) (prior, first iteration)
  size_t nis_count = 0;
};

// One LiDAR sweep in the IMU frame, range-filtered, with absolute point times.
// A sweep may have no points: it still tells the filter that LiDAR saw nothing
// over [start_ns, end_ns].
struct Sweep {
  int64_t start_ns = 0;
  int64_t end_ns = 0;
  bievr::Pointcloud points;      // IMU frame
  std::vector<int64_t> times;    // ns, absolute
  std::vector<float> intensity;
  std::vector<double> ranges;    // |p| in the IMU frame (BIEVR's weighted map update)
  bool in_map = false;           // the initialization sweep seeded the map already
  SweepDiagnostics diag;
};

class Odometry {
 public:
  static constexpr int kInitImuSamples = 15;  // RESPLE initialization()

  // Called once per sweep, in order, when the sweep's poses are final:
  // stamp = sweep end, T_W_I = pose at the sweep end, deskewed_I = every point
  // of the sweep moved to the IMU frame at the sweep end (index-aligned with
  // sweep.points / intensity).
  using SweepObserver = std::function<void(int64_t stamp_ns, const bievr::Transform& T_W_I,
                                           const bievr::Pointcloud& deskewed_I, const Sweep& sweep)>;

  explicit Odometry(const Config& config)
      : cfg_(config), map_(config.map), dt_ns_(int64_t(1000000000) / config.knot_hz) {
    NUM_OF_THREAD = cfg_.max_num_threads > 0 ? cfg_.max_num_threads : omp_get_num_procs();
    estimator_.n_iter = cfg_.n_iter;
    estimator_.gate_chi2 = cfg_.gate_chi2;
    if (cfg_.imu_normalized == 0.0) acc_scale_ = 1.0;
    if (cfg_.imu_normalized > 0.0) acc_scale_ = kGravity;
  }

  void setObserver(SweepObserver observer) { observer_ = std::move(observer); }

  // False if the sample is not strictly after the previous one.
  bool addImu(int64_t t_ns, const Eigen::Vector3d& acc, const Eigen::Vector3d& gyro) {
    if (t_ns <= last_imu_ns_) {
      ++imu_rejected_;
      return false;
    }
    last_imu_ns_ = t_ns;
    if (imu_accepted_++ == 0) first_imu_ns_ = t_ns;
    // The last kInitImuSamples before initialization set gravity (init may
    // come long after the first sample: it waits for a dense sweep).
    init_imu_.emplace_back(t_ns, gyro, acc);
    if (init_imu_.size() > kInitImuSamples) init_imu_.pop_front();
    imu_buff_.emplace_back(t_ns, gyro, acc * (acc_scale_ > 0 ? acc_scale_ : 1.0));
    process();
    return true;
  }

  // False if the sweep starts before the previous one.
  bool addSweep(Sweep&& sweep) {
    if (sweep.start_ns < last_sweep_ns_) {
      ++lidar_rejected_;
      return false;
    }
    last_sweep_ns_ = sweep.start_ns;
    ++lidar_accepted_;
    sweeps_.push_back(std::move(sweep));
    process();
    return true;
  }

  // End of input: report the sweeps the spline covers with its current
  // (no longer refined) estimate. Sweeps it does not reach are dropped.
  void finish() {
    finalizeSweeps(true);
    sweeps_unfinished_ += pending_final_.size() + sweeps_.size();
    pending_final_.clear();
    sweeps_.clear();
  }

  bool initialized() const { return initialized_; }
  const bievr::BIEVRMap& map() const { return map_; }
  Eigen::Vector3d accBias() const { return initialized_ ? estimator_.biasAcc() : Eigen::Vector3d::Zero(); }
  Eigen::Vector3d gyroBias() const { return initialized_ ? estimator_.biasGyro() : Eigen::Vector3d::Zero(); }

  struct Stats {
    size_t imu_accepted, imu_rejected, lidar_accepted, lidar_rejected;
    size_t sweeps_dropped_init, sweeps_finalized, sweeps_unfinished;
    size_t updates, imu_only_updates, gap_knots, points_measured, points_effective, map_voxels;
    double acc_scale;
  };
  Stats stats() const {
    return {imu_accepted_,   imu_rejected_,      lidar_accepted_,   lidar_rejected_,
            sweeps_dropped_init_, sweeps_finalized_, sweeps_unfinished_,
            updates_, imu_only_updates_, gap_knots_, points_measured_, points_effective_,
            map_.size(), acc_scale_ > 0 ? acc_scale_ : 0.0};
  }

 private:
  static constexpr double kGravity = 9.81;  // RESPLE's constant

  void process() {
    while (true) {
      if (!initialized_) {
        const int init = initialize();
        if (init < 0) continue;  // dropped a sweep (before IMU, or sparse), try the next
        if (init == 0) return;
      }
      while (!sweeps_.empty() && needPoints()) {
        promote(sweeps_.front());
        sweeps_.pop_front();
      }
      if (!collectMeasurements()) break;
      if (pt_meas_.empty() && imu_meas_.empty()) continue;

      // RESPLE processData, LiDAR-inertial branch (pt_meas_ may be empty: an
      // IMU-only step through a gap).
      int64_t max_time_ns = pt_meas_.empty() ? spline_->maxTimeNs() : pt_meas_.back().time_ns;
      if (!imu_meas_.empty()) max_time_ns = std::max(imu_meas_.back().time_ns, max_time_ns);
      while (!imu_meas_.empty() &&
             imu_meas_.front().time_ns < spline_->maxTimeNs() - spline_->getKnotTimeIntervalNs()) {
        imu_meas_.pop_front();
      }
      // RESPLE's propRCP also adds the motion-model noise when the spline
      // already reaches max_time_ns, i.e. once per batch. RESPLE sees about
      // one batch per knot; dense clouds give several, which then counted as
      // extra elapsed time (chandv2: ~5 per knot, the filter loosened and
      // wobbled). Process noise is per knot: skip it for later batches of
      // the same knot.
      if (max_time_ns > spline_->maxTimeNs() || spline_->numKnots() != knots_at_last_update_) {
        estimator_.propRCP(max_time_ns);
      }
      knots_at_last_update_ = spline_->numKnots();
      for (PointData& p : pt_meas_) p.var_pt = cfg_.w_pt;
      estimator_.updateIEKFLiDARInertial(pt_meas_, map_, cfg_.association, imu_meas_, gravity_,
                                         cfg_.cov_acc, cfg_.cov_gyro);
      recordUpdate(max_time_ns);
      pt_meas_.clear();
      finalizeSweeps(false);
    }
  }

  // Adds this update to the diagnostics of the sweep its batch ends in.
  void recordUpdate(int64_t t_ns) {
    size_t effective = 0;
    Eigen::Matrix<double, 6, 6> info = Eigen::Matrix<double, 6, 6>::Zero();
    if (!pt_meas_.empty()) {
      const Eigen::Vector3d pos = spline_->itpPosition(std::min(t_ns, spline_->maxTimeNs()));
      for (const PointData& p : pt_meas_) {
        if (!p.if_valid) continue;
        ++effective;
        // r = n . p_w + d with p_w = R p_b + pos: dr/dtheta = q x n (world-frame
        // rotation about pos, q = p_w - pos), dr/dpos = n.
        Eigen::Matrix<double, 6, 1> J;
        J << (p.pt_w - pos).cross(p.normvec), p.normvec;
        info.noalias() += (1.0 / (p.var_pt * p.var_scale)) * J * J.transpose();
      }
    }
    ++updates_;
    points_measured_ += pt_meas_.size();
    points_effective_ += effective;
    if (effective == 0) ++imu_only_updates_;

    SweepDiagnostics* d = nullptr;
    for (Sweep& s : pending_final_) {
      if (s.end_ns >= t_ns) {
        d = &s.diag;
        break;
      }
    }
    if (!d) d = &carry_diag_;  // a gap before the next sweep arrives
    d->updates += 1;
    d->imu_only_updates += effective == 0;
    d->points_measured += pt_meas_.size();
    d->points_effective += effective;
    d->info += info;
    for (double nis : estimator_.point_nis) {
      d->nis_sum += nis;
      ++d->nis_count;
    }
    const Eigen::Matrix3d P_pos = estimator_.covariance().template block<3, 3>(18, 18);
    d->pos_sigma = std::sqrt(std::max(
        0.0, Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d>(P_pos).eigenvalues().maxCoeff()));
  }

  // 1: initialized, 0: waiting for data, -1: dropped the first sweep (too sparse).
  int initialize() {
    if (sweeps_.empty() || init_imu_.size() < kInitImuSamples) return 0;
    Sweep& sweep = sweeps_.front();
    // Start only with IMU coverage (as the RESPLE bridge): a sweep from
    // before the first IMU sample would run the filter on LiDAR alone and
    // seed the map before gravity is ever measured.
    if (sweep.start_ns < first_imu_ns_) {
      ++sweeps_dropped_init_;
      sweeps_.pop_front();
      return -1;
    }
    // Seed only from a sweep that covers enough distinct map voxels (as the
    // RESPLE bridge waits for a dense window). A sparse start, e.g. a drone
    // on the ground with few returns, would otherwise be dead-reckoned as
    // if moving, and its smeared sweeps would seed the map.
    std::vector<size_t> occupied;
    bievr::voxelDownsample(sweep.points, occupied, cfg_.map.voxel_size);
    if (occupied.size() < cfg_.min_voxels_for_init) {
      ++sweeps_dropped_init_;
      sweeps_.pop_front();
      return -1;
    }

    Eigen::Vector3d gravity_sum = Eigen::Vector3d::Zero();
    for (const ImuData& imu : init_imu_) gravity_sum += imu.accel;
    gravity_sum /= double(init_imu_.size());
    if (acc_scale_ <= 0) {  // BIEVR's autodetection: ~1 means the IMU reports g
      acc_scale_ = gravity_sum.norm() < 0.5 * kGravity ? kGravity : 1.0;
      for (ImuData& imu : imu_buff_) imu.accel *= acc_scale_;
    }
    // RESPLE initialization(): level the first pose on gravity, zero yaw.
    const Eigen::Vector3d gravity_ave = gravity_sum.normalized() * kGravity;
    Eigen::Matrix3d R0 = CommonUtils::g2R(gravity_ave);
    const double yaw = CommonUtils::R2ypr(R0).x();
    R0 = CommonUtils::ypr2R(Eigen::Vector3d{-yaw, 0, 0}) * R0;
    const Eigen::Quaterniond q_WI = Quater::positify(Eigen::Quaterniond(R0));
    gravity_ = q_WI * gravity_ave;

    const int64_t start_t_ns = std::max(sweep.start_ns, int64_t(0));
    initFilter(start_t_ns, q_WI);
    while (!imu_buff_.empty() && imu_buff_.front().time_ns < start_t_ns) imu_buff_.pop_front();
    estimator_.propRCP(start_t_ns);

    // Seed the map with the whole sweep at the initial pose (static start).
    const bievr::Transform T_W_I(q_WI, Eigen::Vector3d::Zero());
    map_.integratePoints(T_W_I * sweep.points, &sweep.ranges);
    sweep.in_map = true;
    covered_until_ = sweep.end_ns;
    pending_final_.push_back(std::move(sweep));
    sweeps_.pop_front();
    initialized_ = true;
    return 1;
  }

  // RESPLE initFilter, with the resple bridge's fix (the new-knot noise goes
  // to the newest control point's block, not the bias block) and a bias
  // random walk (RESPLE: none).
  void initFilter(int64_t start_t_ns, const Eigen::Quaterniond& q_init) {
    const double dt_s = double(dt_ns_) * 1e-9;
    const double cov_P0 = cfg_.cov_P0 * dt_s * dt_s;
    const double cov_sys_pos = cfg_.std_sys_pos * cfg_.std_sys_pos * dt_s * dt_s;
    const double cov_sys_ort = cfg_.std_sys_ort * cfg_.std_sys_ort * dt_s * dt_s;
    Eigen::Matrix<double, 24, 24> cov_RCPs = cov_P0 * Eigen::Matrix<double, 24, 24>::Identity();
    Eigen::Matrix<double, 30, 30> Q = Eigen::Matrix<double, 30, 30>::Zero();
    Eigen::Matrix<double, 6, 6> Q_block_old = Eigen::Matrix<double, 6, 6>::Zero();
    Q_block_old.topLeftCorner<3, 3>() = cfg_.cov_RCP_pos_old * cov_sys_pos * Eigen::Matrix3d::Identity();
    Q_block_old.bottomRightCorner<3, 3>() = cfg_.cov_RCP_ort_old * cov_sys_ort * Eigen::Matrix3d::Identity();
    Eigen::Matrix<double, 6, 6> Q_block_new = Eigen::Matrix<double, 6, 6>::Zero();
    Q_block_new.topLeftCorner<3, 3>() = cfg_.cov_RCP_pos_new * cov_sys_pos * Eigen::Matrix3d::Identity();
    Q_block_new.bottomRightCorner<3, 3>() = cfg_.cov_RCP_ort_new * cov_sys_ort * Eigen::Matrix3d::Identity();
    Q.block<6, 6>(0, 0) = Q_block_old;
    Q.block<6, 6>(6, 6) = Q_block_old;
    Q.block<6, 6>(12, 12) = Q_block_old;
    Q.block<6, 6>(18, 18) = Q_block_new;
    Q.block<3, 3>(24, 24) = cfg_.bias_rw_acc * cfg_.bias_rw_acc * dt_s * Eigen::Matrix3d::Identity();
    Q.block<3, 3>(27, 27) = cfg_.bias_rw_gyro * cfg_.bias_rw_gyro * dt_s * Eigen::Matrix3d::Identity();
    Eigen::Matrix<double, 30, 30> cov_x = Eigen::Matrix<double, 30, 30>::Zero();
    cov_x.topLeftCorner<24, 24>() = cov_RCPs;
    cov_x.block<3, 3>(24, 24) = cfg_.cov_ba.asDiagonal();
    cov_x.block<3, 3>(27, 27) = cfg_.cov_bg.asDiagonal();
    estimator_.setState(dt_ns_, start_t_ns, Eigen::Vector3d::Zero(), q_init, Q, cov_x);
    spline_ = estimator_.getSpline();
  }

  // collectMeasurements() would stop for lack of points.
  bool needPoints() const {
    return pt_buff_.empty() || pt_buff_.back().time_ns <= spline_->maxTimeNs() + dt_ns_;
  }

  // Selects the sweep's measurement points (BIEVR's downsampling and informed
  // sampling) and queues them, time-ordered, for the filter.
  void promote(Sweep& sweep) {
    covered_until_ = std::max(covered_until_, sweep.end_ns);
    sweep.diag = carry_diag_;  // updates in the gap before this sweep
    carry_diag_ = SweepDiagnostics();
    std::vector<size_t> down;
    bievr::voxelDownsample(sweep.points, down, cfg_.downsample_resolution);
    std::vector<size_t> selected;
    if (!down.empty()) {
      // Place the points with the spline, clamped to its span: the spline ends
      // about where this sweep starts. Only voxel lookups depend on this.
      bievr::Pointcloud world;
      world.resize(down.size());
      const int64_t t_lo = spline_->minTimeNs(), t_hi = spline_->maxTimeNs();
      #pragma omp parallel for num_threads(NUM_OF_THREAD)
      for (size_t i = 0; i < down.size(); ++i) {
        const int64_t t = std::clamp(sweep.times[down[i]], t_lo, t_hi);
        world[i] = Association::pointBodyToWorld(t, spline_, sweep.points[down[i]]);
      }
      std::vector<size_t> informed;
      bievr::sampleInformed(map_, world, informed, cfg_.informed_sample_count);
      selected.reserve(informed.size());
      for (size_t i : informed) selected.push_back(down[i]);
    }
    std::sort(selected.begin(), selected.end(), [&](size_t a, size_t b) {
      return sweep.times[a] != sweep.times[b] ? sweep.times[a] < sweep.times[b] : a < b;
    });
    for (size_t i : selected) {
      pt_buff_.emplace_back(Eigen::Vector3d(sweep.points[i]), sweep.times[i], cfg_.w_pt);
    }
    pending_final_.push_back(std::move(sweep));
  }

  // RESPLE collectMeasurements(), one LiDAR, IMU required, plus gap steps.
  bool collectMeasurements() {
    const int64_t next = spline_->maxTimeNs() + dt_ns_;
    // Gap: no point up to the next knot, but the arrived sweeps (or a later
    // point) show LiDAR has nothing there. Step one knot on IMU alone.
    const bool point_before_next = !pt_buff_.empty() && pt_buff_.front().time_ns <= next;
    const int64_t lidar_known =
        std::max(covered_until_, pt_buff_.empty() ? covered_until_ : pt_buff_.front().time_ns);
    if (!point_before_next && lidar_known > next) {
      if (imu_buff_.empty() || imu_buff_.back().time_ns <= next) return false;
      estimator_.propRCP(next);
      ++gap_knots_;
      takeImu(spline_->maxTimeNs());
      return true;
    }

    if (pt_buff_.empty()) return false;
    const int64_t pt_min_time = pt_buff_.front().time_ns;
    const int64_t pt_max_time = pt_buff_.back().time_ns;
    if (pt_max_time <= spline_->maxTimeNs() + dt_ns_) return false;
    if (imu_buff_.empty() || imu_buff_.back().time_ns <= spline_->maxTimeNs()) return false;
    int64_t max_time_ns = std::min(spline_->maxTimeNs(), pt_min_time + dt_ns_);
    if (pt_min_time > max_time_ns) {
      estimator_.propRCP(pt_min_time);
      max_time_ns = spline_->maxTimeNs();
    }
    if (spline_->numKnots() > 4) max_time_ns = spline_->maxTimeNs();
    int cnt = 0;
    while (!pt_buff_.empty() && pt_buff_.front().time_ns <= max_time_ns && cnt < cfg_.num_points_upd) {
      if (spline_->numKnots() < 10 || pt_buff_.front().time_ns >= spline_->maxTimeNs() - dt_ns_) {
        pt_meas_.push_back(pt_buff_.front());
      }
      pt_buff_.pop_front();
      cnt++;
    }
    takeImu(max_time_ns);
    return true;
  }

  void takeImu(int64_t max_time_ns) {
    while (!imu_buff_.empty() && imu_buff_.front().time_ns < spline_->minTimeNs()) imu_buff_.pop_front();
    while (!imu_buff_.empty() && imu_buff_.front().time_ns <= max_time_ns) {
      imu_meas_.push_back(imu_buff_.front());
      imu_buff_.pop_front();
    }
  }

  // Integrates and reports, in order, each sweep whose spline segment left the
  // active window (time t uses knots up to floor((t - t0) / dt) + 1; the last
  // four are still being estimated). With `force`, the rest the spline covers.
  void finalizeSweeps(bool force) {
    while (!pending_final_.empty()) {
      Sweep& sweep = pending_final_.front();
      const bool final_ready = sweep.end_ns + 5 * dt_ns_ <= spline_->maxTimeNs();
      if (!final_ready && !(force && sweep.end_ns <= spline_->maxTimeNs())) return;
      finalize(sweep);
      pending_final_.pop_front();
    }
  }

  static void summarizeInfo(SweepDiagnostics& d) {
    const Eigen::Matrix3d Ir = d.info.topLeftCorner<3, 3>();
    const Eigen::Matrix3d It = d.info.bottomRightCorner<3, 3>();
    d.info_rotation = Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d>(Ir).eigenvalues().minCoeff();
    d.info_translation = Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d>(It).eigenvalues().minCoeff();
  }

  void finalize(Sweep& sweep) {
    summarizeInfo(sweep.diag);
    // Pose grid at 0.5 ms, interpolated per point (spline evaluation per point
    // is the expensive part for dense sweeps).
    constexpr int64_t kStep = 500000;
    const int64_t t0 = sweep.start_ns;
    const size_t n_grid = size_t((sweep.end_ns - t0) / kStep) + 2;
    std::vector<Eigen::Quaterniond, Eigen::aligned_allocator<Eigen::Quaterniond>> q(n_grid);
    std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> p(n_grid);
    #pragma omp parallel for num_threads(NUM_OF_THREAD)
    for (size_t k = 0; k < n_grid; ++k) {
      const int64_t t = std::min(t0 + int64_t(k) * kStep, sweep.end_ns);
      p[k] = spline_->itpPosition(t);
      spline_->itpQuaternion(t, &q[k]);
    }
    Eigen::Quaterniond q_end;
    spline_->itpQuaternion(sweep.end_ns, &q_end);
    const bievr::Transform T_W_I(q_end, spline_->itpPosition(sweep.end_ns));
    const bievr::Transform T_I_W = T_W_I.inverse();

    const size_t n = sweep.points.size();
    bievr::Pointcloud world, deskewed;
    world.resize(n);
    deskewed.resize(n);
    #pragma omp parallel for num_threads(NUM_OF_THREAD)
    for (size_t i = 0; i < n; ++i) {
      const int64_t dt = sweep.times[i] - t0;
      const size_t k = std::min(size_t(dt / kStep), n_grid - 2);
      const int64_t tk = t0 + int64_t(k) * kStep;
      const int64_t tk1 = std::min(tk + kStep, sweep.end_ns);
      const double a = tk1 > tk ? double(sweep.times[i] - tk) / double(tk1 - tk) : 0.0;
      const Eigen::Quaterniond qi = q[k].slerp(a, q[k + 1]);
      const Eigen::Vector3d pi = (1.0 - a) * p[k] + a * p[k + 1];
      world[i] = qi * Eigen::Vector3d(sweep.points[i]) + pi;
      deskewed[i] = T_I_W * Eigen::Vector3d(world[i]);
    }
    if (!sweep.in_map && n > 0) map_.integratePoints(world, &sweep.ranges);
    ++sweeps_finalized_;
    if (observer_) observer_(sweep.end_ns, T_W_I, deskewed, sweep);
  }

  Config cfg_;
  bievr::BIEVRMap map_;
  int64_t dt_ns_;
  Estimator<30> estimator_;
  SplineState* spline_ = nullptr;
  bool initialized_ = false;
  double acc_scale_ = -1.0;  // < 0 until decided
  Eigen::Vector3d gravity_ = Eigen::Vector3d::Zero();
  int64_t covered_until_ = std::numeric_limits<int64_t>::min();  // LiDAR seen up to here
  int64_t knots_at_last_update_ = -1;

  std::deque<ImuData> init_imu_;
  Eigen::aligned_deque<ImuData> imu_buff_;
  Eigen::aligned_deque<ImuData> imu_meas_;
  std::deque<Sweep> sweeps_;         // waiting to be measured
  std::deque<Sweep> pending_final_;  // measured (or seeding), waiting for final poses
  Eigen::aligned_deque<PointData> pt_buff_;
  Eigen::aligned_deque<PointData> pt_meas_;
  SweepDiagnostics carry_diag_;
  SweepObserver observer_;

  int64_t last_imu_ns_ = std::numeric_limits<int64_t>::min();
  int64_t first_imu_ns_ = 0;
  int64_t last_sweep_ns_ = std::numeric_limits<int64_t>::min();
  size_t imu_accepted_ = 0, imu_rejected_ = 0, lidar_accepted_ = 0, lidar_rejected_ = 0;
  size_t sweeps_dropped_init_ = 0, sweeps_finalized_ = 0, sweeps_unfinished_ = 0;
  size_t updates_ = 0, imu_only_updates_ = 0, gap_knots_ = 0;
  size_t points_measured_ = 0, points_effective_ = 0;
};

}  // namespace resbie

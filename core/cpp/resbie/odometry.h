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
// * Initialization: the first pose is levelled on the first 15 IMU samples, as
//   RESPLE, but gravity's direction in the map frame is a state from then on
//   (RESPLE: fixed), so neither a moving start nor a map that tilts against
//   gravity leaves the IMU with a fake acceleration. The first sweep with
//   enough points seeds the map at the initial pose (RESPLE seeds with the
//   first 100 ms of points, BIEVR with its first sweep).
// * LiDAR gaps: RESPLE advances the spline only when points arrive and then
//   jumps the gap on its constant-velocity model, discarding the gap's IMU.
//   Here the spline steps knot by knot through any stretch the arrived sweeps
//   show to be point-free, with IMU-only updates.
// * Diagnostics: per-sweep LiDAR information and consistency (SweepDiagnostics).

#include <omp.h>

#include <Eigen/Eigenvalues>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <deque>
#include <functional>
#include <iomanip>
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
  Eigen::Vector3d weak_translation = Eigen::Vector3d::Zero();  // world direction of info_translation
  size_t matched_voxels = 0;     // distinct map voxels of the matched points, summed over updates
  size_t imu_samples = 0;        // IMU samples measured
  size_t acc_gated = 0;          // of their acceleration axes, dropped by the 10 m/s^2 residual gate
  Eigen::Vector3d acc_bias = Eigen::Vector3d::Zero();  // at the last update, m/s^2
  Eigen::Vector3d velocity = Eigen::Vector3d::Zero();  // final spline, world, mean over the sweep, m/s
  Eigen::Vector3d gyro_bias = Eigen::Vector3d::Zero();  // at the last update, rad/s
  Eigen::Vector3d gravity_up = Eigen::Vector3d::UnitZ();  // world up direction at the last update
  double gravity_sigma = NAN;    // its largest tangent std, rad
  size_t points_gated = 0;       // matched points the Mahalanobis gate rejected (last iteration)
  // A-posteriori IMU check: the final spline's velocity change over the sweep
  // minus the IMU's (world, m/s, final orientation, current acc bias). Each
  // IMU sample is checked against the spline only once, before later updates
  // reshape it; this checks the result. Its std under the accelerometer noise
  // is imu_post_sigma.
  Eigen::Vector3d imu_post_dv = Eigen::Vector3d::Constant(NAN);
  double imu_post_sigma = NAN;
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
  static constexpr double kGravity = 9.81;  // RESPLE's constant

  // Called once per sweep, in order, when the sweep's poses are final:
  // stamp = sweep end, T_W_I = pose at the sweep end, deskewed_I = every point
  // of the sweep moved to the IMU frame at the sweep end (index-aligned with
  // sweep.points / intensity).
  using SweepObserver = std::function<void(int64_t stamp_ns, const bievr::Transform& T_W_I,
                                           const bievr::Pointcloud& deskewed_I, const Sweep& sweep)>;

  explicit Odometry(const Config& config)
      : cfg_(config), map_(config.map), dt_ns_(int64_t(1000000000) / config.knot_hz) {
    NUM_OF_THREAD = cfg_.max_num_threads > 0 ? cfg_.max_num_threads : omp_get_num_procs();
    estimator_.n_iter = cfg_.iterations;
    estimator_.gate_chi2 = cfg_.gate_sigma * cfg_.gate_sigma;
  }

  void setObserver(SweepObserver observer) { observer_ = std::move(observer); }

  // acc in m/s^2, gyro in rad/s. False if the sample is not strictly after
  // the previous one.
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
    imu_buff_.emplace_back(t_ns, gyro, acc);
    imu_hist_.emplace_back(t_ns, acc);
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
  Eigen::Vector3d gravityUp() const { return initialized_ ? estimator_.gravityUp() : Eigen::Vector3d::UnitZ(); }

  struct Stats {
    size_t imu_accepted, imu_rejected, lidar_accepted, lidar_rejected;
    size_t sweeps_dropped_init, sweeps_finalized, sweeps_unfinished;
    size_t updates, imu_only_updates, gap_knots, points_measured, points_effective, map_voxels;
  };
  Stats stats() const {
    return {imu_accepted_,   imu_rejected_,      lidar_accepted_,   lidar_rejected_,
            sweeps_dropped_init_, sweeps_finalized_, sweeps_unfinished_,
            updates_, imu_only_updates_, gap_knots_, points_measured_, points_effective_,
            map_.size()};
  }

 private:

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
             imu_meas_.front().time_ns <= spline_->maxTimeNs() - spline_->getKnotTimeIntervalNs()) {
        imu_meas_.pop_front();
      }
      const Eigen::Matrix<double, 32, 1> state_before = estimator_.getState();
      for (PointData& p : pt_meas_) p.var_pt = pointVariance();
      estimator_.updateIEKFLiDARInertial(pt_meas_, map_, cfg_.association, imu_meas_,
                                         cfg_.acc_noise_std.cwiseAbs2(),
                                         cfg_.gyro_noise_std.cwiseAbs2());
      recordUpdate(max_time_ns);
      if (debugEnabled()) logUpdate(max_time_ns, state_before);
      pt_meas_.clear();
      imu_meas_.clear();
      finalizeSweeps(false);
    }
  }

  // Debug trace, one line per filter update: what it measured and how far it
  // moved the newest knot (position, and velocity (cp_3 - cp_2) / dt).
  void logUpdate(int64_t t_ns, const Eigen::Matrix<double, 32, 1>& before) {
    const Eigen::Matrix<double, 32, 1> after = estimator_.getState();
    const double inv_dt = 1e9 / double(dt_ns_);
    const auto vel = [&](const Eigen::Matrix<double, 32, 1>& s) {
      return Eigen::Vector3d((s.segment<3>(18) - s.segment<3>(12)) * inv_dt);
    };
    size_t matched = 0;
    for (const PointData& p : pt_meas_) matched += p.if_valid;
    double nis_mean = 0.0, nis_max = 0.0;
    for (double n : estimator_.point_nis) {
      nis_mean += n;
      nis_max = std::max(nis_max, n);
    }
    if (!estimator_.point_nis.empty()) nis_mean /= double(estimator_.point_nis.size());
    LOG(D, std::fixed << std::setprecision(4) << "update " << t_ns * 1e-9 << " points " << matched << "/"
        << pt_meas_.size() << " gated " << estimator_.points_gated << " imu " << imu_meas_.size()
        << " nis " << nis_mean << " max " << nis_max
        << " d_pos " << (after.segment<3>(18) - before.segment<3>(18)).norm()
        << " vel [" << vel(after).transpose() << "] d_vel [" << (vel(after) - vel(before)).transpose() << "]");
  }

  // Adds this update to the diagnostics of the sweep its batch ends in.
  void recordUpdate(int64_t t_ns) {
    size_t effective = 0;
    size_t voxels = 0;
    Eigen::Matrix<double, 6, 6> info = Eigen::Matrix<double, 6, 6>::Zero();
    if (!pt_meas_.empty()) {
      const Eigen::Vector3d pos = spline_->itpPosition(std::min(t_ns, spline_->maxTimeNs()));
      std::vector<size_t> matched;
      for (const PointData& p : pt_meas_) {
        if (!p.if_valid) continue;
        ++effective;
        // r = n . p_w + d with p_w = R p_b + pos: dr/dtheta = q x n (world-frame
        // rotation about pos, q = p_w - pos), dr/dpos = n.
        Eigen::Matrix<double, 6, 1> J;
        J << (p.pt_w - pos).cross(p.normvec), p.normvec;
        info.noalias() += (1.0 / (p.var_pt * p.var_scale)) * J * J.transpose();
        matched.push_back(p.voxel);
      }
      std::sort(matched.begin(), matched.end());
      voxels = std::unique(matched.begin(), matched.end()) - matched.begin();
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
    d->matched_voxels += voxels;
    d->info += info;
    for (double nis : estimator_.point_nis) {
      d->nis_sum += nis;
      ++d->nis_count;
    }
    d->imu_samples += estimator_.imu_samples;
    d->acc_gated += estimator_.acc_gated;
    if (effective > 0) d->points_gated += estimator_.points_gated;
    d->acc_bias = estimator_.biasAcc();
    d->gyro_bias = estimator_.biasGyro();
    d->gravity_up = estimator_.gravityUp();
    d->gravity_sigma = std::sqrt(std::max(
        0.0, Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d>(estimator_.gravityCovariance()).eigenvalues().maxCoeff()));
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
    if (occupied.size() < cfg_.init_min_voxels) {
      ++sweeps_dropped_init_;
      sweeps_.pop_front();
      return -1;
    }

    // Each knot interval needs at least one IMU sample to remain constrained
    // when LiDAR points are sparse or dropped: knot_hz <= round(imu_hz).
    const double imu_hz = (init_imu_.size() - 1) * 1e9 / double(init_imu_.back().time_ns - init_imu_.front().time_ns);
    const int max_knot_hz = static_cast<int>(std::lround(imu_hz));
    if (cfg_.knot_hz > max_knot_hz) {
      LOG(W, "spline.knot_hz " << cfg_.knot_hz << " is more than the IMU (" << max_knot_hz
                               << " Hz) can carry: knot intervals go unconstrained when the LiDAR is starved."
                               << " Use " << max_knot_hz << ".");
    }

    Eigen::Vector3d gravity_sum = Eigen::Vector3d::Zero();
    for (const ImuData& imu : init_imu_) gravity_sum += imu.accel;
    gravity_sum /= double(init_imu_.size());
    // RESPLE initialization(): level the first pose on the mean specific
    // force, zero yaw. That pose is the gauge of the map frame; the mean force
    // (motion included on a moving start) only sets gravity's linearisation
    // point, the direction itself is estimated from here on.
    const Eigen::Matrix3d R0 = CommonUtils::g2R(gravity_sum);
    const Eigen::Quaterniond q_WI = Quater::positify(Eigen::Quaterniond(R0));

    const int64_t start_t_ns = std::max(sweep.start_ns, int64_t(0));
    initFilter(start_t_ns, q_WI);
    estimator_.setGravity((q_WI * gravity_sum).normalized(), kGravity);
    while (!imu_buff_.empty() && imu_buff_.front().time_ns < start_t_ns) imu_buff_.pop_front();

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
    const double dt2 = dt_s * dt_s;
    const double cov_P0 = cfg_.init_std * cfg_.init_std * dt2;
    Eigen::Matrix<double, 24, 24> cov_RCPs = cov_P0 * Eigen::Matrix<double, 24, 24>::Identity();
    Eigen::Matrix<double, 32, 32> Q = Eigen::Matrix<double, 32, 32>::Zero();  // gravity: constant
    Eigen::Matrix<double, 6, 6> Q_block_old = Eigen::Matrix<double, 6, 6>::Zero();
    Q_block_old.topLeftCorner<3, 3>() = cfg_.pos_noise * cfg_.pos_noise * dt2 * Eigen::Matrix3d::Identity();
    Q_block_old.bottomRightCorner<3, 3>() = cfg_.ort_noise * cfg_.ort_noise * dt2 * Eigen::Matrix3d::Identity();
    Eigen::Matrix<double, 6, 6> Q_block_new = Eigen::Matrix<double, 6, 6>::Zero();
    Q_block_new.topLeftCorner<3, 3>() = cfg_.new_pos_noise * cfg_.new_pos_noise * dt2 * Eigen::Matrix3d::Identity();
    Q_block_new.bottomRightCorner<3, 3>() = cfg_.new_ort_noise * cfg_.new_ort_noise * dt2 * Eigen::Matrix3d::Identity();
    Q.block<6, 6>(0, 0) = Q_block_old;
    Q.block<6, 6>(6, 6) = Q_block_old;
    Q.block<6, 6>(12, 12) = Q_block_old;
    Q.block<6, 6>(18, 18) = Q_block_new;
    Q.block<3, 3>(24, 24) = cfg_.acc_bias_walk * cfg_.acc_bias_walk * dt_s * Eigen::Matrix3d::Identity();
    Q.block<3, 3>(27, 27) = cfg_.gyro_bias_walk * cfg_.gyro_bias_walk * dt_s * Eigen::Matrix3d::Identity();
    Eigen::Matrix<double, 32, 32> cov_x = Eigen::Matrix<double, 32, 32>::Zero();
    cov_x.topLeftCorner<24, 24>() = cov_RCPs;
    cov_x.block<3, 3>(24, 24) = cfg_.acc_bias_init_std.cwiseAbs2().asDiagonal();
    cov_x.block<3, 3>(27, 27) = cfg_.gyro_bias_init_std.cwiseAbs2().asDiagonal();
    const double grav_std = cfg_.gravity_init_std_deg * M_PI / 180.0;
    cov_x.block<2, 2>(30, 30) = grav_std * grav_std * Eigen::Matrix2d::Identity();
    estimator_.setState(dt_ns_, start_t_ns, Eigen::Vector3d::Zero(), q_init, Q, cov_x);
    spline_ = estimator_.getSpline();
  }

  double pointVariance() const { return cfg_.point_noise_std * cfg_.point_noise_std; }

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
    bievr::voxelDownsample(sweep.points, down, cfg_.downsample);
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
      bievr::sampleInformed(map_, world, informed, cfg_.informed_voxels);
      selected.reserve(informed.size());
      for (size_t i : informed) selected.push_back(down[i]);
    }
    std::sort(selected.begin(), selected.end(), [&](size_t a, size_t b) {
      return sweep.times[a] != sweep.times[b] ? sweep.times[a] < sweep.times[b] : a < b;
    });
    for (size_t i : selected) {
      pt_buff_.emplace_back(Eigen::Vector3d(sweep.points[i]), sweep.times[i], pointVariance());
    }
    pending_final_.push_back(std::move(sweep));
  }

  // RESPLE collectMeasurements(), one LiDAR, IMU required, plus gap steps.
  // Advances at most one knot per call and collects all points in the active
  // knot into a single update.
  bool collectMeasurements() {
    const int64_t next = spline_->maxTimeNs() + dt_ns_;
    // Gap: no point up to the next knot, but the arrived sweeps (or a later
    // point) show LiDAR has nothing there. Step one knot on IMU alone.
    const bool point_before_next = !pt_buff_.empty() && pt_buff_.front().time_ns <= next;
    const int64_t lidar_known =
        std::max(covered_until_, pt_buff_.empty() ? covered_until_ : pt_buff_.front().time_ns);
    if (!point_before_next && lidar_known > next) {
      if (imu_buff_.empty() || imu_buff_.back().time_ns < next) return false;
      estimator_.propRCP(next);
      ++gap_knots_;
      takeImu(spline_->maxTimeNs());
      return true;
    }

    if (pt_buff_.empty()) return false;
    const int64_t pt_min_time = pt_buff_.front().time_ns;
    const int64_t pt_max_time = pt_buff_.back().time_ns;
    if (pt_max_time <= next) return false;
    if (pt_min_time > spline_->maxTimeNs()) {
      if (imu_buff_.empty() || imu_buff_.back().time_ns < next) return false;
      estimator_.propRCP(next);
    } else {
      if (imu_buff_.empty() || imu_buff_.back().time_ns < spline_->maxTimeNs()) return false;
    }
    const int64_t max_time_ns = spline_->maxTimeNs();
    while (!pt_buff_.empty() && pt_buff_.front().time_ns <= max_time_ns) {
      if (spline_->numKnots() < 10 || pt_buff_.front().time_ns >= max_time_ns - dt_ns_) {
        pt_meas_.push_back(pt_buff_.front());
      }
      pt_buff_.pop_front();
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

  // SweepDiagnostics::imu_post_dv. The spline's acceleration is R (a - b_a) - g
  // (Estimator::prepIMU), integrated here over the sweep's IMU samples.
  void checkImuPosterior(Sweep& sweep) {
    while (!imu_hist_.empty() && imu_hist_.front().first < sweep.start_ns - 1000000000LL) imu_hist_.pop_front();
    if (sweep.end_ns <= sweep.start_ns || sweep.start_ns < spline_->minTimeNs()) return;
    const Eigen::Vector3d ba = estimator_.biasAcc();
    const Eigen::Vector3d g = estimator_.gravityVector();
    Eigen::Vector3d dv_imu = Eigen::Vector3d::Zero();
    double sum_dt2 = 0.0;
    int64_t prev = sweep.start_ns;
    for (const auto& [t, a] : imu_hist_) {
      if (t <= sweep.start_ns || t > sweep.end_ns) continue;
      Eigen::Quaterniond q;
      spline_->itpQuaternion(t, &q);
      const double dt = double(t - prev) * 1e-9;
      dv_imu += (q * (a - ba) - g) * dt;
      sum_dt2 += dt * dt;
      prev = t;
    }
    if (prev == sweep.start_ns) return;
    const Eigen::Vector3d dv_spline =
        spline_->itpPosition<1>(prev) - spline_->itpPosition<1>(sweep.start_ns);
    sweep.diag.imu_post_dv = dv_spline - dv_imu;
    sweep.diag.imu_post_sigma = cfg_.acc_noise_std.mean() * std::sqrt(sum_dt2);
  }

  static void summarizeInfo(SweepDiagnostics& d) {
    const Eigen::Matrix3d Ir = d.info.topLeftCorner<3, 3>();
    const Eigen::Matrix3d It = d.info.bottomRightCorner<3, 3>();
    d.info_rotation = Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d>(Ir).eigenvalues().minCoeff();
    const Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> et(It);  // eigenvalues ascending
    d.info_translation = et.eigenvalues()(0);
    d.weak_translation = et.eigenvectors().col(0);
  }

  static void logDiagnostics(int64_t end_ns, const SweepDiagnostics& d) {
    LOG(D, std::fixed << std::setprecision(3) << "sweep " << end_ns * 1e-9
        << " matched " << d.points_effective << "/" << d.points_measured << " points"
        << " updates " << d.updates << " (imu-only " << d.imu_only_updates << ")"
        << std::setprecision(4) << " info rot " << d.info_rotation << " trans " << d.info_translation
        << " weak [" << d.weak_translation.transpose() << "]"
        << " weak_sigma " << (d.info_translation > 0.0 ? 1.0 / std::sqrt(d.info_translation) : INFINITY)
        << " pos_sigma " << d.pos_sigma
        << " nis " << (d.nis_count ? d.nis_sum / double(d.nis_count) : NAN)
        << " voxels " << d.matched_voxels << " imu " << d.imu_samples << " acc_gated " << d.acc_gated
        << " vel [" << d.velocity.transpose() << "] acc_bias [" << d.acc_bias.transpose() << "]"
        << " gyro_bias [" << d.gyro_bias.transpose() << "] up [" << d.gravity_up.transpose() << "] (sigma "
        << d.gravity_sigma * 180.0 / M_PI << " deg)"
        << " gated " << d.points_gated << " imu_post_dv [" << d.imu_post_dv.transpose() << "] (sigma "
        << d.imu_post_sigma << ")");
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
    if (sweep.end_ns > t0) {
      sweep.diag.velocity = (T_W_I.translation() - p[0]) / (double(sweep.end_ns - t0) * 1e-9);
    }
    checkImuPosterior(sweep);
    if (debugEnabled()) logDiagnostics(sweep.end_ns, sweep.diag);

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
  Estimator<32> estimator_;
  SplineState* spline_ = nullptr;
  bool initialized_ = false;
  int64_t covered_until_ = std::numeric_limits<int64_t>::min();  // LiDAR seen up to here

  std::deque<ImuData> init_imu_;
  std::deque<std::pair<int64_t, Eigen::Vector3d>> imu_hist_;  // raw acc, for checkImuPosterior
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

// pybind11 bindings for resbie: RESPLE's spline filter on BIEVR's map, with
// BIEVR's loop closer downstream.
//
// Synchronous like the BIEVR bridge: every push returns after all processing
// it triggered has finished, so results depend only on the pushed data. The
// API mirrors bievr's (push_imu / push_lidar / trajectory /
// keyframe_trajectory / drain_map_scans / status) plus finish().

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <tbb/global_control.h>
#include <tbb/task_arena.h>

#include <cmath>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#include "bievr_pgo/config_loader.h"
#include "bievr_pgo/loop_closer.h"
#include "resbie/config.h"
#include "resbie/odometry.h"

namespace py = pybind11;

namespace {

using Array = py::array_t<double, py::array::c_style | py::array::forcecast>;

int64_t secondsToNs(double s) {
  if (!std::isfinite(s) || s <= 0.0) {
    throw std::invalid_argument("stamp must be a finite, positive time in seconds");
  }
  return static_cast<int64_t>(std::llround(s * 1e9));
}

double nsToS(int64_t ns) { return static_cast<double>(ns) * 1e-9; }

Eigen::Vector3d toV3(const Array& a, const char* name) {
  if (a.size() != 3) throw std::invalid_argument(std::string(name) + " must have 3 elements");
  const double* p = a.data();
  return Eigen::Vector3d(p[0], p[1], p[2]);
}

class Resbie {
 public:
  Resbie(const std::vector<std::string>& config_files, size_t map_point_stride)
      : map_point_stride_(map_point_stride) {
    if (!resbie::loadConfig(config_files, config_)) {
      throw std::invalid_argument("resbie rejected the configuration (see log above)");
    }
    if (config_.max_num_threads > 0) {
      tbb_control_ = std::make_unique<tbb::global_control>(
          tbb::global_control::max_allowed_parallelism, config_.max_num_threads);
    }
    odometry_ = std::make_unique<resbie::Odometry>(config_);

    bievr::LoopClosureConfig loop_closure_config;
    if (!bievr::loadLoopClosureConfig(config_files, loop_closure_config)) {
      throw std::invalid_argument("resbie rejected the loop_closure configuration");
    }
    if (loop_closure_config.enable) {
      loop_closer_ = std::make_unique<bievr::LoopCloser>(loop_closure_config.closer);
    }

    odometry_->setObserver([this](int64_t stamp, const bievr::Transform& T_W_I,
                                  const bievr::Pointcloud& deskewed_I,
                                  const resbie::Sweep& sweep) {
      onSweep(stamp, T_W_I, deskewed_I, sweep);
    });
  }

  bool pushImu(double stamp, const Array& acc, const Array& gyro) {
    const int64_t t = secondsToNs(stamp);
    const Eigen::Vector3d a = toV3(acc, "acceleration");
    const Eigen::Vector3d w = toV3(gyro, "angular_velocity");
    py::gil_scoped_release release;
    std::lock_guard<std::mutex> lock(mutex_);
    return odometry_->addImu(t, a, w);
  }

  bool pushLidar(double stamp, const Array& points, const Array& times) {
    if (points.ndim() != 2 || (points.shape(1) != 3 && points.shape(1) != 4)) {
      throw std::invalid_argument("points must be (N, 3) or (N, 4) [x, y, z(, intensity)]");
    }
    const size_t n = static_cast<size_t>(points.shape(0));
    const size_t cols = static_cast<size_t>(points.shape(1));
    if (times.ndim() != 1 || static_cast<size_t>(times.shape(0)) != n) {
      throw std::invalid_argument("relative_times must be (N,) matching points");
    }
    const int64_t t0 = secondsToNs(stamp) + std::llround(config_.time_offset * 1e9);
    const double* p = points.data();
    const double* t = times.data();
    for (size_t i = 0; i < n; ++i) {
      if (!std::isfinite(t[i]) || t[i] < 0.0) {
        throw std::invalid_argument("relative_times must be finite and >= 0");
      }
    }

    py::gil_scoped_release release;
    // Range filter in the LiDAR frame (BIEVR), then into the IMU frame.
    const double min2 = config_.min_range * config_.min_range;
    const double max2 = config_.max_range * config_.max_range;
    resbie::Sweep sweep;
    sweep.start_ns = t0;
    sweep.end_ns = t0;
    for (size_t i = 0; i < n; ++i) sweep.end_ns = std::max<int64_t>(sweep.end_ns, t0 + std::llround(t[i] * 1e9));
    std::vector<size_t> keep;
    keep.reserve(n);
    for (size_t i = 0; i < n; ++i) {
      const double x = p[i * cols], y = p[i * cols + 1], z = p[i * cols + 2];
      const double r2 = x * x + y * y + z * z;
      if (std::isfinite(r2) && r2 >= min2 && r2 <= max2) keep.push_back(i);
    }
    sweep.points.resize(keep.size());
    sweep.times.resize(keep.size());
    sweep.intensity.resize(keep.size());
    sweep.ranges.resize(keep.size());
    for (size_t j = 0; j < keep.size(); ++j) {
      const size_t i = keep[j];
      const Eigen::Vector3d p_L(p[i * cols], p[i * cols + 1], p[i * cols + 2]);
      const Eigen::Vector3d p_I = config_.T_I_L * p_L;
      sweep.points[j] = p_I;
      sweep.times[j] = t0 + std::llround(t[i] * 1e9);
      sweep.intensity[j] = cols == 4 ? static_cast<float>(p[i * cols + 3]) : 0.0f;
      sweep.ranges[j] = p_I.norm();
    }

    std::lock_guard<std::mutex> lock(mutex_);
    // An empty sweep still goes in: it tells the filter LiDAR saw nothing over
    // [start, end], so the spline may step through on IMU alone.
    if (keep.empty()) ++lidar_empty_;
    return odometry_->addSweep(std::move(sweep));
  }

  void finish() {
    py::gil_scoped_release release;
    std::lock_guard<std::mutex> lock(mutex_);
    odometry_->finish();
  }

  py::object latestPose() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (trajectory_.empty()) return py::none();
    py::array_t<double> out(8);
    std::copy(trajectory_.end() - 8, trajectory_.end(), out.mutable_data());
    return out;
  }

  py::array_t<double> trajectory() {
    std::lock_guard<std::mutex> lock(mutex_);
    const py::ssize_t rows = static_cast<py::ssize_t>(trajectory_.size() / 8);
    py::array_t<double> out({rows, py::ssize_t{8}});
    std::copy(trajectory_.begin(), trajectory_.end(), out.mutable_data());
    return out;
  }

  py::array_t<double> keyframeTrajectory() {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto kfs =
        loop_closer_ ? loop_closer_->keyframes() : std::vector<bievr::LoopCloser::Keyframe>{};
    py::array_t<double> out({static_cast<py::ssize_t>(kfs.size()), py::ssize_t{8}});
    auto r = out.mutable_unchecked<2>();
    for (size_t i = 0; i < kfs.size(); ++i) {
      const Eigen::Vector3d& t = kfs[i].pose.translation();
      const Eigen::Quaterniond q = kfs[i].pose.quaternion();
      const double row[8] = {nsToS(static_cast<int64_t>(kfs[i].stamp)), t.x(), t.y(), t.z(),
                             q.x(), q.y(), q.z(), q.w()};
      std::copy(row, row + 8, r.mutable_data(i, 0));
    }
    return out;
  }

  py::list drainMapScans() {
    std::lock_guard<std::mutex> lock(mutex_);
    py::list out;
    for (const MapScan& scan : map_scans_) {
      const py::ssize_t rows = static_cast<py::ssize_t>(scan.points.size() / 4);
      py::array_t<float> points({rows, py::ssize_t{4}});
      std::copy(scan.points.begin(), scan.points.end(), points.mutable_data());
      out.append(py::make_tuple(scan.stamp, points));
    }
    map_scans_.clear();
    return out;
  }

  py::dict status() {
    std::lock_guard<std::mutex> lock(mutex_);
    const resbie::Odometry::Stats s = odometry_->stats();
    py::dict d;
    d["imu_accepted"] = s.imu_accepted;
    d["imu_rejected"] = s.imu_rejected;
    d["lidar_accepted"] = s.lidar_accepted;
    d["lidar_rejected"] = s.lidar_rejected;
    d["lidar_empty"] = lidar_empty_;
    d["initialized"] = odometry_->initialized();
    d["sweeps_dropped_init"] = s.sweeps_dropped_init;
    d["sweeps_finalized"] = s.sweeps_finalized;
    d["sweeps_unfinished"] = s.sweeps_unfinished;
    d["filter_updates"] = s.updates;
    d["imu_only_updates"] = s.imu_only_updates;
    d["gap_knots"] = s.gap_knots;
    d["loop_closer_skipped"] = loop_closer_skipped_;
    d["points_measured"] = s.points_measured;
    d["points_effective"] = s.points_effective;
    d["map_voxels"] = s.map_voxels;
    d["acc_scale"] = s.acc_scale;
    d["poses"] = trajectory_.size() / 8;
    const Eigen::Vector3d ba = odometry_->accBias(), bg = odometry_->gyroBias();
    d["acc_bias"] = std::vector<double>(ba.data(), ba.data() + 3);
    d["gyro_bias"] = std::vector<double>(bg.data(), bg.data() + 3);
    py::dict pgo;
    pgo["enabled"] = loop_closer_ != nullptr;
    if (loop_closer_) {
      const auto ls = loop_closer_->stats();
      pgo["keyframes"] = ls.num_keyframes;
      pgo["loops"] = ls.num_loops;
      pgo["rejected"] = ls.num_rejected;
    }
    d["loop_closure"] = pgo;
    return d;
  }

 private:
  // Called from inside Odometry (mutex_ held) once a sweep's poses are final.
  void onSweep(int64_t stamp, const bievr::Transform& T_W_I, const bievr::Pointcloud& deskewed_I,
               const resbie::Sweep& sweep) {
    const Eigen::Quaterniond q = T_W_I.quaternion();
    const Eigen::Vector3d& t = T_W_I.translation();
    trajectory_.insert(trajectory_.end(),
                       {nsToS(stamp), t.x(), t.y(), t.z(), q.x(), q.y(), q.z(), q.w()});
    // Loop closer guard: an empty or implausibly fast sweep never becomes a
    // keyframe (a diverged run otherwise stalls the closer forever).
    bool to_closer = loop_closer_ != nullptr && deskewed_I.size() > 0;
    if (has_last_pose_) {
      const double dt = nsToS(stamp) - last_stamp_;
      if (dt > 0 && (T_W_I.translation() - last_pos_).norm() / dt > config_.loop_closure_max_speed_mps) {
        to_closer = false;
      }
    }
    has_last_pose_ = true;
    last_stamp_ = nsToS(stamp);
    last_pos_ = T_W_I.translation();
    if (loop_closer_ && !to_closer) ++loop_closer_skipped_;
    const resbie::SweepDiagnostics& g = sweep.diag;
    diagnostics_.insert(diagnostics_.end(),
                        {nsToS(stamp), double(sweep.points.size()), double(g.points_measured),
                         double(g.points_effective), double(g.updates), double(g.imu_only_updates),
                         g.info_rotation, g.info_translation, g.pos_sigma, double(to_closer),
                         g.nis_count ? g.nis_sum / double(g.nis_count) : NAN});
    if (to_closer) {
      // GTSAM's ISAM2 parallelizes through TBB and then depends on scheduling;
      // one thread keeps the loop closer deterministic (as the bievr bridge).
      loop_closer_arena_.execute(
          [&] { loop_closer_->addFrame(static_cast<uint64_t>(stamp), T_W_I, deskewed_I); });
    }
    if (map_point_stride_ == 0 || deskewed_I.size() == 0) return;
    MapScan& scan = map_scans_.emplace_back();
    scan.stamp = nsToS(stamp);
    scan.points.reserve((deskewed_I.size() / map_point_stride_ + 1) * 4);
    for (size_t i = 0; i < deskewed_I.size(); i += map_point_stride_) {
      const auto pt = deskewed_I[i];
      scan.points.insert(scan.points.end(),
                         {static_cast<float>(pt(0)), static_cast<float>(pt(1)),
                          static_cast<float>(pt(2)), sweep.intensity[i]});
    }
  }

  struct MapScan {
    double stamp;
    std::vector<float> points;  // rows of x, y, z, intensity
  };

  resbie::Config config_;
  size_t map_point_stride_;
  std::unique_ptr<tbb::global_control> tbb_control_;
  std::unique_ptr<resbie::Odometry> odometry_;
  std::unique_ptr<bievr::LoopCloser> loop_closer_;
  tbb::task_arena loop_closer_arena_{1};
  std::mutex mutex_;

  std::vector<double> trajectory_;  // rows of t, x, y, z, qx, qy, qz, qw
  std::vector<MapScan> map_scans_;
  size_t lidar_empty_ = 0;
  std::vector<double> diagnostics_;  // rows of kDiagColumns
  bool has_last_pose_ = false;
  double last_stamp_ = 0.0;
  Eigen::Vector3d last_pos_ = Eigen::Vector3d::Zero();
  size_t loop_closer_skipped_ = 0;

 public:
  static constexpr size_t kDiagColumns = 11;
  py::array_t<double> sweepDiagnostics() {
    std::lock_guard<std::mutex> lock(mutex_);
    const py::ssize_t rows = static_cast<py::ssize_t>(diagnostics_.size() / kDiagColumns);
    py::array_t<double> out({rows, py::ssize_t(kDiagColumns)});
    std::copy(diagnostics_.begin(), diagnostics_.end(), out.mutable_data());
    return out;
  }
};

}  // namespace

// Every default, as the nested dict a YAML config file would hold. The
// single source of defaults; resbie.DEFAULT_CONFIG is this.
py::dict defaultConfig() {
  const resbie::Config c;
  const bievr::LoopCloser::Config lc;
  const auto vec = [](const Eigen::Vector3d& v) { return std::vector<double>{v.x(), v.y(), v.z()}; };
  py::dict d;
  d["calibration"] = py::dict(py::arg("translation") = std::vector<double>{0, 0, 0},
                              py::arg("rotation") = std::vector<double>{1, 0, 0, 0, 1, 0, 0, 0, 1});
  d["lidar"] = py::dict(py::arg("min_range_m") = c.min_range, py::arg("max_range_m") = c.max_range,
                        py::arg("time_offset_s") = c.time_offset);
  d["imu"] = py::dict(py::arg("normalized") = c.imu_normalized,
                      py::arg("acc_noise_std") = vec(c.acc_noise_std),
                      py::arg("gyro_noise_std") = vec(c.gyro_noise_std),
                      py::arg("acc_bias_init_std") = vec(c.acc_bias_init_std),
                      py::arg("gyro_bias_init_std") = vec(c.gyro_bias_init_std),
                      py::arg("acc_bias_walk") = c.acc_bias_walk,
                      py::arg("gyro_bias_walk") = c.gyro_bias_walk);
  d["spline"] = py::dict(py::arg("knot_hz") = c.knot_hz, py::arg("iterations") = c.iterations,
                         py::arg("init_std") = c.init_std, py::arg("pos_noise") = c.pos_noise,
                         py::arg("ort_noise") = c.ort_noise,
                         py::arg("new_pos_noise") = c.new_pos_noise,
                         py::arg("new_ort_noise") = c.new_ort_noise);
  d["points"] = py::dict(py::arg("downsample_m") = c.downsample,
                         py::arg("informed_voxels") = c.informed_voxels,
                         py::arg("per_update") = c.per_update,
                         py::arg("noise_std_m") = c.point_noise_std,
                         py::arg("huber_delta_m") = c.association.huber_delta,
                         py::arg("gate_sigma") = c.gate_sigma);
  d["map"] = py::dict(py::arg("pixel_size_m") = c.map.px_size,
                      py::arg("voxel_size_m") = c.map.voxel_size,
                      py::arg("normal_tolerance_deg") = c.map.norm_tol_deg,
                      py::arg("smooth") = c.map.smooth, py::arg("weighted") = c.map.weighted,
                      py::arg("max_size") = c.map.max_size,
                      py::arg("init_min_voxels") = c.init_min_voxels);
  const auto& sc = lc.scan_context;
  d["loop_closure"] = py::dict(
      py::arg("enable") = bievr::LoopClosureConfig{}.enable,
      py::arg("max_speed_mps") = c.loop_closure_max_speed_mps,
      py::arg("keyframe_meter_gap") = lc.keyframe_meter_gap,
      py::arg("keyframe_deg_gap") = lc.keyframe_deg_gap,
      py::arg("keyframe_filter_size") = lc.keyframe_filter_size,
      py::arg("icp_filter_size") = lc.icp_filter_size, py::arg("sc_num_rings") = sc.num_rings,
      py::arg("sc_num_sectors") = sc.num_sectors, py::arg("sc_max_radius") = sc.max_radius,
      py::arg("sc_lidar_height") = sc.lidar_height,
      py::arg("sc_num_exclude_recent") = sc.num_exclude_recent,
      py::arg("sc_num_candidates") = sc.num_candidates,
      py::arg("sc_search_ratio") = sc.search_ratio,
      py::arg("sc_dist_threshold") = sc.dist_threshold,
      py::arg("sc_tree_making_period") = sc.tree_making_period,
      py::arg("history_keyframe_search_num") = lc.history_keyframe_search_num,
      py::arg("icp_max_correspondence_distance") = lc.icp_max_correspondence_distance,
      py::arg("icp_max_iterations") = lc.icp_max_iterations,
      py::arg("icp_transformation_epsilon") = lc.icp_transformation_epsilon,
      py::arg("icp_euclidean_fitness_epsilon") = lc.icp_euclidean_fitness_epsilon,
      py::arg("icp_ransac_iterations") = lc.icp_ransac_iterations,
      py::arg("loop_fitness_score_threshold") = lc.loop_fitness_score_threshold,
      py::arg("use_sc_yaw_guess") = lc.use_sc_yaw_guess,
      py::arg("sc_yaw_guess_min_deg") = lc.sc_yaw_guess_min_deg,
      py::arg("prior_noise_score") = lc.prior_noise_score,
      py::arg("odom_noise_rotation") = lc.odom_noise_rotation,
      py::arg("odom_noise_translation") = lc.odom_noise_translation,
      py::arg("loop_noise_score") = lc.loop_noise_score,
      py::arg("loop_noise_cauchy_c") = lc.loop_noise_cauchy_c,
      py::arg("isam_relinearize_threshold") = lc.isam_relinearize_threshold,
      py::arg("isam_relinearize_skip") = lc.isam_relinearize_skip);
  d["max_num_threads"] = c.max_num_threads;
  return d;
}

PYBIND11_MODULE(_core, m) {
  m.doc() = "resbie: RESPLE spline filter on the BIEVR map";
  m.def("default_config", &defaultConfig, "Every default, as a nested config dict.");
  py::class_<Resbie>(m, "Resbie")
      .def(py::init<const std::vector<std::string>&, size_t>(), py::arg("config_files"),
           py::arg("map_point_stride") = 0)
      .def("push_imu", &Resbie::pushImu, py::arg("stamp"), py::arg("acceleration"),
           py::arg("angular_velocity"))
      .def("push_lidar", &Resbie::pushLidar, py::arg("stamp"), py::arg("points"),
           py::arg("relative_times"))
      .def("finish", &Resbie::finish)
      .def("latest_pose", &Resbie::latestPose)
      .def("trajectory", &Resbie::trajectory)
      .def("keyframe_trajectory", &Resbie::keyframeTrajectory)
      .def("drain_map_scans", &Resbie::drainMapScans)
      .def("status", &Resbie::status)
      .def("sweep_diagnostics", &Resbie::sweepDiagnostics);
}

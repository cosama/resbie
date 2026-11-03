// pybind11 bindings for resbie: a front end and a back end, wired together by
// the caller (python/resbie/__init__.py shows how).
//
// FrontEnd: RESPLE's spline filter on BIEVR's map. IMU and LiDAR in; each
//   finished sweep out as a LIO result: (stamp, pose, points deskewed into the IMU
//   frame there).
// BackEnd: BIEVR's loop closer (Scan Context, ICP, GTSAM graph with IMU
//   factors). IMU and the front end's LIO results in; loop-closed keyframes out.
//
// Each owns its configuration and drains its own outputs. Both are synchronous:
// every push returns after all processing it triggered has finished, so
// results depend only on the pushed data and its order.

#include <dlfcn.h>
#include <link.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <tbb/global_control.h>
#include <tbb/task_arena.h>

#include <cmath>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "bievr_pgo/config_loader.h"
#include "bievr_pgo/loop_closer.h"
#include "resbie/config.h"
#include "resbie/odometry.h"

// Without these, results depend on the core a run starts on and on
// OMP_NUM_THREADS (see core/CMakeLists.txt). A build that lost them must not
// produce numbers.
#if !defined(EIGEN_NO_CPUID) || !defined(EIGEN_DONT_PARALLELIZE)
#error "resbie needs EIGEN_NO_CPUID and EIGEN_DONT_PARALLELIZE (core/CMakeLists.txt)"
#endif

namespace py = pybind11;

namespace {

using Array = py::array_t<double, py::array::c_style | py::array::forcecast>;
using Documents = std::vector<std::string>;  // YAML texts, later ones win

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

std::vector<double> vec(const Eigen::Vector3d& v) { return {v.x(), v.y(), v.z()}; }

// Eigen sizes its matrix-product blocks from the cache sizes CPUID reports on
// the core that runs the first product. On hybrid CPUs (P cores 48 KB L1d, E
// cores 32 KB) that changes the summation order, so a run's last bits depend
// on where it started, and the filter grows them into metres. resbie's own
// Eigen is built with EIGEN_NO_CPUID (CMakeLists.txt). The prebuilt GTSAM and
// PCL keep their own copy of that state (Eigen 3.4: an exported function-local
// static and its guard); pin every loaded copy to the same fixed sizes. Runs
// once, at import: every copy is loaded by then and none has run a product.
void pinEigenCacheSizes() {
  struct CacheSizes {
    std::ptrdiff_t l1, l2, l3;  // Eigen::internal::CacheSizes
  };
  static const CacheSizes fixed{Eigen::l1CacheSize(), Eigen::l2CacheSize(), Eigen::l3CacheSize()};
  dl_iterate_phdr(
      [](dl_phdr_info* info, size_t, void*) -> int {
        if (!info->dlpi_name || !info->dlpi_name[0]) return 0;
        void* lib = dlopen(info->dlpi_name, RTLD_NOLOAD | RTLD_LAZY);
        if (!lib) return 0;
        auto* sizes = static_cast<CacheSizes*>(
            dlsym(lib, "_ZZN5Eigen8internal20manage_caching_sizesENS_6ActionEPlS2_S2_E12m_cacheSizes"));
        auto* guard = static_cast<unsigned char*>(
            dlsym(lib, "_ZGVZN5Eigen8internal20manage_caching_sizesENS_6ActionEPlS2_S2_E12m_cacheSizes"));
        if (sizes && guard) {
          *sizes = fixed;
          __atomic_store_n(guard, 1, __ATOMIC_RELEASE);  // constructed: never queries CPUID
        }
        dlclose(lib);
        return 0;
      },
      nullptr);
}

// Every config error is a ValueError: a value of the wrong type included.
template <typename Config>
Config loadOrThrow(const Documents& documents, bool (*load)(const Documents&, Config&), const char* which) {
  Config config;
  bool ok = false;
  try {
    ok = load(documents, config);
  } catch (const YAML::Exception& e) {
    throw std::invalid_argument(std::string("resbie rejected the ") + which + " configuration: " + e.what());
  }
  if (!ok) throw std::invalid_argument(std::string("resbie rejected the ") + which + " configuration (see log above)");
  return config;
}

// One diagnostics row: column names with their values, in column order.
using Row = std::vector<std::pair<const char*, double>>;

std::vector<std::string> columnsOf(const Row& row) {
  std::vector<std::string> out;
  for (const auto& c : row) out.push_back(c.first);
  return out;
}

// Diagnostics rows since the last drain; none kept unless enabled.
class Diagnostics {
 public:
  Diagnostics(size_t columns, bool enabled) : columns_(columns), enabled_(enabled) {}

  void add(const Row& row) {
    if (!enabled_) return;
    for (const auto& c : row) values_.push_back(c.second);
  }

  py::array_t<double> drain() {
    const py::ssize_t cols = static_cast<py::ssize_t>(columns_);
    py::array_t<double> out({static_cast<py::ssize_t>(values_.size()) / cols, cols});
    std::copy(values_.begin(), values_.end(), out.mutable_data());
    values_.clear();
    return out;
  }

 private:
  size_t columns_;
  bool enabled_;
  std::vector<double> values_;
};

// ---------------------------------------------------------------- front end

// One row per sweep, in the order the sweeps are reported.
Row frontEndRow(const resbie::LioResult& s) {
  const resbie::SweepDiagnostics& g = s.diag;
  return {
      {"stamp", nsToS(s.stamp_ns)},                // sweep end, s
      {"points", double(g.points)},                // range-filtered points in the sweep
      {"points_measured", double(g.points_measured)},    // handed to the filter for it
      {"points_effective", double(g.points_effective)},  // of those, matched to the map
      {"updates", double(g.updates)},              // filter updates ending in the sweep (incl. a preceding gap)
      {"imu_only_updates", double(g.imu_only_updates)},  // updates without a matched point
      {"info_rotation", g.info_rotation},          // smallest eigenvalue of the LiDAR information, rotation, 1/rad^2
      {"info_translation", g.info_translation},    // same, translation, 1/m^2
      {"pos_sigma", g.pos_sigma},                  // position sigma of the newest control point, m
      {"nis_mean", g.nis_count ? g.nis_sum / double(g.nis_count) : NAN},  // mean point zp^2 / (HPH' + R) (prior); ~1 if honest
      {"weak_x", g.weak_translation.x()},          // world direction of info_translation (unit; sign arbitrary)
      {"weak_y", g.weak_translation.y()},
      {"weak_z", g.weak_translation.z()},
      {"imu_samples", double(g.imu_samples)},      // IMU samples measured
      {"acc_gated", double(g.acc_gated)},          // of their acceleration axes, dropped by the 10 m/s^2 residual gate
      {"vel_x", g.velocity.x()},                   // final spline velocity, mean over the sweep, world, m/s
      {"vel_y", g.velocity.y()},
      {"vel_z", g.velocity.z()},
      {"acc_bias_x", g.acc_bias.x()},              // accelerometer bias at the sweep's last update, m/s^2
      {"acc_bias_y", g.acc_bias.y()},
      {"acc_bias_z", g.acc_bias.z()},
      {"matched_voxels", double(g.matched_voxels)},  // distinct map voxels of the matched points, summed over updates
      {"imu_post_dv_x", g.imu_post_dv.x()},        // final spline's velocity change over the sweep minus the IMU's, world, m/s
      {"imu_post_dv_y", g.imu_post_dv.y()},
      {"imu_post_dv_z", g.imu_post_dv.z()},
      {"imu_post_sigma", g.imu_post_sigma},        // its std under the configured accelerometer noise, m/s
      {"gyro_bias_x", g.gyro_bias.x()},            // gyro bias at the sweep's last update, rad/s
      {"gyro_bias_y", g.gyro_bias.y()},
      {"gyro_bias_z", g.gyro_bias.z()},
      {"up_x", g.gravity_up.x()},                  // estimated gravity up direction, world (unit), at the last update
      {"up_y", g.gravity_up.y()},
      {"up_z", g.gravity_up.z()},
      {"up_sigma", g.gravity_sigma},               // its largest std, rad
  };
}

class FrontEnd {
 public:
  FrontEnd(const Documents& config, int max_num_threads, size_t point_stride, bool diagnostics)
      : config_(loadOrThrow(config, resbie::loadConfig, "front-end")),
        diagnostics_(frontEndRow({}).size(), diagnostics) {
    if (max_num_threads < 0) throw std::invalid_argument("max_num_threads must be >= 0");
    if (max_num_threads > 0) {
      tbb_control_ = std::make_unique<tbb::global_control>(
          tbb::global_control::max_allowed_parallelism, max_num_threads);
    }
    odometry_ = std::make_unique<resbie::Odometry>(config_, max_num_threads, point_stride);
    // Runs inside Odometry, mutex_ held.
    odometry_->setObserver([this](resbie::LioResult&& result) {
      diagnostics_.add(frontEndRow(result));
      results_.push_back(std::move(result));
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
      throw std::invalid_argument("points must be (N, 3) or (N, 4): x, y, z[, intensity]");
    }
    const size_t n = static_cast<size_t>(points.shape(0));
    const size_t cols = static_cast<size_t>(points.shape(1));
    if (times.ndim() != 1 || static_cast<size_t>(times.shape(0)) != n) {
      throw std::invalid_argument("relative_times must be (N,) matching points");
    }
    const int64_t t0 = secondsToNs(stamp) + std::llround(config_.time_offset * 1e9);
    const double* p = points.data();
    const double* t = times.data();
    resbie::Sweep sweep;
    sweep.start_ns = sweep.end_ns = t0;
    sweep.times.resize(n);
    for (size_t i = 0; i < n; ++i) {
      if (!std::isfinite(t[i]) || t[i] < 0.0) {
        throw std::invalid_argument("relative_times must be finite and >= 0");
      }
      sweep.times[i] = t0 + std::llround(t[i] * 1e9);
      sweep.end_ns = std::max(sweep.end_ns, sweep.times[i]);
    }

    py::gil_scoped_release release;
    // Range filter in the LiDAR frame (BIEVR), then into the IMU frame.
    const double min2 = config_.min_range * config_.min_range;
    const double max2 = config_.max_range * config_.max_range;
    sweep.points.resize(n);
    sweep.intensity.resize(n);
    sweep.ranges.resize(n);
    size_t kept = 0;  // points kept so far, compacted in place
    for (size_t i = 0; i < n; ++i) {
      const Eigen::Vector3d p_L(p[i * cols], p[i * cols + 1], p[i * cols + 2]);
      const double r2 = p_L.squaredNorm();
      if (!(std::isfinite(r2) && r2 >= min2 && r2 <= max2)) continue;
      const Eigen::Vector3d p_I = config_.T_I_L * p_L;
      sweep.points[kept] = p_I;
      sweep.times[kept] = sweep.times[i];
      sweep.intensity[kept] = cols == 4 ? p[i * cols + 3] : 0.0;
      sweep.ranges[kept] = p_I.norm();
      ++kept;
    }
    sweep.points.resize(kept);
    sweep.times.resize(kept);
    sweep.intensity.resize(kept);
    sweep.ranges.resize(kept);

    std::lock_guard<std::mutex> lock(mutex_);
    // An empty sweep still goes in: it tells the filter LiDAR saw nothing over
    // [start, end], so the spline may step through on IMU alone.
    return odometry_->addSweep(std::move(sweep));
  }

  void finish() {
    py::gil_scoped_release release;
    std::lock_guard<std::mutex> lock(mutex_);
    odometry_->finish();
  }

  // LIO results since the last call: (stamp, pose, points). The point
  // buffers go to numpy as they are, without a copy.
  py::list drain() {
    std::vector<resbie::LioResult> results;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      results.swap(results_);
    }
    py::list out;
    for (resbie::LioResult& result : results) {
      const Eigen::Vector3d& p = result.p_W_I;
      const Eigen::Quaterniond& q = result.q_W_I;
      py::array_t<double> pose(7);
      const double row[7] = {p.x(), p.y(), p.z(), q.x(), q.y(), q.z(), q.w()};
      std::copy(row, row + 7, pose.mutable_data());
      auto* points = new std::vector<double>(std::move(result.points));
      py::capsule owner(points, [](void* v) { delete static_cast<std::vector<double>*>(v); });
      const py::ssize_t rows = static_cast<py::ssize_t>(points->size() / 4);
      out.append(py::make_tuple(nsToS(result.stamp_ns), pose,
                                py::array_t<double>({rows, py::ssize_t{4}}, points->data(), owner)));
    }
    return out;
  }

  py::array_t<double> drainDiagnostics() {
    std::lock_guard<std::mutex> lock(mutex_);
    return diagnostics_.drain();
  }

  py::dict status() {
    std::lock_guard<std::mutex> lock(mutex_);
    const resbie::Odometry::Stats s = odometry_->stats();
    py::dict d;
    d["imu_accepted"] = s.imu_accepted;
    d["imu_rejected"] = s.imu_rejected;
    d["lidar_accepted"] = s.lidar_accepted;
    d["lidar_rejected"] = s.lidar_rejected;
    d["lidar_empty"] = s.lidar_empty;
    d["initialized"] = odometry_->initialized();
    d["sweeps_dropped_init"] = s.sweeps_dropped_init;
    d["sweeps_finalized"] = s.sweeps_finalized;
    d["sweeps_unfinished"] = s.sweeps_unfinished;
    d["filter_updates"] = s.updates;
    d["imu_only_updates"] = s.imu_only_updates;
    d["gap_knots"] = s.gap_knots;
    d["points_measured"] = s.points_measured;
    d["points_effective"] = s.points_effective;
    d["map_voxels"] = s.map_voxels;
    d["acc_bias"] = vec(odometry_->accBias());
    d["gyro_bias"] = vec(odometry_->gyroBias());
    d["gravity_up"] = vec(odometry_->gravityUp());
    return d;
  }

 private:
  resbie::Config config_;
  Diagnostics diagnostics_;
  std::unique_ptr<tbb::global_control> tbb_control_;
  std::unique_ptr<resbie::Odometry> odometry_;
  std::mutex mutex_;

  std::vector<resbie::LioResult> results_;  // not yet drained
};

// ----------------------------------------------------------------- back end

// One row per pushed LIO result.
Row backEndRow(double stamp, size_t points, bool keyframe) {
  return {
      {"stamp", stamp},                 // s, as pushed
      {"points", double(points)},       // 0: never a keyframe
      {"keyframe", double(keyframe)},   // became a pose-graph keyframe
  };
}

class BackEnd {
 public:
  BackEnd(const Documents& config, bool diagnostics)
      : diagnostics_(backEndRow(0.0, 0, false).size(), diagnostics),
        loop_closer_(loadOrThrow(config, bievr::loadLoopClosureConfig, "back-end")) {}

  bool pushImu(double stamp, const Array& acc, const Array& gyro) {
    const int64_t t = secondsToNs(stamp);
    const Eigen::Vector3d a = toV3(acc, "acceleration");
    const Eigen::Vector3d w = toV3(gyro, "angular_velocity");
    py::gil_scoped_release release;
    std::lock_guard<std::mutex> lock(mutex_);
    if (t <= last_imu_ns_) {
      ++imu_rejected_;
      return false;
    }
    last_imu_ns_ = t;
    ++imu_accepted_;
    loop_closer_.addImu(static_cast<uint64_t>(t), a, w);
    return true;
  }

  bool pushLio(double stamp, const Array& pose, const Array& points) {
    const int64_t t = secondsToNs(stamp);
    if (pose.size() != 7) throw std::invalid_argument("pose must be [x, y, z, qx, qy, qz, qw]");
    if (points.ndim() != 2 || points.shape(1) < 3) {
      throw std::invalid_argument("points must be (N, 3) or wider: x, y, z[, ...]");
    }
    const double* v = pose.data();
    const Eigen::Quaterniond q(v[6], v[3], v[4], v[5]);
    if (!(std::abs(q.norm() - 1.0) < 1e-6)) throw std::invalid_argument("pose quaternion must be unit");
    const bievr::Transform T_W_I(q, Eigen::Vector3d(v[0], v[1], v[2]));
    const bievr::LoopCloser::PointsView cloud(
        Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(
            points.data(), points.shape(0), points.shape(1)));

    py::gil_scoped_release release;
    std::lock_guard<std::mutex> lock(mutex_);
    if (t <= last_lio_ns_) {
      ++lio_rejected_;
      return false;
    }
    ++lio_accepted_;
    last_lio_ns_ = t;
    // Without points there is nothing to match: never a keyframe.
    bool keyframe = false;
    if (cloud.rows() > 0) {
      // GTSAM's ISAM2 parallelizes through TBB and then depends on scheduling;
      // one thread keeps the loop closer deterministic.
      arena_.execute([&] { keyframe = loop_closer_.addFrame(static_cast<uint64_t>(t), T_W_I, cloud); });
    }
    // Keyframes report the stamp as pushed, so it matches the caller's exactly.
    if (keyframe) keyframe_stamps_.push_back(stamp);
    diagnostics_.add(backEndRow(stamp, cloud.rows(), keyframe));
    return true;
  }

  void finish() {
    py::gil_scoped_release release;
    std::lock_guard<std::mutex> lock(mutex_);
    arena_.execute([&] { loop_closer_.finish(); });
  }

  py::array_t<double> keyframes() {
    std::lock_guard<std::mutex> lock(mutex_);
    const std::vector<bievr::Transform> poses = loop_closer_.keyframePoses();
    py::array_t<double> out({static_cast<py::ssize_t>(poses.size()), py::ssize_t{8}});
    auto r = out.mutable_unchecked<2>();
    for (size_t i = 0; i < poses.size(); ++i) {
      const Eigen::Vector3d& t = poses[i].translation();
      const Eigen::Quaterniond q = poses[i].quaternion();
      const double row[8] = {keyframe_stamps_[i], t.x(), t.y(), t.z(), q.x(), q.y(), q.z(), q.w()};
      std::copy(row, row + 8, r.mutable_data(i, 0));
    }
    return out;
  }

  py::array_t<double> drainDiagnostics() {
    std::lock_guard<std::mutex> lock(mutex_);
    return diagnostics_.drain();
  }

  py::dict status() {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto ls = loop_closer_.stats();
    py::dict d;
    d["imu_accepted"] = imu_accepted_;
    d["imu_rejected"] = imu_rejected_;
    d["lio_accepted"] = lio_accepted_;
    d["lio_rejected"] = lio_rejected_;
    d["keyframes"] = ls.num_keyframes;
    d["loops"] = ls.num_loops;
    d["rejected"] = ls.num_rejected;
    return d;
  }

 private:
  Diagnostics diagnostics_;
  bievr::LoopCloser loop_closer_;
  tbb::task_arena arena_{1};
  std::mutex mutex_;

  int64_t last_imu_ns_ = std::numeric_limits<int64_t>::min();
  size_t imu_accepted_ = 0, imu_rejected_ = 0;
  size_t lio_accepted_ = 0, lio_rejected_ = 0;
  int64_t last_lio_ns_ = std::numeric_limits<int64_t>::min();
  std::vector<double> keyframe_stamps_;  // s, as pushed, one per loop-closer keyframe
};

// Every default, as the nested dict a YAML config file would hold. The
// single source of defaults; resbie.FrontEnd.DEFAULT_CONFIG is this.
// calibration has none: it must be set.
py::dict frontEndDefaultConfig() {
  const resbie::Config c;
  py::dict d;
  d["lidar"] = py::dict(py::arg("min_range_m") = c.min_range, py::arg("max_range_m") = c.max_range,
                        py::arg("time_offset_s") = c.time_offset);
  d["imu"] = py::dict(py::arg("acc_noise_std") = vec(c.acc_noise_std),
                      py::arg("gyro_noise_std") = vec(c.gyro_noise_std),
                      py::arg("acc_bias_init_std") = vec(c.acc_bias_init_std),
                      py::arg("gyro_bias_init_std") = vec(c.gyro_bias_init_std),
                      py::arg("acc_bias_walk") = c.acc_bias_walk,
                      py::arg("gyro_bias_walk") = c.gyro_bias_walk,
                      py::arg("gravity_init_std_deg") = c.gravity_init_std_deg);
  d["spline"] = py::dict(py::arg("knot_hz") = c.knot_hz, py::arg("iterations") = c.iterations,
                         py::arg("init_std") = c.init_std, py::arg("pos_noise") = c.pos_noise,
                         py::arg("ort_noise") = c.ort_noise,
                         py::arg("new_pos_noise") = c.new_pos_noise,
                         py::arg("new_ort_noise") = c.new_ort_noise);
  d["points"] = py::dict(py::arg("downsample_m") = c.downsample,
                         py::arg("informed_voxels") = c.informed_voxels,
                         py::arg("noise_std_m") = c.point_noise_std,
                         py::arg("huber_delta_m") = c.association.huber_delta);
  d["map"] = py::dict(py::arg("pixel_size_m") = c.map.px_size,
                      py::arg("voxel_size_m") = c.map.voxel_size,
                      py::arg("normal_tolerance_deg") = c.map.norm_tol_deg,
                      py::arg("max_size") = c.map.max_size,
                      py::arg("init_min_voxels") = c.init_min_voxels);
  return d;
}

// Same for the back end (flat keys); resbie.BackEnd.DEFAULT_CONFIG.
py::dict backEndDefaultConfig() {
  const bievr::LoopCloser::Config lc;
  const auto& sc = lc.scan_context;
  return py::dict(
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
      py::arg("loop_max_correction_z") = lc.loop_max_correction_z,
      py::arg("prior_noise_score") = lc.prior_noise_score,
      py::arg("odom_noise_rotation") = lc.odom_noise_rotation,
      py::arg("odom_noise_translation") = lc.odom_noise_translation,
      py::arg("loop_noise_score") = lc.loop_noise_score,
      py::arg("loop_noise_cauchy_c") = lc.loop_noise_cauchy_c,
      py::arg("isam_relinearize_threshold") = lc.isam_relinearize_threshold,
      py::arg("isam_relinearize_skip") = lc.isam_relinearize_skip,
      py::arg("acc_noise_std") = vec(lc.acc_noise_std),
      py::arg("gyro_noise_std") = vec(lc.gyro_noise_std),
      py::arg("acc_bias_walk") = lc.acc_bias_walk,
      py::arg("gyro_bias_walk") = lc.gyro_bias_walk,
      py::arg("acc_bias_init_std") = vec(lc.acc_bias_init_std),
      py::arg("gyro_bias_init_std") = vec(lc.gyro_bias_init_std));
}

}  // namespace

PYBIND11_MODULE(_core, m) {
  m.doc() = "resbie: RESPLE spline filter on the BIEVR map (FrontEnd), BIEVR loop closer (BackEnd)";
  pinEigenCacheSizes();
  m.def(
      "set_log_level",
      [](const std::string& level) {
        if (!resbie::setLogLevel(level)) {
          throw std::invalid_argument("log level must be debug, info, warn or error, not '" + level + "'");
        }
      },
      py::arg("level"));
  py::class_<FrontEnd>(m, "FrontEnd")
      .def(py::init<const Documents&, int, size_t, bool>(), py::arg("config"), py::arg("max_num_threads"),
           py::arg("point_stride"), py::arg("diagnostics"))
      .def("push_imu", &FrontEnd::pushImu, py::arg("stamp"), py::arg("acceleration"),
           py::arg("angular_velocity"))
      .def("push_lidar", &FrontEnd::pushLidar, py::arg("stamp"), py::arg("points"),
           py::arg("relative_times"))
      .def("finish", &FrontEnd::finish)
      .def("drain", &FrontEnd::drain)
      .def("drain_diagnostics", &FrontEnd::drainDiagnostics)
      .def("status", &FrontEnd::status)
      .attr("DEFAULT_CONFIG") = frontEndDefaultConfig();
  m.attr("FrontEnd").attr("DIAGNOSTIC_COLUMNS") = py::tuple(py::cast(columnsOf(frontEndRow({}))));
  py::class_<BackEnd>(m, "BackEnd")
      .def(py::init<const Documents&, bool>(), py::arg("config"), py::arg("diagnostics"))
      .def("push_imu", &BackEnd::pushImu, py::arg("stamp"), py::arg("acceleration"),
           py::arg("angular_velocity"))
      .def("push_lio", &BackEnd::pushLio, py::arg("stamp"), py::arg("pose"), py::arg("points"))
      .def("finish", &BackEnd::finish)
      .def("keyframes", &BackEnd::keyframes)
      .def("drain_diagnostics", &BackEnd::drainDiagnostics)
      .def("status", &BackEnd::status)
      .attr("DEFAULT_CONFIG") = backEndDefaultConfig();
  m.attr("BackEnd").attr("DIAGNOSTIC_COLUMNS") = py::tuple(py::cast(columnsOf(backEndRow(0.0, 0, false))));
}

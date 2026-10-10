# resbie: design notes

The details behind `README.md`. What each part does, where it lives, and what
changed from upstream.

## What comes from where

- **From RESPLE:** the estimator. A cubic B-spline over SE(3) at `spline.knot_hz`
  (100 Hz), updated by an iterated EKF over the four active control points
  plus the IMU biases. IMU readings and LiDAR points are residuals in one
  update, and the spline's motion model holds the directions the LiDAR can't
  see. This is what keeps RESPLE alive in point-depleted scenes.
- **From BIEVR:** the map, the point selection and the loop closer. Per-voxel
  oriented height images with sub-voxel gradients (sharp maps, fine
  structure), BIEVR's voxel downsampling plus informed sampling, Huber
  weighting, and Scan Context + GTSAM pose-graph loop closure.
- **Dropped:** RESPLE's ikd-tree and ROS layer; BIEVR's LiDAR-only LM
  registration, IMU preintegration window, static bias init and pipeline.

## How the two connect

RESPLE's filter needs only one thing per point from the map: a plane
`(normvec, dist)` at the point's spline-interpolated world position. Its
Jacobians with respect to the control points never look at the map.

BIEVR's residual `r(p_w) = p_o.z - I(p_o.x, p_o.y)` (the point's height above
its voxel's smoothed height image, `p_o = T_C_W p_w`) linearizes to exactly
that: `normvec = R_C_W^T (-dI/dx, -dI/dy, 1)` (not unit length) and
`dist = r - normvec . p_w`. `vendor/resple/Association.h` does this lookup in
place of the ikd-tree search. The filter re-associates on every converged
iteration, so the bump-image linearization is refreshed for free.

Huber weighting (`points.huber_delta_m`) enters as IRLS: a
per-point variance scale `PointData::var_scale`.

## Data flow (`cpp/resbie/odometry.h`)

This is RESPLE's `processData` loop made synchronous, with no worker thread.
Every push runs all processing the new data allows before returning.

1. **Init.** The first sweep covering at least `map.init_min_voxels`
   distinct map voxels seeds BIEVR's map at the initial pose (level, zero
   yaw). Gravity comes from the 15 IMU samples before it, as in RESPLE.
   Sparse sweeps before that (a drone on the ground) are dropped, since a
   pose dead-reckoned on them would smear the map. So are sweeps that start
   before the first IMU sample: the filter would
   run on LiDAR alone and seed the map before gravity is measured.
2. **Sweep promotion.** A sweep's points enter the filter's buffer only when
   `collectMeasurements()` would otherwise stop for lack of points. At that
   moment the spline reaches the sweep's start, which BIEVR's informed
   sampling needs: it places points with the spline (clamped to its span) to
   look up voxel scores. Selected points are time-sorted into the buffer.
3. **Filter.** This is RESPLE's `collectMeasurements` and
   `updateIEKFLiDARInertial`, unchanged except for the map.
4. **Finalization.** Once the spline over a sweep has left the 4-knot active
   window (`end + 5 dt <= maxTime`), the sweep is final:
   - every point (the full range-filtered sweep, not just the sampled ones)
     is placed with the spline pose at its own timestamp;
   - the sweep goes into BIEVR's map, weighted by range as in BIEVR;
   - one pose is reported at the sweep end;
   - the sweep, deskewed to that pose, goes to the loop closer and the map
     export.

Results depend only on the pushed data and its order. This is tested
bit-identical across runs and OpenMP/TBB thread counts
(`tests/test_resbie.py`).

## Beyond RESPLE (hard areas)

RESPLE's noise model and filter are kept as they are. The only additions are
for continuing through areas with little or no LiDAR:

- **Gap handling.** RESPLE skipped the update when no point matched, and
  jumped LiDAR gaps on its constant-velocity model, discarding the gap's IMU.
  Here the update runs IMU-only, and the spline steps knot by knot through
  any stretch the arrived sweeps show to be empty. Empty sweeps are passed in
  as gap markers.
- **Mahalanobis point gate,** `zp^2 <= gate_sigma^2 (HPH' + R)` (`points.gate_sigma`, 3),
  using RESPLE's own covariance. RESPLE's gate widened only when the filter
  was certain.
- **Bias drift** (`imu.*_bias_walk`): RESPLE gives the biases no process
  noise, so they freeze.
- **Diagnostics** (`sweep_diagnostics()`): per sweep, matched points,
  IMU-only updates, the smallest eigenvalues of the LiDAR information
  (rotation, translation), position sigma, loop-closer use, point NIS.
- **Loop-closer guard:** empty or implausibly fast sweeps
  (`loop_closure.max_speed_mps`) never become keyframes, so a diverged run
  cannot stall the synchronous closer.

## The pose graph (on by default)

The filter tracks against its own map, so once that map is built slightly
tilted (it happens on stairs), the accelerometer can disagree for minutes
without being able to rotate it back. The graph downstream fixes the
outputs:

- One node per keyframe (every `keyframe_meter_gap` or `keyframe_deg_gap`)
  with pose, velocity and IMU bias.
- Between consecutive keyframes: the odometry's relative pose, and a GTSAM
  `CombinedImuFactor` from every raw IMU sample in between (preintegrated,
  bias random walk, gravity along world z). The noise comes from the `imu`
  section; the per-sample standard deviations become GTSAM's continuous
  densities at the measured IMU rate.
- The first keyframe's prior fixes only yaw and position (the gauge); roll
  and pitch are left to gravity.
- Loop closures (Scan Context + ICP) add relative poses as before.
- iSAM2 updates incrementally per keyframe (loop search uses the corrected
  map); `finish()` runs one batch optimization of the whole graph.

`keyframe_trajectory()` returns the optimized keyframes; carrying their
correction to every pose (blend by distance and angle between keyframes)
gives a gravity-aligned, loop-closed trajectory. Tilt and accelerometer bias
separate only when the sensor rotates; a sensor that never turns keeps
some ambiguity at the start.

## Layout

- `cpp/bindings.cpp`: pybind11 module `resbie._core`. Its API mirrors
  `bievr`'s, plus `finish()`.
- `cpp/resbie/config.h`: YAML loader, the single source of defaults (see
  Configuration).
- `cpp/resbie/odometry.h`: the synchronous driver.
- `python/resbie/__init__.py`: the `Resbie` session and `DEFAULT_CONFIG`.
- `vendor/resple`: edited copies of the RESPLE headers, from upstream
  `2f8b09e` (`resple/include`).
- `vendor/bievr`: edited copies of the BIEVR map, preprocess, and the pgo
  and scancontext modules, from upstream `b458505`
  (`BIEVR-LIO-SLAM`, with the loop closer made synchronous).

Every edit in `vendor/` is marked `resbie:` at the site. Grep for it to
enumerate the delta:

- **RESPLE:** ROS removed; `Association` reads BIEVR's map; points arrive in
  the body frame; Huber variance scale; bias getters.
- **BIEVR:** the samplers return indices so per-point times survive; sort
  tie-breaks; the config loader is reduced to its YAML helpers; the LM
  registration is removed from `ls_optimizer.h`; the loop closer is
  synchronous only (no worker threads, GPS or map bundle) and carries the
  IMU (velocity and bias states, CombinedImuFactor, gravity-only prior).

## Point density

`points.downsample_m` decides how many points the filter
measures (default 0.25 m). The map always gets the full sweep. Informed
sampling keeps the 300 most informative map voxels and *every* downsampled
point inside them, so at 0.1 m the filter sees ~7 points per 0.5 m voxel
(~2100 per sweep).
Those points share one surface and, when point times are synthesized, one
timing error, but the filter counts them as independent, and the flexible
spline bends to fit them. With a partly occluded LiDAR whose synthesized
times are off by tens of ms for half of each scan, that bent the heading by
several degrees and doubled walls. At 0.25 m (a few points per map
voxel) the map stays just as sharp and the trajectory honest. Much coarser
(0.5 m) starves cluttered scenes such as forests: too few points per trunk.

## Configuration

One YAML schema, sections by function, standard deviations in SI units.
Later files override earlier ones per key; unknown keys are errors.
`resbie.DEFAULT_CONFIG` lists every key with its default (taken from the C++
loader); only `calibration` must be set.

| section | keys |
|---|---|
| `calibration` | `translation`, `rotation` (T_I_L, LiDAR to IMU, row-major) |
| `lidar` | `min_range_m`, `max_range_m`, `time_offset_s` |
| `imu` | `normalized`; `acc_noise_std`, `gyro_noise_std` (per sample); `acc_bias_init_std`, `gyro_bias_init_std`; `acc_bias_walk`, `gyro_bias_walk` (per sqrt(s)) |
| `spline` | `knot_hz` (at most the IMU rate, rounded: each knot interval needs >= 1 IMU sample when the LiDAR is starved; init warns otherwise), `iterations`, `init_std`; `pos_noise`, `ort_noise` (motion-model noise of settled control points, m/s and rad/s); `new_pos_noise`, `new_ort_noise` (newest control point) |
| `points` | `downsample_m`, `informed_voxels`, `per_update`, `noise_std_m`, `huber_delta_m`, `gate_sigma` |
| `map` | `pixel_size_m`, `voxel_size_m`, `normal_tolerance_deg`, `smooth`, `weighted`, `max_size`, `init_min_voxels` |
| `loop_closure` | `enable` (the whole graph, default on), `max_speed_mps`, and the closer's keyframe, Scan Context, ICP and pose-graph keys |
| (top level) | `max_num_threads` |

Noises and biases take one number (all axes) or three. RESPLE's parameters
map one to one: `cov_*` variances become `*_std`, and `std_sys` times
`sqrt(cov_RCP_*)` becomes `*_noise`.

## Building

`core/` is a pip-installable package (scikit-build, pybind11):

```bash
cd core
uv venv .venv && uv pip install --python .venv/bin/python . pytest
.venv/bin/python -m pytest tests
```

Native dependencies (Ubuntu 24.04): `libeigen3-dev libgtsam-dev libmetis-dev
libpcl-dev libtbb-dev libyaml-cpp-dev libsuitesparse-dev`. GTSAM's CMake
export references a `libCppUnitLite.a` that the package does not ship; an
empty one satisfies it: `ar cr /usr/lib/x86_64-linux-gnu/libCppUnitLite.a`.
`docker/build.sh` builds an image with the package installed.

Usage: write `resbie.DEFAULT_CONFIG` with your LiDAR -> IMU `calibration` to a
YAML file, create `resbie.Resbie(path)`, push IMU (`push_imu`) and LiDAR
(`push_lidar`: stamp of the earliest point, points, times relative to it) in
time order, call `finish()`, read `trajectory()`. Every call is synchronous.

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
  weighting, and GTSAM pose-graph loop closure (with BTC + `small_gicp` and GNC).
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
   distinct map voxels seeds BIEVR's map at the initial pose, levelled on the
   mean specific force of the 15 IMU samples before it (zero yaw), as in
   RESPLE. That pose is the map frame's gauge; gravity's direction in it is
   then estimated (see below), so a moving start is fine.
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
   `updateIEKFLiDARInertial`, on the map above and with the changes below.
4. **Finalization.** Once the spline over a sweep has left the 4-knot active
   window (`end + 4 dt <= maxTime`), the sweep is final:
   - every point (the full range-filtered sweep, not just the sampled ones)
     is placed with the spline pose at its own timestamp;
   - the sweep goes into BIEVR's map, weighted by range as in BIEVR;
   - one LIO result is reported (`FrontEnd.drain()`): the pose at the
     sweep end and every `point_stride`-th point deskewed to it.

Results depend only on the pushed data and its order. This is tested
bit-identical across runs and OpenMP/TBB thread counts
(`tests/test_resbie.py`).

## Beyond RESPLE (hard areas)

What resbie changes in RESPLE's filter, mostly to continue through areas
with little or no LiDAR:

- **Gap handling.** RESPLE skipped the update when no point matched, and
  jumped LiDAR gaps on its constant-velocity model, discarding the gap's IMU.
  Here the update runs IMU-only, and the spline steps knot by knot through
  any stretch the arrived sweeps show to be empty. Empty sweeps are passed in
  as gap markers.
- **No point gate.** Every matched point is measured; the Huber weight
  (`points.huber_delta_m`) caps each point's pull. RESPLE gated `|zp|` by a
  fixed threshold unless the filter was certain; a Mahalanobis gate tried
  before fired on about one matched point in 10^5-10^6.
- **No IMU gate.** Every IMU sample is measured. RESPLE dropped an axis
  whose residual passed 10 m/s^2 / 5 rad/s, which threw away exactly the
  IMU that pulls a drifted spline back.
- **Fixed iteration count:** `spline.iterations` updates per knot, each
  re-associating and linearizing every point once. RESPLE's convergence test
  never fired, so it ran one update more than configured.
- **Same bits on every core.** Eigen sizes its matrix-product blocks from
  the cache sizes CPUID reports on the core that runs the first product; on
  hybrid CPUs (P/E cores) a run's last bits depended on where it started, and
  the filter grows those into metres. resbie's Eigen is built with
  `EIGEN_NO_CPUID`, and at import the binding pins the cache sizes of
  the Eigen copies inside the prebuilt GTSAM and PCL to the same values.
  `EIGEN_DONT_PARALLELIZE` keeps Eigen's products on one thread: with OpenMP,
  Eigen blocks a large product differently on one thread than on several, so
  `OMP_NUM_THREADS=1` changed the bits. resbie's own loops stay parallel and
  write per element. The binding refuses to compile without both defines.
- **Bias drift** (`imu.*_bias_walk`): RESPLE gives the biases no process
  noise, so they freeze.
- **Gravity direction as a state.** RESPLE fixes gravity at init along the
  map's z. A start that is not static, or a map region built tilted, then
  leaves the IMU with a fake horizontal acceleration g sin(tilt) that the
  body-fixed accelerometer bias cannot absorb. Here the up direction in the
  map frame is a 2-DoF state (S^2 tangent error, constant: no process noise,
  prior `imu.gravity_init_std_deg`, 1 rad; 0 fixes it as RESPLE), with the
  exact specific-force Jacobian, re-anchored once at the exit of every
  update. Tilt separates from the horizontal accelerometer bias only with
  heading changes; until then its std stays near `acc_bias_init_std / g`.
  Poses stay in the map frame (only as level as the start); the estimate is
  reported as `up_*` in the diagnostics and `gravity_up` in `status()`.
- **Diagnostics** (`FrontEnd(..., diagnostics=True)`, then
  `drain_diagnostics()`: one row per sweep since the last call, columns in
  `FrontEnd.DIAGNOSTIC_COLUMNS`, documented in `frontEndRow()` in
  `cpp/bindings.cpp`):
  per sweep, matched points, IMU-only updates, the smallest eigenvalues of the
  LiDAR information (rotation, translation) and the weakest translation
  direction, position sigma, point NIS, IMU samples, the sweep's mean
  velocity, the accelerometer and gyro biases, the estimated up direction
  and its std, and an a-posteriori IMU check (the final trajectory's velocity change per sweep
  minus the IMU's: each IMU sample is otherwise only checked once, before
  later updates reshape the trajectory). `resbie.set_log_level("debug")` (a run
  option, not config) prints the same as one line per sweep, plus one line per
  filter update (what it measured, how far it moved the newest knot); the
  default `info` prints none of it.

## Front end and back end

Two classes wired together by the caller (`python/resbie/__init__.py` shows
the loop). Each owns its configuration, which decides the results; how a run
executes is a constructor argument (`FrontEnd`: `max_num_threads`,
`point_stride`, `diagnostics`; `BackEnd`: `diagnostics`). Neither writes
files: every output is drained or read by the caller.

- `FrontEnd`: the filter above. `push_imu`, `push_lidar`, `finish()`, and
  `drain()`: the LIO results since the last call, one per finished sweep, each
  `(stamp, pose, points)`: the time of the sweep's last point, the pose
  `[x, y, z, qx, qy, qz, qw]` there (the spline's own quaternion, w >= 0), and
  every `point_stride`-th range-filtered point (0: none) deskewed into the IMU
  frame there, xyzi; the filter and its map use every point regardless. A sweep is finished once
  the spline over it is final; one push can finish none or several. Its
  poses are the odometry, its points on them the map; the front end alone is
  a complete odometry. The points are written once, in `Odometry::finalize`
  (only those the stride keeps), and reach numpy and the back end without a
  copy.
- `BackEnd`: the pose graph below. `push_imu`, `push_lio` (every LIO result, not
  only keyframes), `finish()`, `keyframes()`. It needs every IMU sample up to a
  result's stamp before that result, which any time-ordered merge gives. It
  uses every point it is given; thinning them is the caller's choice. Its
  diagnostics: one row per LIO result (points, keyframe).

At the interface every time is float64 seconds and every number float64.
Inside, both halves count integer nanoseconds (rounded once on the way in), so
a LIO result's stamp lands in the back end up to ~120 ns (the resolution of a
float64 second at Unix times) from where the front end had it; `keyframes()`
reports the stamps exactly as pushed, so they match the caller's. Sweeps must
start no earlier and end strictly later than the previous one, so the stamps
increase.

The back end needs nothing else from the front end: velocity comes from
consecutive poses; a result without points never becomes a keyframe. A
diverged front end is the caller's to catch, e.g. by stopping once the
odometry moves implausibly fast: its map is corrupt, nothing downstream
recovers it. Everything is
synchronous, so offline runs are deterministic; live, the back end can run
on its own thread from one queue of IMU samples and LIO results (`push_lio`
releases the GIL) without changing its results.

## The pose graph (the back end)

The filter tracks against its own map, so once that map is built slightly
tilted (it happens on stairs), the accelerometer can disagree for minutes
without being able to rotate it back. The graph downstream fixes the
outputs:

- One node per keyframe (every `keyframe_meter_gap` or `keyframe_deg_gap`)
  with pose, velocity and IMU bias.
- Between consecutive keyframes: the odometry's relative pose, and a GTSAM
  `CombinedImuFactor` from every raw IMU sample in between (preintegrated,
  bias random walk, gravity along world z). The noise comes from the back
  end's own IMU keys (defaults: the front end's); the per-sample standard
  deviations become GTSAM's continuous densities at the measured IMU rate.
- The first keyframe's prior fixes only yaw and position (the gauge); roll
  and pitch are left to gravity.
- Loop closures (Binary Triangle Combined descriptors `BTC` + `small_gicp` over
  local odometry submaps, seeded from BTC's 6-DoF triangle solver) add relative
  poses. BTC sees each local submap cloud levelled on gravity (by the graph's
  corrected attitude, heading kept), not in the body frame, which may be tilted
  or mounted upside down.
- iSAM2 updates incrementally per keyframe (loop search uses the corrected
  map); `finish()` runs Graduated Non-Convexity (GNC) over the loop candidates
  followed by one batch Levenberg-Marquardt optimization of the whole graph.

`keyframes()` returns the optimized keyframes; carrying their
correction to every pose (blend by distance and angle between keyframes)
gives a gravity-aligned, loop-closed trajectory. Tilt and accelerometer bias
separate only when the sensor rotates; a sensor that never turns keeps
some ambiguity at the start.

## Layout

- `cpp/bindings.cpp`: pybind11 module `resbie._core`: `FrontEnd`, `BackEnd`,
  their diagnostics columns and default configs.
- `cpp/resbie/config.h`: the front end's YAML loader, the single source of its
  defaults (see Configuration). The back end's is
  `vendor/bievr/modules/pgo/include/bievr_pgo/config_loader.h`.
- `cpp/resbie/odometry.h`: the synchronous driver.
- `python/resbie/__init__.py`: `FrontEnd` and `BackEnd` (configs from files or
  dicts).
- `vendor/resple`: edited copies of the RESPLE headers, from upstream
  `2f8b09e` (`resple/include`).
- `vendor/bievr`: edited copies of the BIEVR map, preprocess, and the pgo
  module, from upstream `b458505`
  (`BIEVR-LIO-SLAM`, with the loop closer made synchronous).
- `vendor/btc`: Binary Triangle Combined (BTC) descriptor and 6-DoF loop
  detector (`BTC.h`, `BTC.cpp`) with per-query vote deduplication, density
  normalization, and temporal NMS.
- `vendor/small_gicp`: unmodified GICP headers from upstream `57c1106` (`v1.0.1`).

Every edit in `vendor/` is marked `resbie:` at the site. Grep for it to
enumerate the delta:

- **RESPLE:** ROS and the ikd-tree removed; `Association` reads BIEVR's map;
  points arrive in the body frame; Huber variance scale; the IMU-only update
  (one update routine for points, IMU or both); the Mahalanobis point gate;
  bias random walk; the gravity direction state; the posterior covariance in
  square-root information form, kept from the last iteration; bias, gravity
  and covariance getters for the diagnostics.
- **BIEVR:** the samplers return indices so per-point times survive; sort
  tie-breaks; a bounds check on the bump-image write; smoothing and range
  weighting always on; the config loader is reduced to its YAML helpers; the
  LM registration is removed (its bump-image sampler is in `bievr_map.h`);
  log++ is replaced by a one-sink logger (`bievr_lio/log.h`); the loop closer
  is synchronous only, carries the IMU (velocity and bias states,
  CombinedImuFactor, gravity-only prior), detects loops with BTC, and verifies
  them with GICP over local odometry submaps and GNC at `finish()`.

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

Each class owns its configuration, everything that decides the results:
one or more YAML files or dicts, later ones overriding earlier ones per key,
standard deviations in SI units.
Unknown keys are errors (the other class's included). `FrontEnd.DEFAULT_CONFIG`
and `BackEnd.DEFAULT_CONFIG` list every key with its default (taken from the
C++ loaders), except the front end's `calibration`, which has none and must be
set. Values are checked: out of range or of the wrong type is a `ValueError`.

Front end, sections by function:

| section | keys |
|---|---|
| `calibration` | `translation`, `rotation` (T_I_L, LiDAR to IMU, row-major) |
| `lidar` | `min_range_m`, `max_range_m`, `time_offset_s` |
| `imu` | `acc_noise_std`, `gyro_noise_std` (per sample); `acc_bias_init_std`, `gyro_bias_init_std`; `acc_bias_walk`, `gyro_bias_walk` (per sqrt(s)); `gravity_init_std_deg` (prior on gravity's direction about the start's mean force; 0 fixes it) |
| `spline` | `knot_hz` (at most the IMU rate, rounded: each knot interval needs >= 1 IMU sample when the LiDAR is starved; init warns otherwise), `iterations` (filter updates per knot, each re-associating), `init_std`; `pos_noise`, `ort_noise` (motion-model noise of settled control points, m/s and rad/s of the quaternion log, i.e. half the rotation rate); `new_pos_noise`, `new_ort_noise` (newest control point) |
| `points` | `downsample_m`, `informed_voxels`, `noise_std_m`, `huber_delta_m` |
| `map` | `pixel_size_m`, `voxel_size_m`, `normal_tolerance_deg`, `max_size` (voxels; least recently used ones are evicted beyond it), `init_min_voxels` |

Back end, flat keys: `keyframe_meter_gap`,
`keyframe_deg_gap`, `keyframe_filter_size`; loop search (`sc_max_radius`,
`sc_num_exclude_recent`); GICP
(`icp_filter_size`, `loop_fitness_score_threshold`,
`loop_max_correction_z`); the graph's noises (`prior_noise_score`,
`odom_noise_*`, `loop_noise_*`), iSAM2 (`isam_relinearize_*`) and IMU
(`acc_noise_std`, `gyro_noise_std`, `acc_bias_walk`, `gyro_bias_walk`,
`acc_bias_init_std`, `gyro_bias_init_std`).

The IMU noise and initial-bias stds take one number (all axes) or three; the
bias walks take one.

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

Usage: create `resbie.FrontEnd` from `FrontEnd.DEFAULT_CONFIG` with your
LiDAR -> IMU `calibration`, push IMU (`push_imu`) and LiDAR (`push_lidar`:
stamp of the earliest point, points, times relative to it) in time order,
`drain()` the LIO results after each push, call `finish()` and drain
once more. Hand each one to a `resbie.BackEnd` (and each IMU sample) for
loop closure. Every call is synchronous.

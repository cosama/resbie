"""Synthetic lifecycle and determinism tests for resbie."""

from __future__ import annotations

import numpy as np
import pytest

import resbie

IMU_DT = 0.005
SWEEP_IMU = 20  # 10 Hz sweeps at 200 Hz IMU
HALF = np.array([5.0, 5.0, 2.0])  # box room half extents


def _room(rng: np.random.Generator, n: int) -> np.ndarray:
    pts = rng.uniform(-1.0, 1.0, (n, 3)) * HALF
    face = rng.integers(0, 6, n)
    axis = face // 2
    pts[np.arange(n), axis] = HALF[axis] * np.where(face % 2, 1.0, -1.0)
    return pts


def _config(tmp_path, extra: str = "") -> str:
    path = tmp_path / f"resbie_{abs(hash(extra))}.yaml"
    path.write_text("calibration: {translation: [0, 0, 0], rotation: [1, 0, 0, 0, 1, 0, 0, 0, 1]}\n" + extra)
    return str(path)


class _Run:
    """A FrontEnd, and a BackEnd if ``back`` is a config, wired as a caller does."""

    def __init__(self, config: str, back=None, threads: int = 0):
        self.front = resbie.FrontEnd(config, max_num_threads=threads, diagnostics=True)
        self.back = None if back is None else resbie.BackEnd(back)
        self.results = []
        self.diag = [self.front.drain_diagnostics()]

    def push_imu(self, stamp, acc, gyro) -> bool:
        accepted = self.front.push_imu(stamp, acc, gyro)
        if self.back is not None:
            assert self.back.push_imu(stamp, acc, gyro) == accepted
        self._drain()
        return accepted

    def push_lidar(self, stamp, points, relative_times) -> bool:
        accepted = self.front.push_lidar(stamp, points, relative_times)
        self._drain()
        return accepted

    def finish(self) -> None:
        self.front.finish()
        self._drain()
        if self.back is not None:
            self.back.finish()

    def _drain(self) -> None:
        for stamp, pose, points in self.front.drain():
            self.results.append((stamp, pose, points))
            if self.back is not None:
                self.back.push_lio(stamp, pose, points)
        self.diag.append(self.front.drain_diagnostics())

    def trajectory(self) -> np.ndarray:
        """(N, 8): t, x, y, z, qx, qy, qz, qw, one row per sweep."""
        return np.array([[t, *pose] for t, pose, _ in self.results]).reshape(-1, 8)

    def diagnostics(self) -> tuple[np.ndarray, dict[str, int]]:
        """Rows and {column name: index}."""
        return np.vstack(self.diag), _columns(resbie.FrontEnd)


def _columns(cls) -> dict[str, int]:
    return {c: i for i, c in enumerate(cls.DIAGNOSTIC_COLUMNS)}


def _run(config: str, seconds: float = 6.0, velocity=(0.0, 0.0, 0.0), back=None, threads: int = 0) -> _Run:
    """Sensor moving at constant velocity through a static room."""
    rng = np.random.default_rng(0)
    v = np.asarray(velocity)
    s = _Run(config, back, threads)
    t0 = 1.0
    n_imu = int(seconds / IMU_DT)
    for k in range(n_imu):
        assert s.push_imu(t0 + k * IMU_DT, [0.0, 0.0, 9.81], [0.0, 0.0, 0.0])
        if k % SWEEP_IMU == SWEEP_IMU - 1:
            start = t0 + (k - SWEEP_IMU + 1) * IMU_DT
            world = _room(rng, 3000)
            rel = np.sort(rng.uniform(0.0, 0.0999, len(world)))
            body = world - v * (start + rel - t0)[:, None]
            assert s.push_lidar(start, body, rel)
    s.finish()
    return s


def test_static_sensor_stays_put(tmp_path):
    s = _run(_config(tmp_path))
    status = s.front.status()
    traj = s.trajectory()
    assert status["initialized"]
    assert status["sweeps_finalized"] == len(traj) > 50
    assert status["points_effective"] > 0.6 * status["points_measured"]
    assert np.all(np.diff(traj[:, 0]) > 0)
    # 5 cm: at the default 0.25 m point filter the small synthetic room keeps
    # only a few hundred points per sweep.
    assert np.abs(traj[:, 1:4]).max() < 0.05
    assert s.front.drain() == []


def _static_room(config: str, point_stride: int) -> list:
    """LIO results of a static sensor in the room, 2990 range-filtered points each."""
    front = resbie.FrontEnd(config, point_stride=point_stride)
    rng = np.random.default_rng(1)
    for k in range(int(3.0 / IMU_DT)):
        assert front.push_imu(1.0 + k * IMU_DT, [0.0, 0.0, 9.81], [0.0, 0.0, 0.0])
        if k % SWEEP_IMU == SWEEP_IMU - 1:
            room = _room(rng, 3000)
            xyzi = np.column_stack((room, rng.uniform(0, 255, len(room))))
            xyzi[:10, :3] = 0.0  # inside min range: dropped
            rel = np.sort(rng.uniform(0.0, 0.0999, len(room)))
            assert front.push_lidar(1.0 + (k - SWEEP_IMU + 1) * IMU_DT, xyzi, rel)
    front.finish()
    return front.drain()


def test_lio_results_carry_every_point_deskewed(tmp_path):
    """Each LIO result holds every range-filtered point, with its intensity, in the
    IMU frame (for a static sensor: the room itself)."""
    results = _static_room(_config(tmp_path), 1)
    assert len(results) > 20
    for stamp, pose, points in results:
        assert isinstance(stamp, float)
        assert pose.shape == (7,) and abs(np.linalg.norm(pose[3:]) - 1.0) < 1e-12 and pose[6] >= 0
        assert points.dtype == np.float64 and points.shape == (2990, 4)
        assert np.all((points[:, 3] >= 0) & (points[:, 3] <= 255))
        # On a wall, to within the pose error (a wrong frame would be metres off).
        assert np.abs(np.abs(points[:, :3]) - HALF).min(axis=1).max() < 0.15


def test_point_stride_thins_the_points_only(tmp_path):
    config = _config(tmp_path)
    full = _static_room(config, 1)
    for stride, n in ((3, 997), (0, 0)):
        thin = _static_room(config, stride)
        assert len(thin) == len(full)
        for (s0, p0, x0), (s1, p1, x1) in zip(full, thin):
            assert s0 == s1
            np.testing.assert_array_equal(p0, p1)  # the estimate does not change
            np.testing.assert_array_equal(x1, x0[::stride] if stride else x0[:0])
            assert x1.shape == (n, 4)


def test_constant_velocity_is_tracked(tmp_path):
    v = np.array([0.4, 0.1, 0.0])
    s = _run(_config(tmp_path), velocity=v)
    traj = s.trajectory()
    expected = v * (traj[:, 0] - 1.0)[:, None]
    # Starts at rest per the filter's prior; judge the second half.
    half = len(traj) // 2
    err = np.linalg.norm(traj[half:, 1:4] - expected[half:], axis=1)
    assert err.max() < 0.1, err.max()


@pytest.mark.parametrize("threads", [1, 3])
def test_deterministic_across_runs_and_threads(tmp_path, threads):
    reference = _run(_config(tmp_path), seconds=3.0, velocity=(0.3, 0.0, 0.0)).trajectory()
    again = _run(_config(tmp_path), seconds=3.0, velocity=(0.3, 0.0, 0.0), threads=threads).trajectory()
    np.testing.assert_array_equal(reference, again)


def test_deterministic_across_p_and_e_cores(tmp_path):
    """Fresh processes pinned to a P-core (CPU 0) vs an E-core produce bit-identical trajectories."""
    import os
    import subprocess
    import sys

    if not hasattr(os, "sched_setaffinity") or (os.cpu_count() or 1) < 2:
        pytest.skip("CPU affinity pinning requires >= 2 cores and sched_setaffinity")

    cfg = _config(tmp_path)
    e_cpu = min(12, (os.cpu_count() or 2) - 1)
    test_dir = str(os.path.dirname(os.path.abspath(__file__)))

    def _run_on_cpu(cpu: int) -> np.ndarray:
        out_npy = tmp_path / f"traj_cpu{cpu}.npy"
        code = (
            "import os, sys, numpy as np\n"
            f"os.sched_setaffinity(0, {{{cpu}}})\n"
            f"sys.path.insert(0, {test_dir!r})\n"
            "import test_resbie\n"
            f"t = test_resbie._run({cfg!r}, seconds=2.0, velocity=(0.3, 0.0, 0.0)).trajectory()\n"
            f"np.save({str(out_npy)!r}, t)\n"
        )
        subprocess.run([sys.executable, "-c", code], check=True)
        return np.load(out_npy)

    np.testing.assert_array_equal(_run_on_cpu(0), _run_on_cpu(e_cpu))


def test_rejects_out_of_order_input(tmp_path):
    s = resbie.FrontEnd(_config(tmp_path))
    assert s.push_imu(2.0, [0, 0, 9.81], [0, 0, 0])
    assert not s.push_imu(2.0, [0, 0, 9.81], [0, 0, 0])
    pts = np.ones((10, 3))
    assert s.push_lidar(2.0, pts, np.zeros(10))
    assert not s.push_lidar(1.5, pts, np.zeros(10))
    # all inside min range: kept as an empty sweep (a gap marker for the filter)
    assert s.push_lidar(3.0, np.zeros((10, 3)), np.zeros(10))
    # same start, but not ending after the previous sweep: its stamp would not increase
    assert not s.push_lidar(3.0, pts, np.zeros(10))
    status = s.status()
    assert (status["imu_rejected"], status["lidar_rejected"], status["lidar_empty"]) == (1, 2, 1)


def test_bad_config_raises(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("calibration: {translation: [0, 0], rotation: [1, 0, 0, 0, 1, 0, 0, 0, 1]}\n")
    with pytest.raises(ValueError):
        resbie.FrontEnd(str(path))


def test_blackout_is_bridged_by_imu(tmp_path):
    """1.5 s of empty sweeps mid-run: the spline steps through on IMU alone."""
    rng = np.random.default_rng(0)
    v = np.array([0.4, 0.0, 0.0])
    s = _Run(_config(tmp_path))
    t0 = 1.0
    for k in range(int(8.0 / IMU_DT)):
        assert s.push_imu(t0 + k * IMU_DT, [0.0, 0.0, 9.81], [0.0, 0.0, 0.0])
        if k % SWEEP_IMU == SWEEP_IMU - 1:
            start = t0 + (k - SWEEP_IMU + 1) * IMU_DT
            rel = np.sort(rng.uniform(0.0, 0.0999, 3000))
            body = _room(rng, 3000) - v * (start + rel - t0)[:, None]
            if 4.0 <= start - t0 < 5.5:
                body = np.zeros((3000, 3))  # all inside min range: empty sweep
            assert s.push_lidar(start, body, rel)
    s.finish()
    status = s.front.status()
    assert status["gap_knots"] > 100 and status["imu_only_updates"] > 100
    traj = s.trajectory()
    err = traj[:, 1:4] - v * (traj[:, 0] - 1.0)[:, None]
    # What the blackout adds: the error after it vs right before it (the
    # absolute error also holds ordinary odometry drift on noise-free data).
    before = err[(traj[:, 0] > t0 + 3.5) & (traj[:, 0] < t0 + 4.0)].mean(0)
    after = err[traj[:, 0] > t0 + 6.0]
    assert np.linalg.norm(after - before, axis=1).max() < 0.05
    diag, col = s.diagnostics()
    assert diag.shape == (len(traj), len(resbie.FrontEnd.DIAGNOSTIC_COLUMNS))
    assert s.front.drain_diagnostics().shape == (0, len(col))
    np.testing.assert_array_equal(diag[:, col["stamp"]], traj[:, 0])
    blackout = (diag[:, 0] > t0 + 4.2) & (diag[:, 0] < t0 + 5.4)
    # no matched point, IMU-only updates
    assert np.all(diag[blackout, col["points_effective"]] == 0)
    assert np.all(diag[blackout, col["imu_only_updates"]] > 0)


@pytest.mark.parametrize(
    "extra",
    [
        "spline: {knot_rate: 50}\n",
        "points: {downsample: 0.3}\n",
        "map: {smooth: false}\n",  # removed: always on
        "loop_closure: {enable: false}\n",  # the back end's, not the front end's
        "max_num_threads: 2\n",  # how it runs, a constructor argument
    ],
)
def test_front_end_rejects_unknown_keys(tmp_path, extra):
    # Misspelled, removed or misplaced keys must not silently fall back to defaults.
    with pytest.raises(ValueError):
        resbie.FrontEnd(_config(tmp_path, extra=extra))


@pytest.mark.parametrize(
    "config",
    [{"enable": False}, {"loop_closure": {"keyframe_meter_gap": 2.0}}, {"knot_hz": 100},
     {"max_speed_mps": 30.0}, {"use_sc_yaw_guess": False}, {"icp_max_correspondence_distance": 5.0}],
)
def test_back_end_rejects_unknown_keys(config):
    with pytest.raises(ValueError):
        resbie.BackEnd(config)


def test_default_configs_load_as_is():
    # calibration has no default: it must be set.
    calibration = {"calibration": {"translation": [0, 0, 0], "rotation": [1, 0, 0, 0, 1, 0, 0, 0, 1]}}
    with pytest.raises(ValueError):
        resbie.FrontEnd(resbie.FrontEnd.DEFAULT_CONFIG)
    resbie.FrontEnd([resbie.FrontEnd.DEFAULT_CONFIG, calibration])
    resbie.BackEnd(resbie.BackEnd.DEFAULT_CONFIG)
    assert set(resbie.FrontEnd.DEFAULT_CONFIG) == {"lidar", "imu", "spline", "points", "map"}
    assert "keyframe_meter_gap" in resbie.BackEnd.DEFAULT_CONFIG


def test_configs_merge_files_and_dicts(tmp_path):
    # A later document wins per key; an invalid value in it is still caught.
    base = _config(tmp_path)
    resbie.FrontEnd([base, {"points": {"noise_std_m": 0.05}}])
    with pytest.raises(ValueError):
        resbie.FrontEnd([base, {"points": {"noise_std_m": -1.0}}])
    resbie.BackEnd([{"keyframe_meter_gap": 2.0}, {"keyframe_meter_gap": 3.0}])


@pytest.mark.parametrize(
    "lidar", [{"min_range_m": -1.0}, {"min_range_m": 5.0, "max_range_m": 5.0}, {"max_range_m": 0.1}]
)
def test_front_end_rejects_bad_ranges(tmp_path, lidar):
    with pytest.raises(ValueError):
        resbie.FrontEnd([_config(tmp_path), {"lidar": lidar}])


@pytest.mark.parametrize(
    "config",
    [{"acc_noise_std": -1.0}, {"sc_num_rings": 0}, {"icp_filter_size": -0.1},
     {"keyframe_filter_size": 0.0}, {"loop_noise_cauchy_c": 0.0}, {"gyro_noise_std": [0.1, 0.1]}],
)
def test_back_end_rejects_bad_values(config):
    with pytest.raises(ValueError):
        resbie.BackEnd(config)


@pytest.mark.parametrize(
    "extra",
    [{"points": {"informed_voxels": -1}}, {"map": {"max_size": 0}}, {"spline": {"knot_hz": 1.5}},
     {"imu": {"acc_noise_std": [1.0, 1.0]}}, {"imu": {"gyro_bias_walk": -0.1}},
     {"imu": {"gyro_noise_std": 0.0}}],
)
def test_front_end_rejects_bad_values(tmp_path, extra):
    with pytest.raises(ValueError):
        resbie.FrontEnd([_config(tmp_path), extra])


def test_imu_stds_take_one_number_or_three(tmp_path):
    for std in (0.5, [0.5, 0.6, 0.7]):
        resbie.FrontEnd([_config(tmp_path), {"imu": {"acc_noise_std": std}}])
        resbie.BackEnd({"acc_noise_std": std})


def test_wrong_types_are_value_errors(tmp_path):
    with pytest.raises(ValueError):
        resbie.FrontEnd([_config(tmp_path), {"spline": {"knot_hz": "fast"}}])
    with pytest.raises(ValueError):
        resbie.BackEnd({"sc_num_rings": "many"})


def test_log_level_rejects_unknown_names():
    with pytest.raises(ValueError):
        resbie.set_log_level("verbose")


@pytest.mark.parametrize("level, lines", [("debug", True), ("info", False)])
def test_debug_logs_one_line_per_sweep(tmp_path, capfd, level, lines):
    resbie.set_log_level(level)
    try:
        s = _run(_config(tmp_path), seconds=3.0)
    finally:
        resbie.set_log_level("info")
    out, err = capfd.readouterr()
    n = sum(line.startswith("DEBUG sweep ") for line in out.splitlines())
    assert n == (len(s.trajectory()) if lines else 0)


def test_diagnostics_velocity_follows_the_motion(tmp_path):
    s = _run(_config(tmp_path), seconds=4.0, velocity=(0.5, 0.0, 0.0))
    diag, col = s.diagnostics()
    late = diag[diag[:, 0] > diag[0, 0] + 1.0]
    vel = late[:, [col["vel_x"], col["vel_y"], col["vel_z"]]]
    assert np.abs(vel - [0.5, 0.0, 0.0]).max() < 0.05, vel
    # Consistent data: the final trajectory agrees with the IMU sweep by sweep
    # to within a few accelerometer-noise sigmas (a wrong gravity or sign
    # would be ~40 sigma).
    dv = late[:, [col["imu_post_dv_x"], col["imu_post_dv_y"], col["imu_post_dv_z"]]]
    z = dv / late[:, [col["imu_post_sigma"]]]
    assert np.nanmean(z**2) < 10.0, z
    assert np.all(late[:, col["imu_samples"]] > 0)


def _run_accelerating_start(config: str, accel: float = 0.5, accel_s: float = 1.0,
                            seconds: float = 6.0) -> _Run:
    """Level sensor accelerating along x from rest for accel_s, then coasting."""
    rng = np.random.default_rng(0)
    s = _Run(config)
    t0 = 1.0

    def x(t):
        t = np.asarray(t) - t0
        tc = np.minimum(t, accel_s)
        return 0.5 * accel * tc**2 + accel * accel_s * (t - tc)

    for k in range(int(seconds / IMU_DT)):
        t = t0 + k * IMU_DT
        a = accel if t - t0 < accel_s else 0.0
        assert s.push_imu(t, [a, 0.0, 9.81], [0.0, 0.0, 0.0])
        if k % SWEEP_IMU == SWEEP_IMU - 1:
            start = t0 + (k - SWEEP_IMU + 1) * IMU_DT
            world = _room(rng, 3000)
            rel = np.sort(rng.uniform(0.0, 0.0999, len(world)))
            body = world.copy()
            body[:, 0] -= x(start + rel)
            assert s.push_lidar(start, body, rel)
    s.finish()
    return s


def test_moving_start_learns_gravity(tmp_path):
    # The start is levelled on the mean specific force, here 2.9 deg off true
    # up because the sensor accelerates. The map frame inherits that tilt
    # (a gauge); gravity's direction in it must be learnt, so that once the
    # sensor coasts the IMU agrees with the trajectory. With gravity fixed at
    # the start, an accelerometer bias of ~0.5 m/s^2 would have to absorb it.
    accel = 0.5
    f0 = np.array([accel, 0.0, 9.81])
    f0 /= np.linalg.norm(f0)
    true_tilt = np.degrees(np.arccos(f0[2]))
    diag, col = _run_accelerating_start(_config(tmp_path), accel).diagnostics()
    up = diag[-1, [col["up_x"], col["up_y"], col["up_z"]]]
    assert abs(np.linalg.norm(up) - 1.0) < 1e-9
    tilt = np.degrees(np.arccos(np.clip(up[2], -1.0, 1.0)))
    assert abs(tilt - true_tilt) < 0.5, (tilt, true_tilt)
    assert up[0] < 0.0, up  # levelling on f0 tilted the frame toward +x
    # Without a heading change tilt and horizontal accelerometer bias are one
    # combination: the tilt's std stays at the bias prior over g (2.6 deg).
    sigma = np.degrees(diag[-1, col["up_sigma"]])
    assert 2.0 < sigma < 3.0, sigma
    ba = diag[-1, [col["acc_bias_x"], col["acc_bias_y"], col["acc_bias_z"]]]
    assert np.abs(ba).max() < 0.1, ba

    pinned, _ = _run_accelerating_start(
        _config(tmp_path, extra="imu: {gravity_init_std_deg: 0}\n"), accel).diagnostics()
    assert np.allclose(pinned[:, [col["up_x"], col["up_y"], col["up_z"]]], [0.0, 0.0, 1.0], atol=1e-9)


def test_turning_start_separates_gravity_from_bias(tmp_path):
    # Moving start (accelerating) with a real horizontal accelerometer bias,
    # turning at 1 rad/s: the bias rotates with the sensor, gravity does not,
    # so the turn tells them apart (gradually: the tilt's std falls from the
    # 2.6 deg bias floor as the heading sweeps). Both must be recovered, from
    # the start, with no static period.
    accel, accel_s, omega, seconds = 0.5, 0.6, 1.0, 8.0
    bias = np.array([0.2, -0.1, 0.0])
    rng = np.random.default_rng(0)
    s = _Run(_config(tmp_path))
    t0 = 1.0

    def pose(t):
        t = np.asarray(t, dtype=float) - t0
        tc = np.minimum(t, accel_s)
        x = 0.5 * accel * tc**2 + accel * accel_s * (t - tc)
        return x, omega * t

    for k in range(int(seconds / IMU_DT)):
        t = t0 + k * IMU_DT
        _, yaw = pose(t)
        a_w = np.array([accel if t - t0 < accel_s else 0.0, 0.0, 9.81])
        c, sn = np.cos(yaw), np.sin(yaw)
        f = np.array([c * a_w[0] + sn * a_w[1], -sn * a_w[0] + c * a_w[1], a_w[2]]) + bias
        assert s.push_imu(t, f, [0.0, 0.0, omega])
        if k % SWEEP_IMU == SWEEP_IMU - 1:
            start = t0 + (k - SWEEP_IMU + 1) * IMU_DT
            world = _room(rng, 3000)
            rel = np.sort(rng.uniform(0.0, 0.0999, len(world)))
            x, yaw = pose(start + rel)
            d = world - np.stack([x, np.zeros_like(x), np.zeros_like(x)], axis=1)
            c, sn = np.cos(yaw), np.sin(yaw)
            body = np.stack([c * d[:, 0] + sn * d[:, 1], -sn * d[:, 0] + c * d[:, 1], d[:, 2]], axis=1)
            assert s.push_lidar(start, body, rel)
    s.finish()

    # The sensor is level throughout, so in the map frame its true up is its
    # final attitude applied to body up, whatever tilt the map frame has
    # (start levelling, a smeared seed sweep). Gravity must agree with that.
    qx, qy, qz, qw = s.trajectory()[-1, 4:8]
    up_true = np.array([2 * (qx * qz + qw * qy), 2 * (qy * qz - qw * qx), 1 - 2 * (qx**2 + qy**2)])
    diag, col = s.diagnostics()
    up = diag[-1, [col["up_x"], col["up_y"], col["up_z"]]]
    err = np.degrees(np.arccos(np.clip(up @ up_true, -1.0, 1.0)))
    assert err < 0.3, (err, up, up_true)
    assert np.degrees(diag[-1, col["up_sigma"]]) < 1.0
    ba = diag[-1, [col["acc_bias_x"], col["acc_bias_y"], col["acc_bias_z"]]]
    assert np.abs(ba - bias).max() < 0.05, ba


def _tilt_deg(q):
    qx, qy = q[..., 0], q[..., 1]
    return np.degrees(2 * np.arcsin(np.clip(np.hypot(qx, qy), 0, 1)))


def test_graph_levels_the_odometry(tmp_path):
    # The back end's pose graph adds IMU factors between keyframes, with
    # gravity along world z. Here the odometry drifts in tilt by ~1 degree; the
    # graph keyframes must come out level, more level than the odometry, and
    # horizontally where the odometry is. A gravity sign or frame error would
    # break all three. The first keyframe is excused: without any rotation in
    # this test, tilt and accelerometer bias cannot be told apart there yet.
    s = _run(_config(tmp_path), seconds=8.0, velocity=(0.5, 0.0, 0.0), back={})
    kf, odom = s.back.keyframes(), s.trajectory()
    assert len(kf) >= 3
    odom_at_kf = odom[np.searchsorted(odom[:, 0], kf[:, 0])]
    np.testing.assert_array_equal(kf[:, 0], odom_at_kf[:, 0])  # the pushed stamps, exactly
    assert np.abs(kf[:, 1:3] - odom_at_kf[:, 1:3]).max() < 0.02
    tilt_kf, tilt_odom = _tilt_deg(kf[1:, 4:6]), _tilt_deg(odom_at_kf[1:, 4:6])
    assert tilt_kf.max() < 0.5, tilt_kf
    assert tilt_kf.mean() < tilt_odom.mean(), (tilt_kf, tilt_odom)


def test_map_max_size_lru_eviction(tmp_path):
    extra = "map: {max_size: 40, init_min_voxels: 20}\n"
    s1 = _run(_config(tmp_path, extra=extra), seconds=4.0, velocity=(0.5, 0.2, 0.0))
    st = s1.front.status()
    assert st["initialized"]
    assert st["map_voxels"] == 40
    s2 = _run(_config(tmp_path, extra=extra), seconds=4.0, velocity=(0.5, 0.2, 0.0), threads=3)
    np.testing.assert_array_equal(s1.trajectory(), s2.trajectory())
    np.testing.assert_array_equal(s1.diagnostics()[0], s2.diagnostics()[0])


def test_single_voxel_sweep_initializes(tmp_path):
    # A sweep whose points all land in one voxel still reaches the map (the voxel grouping once
    # started from the last hash and skipped the only voxel).
    extra = "map: {voxel_size_m: 100.0, init_min_voxels: 1}\n"
    s = resbie.FrontEnd(_config(tmp_path, extra=extra))
    rng = np.random.default_rng(42)
    pts = rng.uniform(1.5, 4.5, (200, 3))
    rel = np.sort(rng.uniform(0.0, 0.09, len(pts)))
    for k in range(40):
        assert s.push_imu(1.0 + k * IMU_DT, [0.0, 0.0, 9.81], [0.0, 0.0, 0.0])
    assert s.push_lidar(1.0, pts, rel)
    assert s.push_lidar(1.1, pts, rel)
    st = s.status()
    assert st["initialized"]
    assert st["map_voxels"] == 1


def test_imu_spike_does_not_throw_the_trajectory(tmp_path):
    # One 50 m/s^2 / 20 rad/s sample, measured like any other (there is no IMU gate).
    rng = np.random.default_rng(42)
    s_spike = _Run(_config(tmp_path))
    t0 = 1.0
    for k in range(int(3.0 / IMU_DT)):
        acc = [50.0, 0.0, 9.81] if k == 250 else [0.0, 0.0, 9.81]
        gyr = [0.0, 20.0, 0.0] if k == 250 else [0.0, 0.0, 0.0]
        assert s_spike.push_imu(t0 + k * IMU_DT, acc, gyr)
        if k % SWEEP_IMU == SWEEP_IMU - 1:
            start = t0 + (k - SWEEP_IMU + 1) * IMU_DT
            world = _room(rng, 800)
            rel_t = np.sort(rng.uniform(0.0, 0.0999, len(world)))
            assert s_spike.push_lidar(start, world, rel_t)
    s_spike.finish()
    traj = s_spike.trajectory()
    assert len(traj) > 10 and np.all(np.isfinite(traj))
    assert np.abs(traj[:, 1:4]).max() < 0.05


def test_one_update_per_knot_and_imu_counted_once(tmp_path):
    assert "per_update" not in resbie.FrontEnd.DEFAULT_CONFIG["points"]
    s = _run(_config(tmp_path), seconds=2.0)
    diag, col = s.diagnostics()
    # Steady-state sweeps (0.1 s at 100 Hz knots, 200 Hz IMU): 10 updates and 20 IMU samples each.
    np.testing.assert_array_equal(diag[2:, col["updates"]], 10)
    np.testing.assert_array_equal(diag[2:, col["imu_samples"]], SWEEP_IMU)


def test_back_end_noise_defaults():
    back = resbie.BackEnd.DEFAULT_CONFIG
    expected = {
        "loop_max_correction_z": 1.5,
        "acc_noise_std": [12.0, 12.0, 12.0],
        "gyro_noise_std": [0.7, 0.7, 0.7],
        "acc_bias_init_std": [0.2, 0.2, 0.2],
        "gyro_bias_init_std": [0.1, 0.1, 0.1],
        "acc_bias_walk": 1.0e-3,
        "gyro_bias_walk": 1.0e-4,
        "odom_noise_rotation": 2.5e-5,
        "odom_noise_translation": 0.00025,
        "loop_noise_score": 0.2,
        "loop_noise_rotation": 0.000402961,
        "loop_noise_translation": 0.0140625,
    }
    for key, val in expected.items():
        assert back[key] == val, key


def test_back_end_one_imu_sample_between_keyframes():
    # One sample gives no IMU rate (the noise densities need one): the keyframes are linked by
    # the bias walk alone instead of a preintegration without parameters (a crash).
    back = resbie.BackEnd({})
    room = _room(np.random.default_rng(0), 500)
    assert back.push_imu(1.0, [0.0, 0.0, 9.81], [0.0, 0.0, 0.0])
    assert back.push_lio(1.05, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], room)
    assert back.push_imu(1.5, [0.0, 0.0, 9.81], [0.0, 0.0, 0.0])
    assert back.push_lio(2.0, [2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], room)
    back.finish()
    np.testing.assert_array_equal(back.keyframes()[:, 0], [1.05, 2.0])


def test_back_end_order_and_diagnostics():
    back = resbie.BackEnd({}, diagnostics=True)
    room = _room(np.random.default_rng(0), 500)
    pose = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    assert back.push_imu(1.0, [0, 0, 9.81], [0, 0, 0])
    assert not back.push_imu(1.0, [0, 0, 9.81], [0, 0, 0])
    assert back.push_lio(1.0, pose, room)
    assert back.push_lio(1.1, pose + [2.0, 0, 0, 0, 0, 0, 0], np.zeros((0, 4)))  # empty
    assert back.push_lio(1.2, pose + [0.5, 0, 0, 0, 0, 0, 0], room)  # < 1 m from the last keyframe
    assert back.push_lio(1.3, pose + [2.5, 0, 0, 0, 0, 0, 0], room)
    assert not back.push_lio(1.3, pose, room)  # stamp did not increase
    for stamp, bad_pose in ((1.4, pose * 2.0), (0.0, pose)):
        with pytest.raises(ValueError):  # not a unit quaternion, not positive
            back.push_lio(stamp, bad_pose, room)
    back.finish()
    status = back.status()
    assert (status["imu_accepted"], status["imu_rejected"]) == (1, 1)
    assert (status["lio_accepted"], status["lio_rejected"], status["keyframes"]) == (4, 1, 2)
    rows, col = back.drain_diagnostics(), _columns(resbie.BackEnd)
    assert back.drain_diagnostics().shape == (0, len(col))
    np.testing.assert_array_equal(rows[:, col["stamp"]], [1.0, 1.1, 1.2, 1.3])
    np.testing.assert_array_equal(rows[:, col["keyframe"]], [1, 0, 0, 1])
    np.testing.assert_array_equal(rows[:, col["points"]], [500, 0, 500, 500])
    np.testing.assert_array_equal(back.keyframes()[:, 0], [1.0, 1.3])


def test_diagnostics_are_a_switch(tmp_path):
    front = resbie.FrontEnd(_config(tmp_path))
    back = resbie.BackEnd({})
    assert front.drain_diagnostics().shape == (0, len(resbie.FrontEnd.DIAGNOSTIC_COLUMNS))
    back.push_lio(1.0, [0, 0, 0, 0, 0, 0, 1], np.ones((10, 3)))
    assert back.drain_diagnostics().shape == (0, len(resbie.BackEnd.DIAGNOSTIC_COLUMNS))


def _btc_room(rng: np.random.Generator) -> np.ndarray:
    """Synthetic structured room with floor, walls, and asymmetric vertical pillars for BTC."""
    parts = []
    # Dense floor at z = -2.0 on a 0.12 m grid so every 1 m^2 voxel has ~70 points (> voxel_init_num=10)
    gx, gy = np.meshgrid(np.arange(-12.0, 12.0, 0.14), np.arange(-12.0, 12.0, 0.14))
    parts.append(np.column_stack([gx.ravel(), gy.ravel(), np.full(gx.size, -2.0)]))
    # 4 outer walls for full 6-DoF GICP observability (low height so they don't trigger binary corners)
    for sign in (-1.0, 1.0):
        wy, wz = np.meshgrid(np.arange(-12.0, 12.0, 0.25), np.arange(-2.0, -1.5, 0.12))
        parts.append(np.column_stack([np.full(wy.size, sign * 12.0), wy.ravel(), wz.ravel()]))
        wx, wz = np.meshgrid(np.arange(-12.0, 12.0, 0.25), np.arange(-2.0, -1.5, 0.12))
        parts.append(np.column_stack([wx.ravel(), np.full(wx.size, sign * 12.0), wz.ravel()]))
    # 18 asymmetric vertical pillars spaced > 3.2 m apart with distinct vertical bin counts
    pillar_xy = [
        (-9.2, -8.8), (-4.6, -9.1), (0.3, -8.7), (5.1, -9.3), (9.4, -8.5),
        (-8.9, -3.8), (-3.7, -4.2), (1.4, -3.6), (6.2, -4.1),
        (-9.1, 1.3), (-4.1, 1.7), (1.1, 1.2), (5.8, 1.6), (9.3, 1.1),
        (-8.7, 6.4), (-3.5, 6.8), (1.8, 6.3), (6.9, 6.7),
    ]
    for idx, (px, py) in enumerate(pillar_xy):
        n_bins = 12 + idx * 2  # 12..46 bins of 0.10 m, each strictly unique
        zs = -1.95 + 0.10 * np.arange(n_bins)
        p = np.column_stack([
            np.full(n_bins, px),
            np.full(n_bins, py),
            zs,
        ])
        parts.append(p)
    return np.vstack(parts)


def test_back_end_loop_rejection_counters_and_conservation():
    reason_keys = (
        "rejected_submap",
        "rejected_icp",
        "rejected_overlap",
        "rejected_fitness",
        "rejected_hessian",
        "rejected_z",
        "rejected_gnc",
    )
    rng = np.random.default_rng(7)
    world_room = _btc_room(rng)
    # 7 keyframes (> kSubmapKfRadius=5) spaced 25 m apart (> 20 m submap radius)
    # so kf=0 and kf=6 have completely disjoint local submaps.
    positions = [
        np.array([0.0, 0.0, 0.0]),
        np.array([25.0, 0.0, 0.0]),
        np.array([50.0, 0.0, 0.0]),
        np.array([75.0, 0.0, 0.0]),
        np.array([50.0, 25.0, 0.0]),
        np.array([25.0, 25.0, 0.0]),
        np.array([0.0, 0.0, 0.0]),
    ]

    # 1. Loop-closing synthetic run
    back_acc = resbie.BackEnd({"sc_num_exclude_recent": 5})
    t = 1.0
    for pos in positions:
        pose = np.array([pos[0], pos[1], pos[2], 0.0, 0.0, 0.0, 1.0])
        back_acc.push_lio(t, pose, world_room)
        t += 1.0
    back_acc.finish()
    back_acc.finish()  # idempotent counter conservation check
    st_acc = back_acc.status()
    for k in ("candidates", "loops", "rejected", *reason_keys):
        assert k in st_acc, k
    assert st_acc["candidates"] == 1, st_acc
    assert st_acc["loops"] == 1, st_acc
    assert st_acc["rejected"] == 0, st_acc
    assert st_acc["candidates"] == st_acc["loops"] + st_acc["rejected"]
    assert st_acc["rejected"] == sum(st_acc[k] for k in reason_keys)

    # 2. Rejected-loop synthetic run (0.6 m vertical drift rejected by 0.3 m z gate)
    back_rej = resbie.BackEnd({"sc_num_exclude_recent": 5, "loop_max_correction_z": 0.3})
    t = 1.0
    for idx, pos in enumerate(positions):
        z_offset = 0.6 if idx == len(positions) - 1 else 0.0
        pose = np.array([pos[0], pos[1], pos[2] + z_offset, 0.0, 0.0, 0.0, 1.0])
        back_rej.push_lio(t, pose, world_room)
        t += 1.0
    back_rej.finish()
    st_rej = back_rej.status()
    for k in ("candidates", "loops", "rejected", *reason_keys):
        assert k in st_rej, k
    assert st_rej["candidates"] == 1, st_rej
    assert st_rej["loops"] == 0, st_rej
    assert st_rej["rejected"] == 1, st_rej
    assert st_rej["rejected_z"] == 1, st_rej
    assert st_rej["candidates"] == st_rej["loops"] + st_rej["rejected"]
    assert st_rej["rejected"] == sum(st_rej[k] for k in reason_keys)


def test_back_end_btc_recovers_5m_and_20deg_yaw_offset():
    """Verify BTC recovers a 6-DoF loop seed when the revisit keyframe has 5 m translation
    drift and 20 deg yaw rotation relative to the initial keyframe."""
    rng = np.random.default_rng(11)
    world_room = _btc_room(rng)

    # True rotation at revisit is +20 deg yaw (so points in body frame are rotated by -20 deg),
    # while odometry pose has drifted to (5.0, 0.0, 0.0) with 0 deg yaw.
    yaw_rad = np.radians(20.0)
    c, s = np.cos(yaw_rad), np.sin(yaw_rad)
    R_body_from_world = np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])
    rotated_room = world_room @ R_body_from_world.T

    positions = [
        (np.array([0.0, 0.0, 0.0]), world_room),
        (np.array([30.0, 0.0, 0.0]), world_room),
        (np.array([60.0, 0.0, 0.0]), world_room),
        (np.array([90.0, 0.0, 0.0]), world_room),
        (np.array([60.0, 30.0, 0.0]), world_room),
        (np.array([30.0, 30.0, 0.0]), world_room),
        (np.array([5.0, 0.0, 0.0]), rotated_room),
    ]

    back = resbie.BackEnd({"sc_num_exclude_recent": 5, "odom_noise_translation": 0.1, "odom_noise_rotation": 0.05})
    t = 1.0
    for pos, scan in positions:
        pose = np.array([pos[0], pos[1], pos[2], 0.0, 0.0, 0.0, 1.0])
        assert back.push_lio(t, pose, scan)
        t += 1.0
    back.finish()
    st = back.status()
    assert st["candidates"] == 1, st
    assert st["loops"] == 1, st
    assert st["rejected"] == 0, st
    kf = back.keyframes()
    # The final keyframe started at x=5.0 m in odometry and must be pulled close to x=0.0 m by the loop factor.
    assert np.linalg.norm(kf[-1, 1:4] - kf[0, 1:4]) < 0.5, (kf[0, 1:4], kf[-1, 1:4])


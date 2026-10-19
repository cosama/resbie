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


def _config(tmp_path, threads: int = 0, extra: str = "") -> str:
    path = tmp_path / f"resbie_{threads}.yaml"
    path.write_text(
        "calibration: {translation: [0, 0, 0], rotation: [1, 0, 0, 0, 1, 0, 0, 0, 1]}\n"
        f"max_num_threads: {threads}\n" + extra
    )
    return str(path)


def _run(config: str, seconds: float = 6.0, velocity=(0.0, 0.0, 0.0)) -> resbie.Resbie:
    """Sensor moving at constant velocity through a static room."""
    rng = np.random.default_rng(0)
    v = np.asarray(velocity)
    s = resbie.Resbie(config, map_point_stride=7)
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
    status = s.status()
    assert status["initialized"]
    assert status["sweeps_finalized"] == status["poses"] > 50
    assert status["points_effective"] > 0.6 * status["points_measured"]
    traj = s.trajectory()
    assert np.all(np.diff(traj[:, 0]) > 0)
    # 5 cm: at the default 0.2 m point filter the small synthetic room keeps
    # only a few hundred points per sweep.
    assert np.abs(traj[:, 1:4]).max() < 0.05
    assert len(s.drain_map_scans()) == status["poses"]
    assert s.drain_map_scans() == []


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
    reference = _run(_config(tmp_path, 0), seconds=3.0, velocity=(0.3, 0.0, 0.0)).trajectory()
    again = _run(_config(tmp_path, threads), seconds=3.0, velocity=(0.3, 0.0, 0.0)).trajectory()
    np.testing.assert_array_equal(reference, again)


def test_deterministic_across_p_and_e_cores(tmp_path):
    """Fresh processes pinned to a P-core (CPU 0) vs an E-core produce bit-identical trajectories."""
    import os
    import subprocess
    import sys

    if not hasattr(os, "sched_setaffinity") or (os.cpu_count() or 1) < 2:
        pytest.skip("CPU affinity pinning requires >= 2 cores and sched_setaffinity")

    cfg = _config(tmp_path, 0)
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
    s = resbie.Resbie(_config(tmp_path))
    assert s.push_imu(2.0, [0, 0, 9.81], [0, 0, 0])
    assert not s.push_imu(2.0, [0, 0, 9.81], [0, 0, 0])
    pts = np.ones((10, 3))
    assert s.push_lidar(2.0, pts, np.zeros(10))
    assert not s.push_lidar(1.5, pts, np.zeros(10))
    # all inside min range: kept as an empty sweep (a gap marker for the filter)
    assert s.push_lidar(3.0, np.zeros((10, 3)), np.zeros(10))
    status = s.status()
    assert (status["imu_rejected"], status["lidar_rejected"], status["lidar_empty"]) == (1, 1, 1)


def test_bad_config_raises(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("calibration: {translation: [0, 0], rotation: [1, 0, 0, 0, 1, 0, 0, 0, 1]}\n")
    with pytest.raises(ValueError):
        resbie.Resbie(str(path))


def test_blackout_is_bridged_by_imu(tmp_path):
    """1.5 s of empty sweeps mid-run: the spline steps through on IMU alone."""
    rng = np.random.default_rng(0)
    v = np.array([0.4, 0.0, 0.0])
    s = resbie.Resbie(_config(tmp_path))
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
    status = s.status()
    assert status["gap_knots"] > 100 and status["imu_only_updates"] > 100
    traj = s.trajectory()
    err = traj[:, 1:4] - v * (traj[:, 0] - 1.0)[:, None]
    # What the blackout adds: the error after it vs right before it (the
    # absolute error also holds ordinary odometry drift on noise-free data).
    before = err[(traj[:, 0] > t0 + 3.5) & (traj[:, 0] < t0 + 4.0)].mean(0)
    after = err[traj[:, 0] > t0 + 6.0]
    assert np.linalg.norm(after - before, axis=1).max() < 0.05
    diag = s.sweep_diagnostics()
    assert diag.shape == (len(traj), len(resbie.DIAGNOSTIC_COLUMNS))
    blackout = (diag[:, 0] > t0 + 4.2) & (diag[:, 0] < t0 + 5.4)
    assert np.all(diag[blackout, 3] == 0) and np.all(diag[blackout, 5] > 0)  # no matched point, IMU-only updates


@pytest.mark.parametrize("extra", ["resple: {knot_hz: 50}\n", "points: {downsample: 0.3}\n"])
def test_unknown_keys_raise(tmp_path, extra):
    # Old (pre-schema) or misspelled keys must not silently fall back to defaults.
    with pytest.raises(Exception):
        resbie.Resbie(_config(tmp_path, extra=extra))


def test_default_config_loads_as_is(tmp_path):
    import json

    path = tmp_path / "defaults.yaml"
    path.write_text(json.dumps(resbie.DEFAULT_CONFIG))  # JSON is valid YAML
    resbie.Resbie(str(path))
    assert set(resbie.DEFAULT_CONFIG) == {
        "calibration", "lidar", "imu", "spline", "points", "map", "loop_closure", "max_num_threads",
    }


def test_log_level_rejects_unknown_names():
    with pytest.raises(ValueError):
        resbie.set_log_level("verbose")


@pytest.mark.parametrize("level, lines", [("debug", True), ("info", False)])
def test_debug_logs_one_line_per_sweep(tmp_path, capfd, level, lines):
    resbie.set_log_level(level)
    try:
        s = _run(_config(tmp_path, extra="loop_closure: {enable: false}\n"), seconds=3.0)
    finally:
        resbie.set_log_level("info")
    out, err = capfd.readouterr()
    n = sum(line.startswith("DEBUG sweep ") for line in out.splitlines())
    assert n == (len(s.trajectory()) if lines else 0)


def test_diagnostics_velocity_follows_the_motion(tmp_path):
    s = _run(_config(tmp_path, extra="loop_closure: {enable: false}\n"), seconds=4.0,
             velocity=(0.5, 0.0, 0.0))
    diag = s.sweep_diagnostics()
    col = {c: i for i, c in enumerate(resbie.DIAGNOSTIC_COLUMNS)}
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


def _tilt_deg(q):
    qx, qy = q[..., 0], q[..., 1]
    return np.degrees(2 * np.arcsin(np.clip(np.hypot(qx, qy), 0, 1)))


def test_graph_levels_the_odometry(tmp_path):
    # The pose graph (on by default) adds IMU factors between keyframes, with
    # gravity along world z. Here the odometry drifts in tilt by ~1 degree; the
    # graph keyframes must come out level, more level than the odometry, and
    # horizontally where the odometry is. A gravity sign or frame error would
    # break all three. The first keyframe is excused: without any rotation in
    # this test, tilt and accelerometer bias cannot be told apart there yet.
    s = _run(_config(tmp_path), seconds=8.0, velocity=(0.5, 0.0, 0.0))
    kf, odom = s.keyframe_trajectory(), s.trajectory()
    assert len(kf) >= 3
    odom_at_kf = odom[np.searchsorted(odom[:, 0], kf[:, 0])]
    assert np.allclose(kf[:, 0], odom_at_kf[:, 0])
    assert np.abs(kf[:, 1:3] - odom_at_kf[:, 1:3]).max() < 0.02
    tilt_kf, tilt_odom = _tilt_deg(kf[1:, 4:6]), _tilt_deg(odom_at_kf[1:, 4:6])
    assert tilt_kf.max() < 0.5, tilt_kf
    assert tilt_kf.mean() < tilt_odom.mean(), (tilt_kf, tilt_odom)

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
    assert np.abs(traj[:, 1:4]).max() < 0.03
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

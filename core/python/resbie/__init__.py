"""Offline resbie session: RESPLE's B-spline filter on BIEVR's map.

Configuration is one or more YAML files, later ones winning per key, in
BIEVR's sectioned format (``calibration``, ``lidar``, ``map``, ``preprocess``,
``optimization``, ``imu``, ``loop_closure``, ``max_num_threads``) plus a
``resple:`` section with RESPLE's estimator parameters under RESPLE's names.
See ``resbie.DEFAULT_CONFIG``.

Every call is synchronous: when ``push_imu``/``push_lidar`` return, all
processing they triggered is done, so results depend only on the pushed data.

* IMU: seconds, m/s^2 (or g, see ``imu.normalized``), rad/s, in the IMU frame.
* LiDAR: points in the LiDAR frame; ``calibration`` maps LiDAR -> IMU.
* Poses are the IMU frame in a gravity-aligned world frame, one per sweep,
  stamped at the sweep's last point. A pose is reported once the spline over
  its sweep is final (about 50 ms of data later); ``finish()`` reports the
  rest with the current estimate.
* The first sweep covering ``map.min_voxels_for_init`` voxels seeds the map at
  the initial pose; the 15 IMU samples before it set the gravity direction.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ._core import Resbie as _NativeResbie

__all__ = ["Resbie", "DEFAULT_CONFIG", "DIAGNOSTIC_COLUMNS"]

# Columns of Resbie.sweep_diagnostics(), one row per reported sweep.
DIAGNOSTIC_COLUMNS = (
    "stamp",             # sweep end, s (as trajectory())
    "points",            # range-filtered points in the sweep
    "points_measured",   # points handed to the filter for it
    "points_effective",  # of those, matched to the map
    "updates",           # filter updates ending in the sweep (incl. a preceding gap)
    "imu_only_updates",  # updates without a matched point
    "info_rotation",     # smallest eigenvalue of the LiDAR information, rotation, 1/rad^2
    "info_translation",  # same, translation, 1/m^2
    "pos_sigma",         # position sigma of the newest control point, m
    "to_loop_closer",    # 1 if handed to the loop closer
    "nis_mean",          # mean point zp^2 / (HPH' + R) (prior); ~1 if variances are honest
)

PathLike = str | os.PathLike

# Every key the loader reads, with its default. Sections other than `resple`
# use BIEVR's key names; `resple` uses RESPLE's.
DEFAULT_CONFIG: dict[str, Any] = {
    "calibration": {  # T_IMU_LIDAR (LiDAR -> IMU), rotation row-major; required
        "translation": [0.0, 0.0, 0.0],
        "rotation": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
    },
    "lidar": {"min_range_m": 0.5, "max_range_m": 100.0, "time_offset_s": 0.0},
    "map": {
        "pixel_size_m": 0.05,
        "voxel_size_m": 0.5,
        "normal_tolerance_deg": 3.0,
        "smooth": True,
        "weighted": True,
        "max_size": 1500000,
        "min_voxels_for_init": 100,
    },
    "preprocess": {
        "downsample_resolution_m": 0.1,
        "informed_sample_count": 300,
    },
    "optimization": {"huber_delta": 0.1},
    "imu": {"normalized": -1.0},  # < 0 autodetect, 0 m/s^2, > 0 g
    "resple": {
        "knot_hz": 100,
        "cov_P0": 0.02,
        "cov_RCP_pos_old": 0.5,
        "cov_RCP_ort_old": 0.5,
        "cov_RCP_pos_new": 1.0,
        "cov_RCP_ort_new": 1.0,
        "std_sys_pos": 0.1,
        "std_sys_ort": 0.1,
        "cov_acc": [1.0, 1.0, 1.0],
        "cov_gyro": [0.1, 0.1, 0.1],
        "cov_ba": [0.2, 0.2, 0.2],
        "cov_bg": [0.2, 0.2, 0.2],
        "n_iter": 3,
        "num_points_upd": 100,
        "w_pt": 0.01,
        "bias_rw_acc": 0.001,
        "bias_rw_gyro": 0.0001,
        "gate_chi2": 9.0,
    },

    # BIEVR's loop_closure keyframe, Scan Context, ICP and pose-graph keys, plus
    # max_speed_mps (resbie's guard)
    "loop_closure": {"enable": False, "max_speed_mps": 30.0},
    "max_num_threads": 0,
}


class Resbie:
    """One resbie session."""

    def __init__(
        self,
        config_files: PathLike | Sequence[PathLike],
        *,
        map_point_stride: int = 0,
    ):
        """``config_files``: one or more YAML files, later ones win.

        ``map_point_stride > 0`` exports every N-th point of each sweep,
        deskewed into the IMU frame at the sweep's pose (see
        ``drain_map_scans``).
        """
        if isinstance(config_files, (str, os.PathLike)):
            config_files = [config_files]
        self.config_files = [Path(p) for p in config_files]
        for path in self.config_files:
            if not path.is_file():
                raise FileNotFoundError(path)
        if map_point_stride < 0:
            raise ValueError("map_point_stride must be >= 0")
        self._native = _NativeResbie(
            [str(p) for p in self.config_files], int(map_point_stride)
        )

    def push_imu(self, stamp: float, acceleration, angular_velocity) -> bool:
        """Push one IMU sample. False if rejected (not after the previous one)."""
        return self._native.push_imu(float(stamp), acceleration, angular_velocity)

    def push_lidar(
        self, stamp: float, points: np.ndarray, relative_times: np.ndarray
    ) -> bool:
        """Push one sweep. False if dropped (out of order).

        A sweep left empty by the range filter is still accepted: it tells the
        filter LiDAR saw nothing over its time span (counted as lidar_empty).

        ``stamp`` is the time of the earliest point, ``points`` is (N, 3) or
        (N, 4) XYZ[I], ``relative_times`` (N,) are offsets from ``stamp`` in s.
        """
        return self._native.push_lidar(float(stamp), points, relative_times)

    def finish(self) -> None:
        """End of input: report the sweeps still waiting for final poses."""
        self._native.finish()

    def latest_pose(self) -> np.ndarray | None:
        """Latest pose [t, x, y, z, qx, qy, qz, qw], or None before the first."""
        return self._native.latest_pose()

    def trajectory(self) -> np.ndarray:
        """Odometry poses so far, one per sweep, (N, 8): t, x, y, z, qx, qy, qz, qw."""
        return self._native.trajectory()

    def keyframe_trajectory(self) -> np.ndarray:
        """Loop-closed keyframe poses, (K, 8), from BIEVR's pose graph.

        Every keyframe stamp is also a ``trajectory()`` stamp. Empty when
        loop closure is disabled.
        """
        return self._native.keyframe_trajectory()

    def drain_map_scans(self) -> list[tuple[float, np.ndarray]]:
        """Sweeps reported since the last call, as ``(stamp, points)``.

        ``points`` (N, 4) float32 [x, y, z, intensity] are deskewed into the
        IMU frame at ``stamp``, the sweep's trajectory stamp.
        """
        return self._native.drain_map_scans()

    def sweep_diagnostics(self) -> np.ndarray:
        """Per-sweep filter diagnostics, (N, len(DIAGNOSTIC_COLUMNS))."""
        return self._native.sweep_diagnostics()

    def status(self) -> dict[str, Any]:
        """Counters, filter statistics, biases and loop-closure state."""
        return dict(self._native.status())

"""Offline resbie session: RESPLE's B-spline filter on BIEVR's map.

Configuration is one or more YAML files, later ones winning per key, with the
sections ``calibration``, ``lidar``, ``imu``, ``spline``, ``points``, ``map``,
``loop_closure`` and the top-level ``max_num_threads``. Noises are standard
deviations in SI units. Unknown keys are an error. ``resbie.DEFAULT_CONFIG``
holds every key with its default (read from the C++ loader, the single
source of defaults); only ``calibration`` has to be set.

Every call is synchronous: when ``push_imu``/``push_lidar`` return, all
processing they triggered is done, so results depend only on the pushed data.

* IMU: seconds, m/s^2 (or g, see ``imu.normalized``), rad/s, in the IMU frame.
* LiDAR: points in the LiDAR frame; ``calibration`` maps LiDAR -> IMU.
* Poses are the IMU frame in a gravity-aligned world frame, one per sweep,
  stamped at the sweep's last point. A pose is reported once the spline over
  its sweep is final (about 50 ms of data later); ``finish()`` reports the
  rest with the current estimate.
* The filter starts once the IMU has started and a sweep covers
  ``map.init_min_voxels`` map voxels; the 15 IMU samples before it set the
  gravity direction.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ._core import Resbie as _NativeResbie, default_config

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

# Every key the loader reads, with its default (from the C++ loader).
DEFAULT_CONFIG: dict[str, Any] = default_config()


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

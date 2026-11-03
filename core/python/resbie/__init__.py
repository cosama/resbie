"""resbie: RESPLE's B-spline filter on BIEVR's map, BIEVR's loop closer behind it.

Two independent classes, wired together by the caller::

    front = resbie.FrontEnd("front_end.yaml", diagnostics=True)
    back = resbie.BackEnd("back_end.yaml")  # optional
    for kind, stamp, *data in stream:          # time order
        if kind == "imu":
            front.push_imu(stamp, acc, gyro)
            back.push_imu(stamp, acc, gyro)
        else:
            front.push_lidar(stamp, points, relative_times)
        for stamp, pose, points in front.drain():
            back.push_lio(stamp, pose, points)
    front.finish()
    for stamp, pose, points in front.drain():
        back.push_lio(stamp, pose, points)
    back.finish()
    keyframes = back.keyframes()
    diagnostics = front.drain_diagnostics()  # rows since the last call, FrontEnd.DIAGNOSTIC_COLUMNS

The front end alone is a LiDAR-inertial odometry: one LIO result per sweep,
its poses the trajectory, its points placed on them the map. The back end adds
loop closure and a pose graph with IMU factors; it takes every LIO result (not
only keyframes) and every IMU sample up to a result's stamp before that result.

Units: every stamp in s, acceleration in m/s^2 (an IMU reporting g needs it
multiplied by 9.81 first), angular velocity in rad/s; every number float64.
Poses are [x, y, z, qx, qy, qz, qw]: the IMU frame in the map frame.

Each class owns its configuration, which decides the results: one or more
YAML files or dicts, later ones winning per key. Unknown keys are an error;
``DEFAULT_CONFIG`` holds every key with its default (read from the C++
loader, the single source of defaults). How a run executes (threads, which
points come out, diagnostics) are constructor arguments.

Diagnostics (``diagnostics=True``): ``drain_diagnostics()`` returns the rows
since the last call as an (N, len(DIAGNOSTIC_COLUMNS)) array: one row per
sweep (FrontEnd) or per LIO result (BackEnd).

Every call is synchronous: when a push returns, all processing it triggered
is done, so results depend only on the pushed data and its order.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import _core
from ._core import set_log_level

__all__ = ["FrontEnd", "BackEnd", "set_log_level"]

PathLike = str | os.PathLike
Config = PathLike | Mapping[str, Any] | Sequence[PathLike | Mapping[str, Any]]


def _documents(config: Config) -> list[str]:
    """YAML texts from files and dicts (JSON is YAML)."""
    if isinstance(config, (str, os.PathLike, Mapping)):
        config = [config]
    documents = []
    for item in config:
        if isinstance(item, Mapping):
            documents.append(json.dumps(item))
        else:
            documents.append(Path(item).read_text())
    return documents


class FrontEnd(_core.FrontEnd):
    """The odometry: IMU and LiDAR in, one LIO result per sweep out.

    * ``push_imu(stamp, acceleration, angular_velocity) -> bool``: False if
      not after the previous sample.
    * ``push_lidar(stamp, points, relative_times) -> bool``: ``stamp`` is the
      earliest point's time, ``points`` (N, 3) or (N, 4) XYZ[I] in the LiDAR
      frame (``calibration`` maps LiDAR -> IMU), ``relative_times`` (N,) s
      after ``stamp``. False if it starts before the previous sweep or does
      not end after it. A sweep left empty by the range filter still counts:
      LiDAR saw nothing then.
    * ``drain() -> [(stamp, pose, points)]``: the LIO results since the last
      call, one per sweep. ``stamp`` is the sweep's last point, ``pose`` (7,)
      the IMU there, ``points`` (N, 4) [x, y, z, intensity] every
      ``point_stride``-th range-filtered point (0: none) deskewed into the IMU
      frame at ``stamp``. A sweep is finished once the spline over it is final
      (about 50 ms of data later); ``finish()`` finishes the rest with the
      current estimate.
    * ``finish()``: end of input.
    * ``drain_diagnostics() -> (N, C)``: per sweep, how the filter fared
      (``DIAGNOSTIC_COLUMNS``; documented in ``frontEndRow()``,
      cpp/bindings.cpp). Empty unless ``diagnostics``.
    * ``status() -> dict``: counters, biases, gravity.
    * ``DEFAULT_CONFIG``, ``DIAGNOSTIC_COLUMNS``: every config key with its
      default; the diagnostics' column names.

    The filter starts once the IMU has started and a sweep covers
    ``map.init_min_voxels`` map voxels. The map frame is the first pose,
    levelled on the 15 IMU samples before it (yaw zero); gravity's direction
    in it is estimated from then on (``up_*`` in the diagnostics), so the map
    frame is only as level as that start.
    """

    def __init__(
        self,
        config: Config,
        *,
        max_num_threads: int = 0,
        point_stride: int = 1,
        diagnostics: bool = False,
    ):
        """``max_num_threads``: 0 = all cores (results do not depend on it).
        ``point_stride``: every N-th point of a sweep goes into its result, 0:
        none (the filter and its map use every point regardless)."""
        super().__init__(_documents(config), max_num_threads, point_stride, diagnostics)


class BackEnd(_core.BackEnd):
    """The loop closer: IMU and the front end's LIO results in, keyframes out.

    * ``push_imu(stamp, acceleration, angular_velocity) -> bool``: as the
      front end's.
    * ``push_lio(stamp, pose, points) -> bool``: one LIO result, as the front
      end's ``drain()`` returns it; every point given is used (``points``
      (N, 3) or wider; without points never a keyframe). False if not after
      the previous result.
    * ``finish()``: one batch optimization of the whole graph.
    * ``keyframes() -> (K, 8)``: t, x, y, z, qx, qy, qz, qw, the current
      loop-closed keyframe poses; each t is the pushed stamp, exactly.
    * ``drain_diagnostics() -> (N, C)``: per LIO result, whether it became a
      keyframe (``DIAGNOSTIC_COLUMNS``). Empty unless ``diagnostics``.
    * ``status() -> dict``: counters, keyframes, loops accepted and rejected.
    * ``DEFAULT_CONFIG``, ``DIAGNOSTIC_COLUMNS``: as the front end's.
    """

    def __init__(self, config: Config, *, diagnostics: bool = False):
        super().__init__(_documents(config), diagnostics)

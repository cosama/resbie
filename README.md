<p align="center"><img src="doc/logo.svg" alt="resbie: a baby beaver wearing a spinning LiDAR as a party hat" width="560"></p>

# resbie

> **RES**(PLE) + **BIE**(VR) = resbie. Nobody asked for this baby, and it's
> here anyway.

Two LiDAR-inertial odometry frameworks met in a [vendor folder](https://github.com/cosama/resbie/tree/main/core/vendor):

- **[RESPLE](https://github.com/ASIG-X/RESPLE)**: a continuous-time B-spline
  filter, the parent who keeps walking in the dark. Take away its points and
  it shrugs and keeps going on the IMU.
- **[BIEVR](https://github.com/S0UL4/BIEVR-LIO-SLAM)**: bump-image voxel maps,
  the parent who notices when a wall is 2 cm off and won't let it go.

resbie got RESPLE's spline brain and BIEVR's eyes. The two get along
surprisingly well, most of the time.

## Personality traits

- **Stubborn.** LiDAR goes dark? It keeps going on the IMU and usually
  comes out roughly where it went in. Usually.
- **Detail-obsessed.** It measures every point against sub-voxel height images,
  not some nearest-neighbour plane fitted in a hurry.
- **A terrible liar detector.** Every point is placed at its own timestamp,
  so if your LiDAR lies about time, resbie believes it and bends over
  backwards trying to make the lie work. Feed it honest point times.
- **Overeager.** Give it too many points and it reads too much into them.
  The default point filter (`points.downsample_m`, 0.25 m) keeps it calm;
  going much finer is asking for trouble.
- **Knows which way is up.** A pose graph behind it listens to the
  accelerometer the whole way, so even when the map drifts off level on a
  staircase, what comes out is level again.
- **Deterministic.** Same input, same output, bit for bit, however many
  threads you throw at it. Neither parent promised that out of the box,
  so this one it worked out on its own.

How the family actually works is in [`DESIGN.md`](DESIGN.md), for the
grown-ups.

## Feeding time

The whole API, in one sitting. Two halves, and you hold the hand that
connects them:

```python
import resbie

# Every knob with its default. The only thing it can't guess is where the
# LiDAR sits on the IMU (LiDAR -> IMU, rotation row-major). YAML files work too.
front_config = resbie.FrontEnd.DEFAULT_CONFIG | {
    "calibration": {"translation": [0.0, 0.0, 0.1], "rotation": [1, 0, 0, 0, 1, 0, 0, 0, 1]}
}

baby = resbie.FrontEnd(front_config, diagnostics=True)  # point_stride=N: fewer points out
nanny = resbie.BackEnd(resbie.BackEnd.DEFAULT_CONFIG)    # loop closure; optional

def burp():  # LIO results: pose at the sweep's end, every point deskewed there
    for stamp, pose, points in baby.drain():  # pose [x, y, z, qx, qy, qz, qw]
        nanny.push_lio(stamp, pose, points)   # points (M, 4) xyzi
        my_map.add(stamp, pose, points[::10])  # yours: odometry poses and a lighter map

for msg in my_time_ordered_sensor_stream():  # bring your own sensors
    if msg.is_imu:
        baby.push_imu(msg.stamp, msg.acceleration, msg.angular_velocity)
        nanny.push_imu(msg.stamp, msg.acceleration, msg.angular_velocity)
    else:  # stamp = earliest point; times relative to it (honest ones, please)
        baby.push_lidar(msg.stamp, msg.points_xyzi, msg.relative_times)
    burp()

baby.finish()  # bedtime: the last few sweeps get their poses
burp()
nanny.finish()

keyframes = nanny.keyframes()            # (K, 8) loop-closed keyframe poses
diag = baby.drain_diagnostics()          # how it felt about each sweep:
print(resbie.FrontEnd.DIAGNOSTIC_COLUMNS)  # points matched, IMU-only updates, sigma, ...
print(baby.status(), nanny.status())     # counters, biases, loop closures
```

The front end alone is a complete odometry; the back end only adds loop
closure and a gravity-aware pose graph, and runs just as well on its own
thread.

Every call is synchronous: when a push returns, the thinking is done, and
the same input always gives the same output.

## Raise one yourself

Ask your coding agent of choice. resbie was raised by one, so it will
probably have an easier time with this than with any other SLAM framework
out there. Point it at `core/` (a pip-installable Python package with a C++
core) and `DESIGN.md`, and tell it to feed the baby IMU and LiDAR in time
order. Honest point timestamps only.

## Family tree

- **Mum:** [RESPLE](https://github.com/ASIG-X/RESPLE) by Ziyu Cao, William
  Talbot and Kailai Li ("RESPLE: Recursive Spline Estimation for LiDAR-Based
  Odometry", RA-L 2025). GPL-3.0, so resbie is GPL-3.0 too: see
  [`LICENSE`](LICENSE). It's in the genes.
- **Dad:** [BIEVR-LIO](https://github.com/ethz-asl/BIEVR-LIO) by Patrick
  Pfreundschuh, Turcan Tuna, Cedric Le Gentil, Roland Siegwart, Cesar Cadena
  and Helen Oleynikova, ETH Zurich ASL ("BIEVR-LIO: Robust LiDAR-Inertial
  Odometry through Bump-Image-Enhanced Voxel Maps", RSS 2026). BSD-3-Clause,
  notice kept in [`core/vendor/bievr/LICENSE`](core/vendor/bievr/LICENSE).
- **The fun uncle:** Iheb Soula, whose
  [BIEVR-LIO-SLAM](https://github.com/S0UL4/BIEVR-LIO-SLAM) fork brought the
  pose-graph loop closer to the party. He forked dad one commit before
  dad put his license on, so the uncle showed up without one. We're
  assuming he meant dad's BSD-3, like any good uncle would.
- **Distant relatives:** [BTC](https://github.com/hku-mars/BTC_descriptor) by
  Chongjian Yuan, Jiarong Lin, Zuhao Zou, Xiaoping Hong and Fu Zhang ("BTC: A
  Binary and Triangle Combined Descriptor for 3-D Place Recognition", T-RO
  2024); [small_gicp](https://github.com/koide3/small_gicp) by Kenji Koide
  (MIT, [`core/vendor/small_gicp/LICENSE`](core/vendor/small_gicp/LICENSE));
  and helpers from basalt-headers (BSD-3), each with its license in its header.

## Credits

Written by Claude (Anthropic), in Claude Code, with great confidence and
the occasional wrong turn. Supervised by a human who kept saying "are you
sure?" at exactly the right moments.

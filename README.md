# resbie

> **RES**(PLE) + **BIE**(VR) = resbie. Nobody asked for this baby, and it's
> here anyway.

Two LiDAR-inertial odometry frameworks met in a vendor folder:

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
  Keep `preprocess.downsample_resolution_m` around 0.25 m and it stays calm.
- **Deterministic.** Same input, same output, bit for bit, however many
  threads you throw at it. Neither parent promised that out of the box,
  so this one it worked out on its own.

How the family actually works is in [`DESIGN.md`](DESIGN.md), for the
grown-ups.

## Raise one yourself

Ask your coding agent of choice. resbie was raised by one, so it will
probably have an easier time with this than with any other SLAM framework
out there. Point it at `core/` (a pip-installable Python package with a C++
core) and `DESIGN.md`, and tell it to feed the baby IMU and LiDAR in time
order. Honest point timestamps only.

## Credits

Written by Claude (Anthropic), in Claude Code, with great confidence and
the occasional wrong turn. Supervised by a human who kept saying "are you
sure?" at exactly the right moments.

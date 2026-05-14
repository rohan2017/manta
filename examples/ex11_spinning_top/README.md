# ex11 — Spinning-top precession

Pure C++ demo of the gyroscopic precession physics from the articulation
patches. No codegen, no Zenoh, no viewer — text output to stdout.

## What it shows

A thin stick (0.8 m, 0.1 kg) carries a heavy flywheel (2 kg disk, r =
0.15 m) on a `Motor` whose joint axis is parallel to the stick. The
stick's two ends are `Collider` spheres that rest on a flat
`CollisionField` ground plane. Gravity is uniform −9.81 ẑ.

Initial state: stick vertical, bottom collider touching the floor,
flywheel pre-spun to 200 rad/s about body +z.

At **t = 3 s** a small `Thruster` at the top of the stick fires for
50 ms in body +x, tipping the craft a few degrees.

Without the rotor-gyro torque correction in `Craft::sense_and_aggregate`,
the kick would tip the stick straight over and it would fall in +x.
With it, the body's Euler equation sees `−ω × h_rotor` and `dL/dt = τ`
makes the tilt direction *precess* around the vertical instead — the
expected gyroscopic top behavior.

## Theoretical precession rate

For a regular-precession heavy symmetric top (nutation neglected):

```
Ω = M · g · h / (I_axial · θ̇)
  ≈ 2.1 kg · 9.81 m/s² · 0.6 m / (0.0225 kg·m² · 200 rad/s)
  ≈ 2.7 rad/s   ⇒   period ≈ 2.3 s
```

The actual sim measures the precession azimuth advancing at ~2.8 rad/s
in the linear-tilt regime (3.2 s < t < 4.4 s).

## Build & run

```bash
cmake --build build --target ex11_spinning_top -j
./build/examples/ex11_spinning_top
```

Output columns:
- `t` — sim time, s
- `x y z` — body origin in scene frame, m
- `tilt[deg]` — angle of body +z from scene +z
- `azim[deg]` — azimuth of the tilt direction in scene xy
- `fly_rate` — motor joint rate, rad/s

The interesting window is t = 3 → 6 s: tilt grows, azimuth sweeps
through a full rotation, then the stick lays flat and slides.

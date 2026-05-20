# ex11 — Spinning-top precession (with Rerun viewer)

Library-workflow codegen demo of the gyroscopic precession physics from
the articulation patches. The Python `config.py` describes the craft;
manta-codegen emits the C++ Craft + telemetry; the user-written
`main.cpp` drives the sim and publishes telemetry over Zenoh; `viewer.py`
subscribes and renders in Rerun.

## What it shows

A thin stick (0.8 m, 0.1 kg) carries a heavy flywheel (2 kg disk,
r = 0.15 m) on a `Motor` whose joint axis is parallel to the stick.
The stick's two ends are `Collider` spheres that rest on a flat
`CollisionField` ground plane. Gravity is uniform −9.81 ẑ.

Initial state (set in `main.cpp`): stick vertical, bottom collider on
the floor, flywheel pre-spun to **350 rad/s** about body +z.

At **t = 3 s** the lateral `Thruster` at the top of the stick fires for
50 ms in body +x — a **4 N** kick (impulse 0.2 N·s) that gives the body
an initial tilt rate of ~13 °/s.

Without the rotor-gyro torque correction in `Craft::sense_and_aggregate`
the kick would tip the stick straight over in +x. With it, the body's
Euler equation sees `−ω × h_rotor` and `dL/dt = τ` makes the tilt
direction *precess* around the vertical instead — the expected
gyroscopic top behavior. The sim **runs indefinitely** until you Ctrl-C
the binary; the top precesses for many cycles and eventually settles
toward the ground as contact friction dissipates the rotational energy.

## Theoretical precession rate

For a regular-precession heavy symmetric top (nutation neglected):

```
Ω = M · g · h / (I_axial · θ̇)
  ≈ 2.1 kg · 9.81 m/s² · 0.6 m / (0.0225 kg·m² · 350 rad/s)
  ≈ 1.6 rad/s   ⇒   period ≈ 4 s
```

Tuning notes: higher flywheel RPM → more stored `h_rotor` → slower
precession AND smaller tilt amplitude for the same kick. Larger kick
→ bigger tilt → easier to see but more chance of tipping over before
the gyro engages. The current values are a balance: ~15° tilt
amplitude with a ~4 s precession period.

## Regenerate the codegen artifacts

```bash
.venv/bin/python -m manta_codegen.cli \
    examples/ex11_spinning_top/config.py --workflow library
```

This (re)emits `generated/ex11/{ex11.hpp,ex11.cpp,ex11_craft.hpp,
ex11_craft.cpp,ex11_telemetry.hpp,ex11_config.h,ex11.cmake}`.

## Build and run

Three terminals at the repo root:

**Terminal 1 — viewer (start FIRST):**

```bash
.venv/bin/python examples/ex11_spinning_top/viewer.py
```

Wait until you see `viewer: listening on manta/ex11/state` and a Rerun
GUI window opens.

**Terminal 2 — sim:**

```bash
cmake --build build --target ex11_spinning_top -j
./build/examples/ex11_spinning_top
```

You'll see the stick stand still for 3 seconds, then the thruster fires
(brief jolt in +x), then the craft starts precessing — the tilt vector
sweeps a horizontal circle around the vertical rather than just falling
straight in the kick direction.

The Rerun window shows:
- the body Transform3D (stick, flywheel, end-cap colliders, thruster),
- the flywheel's two yellow/cyan halves spinning at 200 rad/s,
- a red tilt-arrow at the body origin pointing along body +z,
- scalar plots for `fly_rate` and `tilt_deg`.

## Restarting

If a run gets into a weird state, kill everything and start clean:

1. `Ctrl-C` the sim and the viewer.
2. Close the Rerun GUI window (otherwise the next `spawn=True` may bind
   to the stale process).
3. Restart in the same order: viewer → sim.

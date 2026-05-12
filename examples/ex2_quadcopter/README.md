# ex2 — Quadcopter with rate-control PIDs

X-config quadcopter (1 kg body, 0.25 m arm) lifting off a flat ground
plane. Each of the four rotors is a `Motor` (revolute joint about
body +z) carrying two `Naca00xx` blades 180° apart, pitched 12° about
their span axis. Lift is real blade-element aerodynamics: drag spins
the body opposite to motor rotation; CW and CCW rotors pair up for
yaw cancellation. A `Collider` + ground `CollisionField` lets the
craft rest on the ground.

Codegen emits the `Ex2Craft` C++ type and a JSON telemetry struct;
the user's `main.cpp` runs the rate PIDs, X-config mixing, per-motor
velocity PI, and Zenoh wiring (library workflow).

## Build

```
cmake -S . -B build -DMANTA_DOUBLE_PRECISION=ON
cmake --build build --target ex2_quadcopter -j
```

## Running

Open **three** terminals at the repo root and start them **in this
order**.

**Terminal 1 — Rerun viewer (start FIRST):**

```
.venv/bin/python examples/ex2_quadcopter/viewer.py
```

Wait until you see `viewer: listening on manta/ex2/state` and a
Rerun GUI window opens. The window shows the ground grid, the craft
body, four spinning rotors with yellow/cyan flat-plate blades, and a
thrust arrow per rotor.

**Terminal 2 — Simulator:**

```
./build/examples/ex2_quadcopter
```

Wait for `ex2: ready. Publish [thr, roll_rate, pitch_rate, yaw_rate]
on manta/ex2/cmd.`

Within ~1 second the viewer should print
`viewer: first state received  t=...` and start streaming poses to
Rerun.

**Terminal 3 — Keyboard controller:**

```
.venv/bin/python examples/ex2_quadcopter/keyboard_controller.py
```

Click this terminal to give it focus before typing commands.

## Restarting

If a run gets into a weird state (rotors stuck, no streaming, stale
GUI), kill everything and start clean:

1. `Ctrl-C` the controller, the sim, and the viewer.
2. Close the Rerun GUI window (otherwise the next `spawn=True` may
   bind to the stale process).
3. Restart in the same order: viewer → sim → controller.

Common gotchas:

- The viewer needs `.venv/bin/python` (manta_codegen is installed
  there and the Rerun SDK ships the `rerun` GUI binary next to that
  interpreter). System Python won't work.
- Don't start the sim before the viewer's `listening on` message —
  Zenoh pub/sub will deliver new samples regardless of subscriber
  order, but Rerun won't display historical poses, so the first
  second of telemetry can look like nothing happened.

## Keyboard controls

Throttle is held-stateful (held = continuously ramping); rate
commands are held-only (release → 0).

| Key       | Action                                       |
| --------- | -------------------------------------------- |
| `I` / `K` | Throttle up / down (continuous while held)   |
| `W` / `S` | Pitch nose-down / nose-up                    |
| `A` / `D` | Roll  left / right                           |
| `Q` / `E` | Yaw   left / right                           |
| `SPACE`   | Zero rates (keep throttle)                   |
| `X`       | Zero everything                              |
| `Ctrl-C`  | Quit                                         |

Hover is roughly `thr = 0.5` (at that command the velocity PI drives
each motor to `OMEGA_HOVER` ≈ 90 rad/s, where the prop's blade
aerodynamics balance gravity).

## Wire format

```
manta/ex2/cmd    →  [thr, roll_rate, pitch_rate, yaw_rate]   (4 floats)
manta/ex2/state  ←  { t, p, q, v, w, imu, fr_motor, fl_motor, bl_motor, br_motor }
                    where <name>_motor: { angle, rate, accel }
```

## Regenerating after a config edit

```
.venv/bin/python -m manta_codegen.cli examples/ex2_quadcopter/config.py --workflow library
cmake --build build --target ex2_quadcopter -j
```

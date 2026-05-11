# ex2 — Quadcopter with rate-control PIDs

X-config quadcopter (1 kg, 0.25 m arm) lifting off a flat ground plane.
Four `Thruster1` rotors with a reaction-torque coefficient; a
hand-written rate-PID loop in `main.cpp` maps `[thr, roll_rate,
pitch_rate, yaw_rate]` commands to per-rotor throttles. Codegen emits
the `Ex2Craft` C++ type and a JSON telemetry struct; the user's
`main.cpp` does the control + Zenoh wiring (library workflow).

## Running

Build (one-time):

```
cmake -S . -B build -DMANTA_DOUBLE_PRECISION=ON
cmake --build build -j
```

Three terminals at the repo root:

```
./build/examples/ex2_quadcopter                                # sim + PID
.venv/bin/python examples/ex2_quadcopter/viewer.py             # Rerun
.venv/bin/python examples/ex2_quadcopter/keyboard_controller.py  # keys
```

The controller terminal must have keyboard focus.

## Keyboard controls

Throttle is held-stateful (held = continuously ramping); rate commands
are held-only (release → 0).

| Key       | Action                                       |
| --------- | -------------------------------------------- |
| `I` / `K` | Throttle up / down (continuous while held)   |
| `W` / `S` | Pitch nose-down / nose-up                    |
| `A` / `D` | Roll  left / right                           |
| `Q` / `E` | Yaw   left / right                           |
| `SPACE`   | Zero rates (keep throttle)                   |
| `X`       | Zero everything                              |
| `Ctrl-C`  | Quit                                         |

To hover, hold `I` until the throttle is ~0.5, then release.
`MAX_THRUST_PER_PROP = 2 × hover` per rotor, so half-throttle on all
four rotors holds 1 g.

## Wire format

```
manta/ex2/cmd    →  [thr, roll_rate, pitch_rate, yaw_rate]   (4 floats)
manta/ex2/state  ←  { t, p, q, v, w, fr:{throttle}, fl:..., bl:..., br:... }
```

## Regenerating after a config edit

```
.venv/bin/python -m manta_codegen.cli examples/ex2_quadcopter/config.py --workflow library
cmake --build build --target ex2_quadcopter -j
```

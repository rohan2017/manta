# ex8 — Underwater AUV with sim + EKF

Binary-workflow codegen demo. A small submarine with full underwater
dynamics (mass + buoyancy + drag + 2× thruster) runs in parallel with
an Extended Kalman Filter that ingests its noisy IMU, DVL, and
magnetometer outputs. Both sim and filter live in **one binary**, the
codegen wires up the Zenoh I/O, and a Rerun viewer shows the truth and
the EKF estimate side by side with the position uncertainty drawn as a
sphere around the estimate.

## What it shows

- **Underwater physics**: `Mass` + `PointBuoy` (slight negative buoyancy)
  + `Surface1` (anisotropic linear drag) + two `Thruster1` actuators
  (`thrust_x` forward, `thrust_z` vertical) in a `FluidField` at
  seawater density (1025 kg/m³).
- **Sensors**: `IMU` (100 Hz, accel σ = 0.05 m/s²/√Hz, gyro σ = 0.005
  rad/s/√Hz), `DVL` (10 Hz, σ = 2 cm/s on velocity), `Magnetometer`
  (50 Hz, σ = 0.2 µT) on a uniform `MagField` mid-latitude rough.
- **Estimator**: `EKFGeneric<StateSpec, MeasDim, NoiseSlots>` wrapping a
  matched estimator craft with `Mass` + two thrusters (mirrors of the
  sim thrusters via codegen `connect()`). Predict uses force-only
  dynamics (no buoy / drag); measurement updates from IMU + DVL + Mag
  keep the position bounded.
- **Initial conditions**: 5 m below the surface, facing +x, at rest.
- **Tight initial covariance** (1e-4 on pos, 1e-4 on attitude, 1e-2 on
  vel, 1e-4 on ω): the EKF "knows where the vehicle started."

## Topic map

The codegen-emitted binary publishes and subscribes on these Zenoh
topics:

| Direction  | Topic                       | Payload                                                           |
|------------|-----------------------------|-------------------------------------------------------------------|
| publish    | `manta/ex8/state`           | `{p[3], q[4]=wxyz, v[3], w[3]}` — truth pose @ ~50 Hz             |
| publish    | `manta/ex8/estimate`        | `{p, q, v, w, p_stddev, v_stddev}` — EKF output @ ~50 Hz          |
| subscribe  | `manta/ex8/thrust_x/cmd`    | `[throttle]` ∈ [-1, +1] — forward (+) / reverse (-)               |
| subscribe  | `manta/ex8/thrust_z/cmd`    | `[throttle]` ∈ [-1, +1] — rise (+) / dive (-)                     |

`thrust_x`'s `+x` direction is the sub's nose; `thrust_z`'s `+z`
direction is up. The submarine is slightly negatively buoyant (it sinks
slowly without commanded thrust); `+z` throttle counters that to hold
depth or rise.

## Build

```bash
cmake --build build --target ex8 -j
```

## Regenerate the codegen artifacts

```bash
.venv/bin/python -m manta_codegen.cli \
    examples/ex8_submarine/config.py --workflow binary
```

This (re)emits `generated/ex8/` (sim craft, estimator craft, main,
telemetry, cmake fragment).

## Run

Open **three** terminals at the repo root and start them **in this
order**.

**Terminal 1 — Rerun viewer (start FIRST):**

```bash
.venv/bin/python examples/ex8_submarine/viewer.py
```

Wait until you see `viewer: listening on manta/ex8/{state,estimate}`
and a Rerun GUI window opens.

**Terminal 2 — Simulator:**

```bash
./build/examples/ex8
```

You'll see `ex8: ready (sim+filter).` and the sim starts ticking at
1 kHz. The viewer should immediately start showing the submarine 5 m
below the surface, with the EKF estimate (cyan wireframe) and a small
uncertainty sphere around it.

**Terminal 3 — Keyboard controller:**

```bash
.venv/bin/python examples/ex8_submarine/keyboard_controller.py
```

Click this terminal to give it focus before typing commands.

### Controls

| Key       | Action                                  |
|-----------|-----------------------------------------|
| W / S     | forward / reverse throttle (thrust_x)   |
| I / K     | rise / dive throttle      (thrust_z)    |
| SPACE     | zero both thrusters                     |
| Ctrl-C    | quit                                    |

Throttle ramps continuously while a key is held (auto-repeat keeps
it alive within ~150 ms); release lets it decay back to zero. Bipolar
range [-1, +1] gets clamped before publication.

## What to watch for

- **Steady drift down** in the first few seconds: the sub is negatively
  buoyant. Tap `I` to hold depth.
- **Position uncertainty sphere shrinks rapidly**: the DVL bounds
  velocity error, the IMU bounds attitude error (via gravity in the
  accel reading), and the magnetometer's absolute-attitude reference
  keeps yaw observable. After ~1 s the uncertainty sphere is well
  under 0.1 m.
- **Estimate tracks truth tightly**: with the measurement updates the
  cyan wireframe sits inside the red truth body even when you maneuver
  hard. Watch the `plots/p_err_*` scalar panels for the per-axis
  residuals.
- **Magnetometer earns its keep**: without it, body-frame-only sensors
  (IMU + DVL) wouldn't observe absolute attitude, and the EKF would
  develop a yaw drift. With it, q stays observable through the dive.

## Restarting

If a run gets into a weird state (no streams in the viewer, controller
unresponsive, stale GUI):

1. `Ctrl-C` the controller, the sim, and the viewer.
2. Close the Rerun GUI window (otherwise the next `spawn=True` may bind
   to the stale process).
3. Restart in the same order: viewer → sim → controller.

## Files

- `config.py` — Python codegen descriptor (sim craft, est craft, EKF
  wiring, Zenoh I/O).
- `generated/ex8/` — codegen-emitted C++ (sim, estimator, main,
  telemetry, cmake).
- `viewer.py` — Rerun viewer subscribing to truth + estimate.
- `keyboard_controller.py` — held-key thruster control.
- `smoke_test.py` — non-interactive bounds check that the truth stays
  put and the EKF tracks it. Run while the sim is up.

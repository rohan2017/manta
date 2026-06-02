# manta

A Python-first, CasADi-backed framework for small vehicles — drones,
rockets, underwater vehicles, satellites — spanning rigid-body
simulation, state estimation (EKF), and control synthesis (LQR). You
declare the craft and its physics; manta builds the symbolic graph,
runs it natively in Python, and (when you're ready) lowers it to
embedded C++ via codegen.

## The pipeline

Three layers, explicit at every boundary. The model is declarative; the
IR layer is a set of **analysis transforms** over it — `Sim`, `EKF`,
`LQR` are siblings, each consuming the model and the shared
`Linearization` seam; a `Target*` lowers any transform's IR.

```
Model            IR (transform)       Target              Runtime
─────────────────────────────────────────────────────────────────
World        →   Sim(world)       →   TargetNumpy(sim) →  NumpyWorld
                  (CasADi tick)                            .step() / .initial_state()

World        →   EKF(world)       →   TargetNumpy(ekf) →  NumpyEKF
                  (sym predict +                            .predict() / .update()
                   sensor table)                            .step() / .state_dict()

World        →   LQR(world, …)    →   TargetNumpy(lqr) →  NumpyLQR
                  (gain K + baked                           .control(state) → {input: u}
                   control law)

World        →   Sim(world)       →   TargetCpp(sim,…) →  <basename>.cpp/.hpp
                                                            + flat-C kernels + CMake
```

`Sim(world)`, `EKF(world)`, and `LQR(world, …)` are pure compile-time.
They produce *descriptions* — CasADi function bundles, state specs,
sensor tables, gains — that aren't directly callable. A `Target*`
constructor lowers an IR to a backend-specific runtime; today that's
native-Python (`TargetNumpy`, all three transforms) and Eigen-typed C++
(`TargetCpp`, the sim today). The Jacobian machinery they share lives in
one place — `manta.linearization.Linearization` (manifold-aware
F / B / H / L, emitted as symbolic functions so each consumer evaluates
at its own operating point).

## Quick example

```python
from manta import World, Craft, Sim, EKF, TargetNumpy
from manta.fields import GravityField
from manta.parts import IMU, Mass, PositionSensor, Thruster

# Model.
drone = Craft("drone")
drone.add(Mass("body", mass=1.5, moi=(0.05, 0.05, 0.08)))
drone.add(Thruster("t", force=(0, 0, 1)))
drone.add(IMU("imu", gyro_noise_sigma=0.005, gyro_bias_sigma=1e-4))
drone.add(PositionSensor("gps"))

w = World().add_field(GravityField(g=(0, 0, -9.81)))
w.add_craft(drone, position=(0, 0, 5))

# Build the IR transforms, lower to native-Python.
sim = TargetNumpy(Sim(w))
ekf = TargetNumpy(EKF(w))

# Run.
state = sim.initial_state()
state["drone"]["t.throttle"] = 1.5 * 9.81           # hover
for t in np.arange(0, 3, 0.005):
    state = sim.step(state, dt=0.005, t=t)
    ekf.predict(dt=0.005, t=t, u={"t.throttle": 1.5 * 9.81})
    ekf.update(drone.parts[-2], gyro=state["drone"]["imu.gyro"])
    ekf.update(drone.parts[-1], position=state["drone"]["gps.position"])

print(ekf.state_dict()["drone"]["position"])
```

To lower the same model to C++ for embedded use:

```python
from manta import TargetCpp
TargetCpp(Sim(w), "out", class_name="Drone")
# → out/{drone.hpp, drone.cpp, drone_kernels.c/h, CMakeLists.txt}
```

### Closed-loop control

`LQR(world, …)` synthesizes a state-feedback regulator about an operating
point — the third sibling transform. It regulates a controllable subset
(`track=`), freezing the rest; the runtime maps a state estimate to
commands:

```python
from manta import LQR
import numpy as np

lqr = TargetNumpy(LQR(
    w,
    x_ref={"drone": {"position": (0, 0, 10), "velocity": (0, 0, 0)}},
    u_ref={"t.throttle": 1.5 * 9.81},          # hover trim
    track=["drone.position", "drone.velocity"],
    Q=np.diag([10, 10, 10, 1, 1, 1]), R=np.eye(1), dt=0.02))

u = lqr.control(ekf.state_dict())              # {input_name: command}
```

(Q/R here are LQR *cost* weights, not the EKF's noise. A free rigid body
is underactuated, so full-state LQR isn't stabilizable — regulate the
controllable subspace via `track=`.)

## Core concepts

### Parts

A `Part` is an atomic unit of behavior on a craft. Declares at class
scope:

- `Parameter(default)` — frozen at construction, baked into the graph.
- `State(init, manifold="R1"|"R3"|SO3Manifold(...))` — mutable per-tick
  state; SO(3) slots carry an orientation (IMU integrators, attitude
  filters) with manifold-correct boxplus.
- `Input(default)` — per-tick user-supplied value (e.g. throttle).
- `Output(shape)` — per-tick observable (sensor reading).
- `Noise(shape, kind="white"|"rw", sigma)` — per-tick white noise OR
  RW bias state (synthesizes its own state slot + driver input).

Stock parts: `Mass`, `PointBuoy`, `Collider`, `Thruster` (polynomial
in throttle), `Joint` (1-DOF revolute, with optional Mass child for
rotors), `DragSurface`, `Naca00xx` airfoil, `IMU` (gyro+accel, with
Kalibr 4-parameter noise model), `DVL`, `Magnetometer`,
`PositionSensor`, `TetherEndpoint`.

### Fields and Disturbances

Each `Field` (one of `GravityField`, `FluidField`, `MagField`,
`CollisionField`) is a typed superposition of `Disturbance` objects.
Disturbances combine via per-disturbance flags:

- `"additive"` (default) — linear sum (gravity, B-field, etc.).
- `"averaged"` — mean of running additive + every averaged
  contribution (overlapping wind bubbles compromise on the mean).
- `"projected"` — Gram-Schmidt residual (only add the component
  orthogonal to or extending the running sum).

Disturbances can carry State / Noise declarations like Parts —
this is how `WindBias`, `CraftWindBubble`, and friends become
estimable through the EKF.

### Planets

`Planet` (and the `Earth` preset) is a World-level entity that:

- Holds a planet-fixed frame (axis + rotation rate) and provides
  symbolic + numpy transforms between PlanetFrame and WorldFrame.
- Auto-registers standing disturbances on the world's shared fields
  (Earth: point-mass + optional J2 gravity, ocean + ISA atmosphere
  via `PlanetFrameFluid`, dipole magnetic field).
- Provides initial-state factories — `earth.position(x, y, z)`,
  `earth.velocity(vx, vy, vz)`, `earth.at_rest()` — that resolve to
  WorldFrame seeds at compile time via the planet's transform.

Multiple planets in one world are supported. Each planet's
disturbances superpose into the shared fields.

### EKF

`EKF(world)` builds the Error-State EKF IR over every craft + every
state-bearing disturbance:

- **Q auto-assembled** from declared `Noise` channels: process-noise
  contributions for any noise affecting the next-tick state are
  picked up via autodiff (`L · Σ · Lᵀ`); RW biases get `dt · σ²` on
  their slot diagonal automatically.
- **R auto-assembled** per sensor Output from the noise channels
  feeding that output.
- **State spec auto-built** by walking every craft + every
  disturbance; the EKF estimates per-craft rigid-body slots plus
  any user-declared State or RW-bias slots.
- **Manifold-aware updates** — SO(3) tangent for the
  rigid-body orientation, R3 for vec3 states, R1 for scalars.
  Joseph-form measurement update.

Lower to `TargetNumpy(EKF(w))` for Python; future
`TargetCpp(EKF(w), ...)` for embedded.

### Backends

The `manta.targets` package houses the lowering. Each Target accepts
an IR and produces a runtime:

| Target | Accepts | Produces |
|---|---|---|
| `TargetNumpy(sim)` | Sim | NumpyWorld (`step`/`initial_state`) |
| `TargetNumpy(ekf)` | EKF | NumpyEKF (`predict`/`update`/`step`/`state_dict`/`reset`) |
| `TargetNumpy(lqr)` | LQR | NumpyLQR (`control`/`u`/`K`) |
| `TargetCpp(sim, out_dir, class_name)` | Sim | Buildable C++ static library |

Adding a backend (TensorFlow eager, raw embedded C, GPU CUDA) is one new
`Target` subclass: implement `lower_sim` / `lower_ekf` / `lower_lqr` (the
`Target` ABC holds the IR-type dispatch, so a half-finished backend's
unsupported hook raises `NotImplementedError` at the call rather than
failing silently). Adding a transform (`LQR`, a future `iLQR`/`MPC`)
reuses the shared `Linearization` seam.

## Layout

```
manta/                     library package
    __init__.py            World / Craft / Sim / EKF / LQR / Target* surface
    craft.py               Craft + TickContext + Newton-Euler integrator
    world.py               World (the declarative model)
    sim.py                 Sim (forward-dynamics IR transform)
    world_tick.py          World-tick compile (one CasADi function per world)
    linearization.py       Manifold-aware F / B / H / L (shared seam)
    tick_signature.py      Tick I/O classifier (Inputs / Noise / sensors)
    kinematics.py          Symbolic kinematic-chain pass
    inertia.py             Symbolic inertia rollup
    ir/                    Frames, types, Graph, Manifold, Wrench
    parts/                 Part base + stock parts (sensor/actuation/aero/…)
    fields/                Field + Disturbance + stock + CraftWindBubble
    planets/               Planet ABC, Earth, PlanetFrameFluid, PlanetState
    couplings/             Coupling ABC + Tether
    estimation/            EKF (IR) + StateSpec + measurement helpers
    control/               LQR (IR)
    codegen/               Backends (one subpackage per target language)
        numpy/             TargetNumpy + NumpyWorld + NumpyEKF + NumpyLQR
        cpp/               TargetCpp + extract / kernels / wrapper / cmake
tests/                     358 tests
examples/                  quickstart + physics/ + vehicles/
    _viz.py                rerun visualization helpers
    _control.py            keyboard (pynput) + scripted-fallback control
    quickstart.py          install sanity check (numpy Sim, apex height)
    physics/               bouncing_ball / spinning_top / foucault_pendulum
    vehicles/              quadcopter / glider / submarine / hydrofoil
```

## Demos

Start with the install sanity check — a ball thrown into uniform gravity,
run on the numpy backend, reporting the apex height it reached:

```bash
.venv/bin/python -m examples.quickstart
```

The rest are organized into **physics** (each with a [rerun](https://rerun.io)
3-D visualization) and **vehicles** (visualization + keyboard control, with
a self-running scripted fallback so they work unattended):

```bash
# physics/ — visualized
.venv/bin/python -m examples.physics.bouncing_ball       # Collider + CollisionField
.venv/bin/python -m examples.physics.spinning_top        # gyroscopic precession (Joint)
.venv/bin/python -m examples.physics.foucault_pendulum   # Planet + Tether + Coriolis

# vehicles/ — visualized + keyboard (add --keyboard for live control)
.venv/bin/python -m examples.vehicles.quadcopter         # Sim + EKF + LQR closed loop
.venv/bin/python -m examples.vehicles.glider             # NACA wing, fly-by-wire
.venv/bin/python -m examples.vehicles.submarine          # PointBuoy + DVL + EKF
.venv/bin/python -m examples.vehicles.hydrofoil          # nested-Joint laser gimbal (PID)
```

Visualized demos need the rerun SDK (`.venv/bin/pip install rerun-sdk`); pass
`--no-viz` to run any of them headless. Vehicle demos take `--keyboard`
(live `pynput` control), `--no-viz`, and `--duration`. Shared helpers live in
`examples/_viz.py` (rerun) and `examples/_control.py` (keyboard / scripted).

## Running tests

```bash
.venv/bin/python -m pytest tests/
```

## Status

In active development. The public API (`World`, `Craft`, `Sim`, `EKF`,
`LQR`, `TargetNumpy`, `TargetCpp`) is settled enough that the demos
and 358 tests don't carry compat shims. Open items:

- **`TargetCpp(ekf, …)` / `TargetCpp(lqr, …)`** — estimator and
  controller lowering to C++. The sim lowers today; the EKF/LQR IRs
  carry the same Jacobian bundles (via the shared `Linearization`), so
  the C++ side needs a mutable-state + Joseph-form-update layer (EKF)
  and a feed-forward control-law emit (LQR). This is the missing piece
  of the deploy-to-robot path — today you can lower the simulator, but
  the estimator/controller you'd actually run onboard stay in Python.
- **`iLQR` / `MPC`** — the `Linearization` seam emits symbolic A/B/H, so
  trajectory-tracking controllers reuse it; the iterative solve lives in
  the backend (not the IR), per the design.
- **Observability analysis** (shipped — `manta.estimation.observability`)
  — a faithful EKF of a correct model is still only as good as the model's
  *observability* (a property of dynamics + sensor set + **operating
  point**, not of the model alone): unobservable modes drift silently
  while the covariance looks tight. `observability(EKF(world))` (or
  `numpy_ekf.observability()`) builds the observability matrix from the
  symbolic `F`/`H` at an operating point and reports rank + which state
  slots are unobservable. It flags, e.g., that GPS + DVL + gyro can't see
  absolute heading **at rest** (only through a maneuver) — so a compass
  earns its place. Local + operating-point-dependent by nature; check a
  few representative points.
- **NEES consistency check** (shipped — `manta.estimation.nees`) — the
  complement to observability: observability asks what you *can* estimate;
  NEES asks whether the filter's reported *covariance* is honest (a
  fully-observable filter can still be overconfident and quietly diverge,
  or conservative and waste information). `nees(world, dt=, steps=,
  control=)` runs a Monte-Carlo ensemble (truth jittered by the model's
  process noise, measurements by their R, the initial estimate drawn from
  P₀) and reports ANEES vs the χ² band: too high ⇒ overconfident, too low
  ⇒ conservative. It already surfaces that the linearized one-step
  auto-`Q` (`L·Σ·Lᵀ`) is mildly optimistic. Follow-up: an
  observable-subspace variant and a Gramian-along-trajectory observability.
- **EKF measurement timing** (fixed) — `NumpyEKF.step` now folds a
  measurement *before* propagating over its interval (update-then-predict),
  because the sim emits sensor outputs from the interval's *start* state.
  The old predict-then-update order met a start-of-interval reading with
  the end-of-interval state, biasing rate-derived states (orientation) by
  O(dt) — a gyro-only EKF drifted heading where a naive integrator didn't.
  Fixing it collapsed that error to ~0 (submarine est error: 2.2 m peak →
  ~4 mm). `DVL` also gained a `velocity_noise` channel (it had none, so its
  EKF R was singular).
- **C++ wrapper `t` plumbing** — the generated C++ `predict()`/`measure_*`
  methods hardcode the world clock to `0.0` (the kernel ABI has the slot,
  the wrapper doesn't expose it). Harmless for time-invariant dynamics,
  wrong for time-varying ones. Fix before relying on the C++ backend.
- **Multi-craft EKF over coupled worlds** — works for parallel
  independent crafts (block-decomposed predict is wired); field-mediated
  cross-craft coupling in the *estimator* is untested at scale.
- **Parameter tuning** (`manta.tuning`) — placeholder; the design is an
  offline IPOPT fit of `tunable` Parameters against logged trajectories.

## Design docs

`prompts/` holds the design discussions that informed the current
shape — frame hierarchy, estimator design, field-bus combining,
disturbance state-carrier lift, etc.

## License

See `LICENSE`.

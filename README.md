# manta

A Python-first, CasADi-backed framework for small vehicles — drones,
rockets, underwater vehicles, satellites — spanning rigid-body
simulation, state estimation (EKF), and control synthesis (LQR). You
declare the craft and its physics; manta builds the symbolic graph,
runs it natively in Python, and (when you're ready) lowers it to
embedded C++ via codegen.

## The pipeline

Three layers, explicit at every boundary. The model is declarative; the
**transforms** (`Sim`, `EKF`, `LQR`, the recurrence blocks) are siblings
over it, each owning its math and emitting a typed `Module` IR; a
`Target*` lowers any Module to a backend.

```
Model            Transform            Target              Result
─────────────────────────────────────────────────────────────────
World        →   Sim(world)       →   TargetNumpy(sim) →  NumpyRuntime
                  (linearized tick)                        .step() / .outputs()

World        →   EKF(world)       →   TargetNumpy(ekf) →  NumpyRuntime
                  (Kalman recursion,                       .predict() / .update()
                   baked kernels)                          .feed() / .step()

World        →   LQR(world, …)    →   TargetNumpy(lqr) →  NumpyRuntime
                  (Riccati → gain K,                       .control(state) → {input: u}
                   baked control law)

any of these →   .module()        →   TargetCpp(x, …)  →  <basename>.cpp/.hpp
                                                           + flat-C kernels + CMake
```

`Sim(world)`, `EKF(world)`, and `LQR(world, …)` are pure compile-time.
Each writes its math symbolically over the shared `LinearizedSystem`
(manifold-aware F / B / H / L over the compiled world tick) and emits a
typed `Module` — state layout + named CasADi kernels + typed entry
points — that isn't directly callable. A `Target*` lowers the Module:
`TargetNumpy` to the one native-Python `NumpyRuntime` (its surface is
derived from the Module's shape), `TargetCpp` to a typed Eigen C++
class. Backends contain no per-transform code.

## Quick example

```python
import numpy as np

from manta import World, Craft, Sim, EKF, TargetNumpy
from manta.fields import GravityField
from manta.parts import IMU, Mass, PositionSensor, Thruster

# Model.
drone = Craft("drone")
drone.add(Mass("body", mass=1.5, moi=(0.05, 0.05, 0.08)))
drone.add(Thruster("t", force=(0, 0, 1)))
drone.add(IMU("imu", gyro_noise_sigma=0.005, accel_noise_sigma=0.05,
              gyro_bias_sigma=1e-4))
drone.add(PositionSensor("gps", position_noise_sigma=0.02))

w = World().add_field(GravityField(g=(0, 0, -9.81)))
w.add_craft(drone, position=(0, 0, 5))

# Build the transforms, lower to native-Python.
sim = TargetNumpy(Sim(w))
ekf = TargetNumpy(EKF(w))

# Run. The sim runtime holds the state: mutate `sim.state`, step by dt.
sim.state["drone"]["t.throttle"] = 1.5 * 9.81       # hover
for t in np.arange(0, 3, 0.005):
    sim.step(0.005, t=t)                            # advance truth
    reading = sim.outputs()                         # sensor readings, this step
    ekf.predict(dt=0.005, t=t, u={"t.throttle": 1.5 * 9.81})
    ekf.update("imu.gyro", reading["drone"]["imu.gyro"])
    ekf.update("gps.position", reading["drone"]["gps.position"])

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

# 3-axis thrust makes position + velocity controllable with attitude
# frozen at the operating point. (A single-thruster craft regulates
# through attitude instead — see the quadcopter demo.)
drone.add(Thruster("tx", force=(1, 0, 0)))
drone.add(Thruster("ty", force=(0, 1, 0)))

lqr = TargetNumpy(LQR(
    w,
    x_ref={"drone": {"position": (0, 0, 10), "velocity": (0, 0, 0)}},
    u_ref={"t.throttle": 1.5 * 9.81},          # hover trim
    track=["drone.position", "drone.velocity"],
    Q=np.diag([10, 10, 10, 1, 1, 1]), R=np.eye(3), dt=0.02))

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

Lower to `TargetNumpy(EKF(w))` for Python or `TargetCpp(EKF(w), ...)`
for embedded.

### Backends

The `manta.codegen` package houses the lowering. Every transform emits
the same typed `Module` IR (state layout + named CasADi kernels + typed
entry points), and each backend implements exactly ONE generic lowering
of a Module — no per-transform code anywhere:

| Target | Accepts | Produces |
|---|---|---|
| `TargetNumpy(x)` | any Module / transform | `NumpyRuntime` — surface derived from the Module's shape (sim `step`/`outputs`, filter `predict`/`update`/`feed`/`step`, regulator `control`, recurrence `step`) |
| `TargetCpp(x, out_dir, class_name)` | any Module / transform | C++ static lib: typed Eigen class over flat-C kernels (+ CMake) |

Adding a backend (torch, JAX, raw embedded C) = a way to run/translate
a `ca.Function` plus one generic `Module` lowering. Adding a transform
(a future `iLQR`/`MPC`) reuses the shared `LinearizedSystem` and gets
every backend for free.

## Layout

```
manta/                     library package
    __init__.py            World / Craft / Sim / EKF / LQR / Target* surface
    craft.py               Craft + TickContext + inertial/wrench helpers
    world.py               World (the declarative model)
    sim.py                 Sim (forward-dynamics transform)
    recurrence.py          RecurrenceBlock base (PID/Madgwick/Mahony/IMU)
    linearized_system.py   LinearizedSystem — manifold-aware F/B/H/L +
                           slot/sensor/subset machinery (the shared seam)
    bus.py                 MeasurementBus + PortSet (backend-agnostic bus)
    signal.py              Signal value-channel + wire()
    tick/                  World-tick compile + kinematics/inertia/signature
    ir/                    Frames, types, Graph, Manifold, Wrench, Module
    parts/                 Part base + stock parts (sensor/actuation/aero/…)
    fields/                Field + Disturbance + stock + CraftWindBubble
    planets/               Planet base, Earth, PlanetFrameFluid, PlanetState
    couplings/             Coupling ABC + Tether
    estimation/            EKF + StateSpec + observability/NEES + filters
    control/               LQR + PID
    codegen/               Backends (one generic Module lowering per target)
        target.py          as_module — the backend entry-point contract
        numpy/             TargetNumpy + the one NumpyRuntime + NoiseDriver
        cpp/               TargetCpp + the generic module_emit emitter
tests/                     407 tests
examples/                  quickstart + physics/ + vehicles/
    _viz.py                rerun visualization helpers
    _control.py            keyboard (pynput) + scripted-fallback control
    quickstart.py          install sanity check (numpy Sim, apex height)
    physics/               bouncing_ball / spinning_top / foucault_pendulum
    vehicles/              quadcopter / airplane / submarine / hydrofoil
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
.venv/bin/python -m examples.vehicles.airplane           # control surfaces on Joint hinges
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
and 407 tests don't carry compat shims. The full deploy-to-robot path
lowers to C++ — `TargetCpp` handles `Sim`, `EKF` (mutable state + Joseph
update), and `LQR` (feed-forward control law), each verified against the
numpy backend by a compile-and-run roundtrip test. Open items:

- **`iLQR` / `MPC`** — `LinearizedSystem` emits symbolic A/B/H, so
  trajectory-tracking controllers reuse it; the iterative solve lives in
  the backend (not the IR), per the design.
- **Observability analysis** (shipped — `manta.estimation.observability`)
  — a faithful EKF of a correct model is still only as good as the model's
  *observability* (a property of dynamics + sensor set + **operating
  point**, not of the model alone): unobservable modes drift silently
  while the covariance looks tight. `observability(EKF(world))` builds
  the observability matrix from the
  symbolic `F`/`H` at an operating point and reports rank + which state
  slots are unobservable (+ an orthonormal **observable basis**). It flags,
  e.g., that GPS + DVL + gyro can't see absolute heading **at rest**. Local
  by nature; `observability_trajectory(world, dt=, steps=, control=)` rolls
  out a maneuver and reports the **union** of local observability over it —
  capturing observability-through-motion (that same heading *is* observable
  while the vehicle moves, rank 11→12).
- **NEES consistency check** (shipped — `manta.estimation.nees`) — the
  complement to observability: observability asks what you *can* estimate;
  NEES asks whether the filter's reported *covariance* is honest (a
  fully-observable filter can still be overconfident and quietly diverge,
  or conservative and waste information). `nees(world, dt=, steps=,
  control=)` runs a Monte-Carlo ensemble (truth jittered by the model's
  process noise, measurements by their R, the initial estimate drawn from
  P₀) and reports ANEES vs the χ² band. Pass `observable_basis=` (from an
  observability report) to check consistency only where the state is
  observable. **This settled the auto-`Q` question:** the *full-state* NEES
  reads overconfident only because the EKF shrinks covariance on the
  unobservable attitude; in the **observable subspace the filter is
  consistent**, and the auto-`Q` (`L·Σ·Lᵀ`) is exact for the dynamics-noise
  states — so it was left as-is (tightening it would have masked a sensor
  observability issue). The residual overconfidence on unobservable
  directions is the known EKF-inconsistency-on-unobservable-modes problem;
  FEJ / observability-constrained EKF is the principled fix (future).
- **EKF measurement timing** (fixed) — the filter's `step` folds a
  measurement *before* propagating over its interval (update-then-predict),
  because the sim emits sensor outputs from the interval's *start* state.
  The old predict-then-update order met a start-of-interval reading with
  the end-of-interval state, biasing rate-derived states (orientation) by
  O(dt) — a gyro-only EKF drifted heading where a naive integrator didn't.
  Fixing it collapsed that error to ~0 (submarine est error: 2.2 m peak →
  ~4 mm). `DVL` also gained a `velocity_noise` channel (it had none, so its
  EKF R was singular).
- **Multi-craft EKF over coupled worlds** — works for parallel
  independent crafts (block-decomposed predict is wired); field-mediated
  cross-craft coupling in the *estimator* is untested at scale.
- **Parameter tuning** — planned; the design is an
  offline IPOPT fit of `tunable` Parameters against logged trajectories.

## License

See `LICENSE`.

<p align="center">
  <a href="https://mantapilot.org">
    <img src="web/manta-logo.svg" alt="manta" width="360">
  </a>
</p>

<p align="center">
  <a href="https://mantapilot.org"><b>mantapilot.org</b></a>
</p>

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

<p align="center">
  <img src="web/pipeline.svg" alt="model (Quadcopter, Airplane, Submarine) → transform (Sim, EKF, LQR; model-free PID, Madgwick/Mahony) → targets (TargetNumpy, TargetJax, TargetCpp, …)" width="880">
</p>

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
(`regulate=`), freezing the rest; the runtime maps a state estimate to
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
    regulate=["drone.position", "drone.velocity"],
    Q=np.diag([10, 10, 10, 1, 1, 1]), R=np.eye(3), dt=0.02))

u = lqr.control(ekf.state_dict())              # {input_name: command}
```

(Q/R here are LQR *cost* weights, not the EKF's noise. A free rigid body
is underactuated, so full-state LQR isn't stabilizable — regulate the
controllable subspace via `regulate=`.)

## Core concepts

### Parts

A `Part` is an atomic unit of behavior on a craft. Declares at class
scope:

- `Parameter(default)` — frozen at construction, baked into the graph.
- `State(init, manifold="R1"|"R3"|SO3Manifold(...))` — mutable per-tick
  state; SO(3) slots carry an orientation (IMU integrators, attitude
  filters) with manifold-correct boxplus.
- `Input(default)` — per-tick user-supplied value (e.g. throttle).
- `Output()` — per-tick observable (sensor reading); shape is inferred
  from what the part writes into `PartUpdate.outputs`.
- `WhiteNoise(signal_manifold="R3", *, frame=None, sigma=...)` — per-tick
  i.i.d. Gaussian noise. `RandomWalkNoise(...)` — RW bias state
  (synthesizes its own state slot + driver input). Both subclass `Noise`.

Stock parts: `Mass`, `PointBuoy`, `Collider`, `Thruster` (polynomial
in throttle), `RevoluteJoint` and `PrismaticJoint` (1-DOF joints, with Mass children for
rotors), `DragSurface`, `Aerofoil` (Re-aware, with the `naca()` helper)
and `ControlSurface`, `IMU` (gyro+accel, with Kalibr 4-parameter noise
model), `VelocitySensor`, `Magnetometer`, `PositionSensor`, `Barometer`,
`TetherEndpoint`.

### Fields and Disturbances

Each `Field` (one of `GravityField`, `FluidField`, `MagField`,
`CollisionField`) is a typed superposition of `Disturbance` objects.
Disturbances combine via per-disturbance flags:

- `"additive"` (default) — linear sum (gravity, B-field, a current or
  thruster wake on top of a regime).
- `"averaged"` — membership-weighted mean among the averaged
  contributions (overlapping wind bubbles compromise on the mean).
- `"baseline"` — a regime medium (an ocean, an atmosphere). Baselines
  layer by spatial membership rather than summing
  (`base ← (1 − w)·base + w·value`), so "which fluid am I in" is an
  alpha-composite override, not a sum of 1025 + 1.225 kg/m³.

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

### System identification (Fit)

`Fit(world, parameters={...})` fits a model's physical parameters to
recorded data. Promotable `Parameter`s (those declared with a manifold:
thruster gains, every part's `transform` mount, `Mass.mass`) are
promoted from baked graph constants to a live parameter vector
(`Sim(world, parameters=[...])` → a `params` port on every kernel), and
the fit minimizes windowed prediction error against logged controls +
sensor readings, MAP-regularized by per-parameter Gaussian priors:

```python
fit = Fit(world, parameters={
    "body.mass":     Prior(sigma=0.05, log=True),   # weighed: ±5%
    "t1.force_quad": Prior(sigma=4.0),              # datasheet: loose
    "imu.transform": Prior(sigma=0.10),             # lever arm: ±10 cm
})
result = fit.solve(windows, weights={"imu.gyro": 1/σg**2,
                                     "imu.accel": 1/σa**2})
print(result.summary())     # fitted values + prior σ vs posterior σ
result.apply()              # bake fitted values back into the model
```

Gradients are exact (the oracle `step` kernel folded over each window
via `mapaccum`), IPOPT solves the NLP, and the Gauss-Newton posterior
`(JᵀJ + Σ₀⁻¹)⁻¹` reports which parameters the data actually informed:
`post/prior ≈ 1` means that number came from your prior, not the data,
and `result.weak_directions()` names the unidentifiable parameter
combinations (e.g. the thrust/mass scale). See
`examples/vehicles/sysid_drone.py` for the full recoverability demo.

Noise σ values are fit separately — a mean-prediction L2 loss has zero
gradient in σ. `NoiseFit(world, noise={...})` runs a symbolic Kalman
filter over the same `Window`s and minimizes the innovation
negative-log-likelihood (σ enters through the filter's `Q = LΣLᵀ` and
`R = L_hΣL_hᵀ`), fitting log-σ with relative priors:

```python
nres = NoiseFit(world, noise={"imu.gyro_noise": Prior(sigma=2.0),
                              "imu.accel_noise": Prior(sigma=2.0)})\
    .solve(windows)
nres.apply()      # EKF(world) now auto-builds Q/R from the fitted σ
```

### Backends

The `manta.codegen` package houses the lowering. Every transform emits
the same typed `Module` IR (state layout + named CasADi kernels + typed
entry points), and each backend implements exactly ONE generic lowering
of a Module — no per-transform code anywhere:

| Target | Accepts | Produces |
|---|---|---|
| `TargetNumpy(x)` | any Module / transform | `NumpyRuntime` — surface derived from the Module's shape (sim `step`/`outputs`, filter `predict`/`update`, regulator `control`, recurrence `step`) |
| `TargetCpp(x, out_dir, class_name)` | any Module / transform | C++ static lib: typed Eigen class over flat-C kernels (+ CMake) |
| `TargetJax(x)` | any Module / transform (flat crafts) | `JaxModule` — every kernel as a jitted JAX function + a `lax.scan` rollout you can `jax.grad`/`jax.vmap` through (needs `pip install jax`; not a core dependency) |
| `TargetWasm(x, out_dir, class_name)` | any Module / transform | browser bundle: the C++ backend's flat-C kernels behind a flat-double C ABI + Emscripten `build.sh`, a JSON descriptor, and an ES-module JS runtime (generic `Runtime.call` + typed `Sim`/`Filter`/`Regulator` views mirroring the numpy ones) |

`TargetJax` lowers by expanding each kernel to a CasADi SX instruction
tape and emitting equivalent JAX source (one line per scalar op) —
outputs match CasADi to machine precision, and `jax.grad` matches
CasADi jacobians exactly. Limitation: kernels must SX-expand, so
articulated (jointed) crafts — whose joint-space solve needs a
runtime-pivoting Linsol — stay on numpy/C++.

`TargetWasm` reuses `TargetCpp`'s exact math path (same densified flat-C
kernels) and adds only the marshalling glue, so the numbers match every
other backend bit-for-bit; it powers the live examples on
[mantapilot.org](https://mantapilot.org). The emitted JS dispatches purely
on the descriptor — no per-transform code — so `Sim`, `EKF`, and `LQR` all
get the same browser-ready surface.

Adding a backend (torch, raw embedded C) = a way to run/translate
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
    fit/                   Fit (MAP system ID) + NoiseFit (innovation-NLL σ fit)
    recurrence.py          RecurrenceBlock base (PID/Madgwick/Mahony/IMU)
    linearization/         LinearizedSystem (system) + TickLinearizer
                           (engine) + closure/partition + name helpers —
                           the shared seam every transform reads
    smoothing.py           Shared softened-norm / smooth-max primitives
    rates.py               RateGate + CommandLatch (loop-level rate gating)
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
        numpy/             TargetNumpy + NumpyRuntime engine + the four
                           views (_sim/_filter/_recurrence/_regulator)
        cpp/               TargetCpp + the generic module_emit emitter
        jax/               TargetJax (CasADi SX tape → jitted JAX source)
        wasm/              TargetWasm (flat-C kernels + C ABI + JS runtime)
tests/                     618 tests
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
.venv/bin/python -m examples.physics.spinning_top        # gyroscopic precession (RevoluteJoint)
.venv/bin/python -m examples.physics.foucault_pendulum   # Planet + Tether + Coriolis

# vehicles/ — visualized + keyboard (add --keyboard for live control)
.venv/bin/python -m examples.vehicles.quadcopter         # Sim + EKF + LQR closed loop
.venv/bin/python -m examples.vehicles.airplane           # control surfaces on RevoluteJoint hinges
.venv/bin/python -m examples.vehicles.submarine          # PointBuoy + VelocitySensor + EKF
.venv/bin/python -m examples.vehicles.hydrofoil          # nested-RevoluteJoint laser gimbal (PID)

# system identification — headless, no rerun needed
.venv/bin/python -m examples.vehicles.sysid_drone        # Fit + NoiseFit: thrust/mass/mounts/σ from IMU logs
```

Visualized demos need the rerun SDK (`.venv/bin/pip install rerun-sdk`); pass
`--no-viz` to run any of them headless. Vehicle demos take `--keyboard`
(live control: reads the launching terminal on WSL/headless boxes, a global
`pynput` listener on native desktops), `--no-viz`, `--duration`, and
`--viz-addr HOST[:PORT]` to stream to an already-running viewer (e.g. a
GPU-rendered Windows-native viewer from WSL). Shared helpers live in
`examples/_viz.py` (rerun) and `examples/_control.py` (keyboard / scripted).

## Running tests

```bash
.venv/bin/python -m pytest tests/
```

## Status

In active development. The public API (`World`, `Craft`, `Sim`, `EKF`,
`LQR`, `TargetNumpy`, `TargetCpp`) is settled enough that the demos
and 567 tests don't carry compat shims. The full deploy-to-robot path
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
  ~4 mm). `VelocitySensor` also gained a `velocity_noise` channel (it had
  none, so its EKF R was singular).
- **Multi-craft EKF over coupled worlds** — works for parallel
  independent crafts (block-decomposed predict is wired); field-mediated
  cross-craft coupling in the *estimator* is untested at scale.
- **Parameter tuning** — planned; the design is an
  offline IPOPT fit of `tunable` Parameters against logged trajectories.

## License

See `LICENSE`.

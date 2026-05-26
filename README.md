# manta

A Python-first, CasADi-backed rigid-body sim + EKF framework for small
vehicles — drones, rockets, underwater vehicles, satellites. You
declare the craft and its physics; manta builds the symbolic graph,
runs it natively in Python, and (when you're ready) lowers it to
embedded C++ via codegen.

## The pipeline

Three layers, explicit at every boundary:

```
Model            IR                  Target               Runtime
─────────────────────────────────────────────────────────────────
World        →   CompiledWorld   →   TargetNumpy(cw)  →   NumpyWorld
                  (CasADi tick)                            .step() / .initial_state()

EKF(world)   →   EKF             →   TargetNumpy(ekf) →   NumpyEKF
                  (sym predict +                            .predict() / .update()
                   sensor table)                            .state_dict() / .reset()

CompiledWorld          →           TargetCpp(cw, ...)  →   <basename>.cpp/.hpp
                                                            + flat-C kernels
                                                            + CMakeLists.txt
```

`World.compile()` and `EKF(world)` are pure compile-time. They produce
*descriptions* — CasADi function bundles, state specs, sensor tables —
that aren't directly callable. A `Target*` constructor lowers an IR to
a backend-specific runtime; today that's native-Python (`TargetNumpy`)
and Eigen-typed C++ (`TargetCpp`).

## Quick example

```python
from manta import World, Craft, EKF, TargetNumpy
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

# Compile to IR, lower to native-Python.
sim = TargetNumpy(w.compile())
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
TargetCpp(w.compile(), "out", class_name="Drone")
# → out/{drone.hpp, drone.cpp, drone_kernels.c/h, CMakeLists.txt}
```

## Core concepts

### Parts

A `Part` is an atomic unit of behavior on a craft. Declares at class
scope:

- `Parameter(default)` — frozen at construction, baked into the graph.
- `State(init, manifold="R1"|"R3")` — mutable per-tick state.
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
| `TargetNumpy(cw)` | CompiledWorld | NumpyWorld (`step`/`initial_state`) |
| `TargetNumpy(ekf)` | EKF | NumpyEKF (`predict`/`update`/`state_dict`/`reset`) |
| `TargetCpp(cw, out_dir, class_name)` | CompiledWorld | Buildable C++ static library |

Adding a backend (TensorFlow eager, raw embedded C, GPU CUDA) is one
new `Target*` constructor consuming the same IR.

## Layout

```
manta/                     library package
    __init__.py            World / Craft / EKF / Target* surface
    craft.py               Craft + TickContext + Newton-Euler integrator
    world.py               World + CompiledWorld (IR)
    coupled_tick.py        World-tick compile (one CasADi function per world)
    kinematics.py          Symbolic kinematic-chain pass
    inertia.py             Symbolic inertia rollup
    ir/                    Frames, types, Graph, manifold ops, Wrench
    parts/                 Part base + stock parts (sensor/actuation/aero/…)
    fields/                Field + Disturbance + stock + CraftWindBubble
    planets/               Planet ABC, Earth, PlanetFrameFluid, PlanetState
    couplings/             Coupling ABC + Tether
    estimation/            EKF (IR) + StateSpec + measurement helpers
    codegen/               Backends (one subpackage per target language)
        numpy/             TargetNumpy + NumpyWorld + NumpyEKF
        cpp/               TargetCpp + extract / kernels / wrapper / cmake
tests/                     274 tests
examples/                  9 runnable demos
```

## Demos

```bash
.venv/bin/python -m examples.hover_demo            # IMU + EKF, RW bias
.venv/bin/python -m examples.submarine_demo        # PointBuoy + DragSurface
.venv/bin/python -m examples.spinning_top_demo     # gyroscopic precession
.venv/bin/python -m examples.dual_craft_demo       # multi-craft + Tether
.venv/bin/python -m examples.quadcopter_demo       # polynomial Thruster
.venv/bin/python -m examples.glider_demo           # NACA airfoil
.venv/bin/python -m examples.bouncing_ball_demo    # CollisionField
.venv/bin/python -m examples.pan_tilt_gimbal_demo  # nested Joints
.venv/bin/python -m examples.coriolis_drop_demo    # Planet-frame init
```

## Running tests

```bash
.venv/bin/python -m pytest tests/
```

## Status

In active development. The public API (`World`, `Craft`, `EKF`,
`TargetNumpy`, `TargetCpp`) is settled enough that the 9 demos and
274 tests don't carry compat shims. Open items called out in the
audit:

- **Multi-craft EKF over coupled worlds** — works for parallel
  independent crafts today; field-mediated coupling (e.g. one
  craft's wake disrupting another's drag) works in sim and the EKF
  picks up the joint state, but multi-craft `block_decomposed`
  optimization isn't wired.
- **SO3 on user-declared Part state** — restricted to R1 + R3
  today; SO3 needs the dual-frame parametrization design.
- **`TargetCpp(ekf, ...)`** — EKF lowering to C++. The EKF IR
  carries the same per-sensor h/H bundles as the world tick; the
  C++ wrapper needs a mutable-state + Joseph-form-update layer on
  top.
- **`PointGravitySrc` Part** — users can call
  `gravity_field.add(PointMassGravity(...))` directly; the wrapper
  Part isn't worth a separate class today.

## Design docs

`prompts/` holds the design discussions that informed the current
shape — frame hierarchy, estimator design, field-bus combining,
disturbance state-carrier lift, etc.

## License

See `LICENSE`.

# manta

A C++ robotics simulation library with Python codegen and built-in autodiff
estimators (EKF / UKF). Designed for small-vehicle swarms — drones, rockets,
underwater vehicles, satellites — where you want to specify the craft once
and get the dynamics, telemetry, and estimator without hand-writing them.

The Python side describes the craft and the simulation; the C++ side runs
it. There's no runtime overhead on the C++ side — codegen produces plain
templated C++ that compiles into a regular binary.

## What's in here

```
include/manta/        C++ runtime: parts, fields, planets, scenes, EKF/UKF
python/manta_codegen/ Python descriptors + emitter
examples/             ex0..ex9 — increasing complexity
tests/                doctest unit + integration tests (199 cases)
```

## Quick example

A minimal craft with a 6-axis IMU and one thruster, publishing telemetry
on a single Zenoh topic and accepting throttle commands on another:

```python
# my_craft.py
from manta_codegen import Craft, World
from manta_codegen.parts import IMU, PointMass, Thruster

def make_world() -> World:
    body = PointMass("body", mass=1.0)
    imu  = IMU("imu", accel_sigma=0.05, gyro_sigma=0.005)
    thr  = Thruster("fwd", max_thrust=5.0, direction=(1, 0, 0))

    c = Craft("my_craft")
    c.add(body)
    c.add(imu)
    c.add(thr)

    # Bundled state topic — pose + sensor outputs + thrust.
    c.publish({
        "p": c.position,        "q": c.orientation,
        "v": c.vel_linear,      "w": c.vel_angular,
        "imu_accel": imu.last_accel,
        "imu_gyro":  imu.last_gyro,
        "throttle":  thr.throttle,
    }, "manta/my_craft/state")

    # Subscribe to throttle commands.
    c.subscribe(thr.set_throttle, "manta/my_craft/cmd")

    return (World()
            .add_craft(c)
            .run(dt=0.001, sim_rate_mult=1.0))
```

Run codegen:

```bash
PYTHONPATH=python/manta_codegen/src python -m manta_codegen.cli \
    my_craft.py --workflow binary
```

That writes `my_craft.hpp / .cpp / _main.cpp / _config.h / _telemetry.hpp /
.cmake` — a complete sim binary that publishes the state struct as JSON
over Zenoh and accepts throttle commands. Compile and run it.

## Core concepts

### Parts
Atomic unit of simulation. Each part exposes:
- A C++ class (`manta::parts::IMU`) with templated scalar (`IMUT<Scalar>`)
  for autodiff estimators.
- A Python descriptor (`IMU(...)`) that codegen reads to build the C++ craft.
- A signals table — values the user can publish or subscribe-into via
  bindings (e.g. `imu.last_accel`, `thruster.set_throttle`).

Stock parts: `PointMass`, `Mass`, `Hull` (buoyancy), `Surface1..4` (drag),
`PointBuoy`, `Thruster`, `PropThruster`, `GimbaledThruster`, `IMU`, `DVL`,
`Motor` (1-DOF revolute joint), `GravityPart`, `PointGravityPart`,
`TetherEndpoint`.

### Bindings
Each `c.publish(...)` and `c.subscribe(...)` records a `Binding`:
the (signal-or-struct → topic → encoding) mapping that codegen turns into
a Zenoh publisher/subscriber. Single-signal and bundled-struct bindings
share one path:

```python
c.publish(imu.last_accel)                       # one signal, default topic
c.publish({"a": imu.last_accel}, "topic")       # one-member struct
c.publish({"a": imu.last_accel,                 # multi-member bundle
           "g": imu.last_gyro}, "manta/state")
```

Direction is enforced (publishing a direction-in signal raises). All
members of a struct must agree on direction.

### World
Top-level container holding fields, planets, crafts (with their initial
state), and the sim loop config:

```python
earth = Earth(sea_level=0.0)
return (World()
        .add_planet(earth)
        .add_field(GravityField())
        .add_craft(c, on=earth, pos=(0, 0, -0.5))
        .run(dt=0.001, sim_rate_mult=1.0))
```

`on=planet` puts the craft's initial state in that planet's frame —
the scene anchors to the planet and the craft co-rotates with it. Use
`on=None` (default) for world-frame initial conditions.

Cartesian only — manta core doesn't know about LLA / orbital elements.
Convert on the user side:

```python
world.add_craft(drone, on=earth, pos=from_wgs84_lla(37.4, -122.1, 100))
```

(`from_wgs84_lla` lives in user code or a future optional `manta.geodesy`
helper. Core stays Cartesian.)

### Scalar templating
Most parts are templated on `Scalar` so they work under both `double` (sim)
and `ceres::Jet<double, N>` (autodiff Jacobians for the EKF). Set
`craft.scalar_templated = True` to opt in; the codegen emits the craft as a
class template with a `using FooCraft = FooCraftT<Real>` alias. Required to
plug into `manta::estimation::CraftEKF<MyCraftT, MeasDim>`.

### Estimators
Two flavors share an API:

- **`CraftEKF<MyCraftT, MeasDim>`** — Jet-based autodiff EKF. No
  hand-written process model; the same templated craft runs both for the
  value step (`double`) and the Jacobian step (`Jet`). Best when the
  craft's evaluate() is autodiffable and reasonably linear locally.

- **`CraftUKF<MyCraftT, MeasDim>` / `CraftUKFOf<PlainCraft, MeasDim>`** —
  sigma-point UKF. No autodiff, so the craft doesn't need to be
  scalar-templated — works on plain `manta::Craft` too. Captures
  second-order nonlinearity. 2N+1 evaluates per predict step vs. EKF's
  one-evaluate-plus-Jet-pass.

Same predict/update API on both. See `tests/test_ekf_against_sim.cpp` and
`tests/test_ukf.cpp` for end-to-end demos.

### Frame hierarchy
`World → Planet → Scene → Craft → Part`. Frames are static type tags
(`WorldFrame`, `PlanetFrame`, `SceneFrame`, `CraftFrame`, `PartFrame`)
that prevent accidental mixing — `Vec3<SceneFrame>` and `Vec3<PlanetFrame>`
are distinct types.

When the scene anchors to a rotating, accelerating planet, the integrator
injects all four pseudo-forces in the rigid-body equation:

- Translational: `-a_S` (scene origin's linear acceleration in world)
- Euler: `-α × r` (scene's angular acceleration × craft position)
- Centrifugal: `-ω × (ω × r)`
- Coriolis: `-2 ω × v`

All four are derived automatically from the kinematic-link composition
chain — the user just sets up the planet's rotation rate.

## Workflows

The codegen has two output modes:

- **`--workflow library`** — emits the typed Craft + telemetry + cmake
  fragment. The user provides their own `main.cpp` (e.g. for hand-tuned
  controllers, EKF wiring, multi-craft instantiation). Used by examples
  ex2, ex3, ex6 (with EKF), ex7 (with EKF on real Zenoh data), ex8
  (multi-craft).

- **`--workflow binary`** — additionally emits a complete `<name>_main.cpp`
  with Zenoh I/O wired through `craft.bindings`. Used by ex0, ex1, ex4,
  ex5, ex9.

## Examples

| Example | Workflow | Demonstrates |
|--|--|--|
| ex0 | binary  | 6-thruster free-flight, zero gravity |
| ex1 | binary  | 1km circular orbit, point gravity, 200× sim rate |
| ex2 | library | Quadcopter X-config, 4 PropThrusters, hand-tuned PIDs |
| ex3 | library | TVC rocket hopper, 2-axis gimbaled engine |
| ex4 | binary  | Reaction wheel — Motor + flywheel, conservation of L |
| ex5 | binary  | Underwater sub on Earth: Hull + IMU + DVL + sea surface |
| ex6 | library | Sim + EKF side-by-side, Pattern A (matching part names) |
| ex7 | library | Real-data-only EKF, fed from external Zenoh topics |
| ex8 | library | Multi-craft swarm: 3 drones tethered in a chain |
| ex9 | binary  | Minimal demo of the explicit Binding API |

## Building

```bash
cmake -S . -B build && cmake --build build -j
./build/tests/manta_tests
```

Each example has a CMake target named after its directory (e.g. `ex5`).
Some examples have `smoke_test.py` scripts that exercise the generated
binary over Zenoh.

## Status

Early/active development. The public API is stable enough for examples
ex0..ex9 to drive it without back-compat shims, but any of it can still
move. Notable items not yet built:

- Multi-craft codegen (today the user main instantiates the swarm; ex8
  is hand-written for that reason).
- System identification / parameter optimization layer on top of the
  autodiff infrastructure.
- Distributed Field backend over Zenoh (cross-process FluidField sharing).
- Position-dependent atmosphere/gravity models on Planet (J2 gravity,
  ECEF magnetics, etc.).

## Design docs

`prompts/` holds design discussions on planet anchoring, estimation
workflows, telemetry codec choice (JSON vs binary), and other decisions
the code reflects but doesn't explain in line.

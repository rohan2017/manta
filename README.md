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
examples/             ex0..ex11 + sync_smoke
tests/                doctest unit + integration tests (219 cases)
```

## Quick example

A minimal craft with a 6-axis IMU and one thruster, publishing telemetry
on a single Zenoh topic and accepting throttle commands on another:

```python
# my_craft.py
from manta_codegen import Craft, World
from manta_codegen.parts  import IMU, Mass, Thruster
from manta_codegen.fields import GravityField

def make_world() -> World:
    body = Mass("body", mass=1.0)              # auto-applies m·g if a GravityField is registered
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
            .add_field(GravityField(g=(0, 0, -9.81)))
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

Stock parts:
- **Structure**: `Mass` (with optional MOI tensor and auto-gravity),
  `PointBuoy` (single-point buoyancy), `Surface1..4` (N-th order
  velocity-driven force/torque tensors).
- **Actuator**: `Thruster1..4` — polynomial-in-throttle force/torque
  vectors (`F = Σ_k F_k · throttle^k`). `Thruster` is an alias for the
  common `Thruster1` linear case with a `(name, max_thrust, direction)`
  shorthand constructor.
- **Sensor**: `IMU`, `DVL`, `Magnetometer`.
- **Articulation**: `Motor` (1-DOF revolute joint, torque-controlled).
- **Coupling**: `TetherEndpoint`.
- **Field source**: `PointGravitySrc` — adds an inverse-square
  disturbance to the registered `GravityField` (for self-gravitating
  craft like asteroids or space stations).

Compositions replace some deleted parts: a gimballed thruster is
`Motor → Motor → Thruster1` (see ex3); a buoyant hull is a list of
`PointBuoy`s along the body axis (see ex5); a propeller with reaction
torque is `Thruster1` with explicit `F_1 = (0,0,K)` and
`τ_1 = (0,0,±k_t·K)` (see ex2).

### Fields
A field is a single concrete class per physics (`GravityField`,
`FluidField`, `MagField`) holding a list of additive `Disturbance`
objects. Each Disturbance is `(origin, lambda, lifetime, tag)`:

```python
g = (GravityField()
        .add_uniform((0, 0, -9.81))                          # flat-earth gravity
        .add_point_mass(mu=3.986e14, origin=(0, 0, 0))       # + Earth point-mass
        .add_point_mass(mu=4.9e12,   origin=(385e6, 0, 0)))  # + Moon
```

`state_at(query_pos)` walks all live disturbances and sums their
contributions. Lifetime defaults to one tick (re-add per tick to track
moving sources); `PERSISTENT` keeps a disturbance forever; finite N
gives a fixed-duration disturbance (e.g. an explosion shock).

Disturbances built by stock factories carry a `tag` and a serializable
`params` POD blob — they can be replicated cross-process (see Field
sync below). User-defined lambdas are local-only by default; register
a custom factory to make them syncable.

`FluidField` uses two pools: incompressible (`R = -1`, water-like) and
gas (`R = R_specific`, follows `p = ρRT`). On query, gas pool wins if
any gas disturbance is in influence and density is *derived* as
`ρ = p / (RT)`; otherwise the incompressible pool sums directly. Lets
a planet register both an ocean and an atmosphere on one field.

### Field sync (cross-process)
Set `field.synchronized = True` and codegen emits a Zenoh pub/sub on
`manta/<world>/field_<i>/disturbance`. Stock-tagged disturbances added
on one process replicate to all peers; the receiving side rebuilds the
same Disturbance via a per-Field factory registry. A thread-local
recursion guard prevents the rx side from echoing back to tx.

Wire layout is binary (`uint16 schema | uint16 tag | int32 lifetime |
uint8[96] params`, ~104 B/disturbance). User-defined kinds register
their own factories at `tag >= USER_BASE` (1024) and replicate on the
same path. See `tests/test_field_sync.cpp` and `examples/sync_smoke/`
for the wire round-trip.

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
earth = Earth(sea_level=0.0, gravity_mu=Earth.MU, include_j2=True)
return (World()
        .add_planet(earth)
        .add_craft(c, on=earth, pos=(0, 0, -0.5))
        .run(dt=0.001, sim_rate_mult=1.0))
```

`on=planet` puts the craft's initial state in that planet's frame —
the scene anchors to the planet and the craft co-rotates with it. Use
`on=None` (default) for world-frame initial conditions.

`Earth` automatically registers persistent ocean + atmosphere disturbances
on its FluidField; optional `gravity_mu`, `include_j2`, and
`dipole_moment` flags add point-mass(+J2) gravity and dipole magnetic
disturbances. Use `earth.height_above_surface(p)` /
`earth.height_above_sea_level(p)` for buoyancy / aero queries.

Cartesian only — manta core doesn't know about LLA / orbital elements.
Convert on the user side:

```python
world.add_craft(drone, on=earth, pos=from_wgs84_lla(37.4, -122.1, 100))
```

### Scalar templating
Most parts are templated on `Scalar` so they work under both `double` (sim)
and `ceres::Jet<double, N>` (autodiff Jacobians for the EKF). Set
`craft.scalar_templated = True` to opt in; the codegen emits the craft as a
class template with a `using FooCraft = FooCraftT<Real>` alias. Required to
plug into `manta::estimation::CraftEKF<MyCraftT, MeasDim>`.

For Jet-templated parts that query a Field, use
`fields::state_at_templated<Scalar>(field, pos)` (already wired into
`Mass`, `PointBuoy`, `Magnetometer`) — Real-Scalar takes the cast-only
fast path; Jet-Scalar finite-diffs through the field's spatial gradient
so EKF / system-ID Jacobians capture `∂g/∂pos` for orbital regimes.

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
  ex5, ex9, ex10, ex11.

## Examples

| Example | Workflow | Demonstrates |
|--|--|--|
| ex0 | binary  | 6-thruster free-flight, zero gravity |
| ex1 | binary  | 1km circular orbit, point gravity (synced over Zenoh) |
| ex2 | library | Quadcopter X-config, 4× Thruster1 with reaction torque |
| ex3 | library | TVC rocket hopper, yaw-motor → pitch-motor → engine stack |
| ex4 | binary  | Reaction wheel — Motor + flywheel, conservation of L |
| ex5 | binary  | Underwater sub on Earth: PointBuoy column + IMU + DVL |
| ex6 | library | Sim + EKF side-by-side, Pattern A (matching part names) |
| ex7 | library | Real-data-only EKF, fed from external Zenoh topics |
| ex8 | library | Multi-craft swarm: 3 drones tethered in a chain |
| ex9 | binary  | Minimal demo of the explicit Binding API |
| ex10| binary  | Multi-craft codegen — two Crafts in one binary |
| ex11| binary  | Codegen-driven Tether between two crafts |
| sync_smoke | — | Round-trips a GravityField disturbance through real Zenoh |

## Building

```bash
cmake -S . -B build && cmake --build build -j
ctest --test-dir build       # 220 cases (219 doctest + sync_smoke)
```

Each example has a CMake target named after its directory (e.g. `ex5`).
Some examples have `smoke_test.py` scripts that exercise the generated
binary over Zenoh.

## Status

Early/active development. The public API is stable enough for examples
ex0..ex11 to drive it without back-compat shims, but any of it can still
move. Notable items not yet built:

- A first-class system-ID API. The technique is demonstrated end-to-end
  in `tests/test_system_id.cpp` (mass fit + drag-coefficient fit on
  Surface1), but is not yet wrapped in a `ParameterFit<MyCraftT, NParams>`
  template — users currently write the Ceres residual functor by hand.
- ex8's hand-written 3-drone tether chain has not yet been migrated to
  the codegen-driven Tether API (ex11 demonstrates the pattern; ex8
  works fine as-is).
- Per-disturbance exact `state_at_jet` opt-in. Today the templated
  field query falls back to finite-diff for Jet scalars; an analytic
  override slot on Disturbance can be added when a consumer wants it.

## Design docs

`prompts/` holds design discussions on planet anchoring, estimation
workflows, telemetry codec choice, the disturbance redesign, and other
decisions the code reflects but doesn't explain in line.

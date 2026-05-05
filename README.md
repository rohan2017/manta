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
examples/             ex0 (free flight), ex1 (orbit), ex2 (quad), ex3 (TVC),
                      ex4 (reaction wheel), ex5 (sim+EKF), ex6 (real-data
                      EKF), ex7 (tethered pair), ex8 (submarine + Mag),
                      ex9 (block-decomposed dual-craft EKF), ukf_smoke,
                      connect_demo, sync_smoke, wire_debug
tests/                doctest unit + integration tests (231 cases)
```

## Quick example

The fastest "is this thing alive" check is `examples/ex0_free_flight` —
pure C++, no codegen, no Zenoh. Build, run, watch a drag vs. no-drag
comparison print to the terminal:

```bash
cmake -S . -B build && cmake --build build -j ex0_free_flight
./build/examples/ex0_free_flight
```

A more realistic codegen-driven example: a craft with a 6-axis IMU and
one thruster, publishing telemetry on a single Zenoh topic and accepting
throttle commands on another:

```python
# my_craft.py
from manta_codegen import (Craft, MantaConfig, Target, World,
                           publish, subscribe)
from manta_codegen.parts  import IMU, Mass, Thruster
from manta_codegen.fields import GravityField

def make_config() -> MantaConfig:
    body = Mass("body", mass=1.0)              # auto-applies m·g if a GravityField is registered
    imu  = IMU("imu", accel_sigma=0.05, gyro_sigma=0.005)
    thr  = Thruster("fwd", max_thrust=5.0, direction=(1, 0, 0))

    c = Craft("my_craft")
    c.add(body)
    c.add(imu)
    c.add(thr)

    w = (World()
            .add_field(GravityField(g=(0, 0, -9.81)))
            .add_craft(c))

    # Bundled state topic — pose + sensor outputs + thrust.
    publish({
        "p": c.position,        "q": c.orientation,
        "v": c.vel_linear,      "w": c.vel_angular,
        "imu_accel": imu.last_accel,
        "imu_gyro":  imu.last_gyro,
        "throttle":  thr.throttle,
    }, "manta/my_craft/state")

    # Subscribe to throttle commands.
    subscribe(thr.set_throttle, "manta/my_craft/cmd")

    return MantaConfig(targets=[
        Target("my_craft", drives=[w], dt=0.001, sim_rate_mult=1.0),
    ])
```

Run codegen:

```bash
PYTHONPATH=python/manta_codegen/src python -m manta_codegen.cli \
    my_craft.py --workflow binary
```

That writes `my_craft.hpp / .cpp / _main.cpp / _config.h / _telemetry.hpp /
.cmake` — a complete sim binary that publishes the state struct as JSON
over Zenoh and accepts throttle commands. The cmake fragment auto-emits
`add_executable(my_craft ...)` so the user just `include()`s it.

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
Module-level `publish(...)` / `subscribe(...)` record a `Binding`:
the (signal-or-struct → topic → encoding) mapping that codegen turns into
a Zenoh publisher/subscriber. The owning World is found by walking the
signal's `craft_ref._world` back-pointer, so the user never has to thread
a world handle through their setup. Single-signal and bundled-struct
bindings share one path:

```python
publish(imu.last_accel)                       # one signal, default topic
publish({"a": imu.last_accel}, "topic")       # one-member struct
publish({"a": imu.last_accel,                 # multi-member bundle
         "g": imu.last_gyro}, "manta/state")
```

Direction is enforced (publishing a direction-in signal raises). All
members of a struct must agree on direction.

In-process signal-to-signal piping uses `connect(out, in)` — no Zenoh,
no JSON, just a per-tick C++ copy step:

```python
connect(leader.thrust.throttle, follower.thrust.set_throttle)  # mirror cmd
connect(sim.dvl.last_velocity,  est.dvl.set_measurement)       # sim → est
```

The two signals can live in different worlds in the same Target (used
heavily by the sim+EKF shape — see ex5).

### World, Target, MantaConfig
A `World` holds fields, planets, crafts (with their initial state),
tethers, bindings, and connections. A `Target` wraps one or more
*drives* (a World, an EKF, or one of each in the same binary) plus the
sim cadence. A `MantaConfig` is a list of Targets — one C++ binary per
Target.

```python
earth = Earth(sea_level=0.0, gravity_mu=Earth.MU, include_j2=True)
w = (World()
        .add_planet(earth)
        .add_craft(c, on=earth, pos=(0, 0, -0.5)))

return MantaConfig(targets=[
    Target("my_craft", drives=[w], dt=0.001, sim_rate_mult=1.0),
])
```

`on=planet` puts the craft's initial state in that planet's frame —
the scene anchors to the planet and the craft co-rotates with it. Use
`on=None` (default) for world-frame initial conditions.

Three Target shapes are supported today:

- **`drives=[World]`** — standard sim. Codegen emits the typed crafts +
  a Zenoh main that drives `w.update()` per tick.
- **`drives=[EKF]`** or **`drives=[UKF]`** — pure estimator. The
  filter wraps its own internal World; codegen emits a
  `manta::estimation::WorldEKF<...>` (or `WorldUKFOf<...>`) main. EKF
  uses a fused `begin_step` / `add_update` / `end_step` lifecycle (one
  Jet world pass per tick covering both predict and every fresh
  measurement); UKF uses the legacy `predict(dt, Q)` + per-sensor
  `consume_fresh + update_n<N>` flow (Pattern C — real-data filter fed
  from external Zenoh topics, see ex6).
- **`drives=[World, EKF/UKF]`** — sim + filter in one binary, the two
  worlds ticked from one wall clock. Cross-world `connect()` pipes
  sim sensor outputs into the filter craft's `set_measurement` hooks;
  the sim's commanded throttle mirrors onto the est-side thruster so
  predict's force model matches (Pattern A — see ex5).

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

### Templated World, Scene, Craft
The whole sim core is templated on `Scalar`:

```cpp
template <class Scalar> class WorldT;        // World    = WorldT<Real>
template <class Scalar> class SceneT;        // Scene    = SceneT<Real>
template <class Scalar = Real> class CraftT; // Craft    = CraftT<Real>
template <class Scalar = Real> class PartT;
```

The `Real` instantiation (default `Real = float`) drives the sim. The
Jet instantiation (`ceres::Jet<double, N>`) is what `WorldEKF` runs for
the Jacobian step — the **same physics** is evaluated on Jet scalars,
so the state-transition Jacobian falls out of autodiff for free. No
hand-written process model.

This generalizes to inter-craft physics. Tethers, contacts, fluid
coupling — anything correct in the Real World is automatically correct
in the Jet World. Cross-craft Jacobian entries propagate naturally.

Wrapping a Craft in an `EKF(...)` or `UKF(...)` descriptor flips
`scalar_templated = True` on it automatically — users don't need to
think about codegen-shape details. The emitted craft becomes
`FooCraftT<Scalar>` with a `using FooCraft = FooCraftT<Real>` alias so
the codegen can instantiate it as `<double>` for the Real filter world
and `<Jet>` for the Jet shadow.

For Jet-templated parts that query a Field, use
`fields::state_at_templated<Scalar>(field, pos)` (already wired into
`Mass`, `PointBuoy`, `Magnetometer`) — Real-Scalar takes the cast-only
fast path; Jet-Scalar finite-diffs through the field's spatial gradient
so EKF / system-ID Jacobians capture `∂g/∂pos` for orbital regimes.

### Estimators

Two flavors share an API:

- **`WorldEKF<NumCrafts, MeasDim>`** — autodiff EKF. Wraps a
  `WorldT<double>` (the Real world the user feeds sensor measurements
  into) plus a `WorldT<Jet>` shadow (built identically; runs the Jet
  Jacobian step). State dim is `13 * NumCrafts` — the concat of every
  craft's 13-DOF rigid-body state.

- **`WorldUKF<NumCrafts, MeasDim>`** — sigma-point UKF. No Jet
  shadow — propagates each sigma point through the same Real
  `WorldT<double>::update()`. Captures second-order nonlinearity;
  `2*StateDim+1` World evaluations per predict.

Both expose per-craft slice accessors (`position(idx)`, `vel_linear(idx)`,
etc.). Single-craft callers can omit the index — defaults to craft 0.

The EKF uses a fused predict + update lifecycle that runs **one Jet
world pass per tick**, regardless of how many sensors fire (PyPose-style
linearize-at-`x_{k-1}` formulation; the chained `f` and `h` Jacobians
fall out of one autodiff pass):

```cpp
ekf.bind(w_jet, {&craft_real}, {&craft_jet});
ekf.set_state(...);
ekf.set_covariance(...);

// Per tick:
ekf.begin_step(dt, Q);                      // seed Jets at x_pre, evaluate
if (sensor.consume_fresh())
    ekf.add_update<N>(h_at_pre, z, R);      // read h(x_pre) + H from caches
ekf.end_step();                             // advance, P_pre = F P F^T + Q,
                                            // apply queued updates, mirror
```

For decoupled-craft swarms (no tether/contact/fluid coupling) flip
`block_decomposed=True` on the EKF descriptor: the codegen instantiates
`WorldEKFBlockDecomposed<...>` which runs `NumCrafts` smaller Jet passes
(width 13 each instead of `13·NumCrafts`), giving linear scaling.

UKF keeps the legacy `predict(dt, Q)` + `update_n<N>(h, z, R)` API —
no Jet shadow, no fused step.

### Estimator codegen

An EKF or UKF descriptor wraps a World and a list of measurement parts:

```python
ekf = EKF(est_world, measurements=[imu, dvl, mag])
ukf = UKF(est_world, measurements=[imu, dvl], alpha=1e-3, beta=2.0, kappa=0.0)
```

Both shapes share the same emit pipeline. The codegen-emitted harness:

- Builds the Real `WorldT<double>` + crafts + fields exactly like a sim
  World harness.
- For EKF, builds a Jet shadow `WorldT<Jet>` (identical setup; same
  field instances shared by-pointer between Real and Jet).
- Instantiates `WorldEKF<NumCrafts, MeasDim>` (or `WorldEKFBlockDecomposed`,
  or `WorldUKF<...>`) and binds it to both worlds + per-craft pointer
  arrays.
- For EKF: emits `begin_step / per-sensor add_update / end_step` brackets
  in `tick()`. For UKF: emits `predict + per-sensor update_n`.

Multi-craft worlds work natively. Each measurement part lives on a
specific craft; the codegen routes the per-sensor measurement update
through that craft's state slice (`13 * craft_idx + offset`). See
ex9 for a two-craft EKF (block-decomposed).

#### Per-craft initial state and variance

The EKF/UKF descriptors accept four shapes for every initial-state and
initial-variance knob:

| Form | Behavior |
|---|---|
| `None` (default) | State: inherit from `World.add_craft(c, pos=..., ori=..., vel=...)`. Variance: fall back to `initial_covariance`. |
| Single value | Broadcast to all crafts. |
| `list` of values, length = num_crafts | Positional per-craft (matches `world.crafts` order). |
| `dict` keyed by craft name | Override only the named crafts; others use the broadcast / world-default. |

```python
EKF(world, ...,
    initial_position=[(0,0,0), (5,0,0)],          # list: per-craft positional
    initial_attitude_var={"drone_0": 1e-9},       # dict: only drone_0
    initial_velocity_var=1e-2)                    # scalar: broadcast
```

#### Field requirements per part

Parts declare their field dependencies on the `PartDescriptor`
subclass via `requires_fields = [<FieldClass>, ...]`. Codegen validates
at config time (raises a friendly Python error before C++ compilation
starts); the C++ side carries a mirroring `MANTA_PART_REQUIRES_FIELD`
static_assert as defense-in-depth. Optional augmentations use
`MANTA_PART_AUGMENTS_FIELD(MANTA_HAS_<FIELD>)` inside `if constexpr (...)` —
the inactive branch fully compiles out, no field-registry traffic.

| Part | Required | Optional augmentation |
|---|---|---|
| IMU | — | GravityField (subtracts gravity to report specific force) |
| Mass | — | GravityField (gated by `apply_gravity` flag) |
| DVL | — | — |
| Magnetometer | MagField | — |
| PointGravitySrc | GravityField | — |
| PointBuoy | FluidField + GravityField | — |
| Surface | FluidField | — |

#### Per-sensor measurement functors

Per-sensor measurement-functor templates are built in for:

- **DVL** — `h(x) = R(q)^T · v_scene` read from the Jet sensor's
  `velocity_body()` accessor.
- **IMU** — `h(x) = [specific_force_body; ω_body]` read from the Jet
  IMU's `specific_force_body()` (accel includes gravity contribution
  when a GravityField is registered) and `angular_velocity_body()`.
- **Magnetometer** — `h(x) = R(q)^T · B(p_now)` with B captured
  pre-update from the registered MagField (locally-constant-B:
  ∂h/∂q exact, ∂h/∂p dropped).

R blocks are populated automatically from each part's noise sigmas.
EKF state slices (`position`, `orientation`, `vel_linear`,
`vel_angular`, plus per-component `*_stddev`) are exposed as
BoundSignals — route them through the same `publish/connect` paths
used for craft signals.

### Harness
`manta::Harness` is the polymorphic base class for everything that
drives a `WorldT`. The codegen emits a per-Target subclass:

```cpp
namespace manta_gen::ex1 {
    void setup(); void tick(); void shutdown();   // free functions
    struct Harness : public manta::Harness { /* delegates to free fns */ };
    extern Harness harness;
}

// User code:
manta::Harness& h = manta_gen::ex1::harness;
h.setup();
while (running) h.tick();
h.shutdown();
```

The free functions remain the hot-path entry points (inlinable, no
virtual dispatch); `Harness` is for plugin layers, runtime harness
swapping, or generic multi-Target binaries.

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
  fragment. The user provides their own `main.cpp` (for hand-tuned
  controllers, multi-craft instantiation, etc.). Used by ex2, ex3.

- **`--workflow binary`** — additionally emits a complete `<name>_main.cpp`
  with Zenoh I/O, in-process `connect()` steps, and (for EKF Targets) the
  full estimator predict/update loop. Used by ex1, ex4, ex5, ex6, ex7,
  connect_demo.

## Examples

| Example | Workflow | Demonstrates |
|--|--|--|
| ex0 | none (pure C++) | Drag-comparison freefall — minimal "manta in a terminal" demo |
| ex1 | binary  | 1 km circular orbit, point gravity (synced over Zenoh) |
| ex2 | library | Quadcopter X-config, 4× Thruster1 with reaction torque |
| ex3 | library | TVC rocket hopper, yaw-motor → pitch-motor → engine stack |
| ex4 | binary  | Reaction wheel — Motor + flywheel, conservation of L |
| ex5 | binary  | Sim + EKF side-by-side, fully codegen — cross-world `connect()` |
| ex6 | binary  | Real-data-only EKF, fed from external Zenoh topics (Pattern C) |
| ex7 | binary  | Codegen-driven Tether between two crafts |
| ex8 | binary  | Submarine — Mass + PointBuoy + Surface drag + 2× thruster + IMU/DVL/Mag, sim+EKF |
| ex9 | binary  | Dual-craft EKF — block-decomposed `WorldEKFBlockDecomposed<2, 18>`, per-craft Jet width 13 |
| connect_demo | binary | Two-thruster `connect()` mirror — minimal in-process binding demo |
| ukf_smoke | binary | Minimal UKF codegen smoke — Mass + IMU + DVL |
| sync_smoke | — | Round-trips a GravityField disturbance through real Zenoh |
| wire_debug | — | Pretty-prints field-sync disturbance bytes off the wire |

## Building

```bash
cmake -S . -B build && cmake --build build -j
ctest --test-dir build       # 232 cases (231 doctest + sync_smoke)
```

Each example has a CMake target named after its directory (e.g. `ex5`).
Some examples have `smoke_test.py` scripts that exercise the generated
binary over Zenoh.

## Status

Early/active development. The public API is stable enough for ex0..ex9
to drive it without back-compat shims, but any of it can still move.
Notable items not yet built:

- **Per-disturbance exact `state_at_jet` opt-in.** Today the templated
  field query falls back to finite-diff for Jet scalars; an analytic
  override slot on `Disturbance` can be added when a consumer wants it.
- **Templated `MagField`** for an exact `∂h/∂p` Jacobian in the
  Magnetometer EKF update. Today the codegen captures B at update-time
  and treats it locally constant — fine for typical magnetometer use
  cases, lossy near steep field gradients (close-in dipoles).

## Design docs

`prompts/` holds design discussions on planet anchoring, estimation
workflows, telemetry codec choice, the disturbance redesign, and other
decisions the code reflects but doesn't explain in line.

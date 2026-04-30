# manta — design plan (2026-04-29)

This supersedes `prompts/plan.md`, which is preserved for history but no
longer reflects the current architecture. It captures the design as it
stands today and lays out the next-stage development roadmap.

## What manta is

manta is a C++ library/framework for large-scale robotics swarm simulation
— satellite networks, autonomous tractors, drone swarms, undersea vehicle
fleets. The end user is a programmer who writes part/sensor models;
manta handles dynamics, kinematics, frame transforms, field interactions,
and (uniquely) generates EKF/UKF/MPC code from the same models the user
already wrote for sim. Spiritual ancestor: a "souped-up modular KSP" but
as a library, not an application.

The user-facing surface has two layers: a hand-written **C++ library** and
a **Python codegen** that consumes pure-Python craft descriptions and
emits C++. The codegen is optional — power users can stay in C++ — but
recommended, because it handles wiring (telemetry, Zenoh I/O, EKF state
layout, static-aggregate clustering) that's tedious by hand.

## High-level goals (recap from `prompts/my_prompt.txt`)

1. **Modular simulation.** Many crafts of varying types (boats,
   satellites, planes) in the same sim. When a single process gets too
   slow, crafts split across processes/machines while keeping the same
   Field abstraction (Zenoh-backed instead of in-process).
2. **State estimation and control from the same models.** Specify the
   craft once. Get a state-of-the-art EKF/UKF in a few lines via Ceres
   autodiff. Compiled C++ → fast, embedded-portable.
3. **Model optimization and system ID.** Real telemetry vs. simulated
   telemetry → optimize part parameters. Reuses the same autodiff
   infrastructure.
4. **Networked swarm system models.** Each robot in a swarm runs its own
   manta world model and adds its peers' models to it. Fields are shared
   over Zenoh; sensor disturbances detected by one robot raise Kalman
   confidence in nearby peers' readings of the same disturbance.
5. **Easy interface + full control if wanted.** Python codegen for
   convenience; raw C++ for power users who want to wire everything by
   hand.

## Non-goals

- A self-contained GUI/application. manta is a library + CLI codegen
  tool. Examples and tests are the only binaries shipped.
- Featherstone-style multi-body articulated dynamics. We model joints
  approximately (locked / saturating / passive — see Articulation).
- General-purpose collision detection. Approximate hull-vs-hull at most.
- Runtime YAML or DSL parsing. Configurations are Python → codegen → C++.

---

## Architecture overview

```
                          +-----------+
                          |   World   |  owns Fields, Planets, Scenes, SimClock
                          +-----+-----+
                                |
          +---------------------+---------------------+
          |                     |                     |
      +---+----+            +---+----+            +---+----+
      | Planet | (optional) | Planet | ...        | Scene  | (when no planet)
      +---+----+            +--------+            +---+----+
          |                                           |
       +--+--+        +-----+        +-----+          |
       |Scene|--------|Scene|--------|Scene|----------+
       +--+--+        +--+--+        +--+--+
          |              |              |
       +--+--+        +--+--+        +--+--+
       |Craft|        |Craft|        |Craft|   (each craft pinned to one scene)
       +--+--+        +-----+        +-----+
          |
        Part (RootPart) → Part → Part → ...   (n-ary tree)
```

### Core abstractions

- **Part** — a component of a craft (sensor, actuator, surface, point
  mass, hull). Implements `update()` to read fields and apply wrenches.
  Forms an n-ary tree rooted at the craft's `RootPart`.
- **CompositePart** — a Part with children. Aggregates child mass/MOI/COM
  and child wrenches.
- **ArticulatedPart** — a CompositePart whose joint introduces extra
  degrees of freedom (Motor, Gimbal, Hinge, Bearing). Resolves child
  wrenches into (joint acceleration, parent reaction). Not yet
  implemented.
- **Field** — a shared physical medium (gravity, fluid, magnetic, etc.).
  Exposes a query API and a contribute-disturbance API. Backends: in-
  process, Zenoh-distributed, networked-real.
- **Planet** — composite Field-disturbance source plus an optional
  rotating frame. Not yet implemented; design captured in `planets.md`.
- **Craft** — a vehicle. Owns a `RootPart`, knows its `Scene`, accesses
  fields via the world. The codegen emits a typed Craft subclass per
  user spec.
- **Scene** — a floating-origin frame shared by proximate crafts. When
  a craft drifts too far from its scene's centroid, it's promoted to its
  own scene. Inter-craft physical interactions resolve through the scene
  (or, for cross-scene, planet/world frame).
- **World** — the simulation root. Owns scenes, planets, fields,
  SimClock, drives the per-tick update.

### Two derived layers

- **Estimator** — wraps a Craft, runs an EKF/UKF/MPC against the same
  models. References the Craft externally; the Craft is estimator-
  agnostic. Compile-time mode (`MANTA_ESTIMATION_MODE`) flips the scalar
  type from `float` to `ceres::Jet<double, N>`; user part code is
  identical across modes.
- **Codegen** — Python-driven C++ generation. See "Python codegen".

---

## Build & integration

manta is consumed as a CMake dependency, not vendored. Today it builds
in-tree with `cmake -B build && cmake --build build`. End-user
integration via `find_package(manta)` or `FetchContent_Declare` will
land later; not load-bearing right now.

### Current directory layout

```
manta/
├── include/manta/
│   ├── geom/        # Frame typing, Vec3, Ori, Mat3, KinematicLink, StaticLink
│   ├── core/        # Part, CompositePart, RootPart, Craft, Scene, World,
│   │                # Wrench, Noise, types (Real)
│   ├── parts/       # grouped by role:
│   │                #   sensor/    — IMU, DVL, (GPS, Magnetometer planned)
│   │                #   actuator/  — Thruster, PropThruster, GimbaledThruster
│   │                #   structure/ — PointMass, Mass, Hull, Surface<N>,
│   │                #                PointBuoy
│   │                #   field_src/ — GravityPart, PointGravityPart
│   │                #   coupling/  — TetherEndpoint
│   │                #   articulation/ — Motor, TwoAxisGimbal (planned)
│   ├── fields/      # FluidField, UniformFluidField, OceanAtmosField,
│   │                # GravityField, PointGravityField
│   ├── coupling/    # Tether (free-standing inter-craft coupling)
│   ├── control/     # PID (small, for examples)
│   └── sim/         # SimClock
├── src/             # implementations of the above
├── examples/
│   ├── ex0_free_flight/
│   ├── ex1_orbit/
│   ├── ex2_quadcopter/
│   └── ex3_tvc_rocket/
├── tests/
├── python/manta_codegen/
│   └── src/manta_codegen/
│       ├── core.py            # Craft, PartDescriptor, FieldDescriptor
│       ├── parts/             # one descriptor per stock part
│       ├── fields/            # one descriptor per stock field
│       ├── emit/              # CMake / C++ / telemetry / main emitters
│       ├── _format.py         # cpp_float helpers
│       └── cli.py
└── prompts/
    ├── manta_plan.md   # this document
    ├── plan.md         # historical, superseded
    ├── planets.md      # Planet design note
    ├── stock_parts.txt # part wishlist
    └── ...
```

### CMake helper (planned form)

```cmake
manta_add_craft(my_drone
  CONFIG  crafts/my_drone.py
  WORKFLOW library                 # or "binary"
  PARTS   parts/MyCustomSensor.cpp # user-defined parts
)
```

This is the target API. The codegen package today emits the same
artifacts; the macro just hides the `add_custom_command` plumbing.

---

## Sim vs Estimation modes

A craft binary picks one mode at compile time. Today this is a manual
`MANTA_ESTIMATION_MODE` define; eventually `manta_add_craft` exposes it
as a `MODE` argument.

### Type aliasing

In `manta/core/types.hpp`:

```cpp
#ifdef MANTA_ESTIMATION_MODE
  using Real = ceres::Jet<double, MANTA_STATE_DIM>;
#else
  using Real = float;
#endif
```

`Real` is the project's scalar type. (Renamed from `mFloat` on
2026-04-28.) `MANTA_STATE_DIM` is set by codegen for the craft.

Only `core/` and the stock parts/fields use `Real` today. The geom layer
is already `template<Scalar = Real>` end-to-end; pushing `Scalar` through
`Part`, `Craft`, and stock parts is the eventual refactor that lets sim
and estimator share a process. **This refactor is not yet done** — see
"In-process sim+estimator" under "Open / planned".

### Noise

Noise is a first-class type, not a macro. Each part declares noise
sources as members:

```cpp
class IMU : public Part {
  Noise<WhiteGaussian> gyro_noise_;
  Noise<WhiteGaussian> accel_noise_;
  // future: Noise<RandomWalk> bias for IMU bias estimation
  ...
};
```

`operator+` dispatches on mode:
- **Sim**: samples and adds.
- **Estimation**: returns the value unchanged but registers the noise's
  contribution into a per-craft accumulator the framework reads after
  the update pass.

`RandomWalk`-flavored noise auto-registers as augmented EKF state (the
bias becomes an estimated quantity). `WhiteGaussian` registers into
measurement covariance R.

Noise types are user-extensible.

---

## Geometry — `manta::geom`

### Frame typing

Frames are **type tags**, not values. Mixing kinds is a compile error;
mixing identities of the same kind triggers a debug assertion via runtime
`FrameId`.

```cpp
namespace manta {
struct WorldFrame  {};   // inertial root
struct PlanetFrame {};   // (when planets land) planet-fixed, possibly rotating
struct SceneFrame  {};   // floating origin, scene-local
struct CraftFrame  {};   // craft body
struct PartFrame   {};   // part body
struct ParentFrame {};   // disambiguator for "the parent of this Part"
}
```

### Vec3, Ori, Mat3, Wrench

Thin wrappers over Eigen, frame-tagged at the boundary, with `.raw()` /
`from_raw()` escape hatches for inline math. All templated on `Scalar`
(default `Real`).

### KinematicLink and StaticLink

`KinematicLink<From, To>` carries position + orientation + linear
velocity (in From axes) + angular velocity (in To axes) + linear
acceleration (in From axes) + angular acceleration (in To axes). These
axis conventions matter — see "Frame-query convention" below for the
implications.

`StaticLink<From, To>` is the no-velocity variant for fixed mounts.

Composition rules (operator `*` overloads):

| LHS              | RHS              | Result            |
|------------------|------------------|-------------------|
| `StaticLink`     | `StaticLink`     | `StaticLink`      |
| `StaticLink`     | `KinematicLink`  | `KinematicLink`   |
| `KinematicLink`  | `StaticLink`     | `KinematicLink`   |
| `KinematicLink`  | `KinematicLink`  | `KinematicLink`   |

Naming convention: `world_to_craft`, `craft_to_part` (left-to-right).

### Integrator

`KinematicLink<F, T>::update(dt)` is a single-evaluation 2nd-order
symplectic-flavored integrator:

- Position: midpoint-velocity (`p += v*dt + 0.5*a*dt²`, exact for
  constant `a`).
- Velocity: explicit Euler.
- Orientation: exponential map at midpoint angular velocity.

**Not RK4.** Reasoning: the dynamics function is auto-diffable (Ceres
Jets), used to compute EKF Jacobians at 1 kHz with N≈30 augmented
states. RK4 would 4× the f-evaluations per tick → 4× autodiff cost on
the EKF hot path, for 4th-order local accuracy that's invisible vs.
process noise at dt=1ms. RK4 is also non-symplectic. Single-eval
symplectic-flavored matches what production robotics EKFs do (MSCKF,
ORB-SLAM, Forster IMU pre-integration). If long-horizon orbital sims
later need true symplectic accuracy, drop in Velocity Verlet (2 f-evals,
true 2nd-order symplectic) behind a compile-time switch.

---

## Frame hierarchy

```
World → [Planet →] Scene → Craft → Part [→ Part → ...]
```

The Planet level is **optional**. When no planet is added, scenes parent
directly to World. See `planets.md` for the full Planet design.

### Floating-origin scenes

To preserve `float` precision over kilometers:

- A scene's origin is rebased relative to its parent (planet or world)
  whenever its centroid drifts past a configurable threshold.
- Crafts in proximity share a scene. When a craft drifts >X km from its
  scene's centroid, it's promoted to its own scene automatically.
- Inter-craft / cross-scene interactions resolve through planet/world
  coordinates stored as `(int64_t km, float meters)` to preserve global
  precision.
- Local interactions between crafts in the same scene — fluid wakes, RF
  coupling, near-field forces — are computed in scene coordinates.

### Frame-query convention (locked 2026-04-29)

`Part::quantity<F>()` returns the kinematic quantity of the part **as
observed in frame F** — relative to F's origin and expressed in F's
axes. Therefore:

- `<SceneFrame>` → absolute, in scene axes.
- `<CraftFrame>` → relative to craft body, in craft axes (zero for static
  parts, joint-velocity for articulated parts).
- `<PartFrame>` → **trivially zero/identity** (a part is stationary in
  its own frame).

For the body-frame absolute quantities a sensor on the part physically
reads (DVL, accelerometer, gyro), use the dedicated methods:

- `Part::velocity_body()`
- `Part::acceleration_body()`
- `Part::angular_velocity_body()`
- `Part::angular_acceleration_body()`

All return `Vec3<PartFrame>`. **Do not** be tempted to "fix" the
`<PartFrame>` template specialization to mean "absolute in body axes"
— that mixes two different questions into one API and makes templated
code with frame parameters non-uniform.

### Planet level (when implemented)

`World → Planet → Scene → Craft → Part` when a planet is registered.
Integration always runs in `WorldFrame`; the kinematic pass projects
through `Planet`'s `KinematicLink<WorldFrame, PlanetFrame>` (which
carries the planet's rotation rate). Result: Coriolis / centrifugal
appear automatically in `acceleration_body()` and `angular_velocity_body()`
on Earth-anchored scenes, with no special-case in the IMU code.

`Part::field<F>()` and `Part::planet<P>()` mirror each other; a part
declares `requires_planet = Earth` (in Python codegen) when it needs
planet identity (GPS, geodetic helpers).

---

## Parts — `manta::core` and `manta::parts`

### Part base class

```cpp
class Part {
public:
  explicit Part(std::string name);
  virtual ~Part() = default;
  virtual void update() = 0;

  // Configuration.
  void set_mass(Real m);
  void set_moi(const Mat3<PartFrame>& moi);
  void set_com(const Vec3<PartFrame>& com);
  void set_transform(const StaticLink<ParentFrame, PartFrame>& tf);

  // Wrench application — accumulates within a single tick.
  void apply_force_at(const Vec3<PartFrame>& force,
                      const Vec3<PartFrame>& point = {0,0,0});
  void apply_torque(const Vec3<PartFrame>& torque);
  void apply_wrench(const Wrench<PartFrame>& w);

  // Kinematic queries — F-relative, F-axes.
  template<typename F> Vec3<F> position()              const;
  template<typename F> Vec3<F> velocity()              const;
  template<typename F> Vec3<F> acceleration()          const;
  template<typename F> Vec3<F> angular_velocity()      const;
  template<typename F> Vec3<F> angular_acceleration()  const;
  template<typename F> Ori<F>  orientation()           const;

  // Body-frame absolute (what an onboard sensor reads).
  Vec3<PartFrame> velocity_body()             const;
  Vec3<PartFrame> acceleration_body()         const;
  Vec3<PartFrame> angular_velocity_body()     const;
  Vec3<PartFrame> angular_acceleration_body() const;

  // Field access — typeid-keyed. Register the concrete instance under the
  // base interface slot for cross-craft polymorphism.
  template<typename FieldT> FieldT& field();

  // Context.
  Craft& craft();
  Part*  parent();
};
```

### CompositePart, RootPart

`CompositePart` owns children, aggregates their mass/MOI/COM via
`compute_params()`, and rigidly transforms+sums child wrenches up the
tree. `RootPart` is the topmost composite for a craft; aggregated
wrenches feed the rigid-body integrator there.

### ArticulatedPart (planned)

CompositePart whose joint adds DOFs. Three flavors:

- **Locked** — joint held at commanded position; full reaction → parent.
- **Saturating** — holds until reaction torque exceeds stall torque,
  then slips.
- **Passive** — free integration under whatever wrench it sees.

Each ArticulatedPart implements
`resolve(child_total) → (joint_accel, parent_wrench)`. Joint state is
registered with the craft's integrator at construction. Newton's third
applies: `parent_wrench` includes both constraint reaction and the
actuator's reaction torque. Concrete subclasses planned: `Motor`,
`TwoAxisGimbal`, `FreeHinge`, `Bearing`.

This is **not** Featherstone — too heavy for the autodiff/realtime
constraints. Targeted use cases: control-surface reaction torques,
gimbal counter-torques, articulating drone arms.

### Stock parts (today)

| Part                 | Required Fields           | Notes                                     |
|----------------------|---------------------------|-------------------------------------------|
| `IMU`                | (none)                    | gyro + accel, white noise, body-frame     |
| `DVL`                | (none)                    | body-frame velocity sensor                |
| `Thruster`           | (none)                    | thrust along local Z                      |
| `PropThruster`       | (none)                    | thrust + spin-up dynamics                 |
| `GimbaledThruster`   | (none)                    | thrust on commanded gimbal vector         |
| `Hull`               | `FluidField`              | sampled-points buoyancy/drag              |
| `PointMass`          | (none)                    | optional MOI                              |
| `Mass`               | (none)                    | full MOI tensor                           |
| `PointBuoy`          | `FluidField`,`GravityField`| single-point buoyancy                    |
| `Surface<N>`         | `FluidField`              | N velocity-power tensors (lift/drag)      |
| `GravityPart`        | `GravityField`            | applies gravity on the part               |
| `PointGravityPart`   | `GravityField`            | inverse-square gravity source             |
| `TetherEndpoint`     | (none, uses `Tether`)     | inter-craft tether anchor                 |

### Stock parts (planned, from `prompts/stock_parts.txt`)

`Motor` (ArticulatedPart), `TwoAxisGimbal` (ArticulatedPart), `GPS`
(needs Earth/Planet), `Magnetometer` (needs `MagField`), and others
listed in `stock_parts.txt`.

---

## Wrench aggregation

A part's `update()` may call `apply_force_at`, `apply_torque`, or
`apply_wrench` zero or more times. Calls accumulate into the part's net
wrench (force + torque about the part origin).

During the bottom-up aggregation pass:

- `CompositePart` rigidly transforms child wrenches into its frame and
  sums.
- `ArticulatedPart` calls `resolve()`, splitting child total into
  (joint acceleration, parent wrench).
- Wrenches reach `RootPart` and are converted to net force + torque
  about the **center of mass** (which differs from the geometric origin
  and may move with fuel burn etc.). This feeds the rigid-body
  integrator.

Reported craft position = geometric origin (RootPart frame), unaffected
by COM shifts.

---

## Crafts

```cpp
class Craft {
public:
  explicit Craft(std::string name);
  virtual ~Craft() = default;

  RootPart& root();
  Scene*    scene();
  void      set_scene(Scene&);

  template<typename FieldT> FieldT& field();

  // Three-phase per-tick update (called by Scene).
  void kinematic_pass();
  void sense_and_aggregate();
  void integrate(Real dt);
};
```

### Codegen-emitted Craft subclass

Each user craft definition produces a typed subclass with named accessors:

```cpp
// generated/my_drone.hpp — codegen output, do not edit
class MyDroneCraft : public manta::Craft {
public:
  MyDroneCraft();
  IMU&      imu();
  Thruster& fr();   // front-right motor
  Thruster& fl();
  // ...
};
```

Pointers index into the `unique_ptr` graph owned by `RootPart`. User
code accesses parts by name without runtime string lookups:
`drone.fr().set_throttle(0.7);`. The codegen emits these accessors from
the craft's Python descriptor.

### InitialState lives at the boundary

State is set at `Scene::add_craft` time, not on `Craft` construction:

```cpp
scene.add_craft(craft);                         // pristine, no state
scene.add_craft(craft, manta::InitialState{...}); // applies pos/vel/orientation
```

`InitialState` carries `Vec3<SceneFrame>` position, `Ori<SceneFrame>`
orientation, `Vec3<SceneFrame>` linear velocity, `Vec3<CraftFrame>`
angular velocity. **Do not** add position/velocity arguments to the
`Craft` constructor — codegen reads `Craft.initial_state(...)` from the
Python descriptor and emits the `add_craft(craft, InitialState{...})`
call.

---

## Fields — `manta::fields`

### Protocol

```cpp
class Field {
public:
  virtual void update() = 0;
  // Subclasses define query and disturbance APIs appropriate to the field.
};
```

Each field provides:

- A **query API** (`fluid.state_at(p)`, `gravity.field_at(p)`). Lock-free
  during the craft phase; field state is frozen.
- A **contribute API** (`add_disturbance`, `add_mass`). Calls during the
  craft phase are buffered in per-craft thread-local storage; the field
  drains and applies them during the field phase.

### typeid-based registration

`register_field<T>(f)` keys on `typeid(T)`, **not** the runtime type of
`f`. For polymorphic Fields (e.g. registering `OceanAtmosField` under
the `FluidField` slot), the user must explicitly register under the
base: `register_field<FluidField>(my_ocean)`. The registry does not
walk inheritance — keep it explicit.

### Backends — location-transparent

Every field is an interface with multiple backends:

- **In-process** — shared memory. Default for single-machine sims.
- **Zenoh-distributed** — for multi-machine sims.
- **Networked-real** — for live deployments where a fleet shares field
  state (e.g., a formation sharing inferred wind). Contributions come
  from peer crafts' inferences.

**Fields never ingest raw sensor data.** Sensor data always flows
through a Part. If a sonar return implies a hull is nearby, the *Part*
decides to call `field.add_disturbance(...)`. Keeps Field symmetric
across sim and live use.

### Compile-time vs runtime augmentation — runtime via `dynamic_cast`

Stock parts that adapt to whether a more-specific Field is registered
use the **runtime dynamic_cast** pattern, not compile-time templating
(which was rejected as ugly and asymmetric):

```cpp
void Hull::update() {
  auto& fluid = field<FluidField>();
  if (auto* ocean = dynamic_cast<const OceanAtmosField*>(&fluid)) {
    // augmented branch using sea-surface smoothstep
  } else {
    // base branch, uniform density
  }
}
```

Single registration under the base slot. RTTI is already on (used by
the FrameId system). Do **not** template a Part on its optional fields.

### Scene-local patches

For fields where local interactions dominate (FluidField, near-field
EM), scenes maintain a local patch in scene coordinates. Crafts in the
same scene get high-fidelity local coupling; cross-scene interactions
only see the bulk planet/world-frame state.

GravityField, MagField (when added) always resolve in planet/world
frame; no local patches.

### Disturbances and augmented EKF state

A `Disturbance` contributes to a field's state. Examples:
`PointMassDisturbance`, `WindDisturbance`, `ThrusterPlumeDisturbance`,
`RadioEmitterDisturbance`. Disturbances may declare augmented EKF
state with their own process model — same registration pattern as
parts:

```cpp
WindDisturbance wind{
  .process_model = RandomWalk{sigma: 0.1},
  .initial = {0, 0, 0},
};
fluid.add_disturbance(wind);
```

Three-axis wind becomes 3 augmented states with a random-walk process
model — the EKF estimates wind in real time.

---

## Planets (planned — see `planets.md`)

A `Planet` does three things:

1. Defines a `PlanetFrame` that may rotate/translate w.r.t. `WorldFrame`.
2. Registers disturbances on global Fields. Earth contributes ocean +
   atmosphere on `FluidField`, J2 gravity on `GravityField`, IGRF on
   `MagField`, sea-surface heights on `SeaSurface`.
3. Is itself queryable for parts that need planet identity (GPS,
   geodetic helpers, star tracker). `requires_planet = Earth` mirrors
   `requires_fields = [...]`.

When Planet lands, `OceanAtmosField` is removed — replaced by global
`FluidField` + Earth disturbance. `SeaSurface` becomes its own optional
Field.

Default world has no planets. Coriolis appears automatically when an
Earth-anchored scene is present, because integration always runs in
`WorldFrame` and the kinematic pass projects through Planet's rotating
link.

Implement when the first caller arrives — likely a magnetometer or a
Coriolis-correctness test for IMU. Until then, `OceanAtmosField` and
`UniformFluidField` cover existing use cases.

---

## Three-phase update for inter-craft consistency

Tethers exposed a subtle bug: if craft A's `update()` mutates state, and
craft B's `update()` reads it the same tick, B sees a stale or
half-updated value. Fix: split each craft's per-tick work into three
phases, run barrier-synchronously across all crafts in a scene:

1. **Kinematic pass** — propagate transforms top-down for every craft,
   populating `scene_to_part_` and `craft_to_part_` caches. No mutation.
2. **Sense + aggregate** — each part's `update()` runs (reads cached
   kinematic state from any craft, writes its own wrench accumulator).
   Inter-craft state queries land here.
3. **Integrate** — single combined state vector advances; all crafts'
   rigid-body + joint + augmented EKF states integrate together.

The Scene barrier between phases guarantees globally consistent reads.

**Implication for new Part types**: any cross-craft state query MUST
happen in `update()` (phase 2), not in helpers called outside it.

---

## Threading model and SimClock

One thread per craft. A tick has two phases separated by barriers:

```cpp
void World::update() {
  clock_.lock_dt();

  // Craft phase (parallel) — three-phase update internal to each scene.
  parallel_for(scenes_, [](auto& s){ s.update(); });

  // Field phase (single-threaded per field; fields run in parallel).
  parallel_for(fields_, [](auto& f){ f.drain_contributions(); f.update(); });

  clock_.advance();
}
```

- During the craft phase, field state is frozen; field queries are
  lock-free reads.
- Field contributions buffer in per-craft thread-local storage. The
  user writing a part doesn't think about threads — `add_disturbance`
  is fire-and-forget.
- During the field phase, each field drains buffers single-threaded.

### SimClock (across processes)

In single-process runs SimClock is a thread barrier. In Zenoh-
distributed runs it's a service:

1. Clock publishes `{tick: N, sim_time: T, dt: D}`.
2. Each craft runs its update for tick N.
3. Each craft publishes its field contributions tagged with tick N.
4. Each craft waits until it has received contributions from all peers
   for tick N (or the deadline expires).
5. Crafts assemble the final field view for tick N and signal ready.
6. Clock advances to tick N+1.

A configurable deadline lets the clock advance even if a peer is slow,
reusing the slow peer's previous-tick contributions for the missed
tick. Same `SimClock` interface backs in-process and Zenoh.

---

## Estimation — `manta::estimation` (planned)

### Estimator interface

```cpp
class Estimator {
public:
  virtual void declare_state(StateRef) = 0;
  virtual void predict(Real dt)        = 0;
  virtual void update(MeasurementVec, JacobianMat, NoiseCov) = 0;
  virtual StateView current_estimate() const = 0;
};
```

Concrete implementations:

- `EKF` — primary. Forward-mode autodiff via Ceres Jets, both process
  and measurement Jacobians.
- `UKF` — sigma-point alternative for highly nonlinear models.
- `MPC` — controller, not estimator; consumes an `Estimator`'s state.

### Sim + estimator in one process

Currently `Real` is a global typedef → only one scalar type per binary
→ estimator must be a separate binary. The right fix is templating
`Part`/`Craft`/stock parts on `Scalar` so a single binary holds
`Craft<float> sim` and `Craft<ceres::Jet<double, N>> est` in the same
World, communicating via in-process pointers. Geom is already templated;
core/ and stock parts/fields need it pushed through. Defer until codegen
generalizes; the templating refactor is its own workstream.

**Do not** build a sim ↔ estimator Zenoh bridge as a stopgap.

### Jacobian computation

Default: full Ceres Jets (forward-mode). Each per-tick evaluation
produces value + Jacobian in one pass. Future optimization (not v1):
sparse Jets restricted to active dimensions per evaluation.

---

## Coupling — `manta::coupling`

Inter-craft physical interactions don't multi-parent Parts (one Part →
one Craft is invariant). Instead, a free-standing **coupling object**
holds shared state and communicates via per-craft endpoint Parts.

### Tether

`coupling::Tether` holds spring/damper params and pointers to two
`parts::TetherEndpoint` instances. Each endpoint computes "force on
self" symmetrically in scene frame, rotates to part frame, applies via
`apply_force_at`. Equal-and-opposite forces fall out of the geometry.

Same pattern generalizes to other inter-craft couplings (rigid contact,
magnetic latch, propwash interaction). Write a coupling object + its
endpoint Part(s); don't multi-parent.

---

## Optimizations

### Static-aggregate clustering (planned)

Parts whose mass/MOI/transform don't change at runtime are flagged
static in the Python descriptor. The codegen analyzer walks the part
tree, finds maximal static subtrees, and replaces each with a hidden
`StaticAggregate` `CompositePart` whose mass/COM/MOI tensor is pre-
baked. Static aggregates are leaves in the dirty tree and never trigger
`compute_params` recomputation.

User never calls `mark_static()` from C++. Codegen handles it from the
descriptor. Significant for crafts with hundreds of parts.

### Eager kinematic cache (today)

The kinematic pass populates `scene_to_part_` / `craft_to_part_` for
every part at the start of each tick. Queries like
`position<SceneFrame>()` read the cache. Cheaper in autodiff mode where
lazy + Jets means redundant Jacobian-chain evaluation.

### Flat part array (planned)

At codegen time, the part tree also emits as a flat array indexed by
part ID with parent indices. The two-pass update becomes two cache-
friendly linear scans. Significant for the hot loop with hundreds of
crafts.

### Dirty tracking for `compute_params`

`set_mass`, `set_moi`, `set_transform` mark the part dirty; recompute
runs at most once per tick before integration.

---

## Python codegen — `python/manta_codegen`

### Workflow

1. User writes `crafts/my_drone.py`, building a `Craft` spec
   imperatively.
2. CMake (or the user) invokes `manta_codegen --input crafts/my_drone.py
   --output generated/`.
3. Codegen `import`s the user's module.
4. Analyzer runs (eventually): static-aggregate clustering, EKF state
   layout, validation of `requires_fields` and `requires_planet`.
5. Emitter writes:
   - `<name>.hpp/.cpp`             — typed Craft subclass.
   - `<name>_config.h`             — feature-test macros, force-included
                                     via the CMake fragment.
   - `<name>_telemetry.hpp`        — telemetry struct + JSON encoder
                                     (per-part nested keys).
   - `<name>_main.cpp`             — only if `workflow="binary"`,
                                     contains the sim main with Zenoh
                                     I/O.
   - `<name>.cmake`                — fragment the user's CMake includes.
6. User's CMake target builds the generated C++ alongside their part
   sources.

### Two workflows

- **`library`** — emits only the typed Craft type. Zero Zenoh
  dependency. The library workflow's CMake fragment must not leak
  Zenoh into the user's link line.
- **`binary`** — emits a runnable sim binary with Zenoh telemetry
  publishing and command subscription. The binary workflow self-
  bootstraps Zenoh via `FetchContent` inside an idempotent
  `if(NOT TARGET zenohcxx::zenohc)` guard.

Estimation is **orthogonal** to the workflow choice — same Craft
compiles in `MANTA_ESTIMATION_MODE` either way.

### Pure-Python descriptors

Stock part / field descriptors are real Python classes with `emit_*`
methods, not data containers:

```python
class IMU(PartDescriptor):
    cpp_class  = "manta::parts::IMU"
    cpp_header = "manta/parts/imu.hpp"

    def __init__(self, name, accel_sigma=0.0, gyro_sigma=0.0, **kw):
        super().__init__(name=name, **kw)
        self.accel_sigma = float(accel_sigma)
        self.gyro_sigma  = float(gyro_sigma)

    def emit_constructor_args(self) -> str:
        return f'"{self.name}", manta::parts::ImuNoiseParams{{...}}'

    def telemetry_fields(self) -> list[tuple[str, str]]:
        return [("accel", "manta::geom::Vec3<manta::PartFrame>"),
                ("gyro",  "manta::geom::Vec3<manta::PartFrame>")]

    def emit_telemetry_reads(self) -> list[tuple[str, str]]:
        return [("accel", f"craft.{self.name}().last_accel()"),
                ("gyro",  f"craft.{self.name}().last_gyro()")]

    def render(self, telemetry, path) -> None:
        # rerun.io drawing, owned by the descriptor.
        ...
```

Why pure Python over a JSON DSL: users get inline calculations, for
loops, factory functions for craft variants. The descriptor *is* the
spec; users `import` it and call it.

### Per-part Zenoh I/O

Each part has `publish_state`, `state_topic`, `subscribe_command`,
`command_topic` knobs in the descriptor. Codegen emits typed publishers
and subscribers — no string lookups at runtime.

### Telemetry JSON

Nested per-part: `{"tx_p": {"throttle": 0.5}}` — not flat
`{"tx_p_throttle": 0.5}`. Easier to parse, plays well with rerun.io.

### Per-craft sim config

The Python descriptor declares `Craft.sim_config(dt=..., sim_rate_mult=...)`
which the binary workflow's main respects. Different crafts can run at
different physical rates within the same sim.

### Constraints on craft Python files

The Python file must build and return a spec object. No import-time
side effects (no threads, no sockets, no global state mutation).
Codegen executes it once per build.

### C++ ↔ Python descriptor sync

Currently the C++ part class and its Python descriptor are two
independent sources of truth. Mismatches are detected at compile time
(missing class / wrong constructor signature). When stock parts
proliferate, automate this with macros so the C++ class declares its
descriptor metadata and the Python descriptor reads it. **Defer until
the part list is large enough to make manual sync error-prone.**

---

## Conventions

### Naming

- `world_to_craft`, `craft_to_part` for KinematicLink/StaticLink
  instances — names read left-to-right.
- C++ classes `PascalCase`; methods and free functions `snake_case`.
- Frame tags as type aliases (`WorldFrame`, `CraftFrame`, ...).
- `Part` and `Field` are deliberately kept (not renamed to
  `MantaPart`/`MantaField`); the `manta::` namespace handles
  disambiguation.

### Frame access

Always templated, never suffixed:

```cpp
auto v = part.velocity<SceneFrame>();   // good
auto v = part.velocity_sf();            // not in the API
```

The single exception is the body-frame absolute family:
`velocity_body()`, `acceleration_body()`, `angular_velocity_body()`,
`angular_acceleration_body()`. These are *different questions* than the
templated queries (sensor reading vs. F-relative quantity), so they get
their own names; the `_body` suffix groups them under the `velocity*`
autocomplete prefix.

### Numeric types

- `Real` — primary scalar, mode-dispatched (`float` in sim,
  `ceres::Jet<double, N>` in estimation).
- Plain `float`/`double`/`int` — only for non-physics quantities.
- `Vec3<F>`, `Mat3<F>`, `Ori<F>`, `Wrench<F>` — frame-tagged at API
  boundaries.
- Inside `update()`: free use of Eigen via `.raw()`/`from_raw()` at the
  boundary.

### Modules

- `manta::geom` — frame typing, vectors, orientations, links.
- `manta::core` — Part, CompositePart, RootPart, ArticulatedPart,
  Craft, Scene, World, Wrench, Noise, types (`Real`).
- `manta::parts` — stock parts.
- `manta::fields` — stock fields.
- `manta::coupling` — multi-craft couplings (Tether).
- `manta::control` — small control utilities (`PID`).
- `manta::sim` — SimClock, Zenoh barriers.
- `manta::estimation` — Estimator, EKF, UKF (planned).

---

## Current state (2026-04-29)

### Implemented and tested (153/153 passing)

- Geometry layer end-to-end: `Vec3`, `Ori`, `Mat3`, `KinematicLink`,
  `StaticLink`, frame typing.
- Core: `Part`, `CompositePart`, `RootPart`, `Craft`, `Scene`, `World`,
  `Wrench`, `Noise`, `Real` typedef.
- Three-phase Scene update.
- Stock parts listed in the Parts table above.
- Stock fields: `FluidField`, `UniformFluidField`, `OceanAtmosField`,
  `GravityField`, `PointGravityField`.
- Coupling: `Tether` + `TetherEndpoint`.
- Control: `PID` (small).
- Body-frame absolute accessors (`velocity_body()`, etc.), frame-query
  convention locked.
- Python codegen vertical slice — descriptors, two emit workflows,
  CMake fragments, per-part Zenoh I/O, nested telemetry JSON,
  feature-test macro emission.
- All 4 examples (ex0_free_flight, ex1_orbit, ex2_quadcopter,
  ex3_tvc_rocket) build via codegen and pass smoke tests.

### Known deferred / planned (priority ordered)

1. **Pilot-loop end-to-end smoke test** for ex0 (sim binary + Python
   viewer + Python keyboard controller via Zenoh + rerun.io).
   Currently blocked on Python tooling install (`zenoh`, `rerun-sdk`).
2. **`ArticulatedPart` + `Motor` + `TwoAxisGimbal`.** Unblocks gimbal
   reaction torques, motor counter-torques, articulated arms. Add the
   symmetric-COM articulation invariant test as physics validation
   (see `project_manta.md`).
3. **`Planet` + `Earth`.** Full design in `planets.md`. Trigger:
   first caller (magnetometer, GPS, or Coriolis-correct IMU test).
   Unblocks `GPS`, `Magnetometer` (also needs `MagField`), and
   eliminates `OceanAtmosField` in favor of disturbance-on-`FluidField`.
4. **`Hull` migration to feature-test macros** (`MANTA_HAS_OCEAN_ATMOS_FIELD`)
   once codegen ships and a hull example exists. Removes the
   `dynamic_cast` for the augmented branch.
5. **Scalar templating refactor** — `Part<Scalar>`, `Craft<Scalar>` to
   support sim+estimator in one process. Geom layer is already
   templated; push through core/ and stock parts.
6. **Static-aggregate clustering** in the codegen analyzer. Wait until
   crafts have enough static parts that the speedup is measurable.
7. **Flat part array emission** for cache-friendly hot loop.
8. **Estimator stack** — `EKF`, then `UKF`. Wait until at least one
   non-trivial craft has a clear estimation use case (e.g. an underwater
   vehicle with IMU + DVL fusing depth + heading).
9. **Augmented-state registration plumbing** — auto-registering
   `RandomWalk` noise as EKF state. Tied to Estimator stack landing.
10. **Floating-origin scene rebasing + auto-promotion.** Wait until a
    sim actually drifts far enough to lose precision.
11. **Zenoh-distributed FieldBackend** for cross-machine sims.
12. **MPC** (`Controller` interface sharing Jets infrastructure with
    Estimator).
13. **C++ ↔ Python descriptor metadata sync via macros.** Wait until
    stock part list is large enough to make manual sync painful.
14. **Sparse Jets** for Jacobian acceleration. Profile first; only
    revisit if it shows up in a hot path.

### Unresolved design questions

- **Part ID assignment** (compile-time codegen vs runtime `add()`).
  Affects whether `FrameId` can be `constexpr`.
- **Wrench API for impulses** — current API is force-over-time. Add
  `apply_impulse` if collision modeling needs it.
- **Field migration on craft → new-scene promotion.** What happens to
  the disturbances the craft contributed to its old scene? Likely:
  carry them with the craft into the new scene.
- **MPC integration.** Needs symbolic gradients of dynamics w.r.t.
  inputs as well as state. Probably a parallel `Controller` interface
  reusing Jets.
- **Determinism guarantees.** Every random source must be seedable per
  craft. Need a `RandomSource` abstraction passed via context, not
  pulled from globals.
- **EMField design.** Out of scope for v1; revisit when a comms-link
  modeling use case arrives.

---

## Next-stage roadmap (in execution order)

The following is the recommended order for the next month of work,
biased toward "minimum work to make manta useful for a real project."

### Phase 1 — close the pilot loop

- Install Python deps (`zenoh`, `rerun-sdk`); finish ex0 end-to-end
  smoke test.
- Confirm controller → sim → viewer round-trip works on all 4 examples.
- Document the workflow in a short `README` or inline at the top of
  each example.

### Phase 2 — articulation

- `ArticulatedPart` base class + integrator-state registration.
- `Motor` (1-DOF rotation) and `TwoAxisGimbal` (2-DOF). Stock-parts
  table grows.
- Symmetric-COM articulation invariant test (queued per
  `project_manta.md`).
- Update ex3 (TVC rocket) to use a real `TwoAxisGimbal` instead of a
  scalar gimbal angle parameter. Confirms reaction torques are correct.

### Phase 3 — estimation foundation (non-blocking, can interleave)

- Scalar templating refactor: `template<class Scalar = Real>` on
  `Part`, `Craft`, and stock parts/fields. One example craft templated
  end-to-end as a proof.
- `Estimator` base interface + `EKF` skeleton (no Jacobians yet).
- First measurement update on a synthetic problem (sim + estimator in
  one process).

### Phase 4 — first real EKF demo

- Underwater vehicle example (IMU + DVL + depth + heading) running a
  full EKF in-process. `RandomWalk` noise on IMU bias becomes
  augmented state.
- This exercises the autodiff Jets path and the augmented-state
  plumbing simultaneously.

### Phase 5 — planets

- `Planet` base class, `Earth`, frame plumbing.
- `OceanAtmosField` removed; ocean + atmosphere reborn as Earth
  disturbances on global `FluidField`.
- `SeaSurface` field; `Hull` augmented branch driven by it.
- `GPS` part. `Magnetometer` part if `MagField` lands here.
- Coriolis correctness test on a stationary IMU at the equator.

### Phase 6 — distributed runs

- Zenoh-backed `FieldBackend` for at least `FluidField` and
  `GravityField`.
- SimClock service deployment.
- Two-process example where two crafts in separate processes share a
  fluid field.

### Phase 7 — performance

- Static-aggregate clustering analyzer in the codegen.
- Flat part array emission.
- Profile-driven decisions on sparse Jets, threading granularity.

Phase 1 is the hard near-term blocker. Phases 2 and 3 can run in
parallel once 1 lands. Phases 4–7 are reactive — pick them up when a
real use case surfaces.

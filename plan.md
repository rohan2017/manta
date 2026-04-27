# mantapilot — design plan

## Overview

mantapilot is a C++ library for large-scale robotics swarm simulation — satellite networks, autonomous tractors, drone deliveries, autonomous submarines. It provides a framework for modeling crafts as compositions of parts that interact with shared fields (gravity, EM, fluid, etc.). The end user is a programmer who writes custom part and field models; the library handles kinematics, dynamics, integration, frame transforms, field synchronization, and (uniquely) the autodiff machinery needed to reuse the same models for state estimation and control.

Spiritual reference: a "souped-up modular Kerbal Space Program," but as a library, not an application.

## Goals

- **Hundreds of crafts in real-time** on commodity hardware. Approximate where needed; exact where cheap.
- **Single source of truth for models.** A part written once works in (a) pure simulation with synthetic noise, and (b) state estimation / control on a real craft via Ceres autodiff, with no edits to the part's update logic.
- **Modular and extensible.** Users add custom parts and fields from their own project tree. The library is integrated as a CMake dependency; user code lives outside.
- **Scales from desktop sim to distributed runs.** In-process threaded execution by default; Zenoh-backed field synchronization opt-in for cross-machine simulations.
- **Approximate articulation.** Joints, gimbals, and motors model counter-torques and reaction forces without a full Featherstone-style multi-body algorithm.
- **Floating-origin coordinate handling** so simulations stay accurate over kilometers without losing `float` precision.

## Non-goals

- Self-contained executable application (mantapilot is a library; examples and tests are the only binaries shipped).
- High-fidelity articulated multi-body dynamics (recursive Newton-Euler / Featherstone).
- General-purpose collision detection / physics solver. Collisions are modeled approximately.
- Runtime YAML or DSL parsing — craft configurations are expressed in Python, processed by codegen, compiled to C++.

---

## Architecture overview

Four core abstractions:

- **Part** — a component of a craft (sensor, actuator, surface, mass). Implements an `update()` that computes wrenches and reads fields. Forms an N-ary tree rooted at the craft's `RootPart`.
- **Field** — a shared physical medium that multiple crafts interact with (gravity, EM, fluid). Exposes query and disturbance APIs. Backend may be in-process, Zenoh-distributed, or fed by real sensor inferences.
- **Craft** — a collection of parts forming a vehicle. Owns its `RootPart`, references its `Scene`, and has access to the `World`'s fields.
- **World** — the simulation root. Owns scenes, fields, the `SimClock`, and orchestrates the per-tick update.

Two derived layers:

- **Estimator** — wraps a Craft, runs an EKF / UKF / etc. against the same models. References the Craft externally; the Craft itself is estimator-agnostic.
- **Codegen** — a Python tool that consumes user Python craft definitions + part descriptors and emits C++ crafts, state layouts, and static-aggregate clustering.

The simulation is **two-mode**: at C++ compile time, a craft binary is built either in **Sim mode** (numeric types are plain floats; noise is sampled and added) or **Estimation mode** (numeric types are `ceres::Jet`; noise is unsampled but registered as covariance). The user's part code is identical across modes; the type system and noise machinery dispatch.

---

## Build & integration

mantapilot is consumed as a CMake dependency, not vendored. `find_package(mantapilot)` for installed builds, `FetchContent_Declare` for pinned versions.

### Recommended directory layout

```
mantapilot/                          # the framework, separately installable
├── include/mantapilot/
│   ├── geom/                        # Frame typing, Vec3, Ori, KinematicLink, StaticLink
│   ├── core/                        # Part, CompositePart, RootPart, ArticulatedPart,
│   │                                # Craft, World, Wrench, Noise — abstractions
│   │                                # everything else builds on
│   ├── parts/                       # stock part models (IMU, Surface, Thruster, PointMass, ...)
│   ├── fields/                      # stock field models (Gravity, EM, Fluid)
│   ├── estimation/                  # Estimator interface, EKF, UKF
│   └── sim/
│       ├── dynamics/                # integration, wrench aggregation
│       └── sync/                    # SimClock, Zenoh barriers
├── src/                             # implementations
├── CMakeLists.txt
└── codegen/                         # Python package, pip-installable
    └── mantapilot_codegen/
        ├── descriptors.py           # Part/Field/Craft spec base classes
        ├── analysis.py              # static-aggregate clustering, EKF state layout
        └── emitter.py               # writes generated C++

user-project/
├── crafts/
│   └── my_drone.py                  # craft definition, built via mantapilot_codegen
├── parts/
│   ├── MyCustomSensor.{hpp,cpp}     # update() logic
│   └── MyCustomSensor.py            # descriptor: state, fields, noise
├── fields/
├── generated/                       # codegen output (build tree, gitignored)
├── src/                             # user's autonomy / control code
└── CMakeLists.txt                   # FetchContent mantapilot, calls codegen
```

### CMake helper

```cmake
# In user-project/CMakeLists.txt
mantapilot_add_craft(my_drone
  CONFIG crafts/my_drone.py
  MODE   sim                         # or "estimation"
  PARTS  parts/MyCustomSensor.cpp
)
```

`mantapilot_add_craft` wires the codegen as a custom command, generates C++ into `generated/`, links the user's part sources, and produces a target the user links against.

---

## Sim vs Estimation modes

A craft binary picks one mode at compile time. This is set by `mantapilot_add_craft`'s `MODE` argument, not by the Python craft definition (the same craft definition compiles both ways).

### Type aliasing

In `mantapilot/core/types.hpp`:

```cpp
#ifdef MANTA_ESTIMATION_MODE
  using mFloat = ceres::Jet<double, MANTA_STATE_DIM>;
#else
  using mFloat = float;
#endif
```

`MANTA_STATE_DIM` is set by codegen for that craft. The state dimension is fixed at compile time per binary.

All math inside parts uses `mFloat` (and derived types like `Vec3<F, mFloat>`). Plain `float`, `double`, `int` are reserved for things that genuinely don't participate in autodiff — string IDs, counters, etc.

### Runtime reconfiguration

For events that change the state dimension at runtime (e.g., two crafts dock and become one), the runtime invokes the codegen tool, compiles a new shared library, `dlopen`s it, and hands off state from the old crafts.

- This requires a working compiler available at runtime — fine for desktop sim, not for embedded.
- For crafts that dock often, an **opt-in pre-compiled superset** mode lets the user declare the maximum composition; the binary carries the largest possible state and masks inactive parts. No recompile, but pays autodiff cost for the max state always.

Real-life embedded builds always use the static dev-time codegen path. Runtime codegen is sim-only.

### Noise

Noise is a first-class type, not a macro. Each part declares noise sources as members:

```cpp
class IMU : public Part {
  Noise<WhiteGaussian> gyro_noise_{sigma: 0.01};
  Noise<WhiteGaussian> accel_noise_{sigma: 0.05};
  Noise<RandomWalk>    accel_bias_{sigma: 0.001};
  ...
};
```

Operator overloading dispatches on mode:

```cpp
// In update() — same code in both modes:
auto omega = angular_velocity<PartFrame>() + gyro_noise_;
auto accel = acceleration<PartFrame>() + accel_noise_ + accel_bias_;
```

- **Sim mode**: `operator+` samples and adds.
- **Estimation mode**: `operator+` returns the value unchanged but registers the noise's contribution into a per-craft accumulator the framework reads after the update pass.
- **`RandomWalk`**-type noise auto-registers as augmented EKF state (the bias is estimated).
- **`WhiteGaussian`**-type noise registers into measurement covariance R for the next update step.

Noise subclasses are extensible; users can add `ColoredGaussian`, `OrnsteinUhlenbeck`, sensor-specific failure models, etc., without changing part code.

---

## Geometry — `manta::geom`

### Frame typing

Frame *kind* is checked at compile time; frame *identity* is checked at runtime in debug builds and is zero-cost in release.

```cpp
namespace manta::geom {

enum class FrameKind { World, Planet, Scene, Craft, Part };

template<FrameKind K> struct frame_tag { static constexpr FrameKind kind = K; };
using WorldFrame  = frame_tag<FrameKind::World>;
using PlanetFrame = frame_tag<FrameKind::Planet>;
using SceneFrame  = frame_tag<FrameKind::Scene>;
using CraftFrame  = frame_tag<FrameKind::Craft>;
using PartFrame   = frame_tag<FrameKind::Part>;

}
```

Mixing kinds is a compile error (e.g., adding a `Vec3<WorldFrame>` to a `Vec3<CraftFrame>`). Mixing identities of the same kind (e.g., two different parts' `Vec3<PartFrame>`s) triggers a debug assertion via runtime `FrameId`.

Each Part receives a unique `FrameId` at construction.

### Vec3, Ori, Mat3, Wrench

Thin wrappers over Eigen, frame-tagged at the boundaries.

```cpp
template<typename F, typename Scalar = mFloat>
class Vec3 {
  Eigen::Matrix<Scalar, 3, 1> v_;
#ifdef MANTA_DEBUG
  FrameId id_;
#endif
public:
  // arithmetic — same-frame only, cross-frame is a compile error
  Vec3 operator+(const Vec3& o) const;
  Vec3 operator-(const Vec3& o) const;
  Vec3 operator*(Scalar s) const;
  Scalar norm() const;
  Scalar dot(const Vec3& o) const;
  Vec3   cross(const Vec3& o) const;

  // escape hatches for raw Eigen math inside parts
  const auto& raw() const { return v_; }
  static Vec3 from_raw(Eigen::Matrix<Scalar,3,1> v, FrameId = {});
};

template<typename F, typename Scalar = mFloat>
class Ori {
  // Convention: an orientation expressed with components in F's basis,
  // conventionally representing the orientation of a child (implicit) frame
  // relative to F. Ori is data; it does not rotate vectors by itself.
  // To convert vectors between frames, construct or use a KinematicLink/StaticLink.
  Eigen::Quaternion<Scalar> q_;
  // ... interpolation, normalization, multiplication-by-rate helpers in geom::ori
};

template<typename F, typename Scalar = mFloat>
class Wrench {
  Vec3<F, Scalar> force_;
  Vec3<F, Scalar> torque_;   // about the frame origin
};
```

Inside a part's `update()`, users freely use raw Eigen math and convert at the API boundary via `.raw()` / `from_raw()`. Frame tagging exists for the framework-facing surface, not for inline math.

### KinematicLink and StaticLink

```cpp
template<typename From, typename To, typename Scalar = mFloat>
class KinematicLink {
  Vec3<From, Scalar>  position_;       // To-origin in From frame
  Ori<From, Scalar>   orientation_;    // To-frame attitude, components in From basis
  Vec3<From, Scalar>  vel_linear_;     // velocity of To origin, in From frame
  Vec3<To,   Scalar>  vel_angular_;    // angular velocity of To, in To frame
  Vec3<From, Scalar>  acc_linear_;
  Vec3<To,   Scalar>  acc_angular_;
public:
  Vec3<From> apply_position(const Vec3<To>& p) const;
  Vec3<From> apply_velocity(const Vec3<To>& point_in_To,
                            const Vec3<To>& velocity_in_To) const;
  Vec3<From> rotate(const Vec3<To>& v) const;          // pure rotation (no offset)
  Vec3<To>   rotate_inverse(const Vec3<From>& v) const;

  template<typename C>
  KinematicLink<From, C> operator*(const KinematicLink<To, C>& other) const;
  KinematicLink<To, From> inverse() const;

  void update(Scalar dt);              // RK4 integration
};

template<typename From, typename To, typename Scalar = mFloat>
class StaticLink {
  Vec3<From, Scalar>  position_;
  Ori<From, Scalar>   orientation_;
  // vel/acc implied zero; methods omit them
};
```

Composition rules (operator overloads on `*`):

| LHS              | RHS              | Result            |
|------------------|------------------|-------------------|
| `StaticLink`     | `StaticLink`     | `StaticLink`      |
| `StaticLink`     | `KinematicLink`  | `KinematicLink`   |
| `KinematicLink`  | `StaticLink`     | `KinematicLink`   |
| `KinematicLink`  | `KinematicLink`  | `KinematicLink`   |

Naming convention for instances: `world_to_craft`, `craft_to_part`, etc., reading the type parameters left-to-right.

---

## Frame hierarchy

```
world → planet → scene → craft → part [→ part → ...]
```

- **World**: simulation root. Multiple planets per world.
- **Planet**: planetary body. Multiple scenes per planet.
- **Scene**: floating-origin reference frame. Shared by multiple crafts in proximity.
- **Craft**: vehicle root. Multiple parts.
- **Part**: component of a craft. May contain child parts.

### Floating-origin scenes

To preserve `float` precision over kilometers:

- A scene's origin is rebased relative to the planet whenever its centroid drifts more than a configurable threshold (default 500m).
- Crafts in proximity share a scene. When a craft drifts more than X km from its scene's centroid, it is **promoted to its own scene** automatically.
- Inter-craft / cross-scene physical interactions resolve through planet-frame coordinates stored as `(int64_t km, float meters)` pairs to preserve precision over global distances.
- Local interactions between crafts in the same scene — fluid wakes, RF coupling, collision proximity — are computed in scene coordinates for full fidelity.

Autodiff plays cleanly with the fixed-point representation: gradients are local and operate only on the `float` component.

---

## Parts — `manta::core`

### Part base class

```cpp
namespace manta {

class Part {
public:
  explicit Part(std::string name);
  virtual ~Part() = default;

  // The hook subclasses implement.
  virtual void update() = 0;

  // Configuration (set from subclass ctor or by user).
  void set_mass(mFloat m);
  void set_moi(const Mat3<PartFrame, PartFrame>& moi);
  void set_com(const Vec3<PartFrame>& com);
  void set_transform(const StaticLink<ParentFrame, PartFrame>& tf);

  mFloat                                    get_mass()      const;
  const Mat3<PartFrame, PartFrame>&         get_moi()       const;
  Vec3<PartFrame>                           get_com()       const;
  const StaticLink<ParentFrame, PartFrame>& get_transform() const;

  // Wrench application — used by update() to push forces to the parent.
  // Multiple calls within a single update() accumulate into a single net wrench.
  void apply_force_at(const Vec3<PartFrame>& force,
                      const Vec3<PartFrame>& point = {0,0,0});
  void apply_torque(const Vec3<PartFrame>& torque);
  void apply_wrench(const Wrench<PartFrame>& w);

  // Kinematic queries — read from the eager kinematic-pass cache.
  template<typename F> Vec3<F> position()             const;
  template<typename F> Vec3<F> velocity()             const;
  template<typename F> Vec3<F> acceleration()         const;
  template<typename F> Vec3<F> angular_velocity()     const;
  template<typename F> Vec3<F> angular_acceleration() const;
  template<typename F> Ori<F>  orientation()          const;

  // Context.
  Craft& craft();
  World& world();
  Part*  parent();

protected:
  // Field access — codegen ensures the field is wired and present.
  template<typename FieldT> FieldT& field();

private:
  friend class CompositePart;
  friend class Craft;

  Part*       parent_  = nullptr;
  Craft*      craft_   = nullptr;
  std::string name_;
  // mass, moi, com, transform
  // cached kinematic state (filled by kinematic pass)
  // wrench accumulator (filled by sense+force pass, drained by aggregation)
  // FrameId (debug)
};

}
```

Usage from a sample part:

```cpp
class IMU : public Part {
public:
  IMU(std::string name, ImuNoiseParams p) : Part(std::move(name)), noise_(p) {
    set_mass(0.05);
  }
  void update() override {
    last_accel_ = acceleration<PartFrame>() + noise_.accel + noise_.accel_bias;
    last_gyro_  = angular_velocity<PartFrame>() + noise_.gyro;
    // IMU is passive: doesn't apply wrenches.
  }
  Vec3<PartFrame> last_accel() const { return last_accel_; }
  Vec3<PartFrame> last_gyro()  const { return last_gyro_; }
private:
  ImuNoiseParams noise_;
  Vec3<PartFrame> last_accel_, last_gyro_;
};
```

### CompositePart

Owns children, aggregates physical parameters, delegates wrench combination.

```cpp
class CompositePart : public Part {
public:
  template<typename PartT, typename... Args>
  PartT& add(Args&&... args);             // constructs in place, wires parent/craft

  void compute_params();                  // mass/MOI/COM from children, dirty-aware

  // Default: rigidly transform child wrenches into this frame and sum.
  // ArticulatedPart overrides.
  virtual void aggregate_wrenches();

protected:
  std::vector<std::unique_ptr<Part>> children_;
};
```

`set_mass`, `set_moi`, `set_com` on a `CompositePart` are no-ops; values come from children via `compute_params()`.

### ArticulatedPart

Models simple constrained joints with optional saturation. Three flavors:

- **Locked**: joint is held at a commanded position; reaction wrench transferred entirely to parent.
- **Saturating**: joint holds the commanded position until reaction torque exceeds stall torque, at which point the joint slips and the residual accelerates the joint state.
- **Passive**: no command; joint integrates freely under whatever wrench it sees, parent receives reaction.

```cpp
class ArticulatedPart : public CompositePart {
public:
  // Subclass implements:
  virtual void resolve(const Wrench<PartFrame>& child_total,
                       Wrench<PartFrame>&       parent_out,
                       JointState&              joint_accel_out) = 0;

  // Joint state is registered with the craft's integrator at construction.
};
```

`parent_out` includes both the constraint reaction force AND the actuator's reaction torque (Newton's third). Concrete subclasses include `Motor`, `TwoAxisGimbal`, `FreeHinge`, `Bearing`.

This is intentionally not Featherstone-style multi-body dynamics. Use cases targeted: control surfaces pushing back on actuators, gimbal counter-torques, articulating drone arms exerting reaction torques on the body.

### RootPart

```cpp
class RootPart : public CompositePart {
  // No parent. Topmost in the tree. Aggregated wrenches end here and feed
  // the rigid-body integrator. Frame represents the geometric center of the craft.
};
```

---

## Wrench aggregation

A part's `update()` may call `apply_force_at`, `apply_torque`, or `apply_wrench` zero or more times. All such calls within a single update accumulate into the part's net wrench (a single force + torque about the part origin).

During the aggregation pass:

- Each `CompositePart` collects child wrenches, transforms them into its own frame, and sums.
- Each `ArticulatedPart` calls `resolve()`, which decomposes the child total into (joint acceleration, parent wrench).
- Wrenches reach the `RootPart`, are converted to net force and torque about the **center of mass** (which may differ from the geometric origin and may move as dynamic parts shift), and feed the rigid-body integrator.

Craft physics are computed about the center of mass; reported craft position (geometric origin) is unaffected by COM shifts (e.g., fuel burn, water tank filling).

---

## Crafts

```cpp
class Craft {
public:
  Craft(std::string name);
  virtual ~Craft() = default;

  RootPart& root();
  World&    world();
  Scene&    scene();
  void      set_scene(Scene&);

  template<typename FieldT> FieldT& field();

  void update();   // two-pass + integration; called by World

protected:
  RootPart root_;
  Scene*   scene_ = nullptr;
};
```

Codegen emits a subclass for each user-defined craft, holding parts as members for direct access:

```cpp
// generated/my_drone.hpp — codegen output, do not edit
class MyDroneCraft : public manta::Craft {
public:
  MyDroneCraft();
  IMU&      imu()        { return *imu_; }
  Thruster& motor(int i) { return *motors_[i]; }
private:
  PointMass* fuselage_;
  IMU*       imu_;
  std::array<Thruster*, 4> motors_;
};
```

Pointers index into the `unique_ptr` graph owned by `RootPart`. User code accesses parts by name without runtime string lookups: `drone.motor(0).set_throttle(0.7);`.

The Craft is **estimator-agnostic.** An `Estimator` references a Craft externally and pulls what it needs.

---

## Fields — `manta::fields`

### Protocol

```cpp
class Field {
public:
  virtual void update() = 0;            // called once per tick
  // Subclasses define query and disturbance APIs appropriate to the field.
};
```

Each field provides:

- A **query** API (e.g., `getField(pos)` for gravity, `getState(pos)` for fluid). Lock-free during the craft phase; field state is guaranteed frozen.
- A **contribute** API for adding disturbances (e.g., `addMass`, `addDisturbance`). Calls during the craft phase are buffered in per-craft thread-local storage; the field drains and applies them during the field phase. Users write `field.add_disturbance(...)` and the framework handles the deferral transparently.

### Backends — location-transparent

Every field is an interface with multiple backends:

- **In-process**: shared-memory, default for single-machine sims with multiple craft threads.
- **Zenoh-distributed**: for multi-machine sims. Each craft contributes locally, contributions are pubsubbed, all crafts converge on a consistent field state per tick (see SimClock).
- **Networked-real**: for live deployments where a fleet shares field state (e.g., a formation of drones sharing inferred wind). Contributions come from peer crafts' inferences, not raw sensor data.

**Fields never ingest raw sensor data.** Sensor data always flows through a Part. If a sonar return implies a hull is nearby, the *Part* code decides to call `field.add_disturbance(...)`. This keeps the Field abstraction symmetric across sim and live use.

### Scene-local patches

For fields where local interactions dominate (FluidField, near-field EM coupling), each scene maintains a local patch in scene coordinates:

```cpp
class FluidField : public Field {
public:
  State query(const Vec3<PlanetFrame>& world_pos);
  State query(const Vec3<SceneFrame>& scene_pos, Scene& scene);   // fast path
  void  contribute(const Disturbance& d);
};
```

Crafts in the same scene get high-fidelity local coupling. Crafts in different scenes interact only through the bulk planet-frame state. Since scenes are proximity-based, "in same scene" ≈ "near enough that local field effects matter."

GravityField and EMField (when implemented) always resolve in planet/world frame; no local patches.

### Disturbances

A `Disturbance` is a small object that contributes to a field's state. Examples:

- `PointMassDisturbance` (gravity)
- `WindDisturbance` (fluid)
- `ThrusterPlumeDisturbance` (fluid, scene-local)
- `RadioEmitterDisturbance` (EM)

Disturbances may declare augmented EKF state (see Estimation).

---

## Estimation — `manta::estimation`

### Estimator interface

```cpp
class Estimator {
public:
  virtual void declare_state(StateRef) = 0;          // wired by parts/disturbances
  virtual void predict(mFloat dt)      = 0;          // process model
  virtual void update(MeasurementVec, JacobianMat, NoiseCov) = 0;
  virtual StateView current_estimate() const = 0;
};
```

Concrete implementations:

- `EKF` — primary. Uses Ceres Jets for both process and measurement Jacobians.
- `UKF` — sigma-point alternative for highly nonlinear models.
- `MPC` — controller, not estimator; consumes an Estimator's state.

### Augmented state

Parts and disturbances can declare state variables that the estimator includes. Example: a `WindDisturbance` registered with `FluidField` contributes 3 augmented states with a process model:

```cpp
WindDisturbance wind{
  .process_model = RandomWalk{sigma: 0.1},
  .initial = {0, 0, 0},
};
fluid.add_disturbance(wind);
```

Generalizes to IMU bias, drag coefficient uncertainty, accelerometer scale factor — all via the same registration pattern. `RandomWalk` noise on a part declares augmented state automatically.

### Jacobian computation

Default: full Ceres Jets (forward-mode autodiff, dimension `MANTA_STATE_DIM`). Each per-tick evaluation produces value + Jacobian in one pass.

Future optimization (not for v1): **sparse Jets** restricted to the active dimensions per evaluation. Most measurements depend on a small subset of state; sparse Jets capture the asymptotic win without the engineering cost of a reverse-mode autodiff backend.

---

## Update loop

### Two-pass per craft

`Craft::update()` runs:

1. **Kinematic pass (top-down)** — propagate transforms from the craft's root to every leaf, populating each Part's cached `world↔part` link. This pass is eager; queries like `position<WorldFrame>()` read the cache.
2. **Sense + force pass** — for each part, call `update()`. The order within this pass doesn't matter because parts read cached kinematic state and write to their own wrench accumulator.
3. **Wrench aggregation (bottom-up)** — combine child wrenches at each composite. ArticulatedParts call `resolve()` to split into joint acceleration + parent wrench.
4. **Integration** — RK4 integrate the rigid-body state of the root + every joint state + every augmented EKF state from a single flat state vector.

### World loop and threading

Each craft runs on its own thread. A tick has two phases separated by barriers, designed so that no field state is mutated while any craft is reading it:

```cpp
void World::update() {
  clock_.lock_dt();                 // dt fixed for this tick
  // --- craft phase (parallel) ---
  parallel_for(crafts_, [](auto& c){ c.update(); });
  barrier();
  // --- field phase (single-threaded per field; fields may run in parallel) ---
  parallel_for(fields_, [](auto& f){ f.drain_contributions(); f.update(); });
  barrier();
  clock_.advance();
}
```

Rules enforced by the framework:

- **During craft phase**, field state is frozen. Field queries are lock-free reads.
- Field contributions called from within `update()` (`field.add_disturbance(...)`, `field.add_mass(...)`, etc.) **do not mutate the field immediately**. They are buffered in per-craft thread-local storage. The user writing a part doesn't think about threads.
- **During field phase**, each field drains the buffered contributions from all crafts (single-threaded per field) and runs its `update()`. Different fields may update in parallel since they don't share state.

This eliminates contention without exposing locks or queues to user code. Per-tick barrier cost is negligible compared to the physics work.

### SimClock

Owns simulation time and `dt`. In single-process runs it's a thread barrier. In Zenoh-distributed runs it's a service that publishes tick boundaries and synchronizes peer contributions:

1. Clock publishes `{tick: N, sim_time: T, dt: D}`.
2. Each craft runs its update for tick N.
3. Each craft publishes its field contributions tagged with tick N.
4. Each craft waits until it has received contributions from all peers for tick N (or the deadline expires).
5. Crafts assemble the final field view for tick N and signal ready.
6. Clock advances to tick N+1 once all crafts are ready.

A configurable deadline policy lets the clock advance even if a peer is slow, with that peer's previous-tick contributions reused for the missed tick.

The same `SimClock` interface backs both the in-process barrier and the Zenoh implementation.

---

## Optimizations

### Static-aggregate clustering

Parts whose mass, MOI, and transform don't change at runtime are flagged as static in the Python descriptor. The codegen analyzer walks the part tree, finds maximal static subtrees, and replaces each with a `StaticAggregate` `CompositePart` whose mass, COM, and MOI tensor are pre-baked. Aggregates are leaves in the dirty tree and never trigger `compute_params` recomputation.

The user never calls `mark_static()` from C++. The codegen handles it entirely from the Python descriptor.

### Eager kinematic cache

The kinematic pass populates `world↔part` links for every part at the start of each tick. Queries like `position<WorldFrame>()` and `velocity<WorldFrame>()` read the cache without lazy recomputation. Cheaper in autodiff mode, where lazy + Jets means redundantly evaluating Jacobian chains.

### Flat part array

At codegen time, the part tree is also emitted as a flat array indexed by part ID, with parent indices stored as integers. The two-pass update becomes two cache-friendly linear scans (forward for kinematics, reverse for wrench aggregation) instead of pointer-chasing through `parent->parent->...`. Significant for the hot loop with hundreds of crafts.

### Dirty tracking for `compute_params`

`set_mass`, `set_moi`, `set_transform` mark the part dirty rather than recomputing immediately. `compute_params` runs at most once per tick, just before integration. Avoids O(depth) recomputation when several values change in the same tick.

---

## Python codegen

### Workflow

1. User writes `crafts/my_drone.py`, building a `Craft` spec via the descriptor library.
2. CMake invokes `mantapilot-codegen --input crafts/my_drone.py --output generated/`.
3. Codegen `import`s the user's module, captures the spec object.
4. Analyzer runs: static-aggregate clustering, EKF state layout, validation of `REQUIRE_FIELD` declarations.
5. Emitter writes:
   - `generated/my_drone.hpp` — the codegen-generated `MyDroneCraft` class.
   - `generated/my_drone.cpp` — implementations including pre-baked aggregate parameters.
   - `generated/my_drone.json` (optional) — sidecar manifest for external tooling (visualizers, log replayers). Not used by the runtime.
6. User's CMake target builds the generated C++ alongside their part sources.

### Craft definition

```python
# crafts/my_drone.py
from mantapilot import Craft, PointMass, Surface, Thruster, IMU, ImuNoiseParams, tf
import numpy as np

def make_quadcopter(arm=0.25, motor_kv=920):
    c = Craft("quadcopter")
    body = c.root.add(PointMass("body", mass=0.5, static=True))
    body.add(IMU("imu", noise=ImuNoiseParams(gyro_sigma=0.01)))
    for i, angle in enumerate(np.linspace(0, 360, 4, endpoint=False)):
        pos = (arm * np.cos(np.radians(angle)),
               arm * np.sin(np.radians(angle)), 0)
        c.root.add(Thruster(f"motor_{i}", kv=motor_kv, transform=tf(pos), static=True))
    return c

CRAFT = make_quadcopter()
```

### Part descriptor

For user-defined parts:

```python
# parts/MyCustomSensor.py
from mantapilot import PartDescriptor, Field, RandomWalk, WhiteGaussian

class MyCustomSensor(PartDescriptor):
    cpp_class = "MyCustomSensor"          # matches the C++ class name
    requires_fields = [Field.Magnetic]
    noise = {
        "bias": RandomWalk(sigma=1e-4),
        "white": WhiteGaussian(sigma=0.01),
    }
    state = ["bias_x", "bias_y", "bias_z"]
```

The Python descriptor and the C++ class form a pair. The descriptor tells codegen about state, fields, and noise; the C++ class implements `update()`.

### Constraints on craft Python files

The Python file must build and return a spec object. It must not have import-time side effects (no threads, no sockets, no global state mutation). The codegen executes it once per build.

---

## Conventions

### Naming

- `world_to_craft`, `craft_to_part` for KinematicLink/StaticLink instances — names read left-to-right matching the type parameters.
- C++ classes in `PascalCase`; methods and free functions in `snake_case`.
- Frame tags as type aliases (`WorldFrame`, `CraftFrame`, ...).

### Frame access

Always templated, never suffixed:

```cpp
auto v = part.velocity<WorldFrame>();          // good
auto v = part.velocity_wf();                   // not in the API
```

### Numeric types

- `mFloat` — primary scalar, mode-dispatched.
- Plain `float`, `double`, `int` — only for non-physics quantities.
- `Vec3<F>`, `Mat3<F1, F2>`, `Ori<F>`, `Wrench<F>` — frame-tagged at API boundaries.
- Inside `update()`: free use of `Eigen` via `.raw()`/`from_raw()` at the boundary.

### Modules

- `manta::geom` — frame typing, vectors, orientations, links.
- `manta::core` — Part, CompositePart, RootPart, ArticulatedPart, Craft, World, Wrench, Noise.
- `manta::parts` — stock parts.
- `manta::fields` — stock fields.
- `manta::sim::dynamics` — integration, wrench aggregation.
- `manta::sim::sync` — SimClock, Zenoh barriers.
- `manta::estimation` — Estimator, EKF, UKF.

---

## Open questions

These remain to be resolved during implementation:

- **Part ID assignment scheme.** Compile-time-assigned IDs from codegen vs runtime-assigned at `add()`. Affects whether the `FrameId` debug check can be `constexpr`.
- **Wrench API for impulses.** Current API exposes force-over-time. If users need to model collisions or short impulses cleanly, may need an `apply_impulse` shortcut.
- **Field migration when a craft promotes to its own scene.** What happens to disturbances the craft contributed to its old scene?
- **MPC integration.** The `Estimator` interface is clean, but MPC needs symbolic gradients of the dynamics with respect to inputs as well as state. Likely a parallel `Controller` interface that shares Jets infrastructure.
- **Determinism guarantees.** For reproducible testing, every random source must be seedable per craft. Need a `RandomSource` abstraction passed via context, not pulled from global state.
- **EMField design.** Deferred from initial_plan.md. Will need separate treatment.

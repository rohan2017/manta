# Estimation workflow — design note

How estimator crafts are authored and fed data in manta. The architecture
is uniform; what varies is the **data source** that drives the estimator's
sensor measurement inputs.

## One architecture, three data-source patterns

The constant: an **estimator craft model** (a `Craft<Scalar>` codegen-emitted
from a Python descriptor) wrapped in an `EKF<...>`. The estimator's sensor
parts (IMU, DVL, GPS, etc.) have **measurement input hooks** — typed
setters that whatever data source feeds into. The EKF reads from those
hooks on its update step.

The variable: where the data comes from.

### Pattern A — same craft for sim + est (canonical)

One `craft.py`. The user instantiates it twice — one in a Scene (sim),
one wrapped in an EKF (estimator). Sim sensor outputs flow to estimator
measurement inputs **automatically by matching part names** (codegen
emits the glue). User writes no mapping.

```cpp
Craft sim;                         // codegenerated
Craft est;                         // same type, separate instance
scene.add_craft(sim, ...);
EKF<...> ekf(est);
// each tick: sim.update();  auto_pipe_sensors(sim, est);  ekf.step();
```

### Pattern B — different sim and est crafts

Two `craft.py`s. Sim has full fidelity (PropThruster + drag + counter-
torque). Est has a leaner autodiff-friendly model. Codegen emits two
typed Crafts. The user **specifies the name mapping** because the part
names differ:

```cpp
SimCraft sim;
EstCraft est;
EKF<...> ekf(est);
// each tick:
//   sim.update();
//   est.imu().set_measurement(sim.imu_a().last_accel(), ...);
//   est.gps().set_measurement(sim.gps_alpha().last_fix());
//   ekf.step();
```

The mapping is short and explicit; the user authors it. Codegen can
produce a scaffold that lists every estimator sensor and its expected
input type, but cannot guess the sim-side names — that's the user's
intent.

### Pattern C — real-data only

Same estimator craft. Same EKF. But there is no sim — the data source
is real sensor drivers (Zenoh subscriptions, ROS topics, hardware drivers,
MCAP playback). The user provides a **topic / driver mapping** that
feeds estimator sensor inputs.

For Zenoh-driven real-time deployments, codegen offers a workflow that
generates a runnable main from the topic mapping:

```bash
manta-codegen est_craft.py --workflow real_data \\
    --topics imu=manta/robot/imu,gps=manta/robot/gps,dvl=manta/robot/dvl
```

This emits a `<name>_main.cpp` that subscribes to the listed Zenoh
topics, parses the payloads, and calls `est.<name>().set_measurement(...)`
each time a message arrives. The user just needs `manta-codegen` and
the running estimator binary; no hand-written subscription glue.

For non-Zenoh data sources (direct hardware drivers, ROS, file replay)
the user writes the main by hand — codegen emits only the typed Craft +
config, and the user calls `est.imu().set_measurement(...)` directly
from their driver callbacks. Same estimator architecture, different glue.

### Concrete example: Pattern C codegen flow

```python
# est_craft.py — no sim, robot deployment
from manta_codegen import Craft
from manta_codegen.parts import IMU, GPS, DVL

def make_craft() -> Craft:
    c = Craft("robot_est")
    c.root.add(IMU("imu"))
    c.root.add(GPS("gps"))
    c.root.add(DVL("dvl"))
    return c
```

```bash
manta-codegen robot/est_craft.py --workflow real_data \\
    --topics imu=robot/imu/cooked,gps=robot/gps/fix,dvl=robot/dvl/vel
```

Codegen emits `robot_est_main.cpp` that:
- Declares Zenoh subscribers for the three listed topics.
- Parses their payloads (using the part's documented wire format).
- Calls `est.imu().set_measurement(...)`, `est.gps().set_measurement(...)`,
  `est.dvl().set_measurement(...)` on each incoming message.
- Owns an `EKF<...>` against `est` and runs predict/update at the configured
  rate.

The user compiles and runs the binary — done. Adding new sensors means
authoring a new part, registering it in `est_craft.py`, and adding it to
`--topics`. No subscription boilerplate to maintain by hand.

### What's never a workflow

There is **no** codegen mode that splits one descriptor into sim+est
versions. There is **no** `simple=True` flag that auto-substitutes
simpler parts. There is **no** assumed-canonical mapping when sim and
est crafts differ — the user always specifies the mapping.

## One Craft class, two consumers

The Craft class is a *single* templated type that exposes both:

- `update(dt)` — time-stepped simulation, driven by a Scene. Existing path.
- `evaluate(x, u, dt) → x_new` — Scene-independent dynamics evaluation
  the EKF calls. Sets state from the EKF's vector, runs
  kinematic+aggregate+integrate without Scene barriers, reads the new
  state back out. The EKF instantiates with `Scalar = ceres::Jet` to
  extract Jacobians.

Same Craft, two consumers. Hooking up an EKF is literally:

```cpp
Craft est;             // not in a Scene
EKF<...> ekf(est);     // EKF holds a reference to the est instance
```

For mode 1 (canonical, same craft for sim + est):

```cpp
Craft sim;             // sim instance — added to a Scene
Craft est;             // est instance of the SAME Craft class — not in a Scene
scene.add_craft(sim, ...);
EKF<...> ekf(est);
// each tick: sim.update(); pipe sim sensor outputs into est inputs; ekf.step();
```

For mode 2 (different fidelity sim and est, two descriptors):

```cpp
SimCraft sim;          // high-fidelity, codegenerated from sim_craft.py
EstCraft est;          // lean, codegenerated from est_craft.py
EKF<...> ekf(est);
```

For mode 3 (real-data only, no sim):

```cpp
EstCraft est;
EKF<...> ekf(est);     // measurements come from your real-sensor pipeline
```

## When to use two descriptors

Most users should start with mode 1 (one craft.py, two instances). Two
descriptors are only worth the extra authoring effort when:

- The sim and estimator can have structurally different part trees:
  - sim has full `PropThruster` (thrust + counter-torque + drag),
    estimator has plain `Thruster` (force only).
  - sim has 4 articulated motor parts modeling reaction torques,
    estimator has none — relies on the IMU to capture net acceleration.
  - sim has sensor cross-coupling, drag plate models, prop-wash; the
    estimator omits all of it.
- The user retains full authoring control on both sides, no magic
  mapping.
- The autodiff hot path stays small (estimator only); the sim can be as
  expensive as it wants.

## Wiring: sensor data flows from sim into estimator

The user main creates both crafts and runs them in the same process (or
splits across processes via Zenoh once that backend lands):

```cpp
MyDroneSimCraft sim;
MyDroneEstCraft est;          // estimator-side parts, templated on Scalar

scene.add_craft(sim, InitialState{...});
// `est` is not added to a scene — it's the estimator's model, not a
//  physical entity. It's evaluated on demand by the EKF.

manta::estimation::EKF<EstCraft::StateDim, EstCraft::MeasDim> ekf;
// EKF holds a reference to `est` and evaluates est.predict<Scalar>(...)
// templated on Real (value step) and ceres::Jet (Jacobian step).

while (running) {
    sim.update();
    // Glue: pipe sim.imu().last_accel() → est.imu_input
    //       pipe sim.dvl().last_velocity() → est.dvl_measurement
    ekf.predict(est, dt);
    ekf.update(est);
}
```

The "glue" between the sim's sensor outputs and the estimator's inputs
is **automatable from Python**: when both descriptors use a Part with a
matching name, codegen can emit a Python or C++ adapter that subscribes
to the sim sensor's published topic and feeds it into the estimator's
input hook. The user can override or hand-write the glue when names
differ or when a different transformation is needed (unit conversion,
bias compensation, etc.).

This adapter generation is one workstream (codegen-side) that comes
*after* the basic two-descriptor pattern is established.

## Scalar-templating implications

The *estimator craft* needs its parts templated on `Scalar` so the EKF
can instantiate it with `ceres::Jet<double, N>` to extract Jacobians.
The *sim craft* never needs autodiff — it always runs with `Real =
float` (or `Real = double` if precision is wanted).

Resolution: introduce `template<class Scalar = Real> class PartT` with
`using Part = PartT<Real>` as an alias. Existing non-templated subclasses
remain valid (they implicitly bind `Scalar = Real`). New estimator-side
parts are written as templates; the user instantiates them with `Real`
for the value step and `Jet<double, N>` for the Jacobian step inside
the EKF. Both instantiations live in the same TU.

The same templated part can be used on either side. The user simply
*chooses* simpler templated parts for the estimator descriptor.

## What NOT to build

- A `simple=True` flag on Part descriptors that auto-swaps for a simpler
  version inside one craft.
- A "codegen --split" mode that takes one descriptor and emits sim+est.
- Auto-mapping sim parts to estimator parts inside the codegen (beyond
  matching names for the sensor-data adapter).

These would all violate the principle that sim and estimator are
independent authoring artifacts.

## Today's transitional pattern (ex6)

Until the Scalar-templating refactor lands, the EKF takes a **hand-
written process functor** `f<Scalar>(x, dt, u)` that morally represents
the estimator craft's dynamics. ex6 does this. The functor stays a
first-class authoring path even after templating lands — for users who
want full control or are working with non-Craft state machines.

## Architecture: one templated Craft, two evaluation paths

The estimator does **not** run inside a Scene. But the same Craft
class — the same authoring artifact, the same C++ type — needs to
serve both consumers. So `Craft<Scalar>` exposes two evaluation paths:

(A) **Scene-driven** (`craft.update(dt)`). The Scene's three-phase
    update calls `kinematic_pass()`, `sense_and_aggregate()`,
    `integrate(dt)` in barrier-synchronized order across all sibling
    crafts. This is the existing path; nothing about it changes.

(B) **Estimator-driven** (`craft.evaluate(x, u, dt) → x_new`). Sets
    the craft's state from `x`, runs the same three phases without
    Scene barriers (the estimator only has one craft), reads the new
    state back out. No World, no Scene, no field registry needed in
    the EKF — the estimator's "external inputs" are handed in via
    `u` (e.g. an IMU acceleration reading already corrected for
    bias). When the estimator *does* want field interactions
    (e.g. a gravity-aware step), the EstimatorAdapter hands the
    needed Field instance into the Craft via the same registry the
    sim uses; the Craft doesn't care which.

Both paths run the same code. The difference is just *who calls
update()*: a Scene (path A) vs an EKF (path B). The Scalar template
parameter lets path B work with `ceres::Jet` for Jacobian extraction;
path A always uses `Real`.

```
Part<Scalar>                    (templated base)
   ↓
Craft<Scalar>                   (single class — Scene OR EKF can drive it)
   ├─ used by Scene as Craft<Real>     (sim path)
   └─ used by EKF as Craft<Jet<...>>   (estimator Jacobian path)
```

A typical user binary holds one or more sim Crafts (`Craft<Real>`
inside a Scene) and one or more est Crafts (`Craft<Real>` for the
value step + `Craft<Jet>` for the Jacobian step, both wrapped by a
single `EKF<...>` instance — the EKF maintains both internally).

## Implementation order

1. **Scalar-template `Part`** (with `using Part = PartT<Real>` alias for
   backwards compat). Plus `CompositePart` / `ArticulatedPart` /
   `RootPart`. Move `.cpp` content to headers (header-only) or use
   explicit instantiation per scalar.
2. **Scalar-template `Craft`** with `using Craft = CraftT<Real>`. Add
   the Scene-independent `evaluate(x, u, dt) → x_new` evaluation hook
   that the EKF calls. It must NOT depend on Scene barriers — but it
   may invoke kinematic+aggregate+integrate on its own root.
3. **Scalar-template the stock parts** the estimator side needs first
   (`PointMass`, `Mass`, `Thruster`, `IMU`, `DVL`, `GravityPart`). Each
   becomes `template<class Scalar = Real> class FooT` with a
   `using Foo = FooT<Real>` alias. Codegen emits the alias name in
   subclasses, so existing user crafts continue to work.
4. **Wire the EKF to consume a `Craft<Scalar>` directly.** EKF
   internally keeps two instances: a `Craft<Real>` for the value step
   and a `Craft<Jet<double, N>>` for the Jacobian step. The user
   constructs an `EKF` from a single `Craft<Real>&`; the EKF clones it
   to produce the Jet variant for Jacobian extraction.
5. **Add an `est_craft.py` to ex6** that demonstrates mode 2 (different
   sim and est crafts). Update ex6.cpp to wrap the est instance in an
   EKF directly (no hand-written functor).
6. **Codegen: sensor-glue auto-generation.** When sim and est share
   matching part names, codegen emits a Python or C++ adapter that
   pipes sim sensor outputs into est measurement inputs. Optional
   convenience for mode 2; mode 1 (one descriptor) and mode 3 (real-
   data only) don't need this.
7. **UKF** as an alternative to EKF, sharing the same Craft-driven
   evaluation interface.

Steps 1–4 are the foundational chunk. Step 5 is the visible demo.
Step 6 is polish for mode 2. Step 7 is breadth.

## Today's transitional shim

Until steps 1–5 land, ex6 demonstrates the architecture *in spirit*:
- A single `craft.py` (sim) acts as the sim Craft.
- The EKF uses a hand-written process functor as a stand-in for what
  will eventually be `est_craft.evaluate()`.
- The two are separate functions in ex6.cpp; the file header documents
  the principle that they are *intentionally* independent models.

The code matches the architecture's *seams* even if the codegen-side
plumbing is missing.

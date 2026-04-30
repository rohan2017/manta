# Planets — design note

Design for adding `Planet` to manta. Not yet implemented; this document
captures the agreed approach so the eventual implementer (likely future-me)
has full context.

## Goals

1. **Optional.** A user simulating a quadcopter in a uniform fluid, or two
   spacecraft in vacuum, never has to know `Planet` exists. The default world
   has no planets.
2. **Composable.** A user wanting Earth, Moon, Mars in one `World` should be
   able to add them like any other top-level entity.
3. **Physically faithful.** `Planet` is the natural home for rotating-frame
   effects (Coriolis, centrifugal). Inertial-navigation parts (IMU, AHRS,
   star tracker) require these to behave correctly.
4. **Composable with Fields.** Planets express themselves as **disturbances**
   on top of globally-defined Fields, not as private fields of their own.
   This means a `FluidField` is a single global continuous field, and Earth
   contributes the ocean and atmosphere as a position-dependent disturbance
   that decays with altitude/depth.

## Core idea — Planet is a "composite Field source"

A `Planet` does three things:

1. **Defines a frame** (`PlanetFrame`) that may rotate and/or translate
   relative to `WorldFrame`. The `World` knows about its planets and
   computes their frame transforms each tick (e.g. ephemeris-driven).
2. **Registers disturbances on a set of Fields.** At `Planet` construction
   time, the planet declares which Fields it contributes to and provides the
   disturbance functions. E.g. `Earth` registers a position-dependent ocean
   + atmosphere disturbance on `FluidField`, a J2 gravity field on
   `GravityField`, and an IGRF-style anomaly on `MagField`.
3. **Is itself queryable** for parts that need planet identity (not just a
   sampled field value). GPS needs to know "which planet's geodetic
   ellipsoid am I positioned on"; star trackers need ephemeris; IMU's
   Coriolis correction needs the planet's rotation rate.

## What goes away

- **`OceanAtmosField` is removed.** Replaced by:
  - One global `FluidField` (continuous, default vacuum).
  - `Earth::register_disturbances(World&)` adds a position-dependent
    fluid disturbance: high density inside the ocean shell, lower density
    in the troposphere, → 0 above the Karman line. Smooth blending at the
    sea surface.
- The `Hull` part stops checking "do I have an OceanAtmosField?". It just
  queries `FluidField` everywhere, gets a density, and integrates.
  Buoyancy/drag fall out automatically depending on what part of the
  disturbance the part is sampling.

## What gets added

### `SeaSurface` field

A new optional Field that stores `height_above_surface(position)`. Earth
populates it as a disturbance (zero everywhere except in a band around
each ocean's mean surface, where it returns signed distance to that
surface). The `Hull` part conditionally uses this (under the existing
runtime-or-codegen feature-detection plan) for buoyancy distribution
across a partially-submerged hull. If `SeaSurface` is not registered
(no Earth, or a custom Planet without oceans), `Hull` falls back to
fully-submerged or fully-airborne treatment.

### `Planet` base class

```cpp
class Planet {
public:
    virtual ~Planet() = default;

    // Called once at registration time. Planet calls
    // world.fluid_field().add_disturbance(...), .gravity_field().add_disturbance(...)
    // etc. for each Field it contributes to.
    virtual void register_disturbances(World& w) = 0;

    // Called each tick by World::update() before the field-sampling pass.
    // Updates the PlanetFrame transform (rotation, translation) and any
    // time-varying disturbances (tides, weather patterns, magnetic
    // secular variation, ...).
    virtual void update(Real t, Real dt) = 0;

    // Queries used by parts that requires_planet = ThisPlanetClass.
    // Geodetic helpers, ephemeris, body-fixed coordinates.
    // Specific signatures live on subclasses; the base just exists for
    // dynamic_cast / requires-resolution.

    geom::KinematicLink<WorldFrame, PlanetFrame> world_to_planet() const;
};
```

`Earth : Planet` provides:
- `geodetic_to_ecef(lat, lon, alt) -> Vec3<PlanetFrame>`
- `rotation_rate() -> Real` (≈7.2921e-5 rad/s)
- mean radius, gravitational parameter, J2, …
- ocean shell + atmosphere + magnetic-anomaly disturbance functions.

### Frame chain

When a planet exists:
```
World → Planet → Scene → Craft → Part
```
When no planet exists (default):
```
World → Scene → Craft → Part
```

`Scene` carries an optional `Planet*` parent pointer. If null, the scene's
parent transform into world is identity. If non-null, the parent transform
is the planet's `world_to_planet` inverse composed with whatever
scene-relative offset the user configured (so a "Scene anchored at this
ECEF point" is one line).

### Coriolis & centrifugal

The kinematic pass already operates in `SceneFrame`. With a planet present,
`SceneFrame` ≠ `WorldFrame`, so `scene_to_part_.acc_linear()` does not
include rotational pseudo-forces. The integrator runs in `WorldFrame`
(which by definition is non-rotating), so the rotation terms appear
naturally in the world→scene transform's time-derivative. As long as we
compute `scene_to_part_` from the world-frame state via the planet's
non-trivial `KinematicLink<WorldFrame, PlanetFrame>` (which has nonzero
`vel_angular` and `acc_angular`), the IMU's `acceleration_body()` will
include Coriolis and centrifugal contributions for free.

The reverse: integrator must apply pseudo-forces if it integrates in
`SceneFrame` for performance reasons. To keep the math simple we will
**always integrate in `WorldFrame`** and let the kinematic pass project
into scene/craft/part as needed. This costs one transform per part per
tick but makes Planet correctness automatic.

### Part requirements: `requires_planet`

A part can declare `requires_planet = Earth` (or any `Planet` subclass).
Resolution mirrors `requires_fields`:

- The C++ part queries `craft().planet<Earth>()` which dynamic-casts the
  scene's planet pointer. Compile error / runtime assertion if the scene
  doesn't have an Earth.
- The Python descriptor lists `requires_planet = Earth` and the codegen
  emits the same lookup.

Examples:
- `GPS` requires_planet = `Earth` (uses geodetic conversion + ephemeris).
- `Magnetometer` requires `MagField` only (any planet with a mag
  disturbance works), no specific-planet requirement.
- `IMU` does **not** require a planet — it works fine in vacuum / void.
  But its readings are automatically Coriolis-correct when a planet is
  present, because of where the kinematic pass runs.

### World integration

`World` gains an `add_planet<P>(args...)` method, mirroring
`add_field<F>(args...)`. Order of registration is:
1. User adds fields and planets to the world.
2. World iterates planets, calls `register_disturbances(*this)`.
3. User adds scenes; if a scene wants to be planet-anchored, the user
   calls `scene.set_planet(world.planet<Earth>(), {anchor_lat, anchor_lon})`.
4. User adds crafts to scenes as today.

## Tradeoffs and open questions

**Why not "Planets are just heavy crafts with their own gravity"?**
Tempting, because crafts already have a frame and forces. But planets
don't integrate (their frames are deterministic / ephemeris-driven), they
are queried for geodetic conversions, and their frames need to be the
parent of `Scene` not a sibling. They have a fundamentally different
lifecycle from crafts. A separate base class is cleaner.

**Why disturbances on global Fields, instead of per-planet local Fields?**
A craft transitioning between planets (interplanetary mission) sees
continuous changes in fluid density, gravity, and magnetic field as it
travels. With per-planet local fields you'd need explicit "which planet
am I in" logic. With global fields and disturbances, the answer is just
"sample the field at my position" — both planets' disturbances contribute
where they overlap, and the math works out. This is also how Coriolis
generalizes nicely: each planet's frame contributes its own pseudo-force
through its own world→planet kinematic link.

**Performance.** Disturbances are evaluated per-sample. A craft far from
Earth shouldn't pay the cost of evaluating Earth's atmosphere model. Each
disturbance carries a cheap bounding check (distance from planet center).
If outside, return zero and skip the expensive atmospheric ISA call.

**What lives in `Planet` vs `World`?** World owns the fields and the
planets list. Planet owns its disturbance lambdas and frame transform.
The world's update loop calls `planet.update()` for each planet, then
runs the field-sampling pass with each disturbance contributing.

## Implementation order (when we tackle this)

1. `Planet` base class + `World::add_planet()` + frame plumbing.
2. `VoidPlanet` (no-op) — actually probably never built; absence-of-planet
   handled by the optional `Scene::planet_` pointer.
3. Refactor `FluidField` to support disturbances. (Today it's already
   essentially this; just lift the disturbance API to a uniform pattern
   shared with `GravityField` etc.)
4. Implement `Earth` with: rotating frame, J2 gravity disturbance,
   ocean+atmosphere fluid disturbance, IGRF mag disturbance, sea-surface
   disturbance.
5. `requires_planet` resolution in C++ + Python codegen.
6. First parts that use it: `GPS`, then upgrade `IMU` tests to verify
   Coriolis appears in the readings when an Earth scene is set up.

## When to implement

Now is too early — we have one craft type, no second planet, no GPS, no
mag, and no test that would fail without rotating-frame correctness. The
moment one of these arrives (likely first: a magnetometer for a real
robot, or a desire to test Coriolis-aware IMU integration), we cash in
this design note.

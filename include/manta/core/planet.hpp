#pragma once

#include <string>
#include "frame.hpp"
#include "types.hpp"
#include "../geom/kinematic_link.hpp"

namespace manta {

template <class Scalar> class WorldT;

// Planet — a body that defines a (possibly rotating, possibly translating)
// frame inside the World, and contributes disturbances to the World's global
// fields. See prompts/planets.md for the design.
//
// Three responsibilities:
//   1. Hold a `world_to_planet_` link (PlanetFrame's pose + motion in
//      WorldFrame). Updated each tick by `update(t, dt)`.
//   2. At registration time (`register_disturbances`), add position-dependent
//      contributions to global fields the planet affects (ocean+atmosphere
//      to FluidField, J2 to GravityField, IGRF to MagField, sea-surface
//      heights to SeaSurface, ...).
//   3. Optionally serve as a queryable planet-identity object for parts
//      that declare `requires_planet = ThisPlanetClass`.
//
// Subclasses implement at least `update()` (rotation/translation profile)
// and `register_disturbances()` (the field contributions). For an inertial
// non-moving planet (the default), the base class's update() is a no-op
// and works as-is.
class Planet {
public:
    explicit Planet(std::string name) noexcept : name_(std::move(name)) {}
    virtual ~Planet() = default;

    const std::string& name() const noexcept { return name_; }

    // Pose + motion of PlanetFrame relative to WorldFrame. Subclass `update()`
    // mutates this each tick. Default: identity (planet at world origin, not
    // moving).
    const geom::KinematicLink<WorldFrame, PlanetFrame>& world_to_planet() const noexcept {
        return world_to_planet_;
    }

    // Called once at registration time by `World::add_planet`. Subclasses
    // attach disturbances to the world's fields here. Default: no-op (a
    // featureless planet doesn't affect any field).
    virtual void register_disturbances(WorldT<Real>& /*world*/) {}

    // Called each tick before any craft updates. Subclasses advance their
    // frame transform (rotation, ephemeris) and any time-varying disturbance
    // state here. Default: no-op (stationary planet).
    //
    // `t` is the world clock's current time; `dt` is the upcoming step.
    virtual void update(Real /*t*/, Real /*dt*/) {}

protected:
    // Subclasses set this in their constructor or `update()` to express the
    // planet's pose + motion in WorldFrame.
    geom::KinematicLink<WorldFrame, PlanetFrame> world_to_planet_ =
        geom::KinematicLink<WorldFrame, PlanetFrame>::identity();

private:
    std::string name_;
};

} // namespace manta

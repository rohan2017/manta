#pragma once

#include "field.hpp"
#include "../core/frame.hpp"
#include "../core/types.hpp"
#include "../geom/vec3.hpp"

namespace manta::fields {

// SeaSurface — the boundary between an ocean and atmosphere.
//
// Abstract base. Subclasses implement `height_above_surface(pos)` which
// returns the SIGNED distance from `pos` to the nearest sea surface:
//   * Positive  → above the surface (in air).
//   * Zero      → exactly on the surface.
//   * Negative  → below the surface (in water).
//
// Per `prompts/planets.md`, this exists as a separate Field (rather than
// being baked into a fluid field) so:
//   1. Hull-style buoyancy parts can query just the surface boundary
//      without needing to know a specific fluid implementation.
//   2. Planets (eventually `Earth`) can register a SeaSurface as a
//      *disturbance* on top of a global zero default — same pattern as
//      gravity / fluid fields.
//
// When no Planet contributes a sea surface (e.g. interplanetary flight,
// uniform fluid in a tank), no SeaSurface is registered and parts that
// need it must handle absence gracefully.
class SeaSurface : public Field {
public:
    virtual Real height_above_surface(const geom::Vec3<SceneFrame>& pos) const noexcept = 0;
};

// FlatSeaSurface — a horizontal plane at z = sea_level. Atmosphere above,
// water below. Useful as the default Earth-like surface when curvature
// doesn't matter (most near-shore robotics, tank tests).
class FlatSeaSurface : public SeaSurface {
public:
    explicit FlatSeaSurface(Real sea_level = Real(0)) noexcept
        : sea_level_(sea_level) {}

    Real height_above_surface(const geom::Vec3<SceneFrame>& pos) const noexcept override {
        return pos.z() - sea_level_;
    }

    void update() override {}   // static; nothing to do per tick

    Real sea_level()                 const noexcept { return sea_level_; }
    void set_sea_level(Real z)       noexcept       { sea_level_ = z; }

private:
    Real sea_level_;
};

} // namespace manta::fields

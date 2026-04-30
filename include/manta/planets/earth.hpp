#pragma once

#include <memory>
#include "../core/planet.hpp"
#include "../core/world.hpp"
#include "../fields/ocean_atmos_field.hpp"
#include "../fields/sea_surface.hpp"

namespace manta::planets {

// Earth — concrete Planet for near-Earth simulations.
//
// Owns and registers two fields automatically when added to a World:
//   * `fields::OceanAtmosField` under the `fields::FluidField` slot —
//     water below `sea_level`, air above. Same physics as the legacy
//     standalone OceanAtmosField field.
//   * `fields::FlatSeaSurface` under the `fields::SeaSurface` slot —
//     signed altitude relative to `sea_level`. Hull's smoothstep blend
//     auto-engages.
//
// Optionally rotates `world_to_planet_` about z at Earth's sidereal rate
// (~7.2921e-5 rad/s) so anchored Scenes pick up Coriolis + centrifugal.
//
// Future extensions (deferred):
//   * J2 gravity (currently a stand-alone GravityField is registered by
//     the user at the World level).
//   * Magnetic field (requires MagField).
//   * Position-dependent atmosphere model (vs the current hard-step
//     density profile).
//   * Geodetic helpers (lat/lon/alt ↔ ECEF conversions). Used by GPS-
//     style parts that declare `requires_planet = Earth`.
class Earth : public Planet {
public:
    static constexpr Real kRotationRate = Real(7.2921159e-5f);  // rad/s

    explicit Earth(Real sea_level     = Real(0),
                   Real water_density = Real(1000.0f),
                   Real air_density   = Real(1.225f),
                   Real rotation_rate = Real(0))
        : Planet("earth")
        , fluid_(std::make_unique<fields::OceanAtmosField>(
                     sea_level, water_density, air_density))
        , surface_(std::make_unique<fields::FlatSeaSurface>(sea_level))
        , rotation_rate_(rotation_rate) {
        // Earth's PlanetFrame rotates about its z axis at the configured rate.
        world_to_planet_.set_vel_angular(
            geom::Vec3<PlanetFrame>{Real(0), Real(0), rotation_rate_});
    }

    void register_disturbances(World& world) override {
        world.register_field<fields::FluidField>(*fluid_);
        world.register_field<fields::SeaSurface>(*surface_);
    }

    // Currently a no-op — `world_to_planet_`'s vel_angular already carries
    // the rotation rate, so per-tick advancement of the orientation isn't
    // needed for the rotating-frame-fictitious-force path. When ephemeris-
    // accurate Earth orientation is wanted, integrate the orientation here.
    void update(Real /*t*/, Real /*dt*/) override {}

    fields::OceanAtmosField& fluid() noexcept { return *fluid_; }
    fields::FlatSeaSurface&  surface() noexcept { return *surface_; }
    Real                     rotation_rate() const noexcept { return rotation_rate_; }

private:
    std::unique_ptr<fields::OceanAtmosField> fluid_;
    std::unique_ptr<fields::FlatSeaSurface>  surface_;
    Real                                     rotation_rate_;
};

} // namespace manta::planets

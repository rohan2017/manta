#pragma once

#include <memory>
#include "../core/planet.hpp"
#include "../core/world.hpp"
#include "../fields/mag_field.hpp"
#include "../fields/ocean_atmos_field.hpp"
#include "../fields/point_gravity_field.hpp"
#include "../fields/sea_surface.hpp"

namespace manta::planets {

// Earth — concrete Planet for near-Earth simulations.
//
// Always registers (under their abstract base slots):
//   * `fields::OceanAtmosField` → `fields::FluidField`
//   * `fields::FlatSeaSurface`  → `fields::SeaSurface`
//
// Optionally registers (when constructor flags are non-default):
//   * `fields::PointGravityField` (with optional J2 oblateness) — central
//     inverse-square + J2 perturbation. Activated by `gravity_mu > 0`.
//   * `fields::DipoleMagField`    → `fields::MagField` — leading-term
//     magnetic dipole. Activated by `dipole_moment != 0`.
//
// Optionally rotates `world_to_planet_` about z at Earth's sidereal rate
// (~7.2921e-5 rad/s) so anchored Scenes pick up Coriolis + centrifugal.
//
// Constants (CODATA / WGS84 / IGRF leading-term):
//   sidereal rotation rate     7.2921159e-5  rad/s
//   gravitational parameter    3.986004418e14 m^3/s^2
//   equatorial radius          6.378137e6     m
//   J2                         1.0826267e-3
//   IGRF dipole magnitude     ~7.94e22       A·m^2 (along -z by default)
class Earth : public Planet {
public:
    static constexpr Real kRotationRate    = Real(7.2921159e-5f);   // rad/s
    static constexpr Real kMu              = Real(3.986004418e14f); // m^3/s^2
    static constexpr Real kEquatorialRadius= Real(6.378137e6f);     // m
    static constexpr Real kJ2              = Real(1.0826267e-3f);
    static constexpr Real kDipoleMoment    = Real(7.94e22f);        // A·m^2

    explicit Earth(Real sea_level     = Real(0),
                   Real water_density = Real(1000.0f),
                   Real air_density   = Real(1.225f),
                   Real rotation_rate = Real(0),
                   // Optional gravity model. mu = 0 → no gravity field; the
                   // user can register their own at the World level.
                   Real gravity_mu    = Real(0),
                   // J2 perturbation, only applied when gravity_mu > 0.
                   bool include_j2    = false,
                   // Optional dipole magnetic model. moment = 0 → no MagField.
                   Real dipole_moment = Real(0))
        : Planet("earth")
        , fluid_(std::make_unique<fields::OceanAtmosField>(
                     sea_level, water_density, air_density))
        , surface_(std::make_unique<fields::FlatSeaSurface>(sea_level))
        , rotation_rate_(rotation_rate) {
        // Earth's PlanetFrame rotates about its z axis at the configured rate.
        world_to_planet_.set_vel_angular(
            geom::Vec3<PlanetFrame>{Real(0), Real(0), rotation_rate_});

        if (gravity_mu > Real(0)) {
            gravity_ = std::make_unique<fields::PointGravityField>(gravity_mu);
            if (include_j2) {
                gravity_->set_j2(kJ2, kEquatorialRadius);
            }
        }
        if (dipole_moment != Real(0)) {
            mag_ = std::make_unique<fields::DipoleMagField>(
                geom::Vec3<SceneFrame>{Real(0), Real(0), -dipole_moment});
        }
    }

    void register_disturbances(World& world) override {
        world.register_field<fields::FluidField>(*fluid_);
        world.register_field<fields::SeaSurface>(*surface_);
        if (gravity_) {
            world.register_field<fields::PointGravityField>(*gravity_);
        }
        if (mag_) {
            world.register_field<fields::MagField>(*mag_);
        }
    }

    // Currently a no-op — `world_to_planet_`'s vel_angular already carries
    // the rotation rate, so per-tick advancement of the orientation isn't
    // needed for the rotating-frame-fictitious-force path. When ephemeris-
    // accurate Earth orientation is wanted, integrate the orientation here.
    void update(Real /*t*/, Real /*dt*/) override {}

    fields::OceanAtmosField& fluid() noexcept { return *fluid_; }
    fields::FlatSeaSurface&  surface() noexcept { return *surface_; }
    fields::PointGravityField* gravity() noexcept { return gravity_.get(); }
    fields::DipoleMagField*    mag() noexcept     { return mag_.get(); }
    Real                     rotation_rate() const noexcept { return rotation_rate_; }

private:
    std::unique_ptr<fields::OceanAtmosField>   fluid_;
    std::unique_ptr<fields::FlatSeaSurface>    surface_;
    std::unique_ptr<fields::PointGravityField> gravity_;
    std::unique_ptr<fields::DipoleMagField>    mag_;
    Real                                       rotation_rate_;
};

} // namespace manta::planets

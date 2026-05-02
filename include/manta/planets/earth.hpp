#pragma once

#include <memory>
#include "../core/planet.hpp"
#include "../core/world.hpp"
#include "../fields/fluid_field.hpp"
#include "../fields/gravity_field.hpp"
#include "../fields/mag_field.hpp"

namespace manta::planets {

// Earth — concrete Planet for near-Earth simulations.
//
// Owns one of each global field (gravity, fluid, magnetic) and registers them
// on the world via the new disturbance-based field API. On construction,
// persistent disturbances are added:
//
//   * GravityField : point-mass at planet origin (μ = gravity_mu) plus, when
//                    `include_j2`, a J2 oblateness perturbation. Skipped
//                    entirely when `gravity_mu == 0`.
//   * FluidField   : an ocean disturbance (R = -1, density = water_density)
//                    bounded by `z < sea_level`, and an atmosphere disturbance
//                    (R = 287 J/(kg·K), gas) bounded by `z >= sea_level`. The
//                    in_influence predicates select water vs air at the
//                    query point. Atmosphere parameters are constant by
//                    default (ISA sea-level T, p); can be customized after
//                    construction by walking the disturbance handles.
//   * MagField     : a magnetic dipole at planet origin, aligned along -ẑ
//                    by default. Skipped when `dipole_moment == 0`.
//
// The scene's planet anchor + planet rotation rate produce the four
// fictitious forces in the integrator (Coriolis, centrifugal, Euler,
// translational) for free; no extra wiring needed here.
//
// Cartesian helpers `height_above_surface(p)` and `height_above_sea_level(p)`
// replace the old `SeaSurface` field. For the flat-ocean model used here the
// two functions return the same value; subclasses with bathymetry override
// `height_above_surface` to return distance to the solid floor.
//
// Constants (CODATA / WGS84 / IGRF leading-term):
//   sidereal rotation rate     7.2921159e-5  rad/s
//   gravitational parameter    3.986004418e14 m^3/s^2
//   equatorial radius          6.378137e6     m
//   J2                         1.0826267e-3
//   IGRF dipole magnitude     ~7.94e22       A·m^2 (along -z by default)
class Earth : public Planet {
public:
    static constexpr Real kRotationRate     = Real(7.2921159e-5f);   // rad/s
    static constexpr Real kMu               = Real(3.986004418e14f); // m^3/s^2
    static constexpr Real kEquatorialRadius = Real(6.378137e6f);     // m
    static constexpr Real kJ2               = Real(1.0826267e-3f);
    static constexpr Real kDipoleMoment     = Real(7.94e22f);        // A·m^2

    explicit Earth(Real sea_level     = Real(0),
                   Real water_density = Real(1000.0f),
                   Real /*air_density_unused*/ = Real(1.225f),
                   Real rotation_rate = Real(0),
                   // mu = 0 → no gravity field; user can add one separately.
                   Real gravity_mu    = Real(0),
                   bool include_j2    = false,
                   // moment = 0 → no MagField.
                   Real dipole_moment = Real(0))
        : Planet("earth")
        , gravity_(std::make_unique<fields::GravityField>())
        , fluid_  (std::make_unique<fields::FluidField>())
        , mag_    (std::make_unique<fields::MagField>())
        , sea_level_(sea_level)
        , rotation_rate_(rotation_rate) {
        world_to_planet_.set_vel_angular(
            geom::Vec3<PlanetFrame>{Real(0), Real(0), rotation_rate_});

        // Persistent gravity contributions.
        if (gravity_mu > Real(0)) {
            if (include_j2) {
                gravity_->add(fields::GravityField::Disturbance::point_mass_j2(
                                  geom::Vec3<SceneFrame>::zero(),
                                  gravity_mu, kJ2, kEquatorialRadius),
                              fields::PERSISTENT);
            } else {
                gravity_->add(fields::GravityField::Disturbance::point_mass(
                                  geom::Vec3<SceneFrame>::zero(), gravity_mu),
                              fields::PERSISTENT);
            }
        }

        // Persistent ocean (water below sea level).
        Real sl = sea_level_;
        auto water = fields::FluidField::Disturbance::uniform_incompressible(water_density);
        water.in_influence = [sl](const geom::Vec3<SceneFrame>& off) noexcept {
            return off.z() < sl;
        };
        fluid_->add(water, fields::PERSISTENT);

        // Persistent atmosphere (air above sea level — ISA sea-level T, p).
        auto air = fields::FluidField::Disturbance::uniform_gas(
            Real(287.0f), Real(288.15f), Real(101325.0f));
        air.in_influence = [sl](const geom::Vec3<SceneFrame>& off) noexcept {
            return off.z() >= sl;
        };
        fluid_->add(air, fields::PERSISTENT);

        // Persistent magnetic dipole.
        if (dipole_moment != Real(0)) {
            mag_->add(fields::MagField::Disturbance::dipole(
                          geom::Vec3<SceneFrame>::zero(),
                          geom::Vec3<SceneFrame>{Real(0), Real(0), -dipole_moment}),
                      fields::PERSISTENT);
        }
    }

    void register_disturbances(World& world) override {
        world.register_field<fields::GravityField>(*gravity_);
        world.register_field<fields::FluidField>  (*fluid_);
        world.register_field<fields::MagField>    (*mag_);
    }

    void update(Real /*t*/, Real /*dt*/) override {}

    // Signed altitude above the *solid* surface (negative = inside the
    // planet/ocean floor). Flat-ocean Earth has no bathymetry, so this
    // returns the same value as `height_above_sea_level` and may go
    // arbitrarily negative underwater.
    Real height_above_surface(const geom::Vec3<SceneFrame>& pos) const noexcept {
        return pos.z() - sea_level_;
    }

    // Signed altitude above the sea surface. Positive in air, negative
    // underwater. Flat-ocean approximation; future subclasses may add
    // wave heights.
    Real height_above_sea_level(const geom::Vec3<SceneFrame>& pos) const noexcept {
        return pos.z() - sea_level_;
    }

    fields::GravityField& gravity()        noexcept { return *gravity_; }
    fields::FluidField&   fluid()          noexcept { return *fluid_; }
    fields::MagField&     mag()            noexcept { return *mag_; }
    Real                  sea_level()      const noexcept { return sea_level_; }
    Real                  rotation_rate()  const noexcept { return rotation_rate_; }

private:
    std::unique_ptr<fields::GravityField> gravity_;
    std::unique_ptr<fields::FluidField>   fluid_;
    std::unique_ptr<fields::MagField>     mag_;
    Real                                  sea_level_;
    Real                                  rotation_rate_;
};

} // namespace manta::planets

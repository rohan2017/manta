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
    static constexpr MFloat kRotationRate     = MFloat(7.2921159e-5f);   // rad/s
    static constexpr MFloat kMu               = MFloat(3.986004418e14f); // m^3/s^2
    static constexpr MFloat kEquatorialRadius = MFloat(6.378137e6f);     // m
    static constexpr MFloat kJ2               = MFloat(1.0826267e-3f);
    static constexpr MFloat kDipoleMoment     = MFloat(7.94e22f);        // A·m^2

    explicit Earth(MFloat sea_level     = MFloat(0),
                   MFloat water_density = MFloat(1000.0f),
                   MFloat rotation_rate = MFloat(0),
                   // mu = 0 → no gravity field; user can add one separately.
                   MFloat gravity_mu    = MFloat(0),
                   bool include_j2    = false,
                   // moment = 0 → no MagField.
                   MFloat dipole_moment = MFloat(0))
        : Planet("earth")
        , gravity_(std::make_unique<fields::GravityField>())
        , fluid_  (std::make_unique<fields::FluidField>())
        , mag_    (std::make_unique<fields::MagField>())
        , sea_level_(sea_level)
        , rotation_rate_(rotation_rate) {
        world_to_planet_.set_vel_angular(
            geom::Vec3<PlanetFrame>{MFloat(0), MFloat(0), rotation_rate_});

        // Persistent gravity contributions.
        if (gravity_mu > MFloat(0)) {
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
        MFloat sl = sea_level_;
        auto water = fields::FluidField::Disturbance::uniform_incompressible(water_density);
        water.in_influence = [sl](const geom::Vec3<SceneFrame>& off) noexcept {
            return off.z() < sl;
        };
        fluid_->add(water, fields::PERSISTENT);

        // Persistent atmosphere (air above sea level — ISA sea-level T, p).
        auto air = fields::FluidField::Disturbance::uniform_gas(
            MFloat(287.0f), MFloat(288.15f), MFloat(101325.0f));
        air.in_influence = [sl](const geom::Vec3<SceneFrame>& off) noexcept {
            return off.z() >= sl;
        };
        fluid_->add(air, fields::PERSISTENT);

        // Persistent magnetic dipole.
        if (dipole_moment != MFloat(0)) {
            mag_->add(fields::MagField::Disturbance::dipole(
                          geom::Vec3<SceneFrame>::zero(),
                          geom::Vec3<SceneFrame>{MFloat(0), MFloat(0), -dipole_moment}),
                      fields::PERSISTENT);
        }
    }

    void register_disturbances(World& world) override {
        world.register_field<fields::GravityField>(*gravity_);
        world.register_field<fields::FluidField>  (*fluid_);
        world.register_field<fields::MagField>    (*mag_);
    }

    void update(MFloat /*t*/, MFloat /*dt*/) override {}

    // Signed altitude above the *solid* surface (negative = inside the
    // planet/ocean floor). Flat-ocean Earth has no bathymetry, so this
    // returns the same value as `height_above_sea_level` and may go
    // arbitrarily negative underwater.
    MFloat height_above_surface(const geom::Vec3<SceneFrame>& pos) const noexcept {
        return pos.z() - sea_level_;
    }

    // Signed altitude above the sea surface. Positive in air, negative
    // underwater. Flat-ocean approximation; future subclasses may add
    // wave heights.
    MFloat height_above_sea_level(const geom::Vec3<SceneFrame>& pos) const noexcept {
        return pos.z() - sea_level_;
    }

    fields::GravityField& gravity()        noexcept { return *gravity_; }
    fields::FluidField&   fluid()          noexcept { return *fluid_; }
    fields::MagField&     mag()            noexcept { return *mag_; }
    MFloat                  sea_level()      const noexcept { return sea_level_; }
    MFloat                  rotation_rate()  const noexcept { return rotation_rate_; }

private:
    std::unique_ptr<fields::GravityField> gravity_;
    std::unique_ptr<fields::FluidField>   fluid_;
    std::unique_ptr<fields::MagField>     mag_;
    MFloat                                  sea_level_;
    MFloat                                  rotation_rate_;
};

} // namespace manta::planets

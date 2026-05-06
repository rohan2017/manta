#include <cmath>
#include <doctest/doctest.h>

#include "../include/manta/core/craft.hpp"
#include "../include/manta/core/scene.hpp"
#include "../include/manta/core/world.hpp"
#include "../include/manta/fields/fluid_field.hpp"
#include "../include/manta/fields/gravity_field.hpp"
#include "test_helpers.hpp"

using namespace manta;
using namespace manta::geom;
using namespace manta::fields;

// ---- FluidField — uniform incompressible (replaces UniformFluidField) ----

TEST_CASE("FluidField: uniform incompressible disturbance returns constant density and velocity") {
    FluidField f;
    f.add(FluidField::Disturbance::uniform_incompressible(
              MFloat(1000.0f), Vec3<SceneFrame>{1, 2, 3}),
          PERSISTENT);

    auto s1 = f.state_at(Vec3<SceneFrame>::zero());
    auto s2 = f.state_at(Vec3<SceneFrame>{99, -50, 1000});
    CHECK(s1.R == doctest::Approx(-1));
    CHECK(s1.density == doctest::Approx(1000.0f));
    CHECK(s2.density == doctest::Approx(1000.0f));
    CHECK(test::approx_equal(s1.velocity, Vec3<SceneFrame>{1, 2, 3}));
    CHECK(test::approx_equal(s2.velocity, s1.velocity));
}

// ---- FluidField — co-existing ocean + atmosphere via in_influence (replaces OceanAtmosField) ----

TEST_CASE("FluidField: ocean below sea level, atmosphere above, selected by in_influence") {
    FluidField f;
    MFloat sea = MFloat(0);

    // Water below z=0.
    auto water = FluidField::Disturbance::uniform_incompressible(MFloat(1000.0f));
    water.in_influence = [sea](const Vec3<SceneFrame>& off) noexcept {
        return off.z() < sea;
    };
    f.add(water, PERSISTENT);

    // Air above z=0 (gas with R=287).
    auto air = FluidField::Disturbance::uniform_gas(
        MFloat(287.0f), MFloat(288.15f), MFloat(101325.0f));
    air.in_influence = [sea](const Vec3<SceneFrame>& off) noexcept {
        return off.z() >= sea;
    };
    f.add(air, PERSISTENT);

    auto under = f.state_at(Vec3<SceneFrame>{0, 0, -5});
    CHECK(under.R == doctest::Approx(-1));
    CHECK(under.density == doctest::Approx(1000.0f));

    auto over = f.state_at(Vec3<SceneFrame>{0, 0, 5});
    CHECK(over.R == doctest::Approx(287.0f));
    CHECK(over.density == doctest::Approx(1.225f).epsilon(0.01f));

    auto at_surface = f.state_at(Vec3<SceneFrame>{0, 0, 0});
    CHECK(at_surface.R == doctest::Approx(287.0f));   // sea level → air pool
}

TEST_CASE("FluidField: gas pool derives density from p, T, R via p = ρRT") {
    FluidField f;
    f.add(FluidField::Disturbance::uniform_gas(
              MFloat(287.0f), MFloat(288.15f), MFloat(101325.0f)),
          PERSISTENT);
    auto s = f.state_at(Vec3<SceneFrame>{1, 2, 3});
    CHECK(s.density == doctest::Approx(101325.0 / (287.0 * 288.15)).epsilon(1e-3f));
}

// ---- Velocity composition for currents/winds ----

TEST_CASE("FluidField: currents apply only below surface, wind only above") {
    FluidField f;
    auto current = FluidField::Disturbance::uniform_incompressible(
        MFloat(1000.0f), Vec3<SceneFrame>{0.5f, 0, 0});
    current.in_influence = [](const Vec3<SceneFrame>& off) noexcept {
        return off.z() < MFloat(0);
    };
    f.add(current, PERSISTENT);

    auto wind_dist = FluidField::Disturbance::uniform_gas(
        MFloat(287.0f), MFloat(288.15f), MFloat(101325.0f),
        Vec3<SceneFrame>{0, 10.0f, 0});
    wind_dist.in_influence = [](const Vec3<SceneFrame>& off) noexcept {
        return off.z() >= MFloat(0);
    };
    f.add(wind_dist, PERSISTENT);

    auto under = f.state_at(Vec3<SceneFrame>{0, 0, -1});
    auto over  = f.state_at(Vec3<SceneFrame>{0, 0,  1});
    CHECK(test::approx_equal(under.velocity, Vec3<SceneFrame>{0.5f, 0, 0}));
    CHECK(test::approx_equal(over.velocity,  Vec3<SceneFrame>{0, 10.0f, 0}));
}

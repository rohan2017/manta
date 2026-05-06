// Smoke tests for the new disturbance-based Field API. Exercises:
//   * Disturbance superposition (multiple sources sum correctly).
//   * Lifetime decay (default 1-tick disturbances disappear after update()).
//   * PERSISTENT lifetime (negative values survive update()).
//   * In-influence predicate (ignores disturbances out-of-range).
//   * Factory helpers (uniform / point_mass / dipole / uniform_gas, etc.).
//   * FluidField gas correction (ρ derived from p, T, R).

#include <doctest/doctest.h>

#include "../include/manta/fields/fluid_field.hpp"
#include "../include/manta/fields/gravity_field.hpp"
#include "../include/manta/fields/mag_field.hpp"

using namespace manta;
using namespace manta::fields;

TEST_CASE("GravityField: superposition of two point masses") {
    GravityField g;
    using V = GravityField::Vec;

    // Two equal masses on +x and -x at distance 1; field at origin should
    // sum to zero (the contributions cancel).
    g.add(GravityField::Disturbance::point_mass(V{MFloat(1), MFloat(0), MFloat(0)}, MFloat(1.0f)),
          PERSISTENT);
    g.add(GravityField::Disturbance::point_mass(V{MFloat(-1), MFloat(0), MFloat(0)}, MFloat(1.0f)),
          PERSISTENT);

    auto at_origin = g.state_at(V::zero());
    CHECK(at_origin.x() == doctest::Approx(0.0).epsilon(1e-6));
    CHECK(at_origin.y() == doctest::Approx(0.0).epsilon(1e-6));
    CHECK(at_origin.z() == doctest::Approx(0.0).epsilon(1e-6));

    // Off-axis query: at (0, 1, 0), pull is purely toward x = 0 by symmetry,
    // so y component is negative (toward origin in y), x component is 0.
    auto at_y1 = g.state_at(V{MFloat(0), MFloat(1), MFloat(0)});
    CHECK(at_y1.x() == doctest::Approx(0.0).epsilon(1e-6));
    CHECK(at_y1.y() < MFloat(0));
}

TEST_CASE("GravityField: uniform disturbance is constant everywhere") {
    GravityField g;
    g.add(GravityField::Disturbance::uniform(
              GravityField::Vec{MFloat(0), MFloat(0), MFloat(-9.81f)}),
          PERSISTENT);
    auto at_a = g.state_at(GravityField::Vec{MFloat(0),   MFloat(0),  MFloat(0)});
    auto at_b = g.state_at(GravityField::Vec{MFloat(100), MFloat(50), MFloat(-3)});
    CHECK(at_a.z() == doctest::Approx(-9.81).epsilon(1e-6));
    CHECK(at_b.z() == doctest::Approx(-9.81).epsilon(1e-6));
}

TEST_CASE("Lifetime: default 1-tick disturbance disappears after update()") {
    GravityField g;
    using V = GravityField::Vec;
    g.add(GravityField::Disturbance::uniform(V{MFloat(0), MFloat(0), MFloat(-1)}));  // default lifetime=1

    auto before = g.state_at(V::zero());
    CHECK(before.z() == doctest::Approx(-1.0));

    g.update();   // decrement lifetime → drops the disturbance

    auto after = g.state_at(V::zero());
    CHECK(after.z() == doctest::Approx(0.0));
    CHECK(g.disturbance_count() == 0);
}

TEST_CASE("Lifetime: PERSISTENT disturbance survives many updates") {
    GravityField g;
    using V = GravityField::Vec;
    g.add(GravityField::Disturbance::uniform(V{MFloat(0), MFloat(0), MFloat(-9.81f)}),
          PERSISTENT);

    for (int i = 0; i < 100; ++i) g.update();

    CHECK(g.disturbance_count() == 1);
    CHECK(g.state_at(V::zero()).z() == doctest::Approx(-9.81));
}

TEST_CASE("Lifetime: finite N-tick disturbance counts down") {
    GravityField g;
    using V = GravityField::Vec;
    g.add(GravityField::Disturbance::uniform(V{MFloat(0), MFloat(0), MFloat(-1)}), 3);

    g.update();   // lifetime: 3 → 2
    CHECK(g.disturbance_count() == 1);
    g.update();   // 2 → 1
    CHECK(g.disturbance_count() == 1);
    g.update();   // 1 → 0, dropped
    CHECK(g.disturbance_count() == 0);
}

TEST_CASE("In-influence predicate skips out-of-range queries") {
    GravityField g;
    using V = GravityField::Vec;

    // Spherical influence: only contributes when |off| < 5.
    GravityField::Disturbance d = GravityField::Disturbance::uniform(
        V{MFloat(0), MFloat(0), MFloat(-1)});
    d.in_influence = [](const V& off) {
        return off.raw().squaredNorm() < MFloat(25);   // r < 5
    };
    g.add(d, PERSISTENT);

    CHECK(g.state_at(V::zero()).z()              == doctest::Approx(-1.0));
    CHECK(g.state_at(V{MFloat(10), MFloat(0), MFloat(0)}).z() == doctest::Approx(0.0));
}

TEST_CASE("MagField: dipole vs uniform composition") {
    MagField m;
    using V = MagField::Vec;

    // Aligned-dipole moment along -ẑ. On the equator (x axis), B should
    // point along -ẑ (same direction as moment) — opposite of the
    // off-axis (axial) field.
    MFloat moment_mag = MFloat(7.94e22f);
    m.add(MagField::Disturbance::dipole(
              V::zero(), V{MFloat(0), MFloat(0), -moment_mag}),
          PERSISTENT);

    auto at_eq = m.state_at(V{MFloat(6.378e6f), MFloat(0), MFloat(0)});
    // On equator with moment along -ẑ:
    //   B = (μ0/4π) · (3·(m·r̂)r̂ − m) / r³ = (μ0/4π) · (-m) / r³
    // since m·r̂ = 0 on equator. Thus B has only a -ẑ-direction non-zero
    // component (parallel to moment), and is positive in z if moment is
    // negative in z.
    CHECK(at_eq.x() == doctest::Approx(0.0).epsilon(1e-6));
    CHECK(at_eq.y() == doctest::Approx(0.0).epsilon(1e-6));
    CHECK(at_eq.z() > MFloat(0));   // -(-m_z) flips sign to +z
}

TEST_CASE("FluidField: incompressible water summation") {
    FluidField f;
    using V = FluidField::Vec;

    f.add(FluidField::Disturbance::uniform_incompressible(
              MFloat(1000.0f), V{MFloat(0.5f), MFloat(0), MFloat(0)}),
          PERSISTENT);
    auto s = f.state_at(V::zero());
    CHECK(s.R == doctest::Approx(-1.0));
    CHECK(s.density == doctest::Approx(1000.0));
    CHECK(s.velocity.x() == doctest::Approx(0.5));
}

TEST_CASE("FluidField: gas state derives density from p, R, T") {
    FluidField f;
    using V = FluidField::Vec;

    // Air at sea level: R = 287 J/(kg·K), T = 288 K, p = 101325 Pa.
    // Expected density: 101325 / (287 · 288) ≈ 1.225 kg/m^3.
    f.add(FluidField::Disturbance::uniform_gas(
              MFloat(287.0f), MFloat(288.0f), MFloat(101325.0f)),
          PERSISTENT);

    auto s = f.state_at(V::zero());
    CHECK(s.R == doctest::Approx(287.0));
    CHECK(s.temperature == doctest::Approx(288.0));
    CHECK(s.pressure == doctest::Approx(101325.0));
    CHECK(s.density == doctest::Approx(1.225).epsilon(1e-3));
}

TEST_CASE("FluidField: gas + water co-existing via in_influence selects pool") {
    FluidField f;
    using V = FluidField::Vec;

    // Water disturbance: only below z=0 (sea level).
    auto water = FluidField::Disturbance::uniform_incompressible(MFloat(1000.0f));
    water.in_influence = [](const V& off) { return off.z() < MFloat(0); };
    f.add(water, PERSISTENT);

    // Air disturbance: only at or above z=0.
    auto air = FluidField::Disturbance::uniform_gas(
        MFloat(287.0f), MFloat(288.0f), MFloat(101325.0f));
    air.in_influence = [](const V& off) { return off.z() >= MFloat(0); };
    f.add(air, PERSISTENT);

    auto in_water = f.state_at(V{MFloat(0), MFloat(0), MFloat(-10)});
    CHECK(in_water.R == doctest::Approx(-1.0));
    CHECK(in_water.density == doctest::Approx(1000.0));

    auto in_air = f.state_at(V{MFloat(0), MFloat(0), MFloat(100)});
    CHECK(in_air.R == doctest::Approx(287.0));
    CHECK(in_air.density == doctest::Approx(1.225).epsilon(1e-3));
}

TEST_CASE("FluidField: vacuum default when no disturbance is in influence") {
    FluidField f;
    using V = FluidField::Vec;

    auto bounded = FluidField::Disturbance::uniform_incompressible(MFloat(1000.0f));
    bounded.in_influence = [](const V& off) {
        return off.raw().norm() < MFloat(1.0f);
    };
    f.add(bounded, PERSISTENT);

    auto far = f.state_at(V{MFloat(10), MFloat(0), MFloat(0)});
    CHECK(far.density == doctest::Approx(0.0));
    CHECK(far.pressure == doctest::Approx(0.0));
    CHECK(far.temperature == doctest::Approx(0.0));
    CHECK(far.velocity.raw().norm() == doctest::Approx(0.0));
}

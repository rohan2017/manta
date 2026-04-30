#include <cmath>
#include <doctest/doctest.h>

#include "../include/manta/core/craft.hpp"
#include "../include/manta/core/scene.hpp"
#include "../include/manta/core/world.hpp"
#include "../include/manta/fields/point_gravity_field.hpp"
#include "../include/manta/parts/field_src/point_gravity_part.hpp"
#include "../include/manta/parts/structure/point_mass.hpp"
#include "test_helpers.hpp"

using namespace manta;
using namespace manta::geom;
using namespace manta::parts;
using namespace manta::fields;

// ---- PointGravityField unit tests ----

TEST_CASE("PointGravityField: g points toward center, inverse-square magnitude") {
    PointGravityField pg{4.0e14f};  // ~Earth
    // 1e7 m above center along +x
    Vec3<SceneFrame> p{1.0e7f, 0, 0};
    auto g = pg.g_at(p);
    // |g| = mu / r^2 = 4e14 / 1e14 = 4 m/s^2; direction = -x
    CHECK(g.x() == doctest::Approx(-4.0f).epsilon(1e-4f));
    CHECK(g.y() == doctest::Approx(0.0f).epsilon(1e-4f));
    CHECK(g.z() == doctest::Approx(0.0f).epsilon(1e-4f));
}

TEST_CASE("PointGravityField: at center returns zero (no NaN)") {
    PointGravityField pg{1.0e14f};
    auto g = pg.g_at(Vec3<SceneFrame>::zero());
    CHECK(test::approx_equal(g, Vec3<SceneFrame>::zero()));
}

TEST_CASE("PointGravityField: respects center offset") {
    PointGravityField pg{4.0e14f, Vec3<SceneFrame>{1.0e7f, 0, 0}};
    // sample at +2e7 in x → r = (1e7, 0, 0) → |r|=1e7 → g_x = -4 m/s^2
    auto g = pg.g_at(Vec3<SceneFrame>{2.0e7f, 0, 0});
    CHECK(g.x() == doctest::Approx(-4.0f).epsilon(1e-4f));
}

// ---- Circular-orbit dynamics ----
//
// For a circular orbit at radius r with mu = GM:
//   v_circ = sqrt(mu / r)
//   period T = 2 * pi * sqrt(r^3 / mu)
//
// Set up tangential initial velocity and verify the craft stays close to
// circular over a single period.

TEST_CASE("PointGravityPart: circular orbit holds altitude over one period") {
    // Use a small body so the period is short and the test is fast.
    // mu = 1e6 (toy), r = 100 m → v_circ = sqrt(1e6/100) = 100 m/s.
    // T = 2π * sqrt(100^3 / 1e6) = 2π s ≈ 6.283 s.
    constexpr float mu = 1.0e6f;
    constexpr float r  = 100.0f;
    const float v_circ = std::sqrt(mu / r);
    const float T      = 2.0f * 3.14159265358979f * std::sqrt(r * r * r / mu);

    PointGravityField pgf{mu};
    World w;
    w.register_field(pgf);
    w.clock().set_dt(0.001f);  // 1 ms

    auto& scene = w.create_scene();

    Craft c("orbiter");
    c.root().add<PointMass>("body", 1.0f);
    c.root().add<PointGravityPart>("grav");
    c.root().compute_params();

    // Place at (r, 0, 0) with velocity (0, v_circ, 0) → counter-clockwise in xy.
    c.set_position   (Vec3<SceneFrame>{r, 0, 0});
    c.set_vel_linear (Vec3<SceneFrame>{0, v_circ, 0});
    scene.add_craft(c);

    int steps = static_cast<int>(T / 0.001f);  // 1 full period
    for (int i = 0; i < steps; ++i) w.update();

    // Symplectic Euler isn't symplectic — explicit-Euler will spiral.
    // Use loose tolerance: distance from center stays within 5% over 1 period.
    auto p = c.scene_to_craft().position();
    float r_now = std::sqrt(p.x()*p.x() + p.y()*p.y() + p.z()*p.z());
    CHECK(r_now == doctest::Approx(r).epsilon(0.05f));
}

TEST_CASE("PointGravityPart: surface gravity recovers Earth-like 9.81 m/s^2") {
    // mu = 3.986e14, r = 6.371e6 → g ≈ 9.82 m/s^2
    constexpr float mu = 3.986e14f;
    constexpr float r  = 6.371e6f;

    PointGravityField pgf{mu};
    World w;
    w.register_field(pgf);
    w.clock().set_dt(0.01f);

    auto& scene = w.create_scene();
    Craft c("test");
    c.root().add<PointMass>("body", 1.0f);
    c.root().add<PointGravityPart>("grav");
    c.root().compute_params();

    c.set_position(Vec3<SceneFrame>{r, 0, 0});
    scene.add_craft(c);

    // Take 1 tick; the wrench accumulator is drained by the dynamics pass,
    // so check derived quantity: free-fall acceleration this tick.
    w.update();
    auto a = c.scene_to_craft().acc_linear();
    float a_mag = std::sqrt(a.x()*a.x() + a.y()*a.y() + a.z()*a.z());
    CHECK(a_mag == doctest::Approx(9.82f).epsilon(0.01f));
}

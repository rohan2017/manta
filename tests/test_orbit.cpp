#include <cmath>
#include <doctest/doctest.h>

#include "../include/manta/core/craft.hpp"
#include "../include/manta/core/scene.hpp"
#include "../include/manta/core/world.hpp"
#include "../include/manta/fields/gravity_field.hpp"
#include "../include/manta/fields/mag_field.hpp"
#include "../include/manta/parts/structure/mass.hpp"
#include "test_helpers.hpp"

using namespace manta;
using namespace manta::geom;
using namespace manta::parts;
using namespace manta::fields;

// ---- Point-mass gravity disturbance unit tests ----

TEST_CASE("GravityField/point_mass: g points toward center, inverse-square magnitude") {
    GravityField gf;
    gf.add(GravityField::Disturbance::point_mass(
               Vec3<SceneFrame>::zero(), Real(4.0e14f)),  // ~Earth μ
           PERSISTENT);
    Vec3<SceneFrame> p{1.0e7f, 0, 0};
    auto g = gf.state_at(p);
    CHECK(g.x() == doctest::Approx(-4.0f).epsilon(1e-4f));
    CHECK(g.y() == doctest::Approx(0.0f).epsilon(1e-4f));
    CHECK(g.z() == doctest::Approx(0.0f).epsilon(1e-4f));
}

TEST_CASE("GravityField/point_mass: at center returns zero (no NaN)") {
    GravityField gf;
    gf.add(GravityField::Disturbance::point_mass(
               Vec3<SceneFrame>::zero(), Real(1.0e14f)),
           PERSISTENT);
    auto g = gf.state_at(Vec3<SceneFrame>::zero());
    CHECK(test::approx_equal(g, Vec3<SceneFrame>::zero()));
}

TEST_CASE("GravityField/point_mass: respects center offset") {
    GravityField gf;
    gf.add(GravityField::Disturbance::point_mass(
               Vec3<SceneFrame>{1.0e7f, 0, 0}, Real(4.0e14f)),
           PERSISTENT);
    auto g = gf.state_at(Vec3<SceneFrame>{2.0e7f, 0, 0});
    CHECK(g.x() == doctest::Approx(-4.0f).epsilon(1e-4f));
}

// ---- J2 perturbation ----

TEST_CASE("GravityField/J2: reduces effective gravity at the equator") {
    constexpr Real MU   = Real(3.986004418e14f);
    constexpr Real R_EQ = Real(6.378137e6f);
    constexpr Real J2   = Real(1.0826267e-3f);

    GravityField pure;
    pure.add(GravityField::Disturbance::point_mass(
                 Vec3<SceneFrame>::zero(), MU),
             PERSISTENT);
    GravityField with_j2;
    with_j2.add(GravityField::Disturbance::point_mass_j2(
                    Vec3<SceneFrame>::zero(), MU, J2, R_EQ),
                PERSISTENT);

    Vec3<SceneFrame> equator{R_EQ, 0, 0};
    Real dg = with_j2.state_at(equator).x() - pure.state_at(equator).x();
    CHECK(dg == doctest::Approx(0.0162f).epsilon(1e-3f));
    auto g = with_j2.state_at(equator);
    CHECK(std::abs(g.y()) < 1e-5f);
    CHECK(std::abs(g.z()) < 1e-5f);
}

TEST_CASE("GravityField/J2: increases effective gravity at the pole") {
    constexpr Real MU   = Real(3.986004418e14f);
    constexpr Real R_EQ = Real(6.378137e6f);
    constexpr Real J2   = Real(1.0826267e-3f);

    GravityField pure;
    pure.add(GravityField::Disturbance::point_mass(
                 Vec3<SceneFrame>::zero(), MU),
             PERSISTENT);
    GravityField with_j2;
    with_j2.add(GravityField::Disturbance::point_mass_j2(
                    Vec3<SceneFrame>::zero(), MU, J2, R_EQ),
                PERSISTENT);

    Vec3<SceneFrame> pole{0, 0, R_EQ};
    Real dg = with_j2.state_at(pole).z() - pure.state_at(pole).z();
    CHECK(dg == doctest::Approx(-0.0324f).epsilon(1e-3f));
    auto g = with_j2.state_at(pole);
    CHECK(std::abs(g.x()) < 1e-5f);
    CHECK(std::abs(g.y()) < 1e-5f);
}

// ---- Dipole magnetic disturbance ----

TEST_CASE("MagField/dipole: at the magnetic pole, B is parallel to the moment") {
    Vec3<SceneFrame> moment{0, 0, -7.94e22f};
    MagField mf;
    mf.add(MagField::Disturbance::dipole(Vec3<SceneFrame>::zero(), moment),
           PERSISTENT);

    float r = 1.0e7f;
    auto b = mf.state_at(Vec3<SceneFrame>{0, 0, r});
    float expected = 2.0f * 1.0e-7f * 7.94e22f / (r*r*r);
    CHECK(b.z() == doctest::Approx(-expected).epsilon(1e-3f));
    CHECK(std::abs(b.x()) < 1e-6f);
    CHECK(std::abs(b.y()) < 1e-6f);
}

TEST_CASE("MagField/dipole: at the equator, B is antiparallel to the moment, half-magnitude") {
    Vec3<SceneFrame> moment{0, 0, -7.94e22f};
    MagField mf;
    mf.add(MagField::Disturbance::dipole(Vec3<SceneFrame>::zero(), moment),
           PERSISTENT);

    float r = 1.0e7f;
    auto b = mf.state_at(Vec3<SceneFrame>{r, 0, 0});
    float expected = 1.0e-7f * 7.94e22f / (r*r*r);
    CHECK(b.z() == doctest::Approx(expected).epsilon(1e-3f));
    CHECK(std::abs(b.x()) < 1e-6f);
    CHECK(std::abs(b.y()) < 1e-6f);
}

// ---- End-to-end orbital dynamics under Mass auto-gravity ----
//
// With the redesign, `Mass` queries the registered GravityField at its CoM
// each tick and applies m·g to itself — the dedicated `PointGravityPart` is
// gone. So any craft with a single Mass part picks up gravity automatically.

TEST_CASE("Orbital: circular orbit holds altitude over one period") {
    constexpr float mu = 1.0e6f;
    constexpr float r  = 100.0f;
    const float v_circ = std::sqrt(mu / r);
    const float T      = 2.0f * 3.14159265358979f * std::sqrt(r * r * r / mu);

    GravityField gf;
    gf.add(GravityField::Disturbance::point_mass(Vec3<SceneFrame>::zero(), Real(mu)),
           PERSISTENT);

    World w;
    w.register_field(gf);
    w.clock().set_dt(0.001f);
    auto& scene = w.create_scene();

    Craft c("orbiter");
    c.root().add<Mass>("body", 1.0f);
    c.root().compute_params();
    c.set_position  (Vec3<SceneFrame>{r, 0, 0});
    c.set_vel_linear(Vec3<SceneFrame>{0, v_circ, 0});
    scene.add_craft(c);

    int steps = static_cast<int>(T / 0.001f);
    for (int i = 0; i < steps; ++i) w.step();

    auto p = c.scene_to_craft().position();
    float r_now = std::sqrt(p.x()*p.x() + p.y()*p.y() + p.z()*p.z());
    CHECK(r_now == doctest::Approx(r).epsilon(0.05f));
}

TEST_CASE("Orbital: surface gravity recovers Earth-like 9.81 m/s^2") {
    constexpr float mu = 3.986e14f;
    constexpr float r  = 6.371e6f;

    GravityField gf;
    gf.add(GravityField::Disturbance::point_mass(Vec3<SceneFrame>::zero(), Real(mu)),
           PERSISTENT);

    World w;
    w.register_field(gf);
    w.clock().set_dt(0.01f);
    auto& scene = w.create_scene();

    Craft c("test");
    c.root().add<Mass>("body", 1.0f);
    c.root().compute_params();
    c.set_position(Vec3<SceneFrame>{r, 0, 0});
    scene.add_craft(c);

    w.step();
    auto a = c.scene_to_craft().acc_linear();
    float a_mag = std::sqrt(a.x()*a.x() + a.y()*a.y() + a.z()*a.z());
    CHECK(a_mag == doctest::Approx(9.82f).epsilon(0.01f));
}

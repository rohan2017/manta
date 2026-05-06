// Tests for the surviving stock part models after the 2026-05 cleanup:
//   * Mass            — point mass / full MOI, with optional auto-gravity
//   * PointBuoy       — single-point buoyancy
//   * Surface1..4     — N-power velocity-driven force/torque
//   * DVL             — Doppler velocity log sensor
//
// `Hull` and `PointMass` were removed (Hull replaced by PointBuoy compositions;
// PointMass replaced by `Mass(name, mass)` shorthand with zero MOI).

#include <array>
#include <cmath>
#include <doctest/doctest.h>

#include "manta/core/craft.hpp"
#include "manta/core/scene.hpp"
#include "manta/core/world.hpp"
#include "manta/fields/fluid_field.hpp"
#include "manta/fields/gravity_field.hpp"
#include "manta/parts/sensor/dvl.hpp"
#include "manta/parts/structure/mass.hpp"
#include "manta/parts/structure/point_buoy.hpp"
#include "manta/parts/structure/surface.hpp"
#include "manta/parts/actuator/thruster.hpp"
#include "test_helpers.hpp"

using namespace manta;
using namespace manta::geom;
using namespace manta::parts;
using namespace manta::fields;

namespace {
FluidField uniform_water(Real density = Real(1000.0f),
                         Vec3<SceneFrame> velocity = Vec3<SceneFrame>::zero()) {
    FluidField f;
    f.add(FluidField::Disturbance::uniform_incompressible(density, velocity),
          PERSISTENT);
    return f;
}
}

// ---- Mass ----

TEST_CASE("Mass: contributes mass and MOI to root compute_params") {
    Craft c("test");
    Mat3<PartFrame> I = Mat3<PartFrame>::identity();
    I.raw()(0,0) = 0.5f; I.raw()(1,1) = 0.7f; I.raw()(2,2) = 0.9f;
    c.root().add<Mass>("body", 2.0f, I, /*apply_gravity=*/false);
    c.root().compute_params();
    CHECK(c.root().get_mass() == doctest::Approx(2.0f));
    CHECK(c.root().get_moi().raw()(0,0) == doctest::Approx(0.5f));
    CHECK(c.root().get_moi().raw()(2,2) == doctest::Approx(0.9f));
}

TEST_CASE("Mass: passive when apply_gravity=false and no field registered") {
    Craft c("test");
    Mat3<PartFrame> I = Mat3<PartFrame>::identity();
    c.root().add<Mass>("body", 1.0f, I, /*apply_gravity=*/false);
    c.update();
    CHECK(test::approx_equal(c.root().net_wrench().force(),  Vec3<PartFrame>::zero()));
    CHECK(test::approx_equal(c.root().net_wrench().torque(), Vec3<PartFrame>::zero()));
}

// ---- PointBuoy ----

TEST_CASE("PointBuoy: F = -ρ·V·g, opposes gravity in fluid") {
    auto water = uniform_water();
    GravityField gf{Vec3<SceneFrame>{0, 0, -9.81f}};

    Craft c("test");
    c.register_field(water);
    c.register_field(gf);
    c.root().add<Mass>("body", 0.0f, Mat3<PartFrame>::identity(),
                       /*apply_gravity=*/false);
    c.root().add<PointBuoy>("buoy", 0.001f);
    c.root().compute_params();
    c.update();

    auto F = c.root().net_wrench().force();
    CHECK(F.z() == doctest::Approx(9.81f).epsilon(1e-4f));
    CHECK(F.x() == doctest::Approx(0.0f).epsilon(1e-4f));
}

TEST_CASE("PointBuoy: zero volume → zero force") {
    auto water = uniform_water();
    GravityField gf{Vec3<SceneFrame>{0, 0, -9.81f}};
    Craft c("test");
    c.register_field(water);
    c.register_field(gf);
    c.root().add<PointBuoy>("buoy", 0.0f);
    c.root().compute_params();
    c.update();
    CHECK(test::approx_equal(c.root().net_wrench().force(), Vec3<PartFrame>::zero()));
}

TEST_CASE("PointBuoy composition: 4 buoys ~ neutral buoyancy at depth") {
    // 4 buoys of 0.00025 m^3 each = 0.001 m^3 total, with a 1 kg body.
    // Underwater: F_buoy = 1000 · 0.001 · 9.81 = 9.81 N up; weight = 9.81 N
    // down → net zero, craft stays put.
    auto water = uniform_water();
    GravityField gf{Vec3<SceneFrame>{0, 0, -9.81f}};

    World w;
    w.register_field(water);
    w.register_field(gf);
    w.clock().set_dt(0.01f);

    auto& scene = w.create_scene();
    Craft c("boat");
    c.root().add<Mass>("body", 1.0f);   // auto-gravity on
    // Four PointBuoys evenly spaced — same-symmetry as the deleted Hull's
    // 4-sample column, but composed from atomic parts.
    Real V_per = Real(0.001f / 4.0f);
    auto& b1 = c.root().add<PointBuoy>("b1", V_per);
    auto& b2 = c.root().add<PointBuoy>("b2", V_per);
    auto& b3 = c.root().add<PointBuoy>("b3", V_per);
    auto& b4 = c.root().add<PointBuoy>("b4", V_per);
    StaticLink<ParentFrame, PartFrame> tf1{Vec3<ParentFrame>{0, 0,  0.05f}, Ori<ParentFrame>::identity()};
    StaticLink<ParentFrame, PartFrame> tf2{Vec3<ParentFrame>{0, 0,  0.02f}, Ori<ParentFrame>::identity()};
    StaticLink<ParentFrame, PartFrame> tf3{Vec3<ParentFrame>{0, 0, -0.02f}, Ori<ParentFrame>::identity()};
    StaticLink<ParentFrame, PartFrame> tf4{Vec3<ParentFrame>{0, 0, -0.05f}, Ori<ParentFrame>::identity()};
    b1.set_transform(tf1); b2.set_transform(tf2);
    b3.set_transform(tf3); b4.set_transform(tf4);
    c.root().compute_params();
    c.set_position(Vec3<SceneFrame>{0, 0, -1.0f});
    scene.add_craft(c);

    for (int i = 0; i < 100; ++i) w.step();

    auto v = c.scene_to_craft().vel_linear();
    auto p = c.scene_to_craft().position();
    CHECK(v.z() == doctest::Approx(0.0f).epsilon(1e-3f));
    CHECK(p.z() == doctest::Approx(-1.0f).epsilon(1e-3f));
}

// ---- Surface1..Surface4 ----

namespace {
Mat3<PartFrame> diag_mat(float a, float b, float cc) {
    Mat3<PartFrame> m = Mat3<PartFrame>::identity();
    m.raw()(0,0) = a; m.raw()(1,1) = b; m.raw()(2,2) = cc;
    return m;
}
}

TEST_CASE("Surface1: linear drag — F = -k * v_rel") {
    auto fluid = uniform_water(Real(1.0f));
    Craft c("test");
    c.register_field(fluid);
    auto A = std::array<Mat3<PartFrame>, 1>{ diag_mat(-2.0f, -2.0f, -2.0f) };
    auto B = std::array<Mat3<PartFrame>, 1>{ diag_mat( 0.0f,  0.0f,  0.0f) };
    c.root().add<Surface1>("surf", A, B);
    c.root().compute_params();
    c.update();
    CHECK(test::approx_equal(c.root().net_wrench().force(), Vec3<PartFrame>::zero()));
}

TEST_CASE("Surface1: fluid moving relative to body → force") {
    auto wind = uniform_water(Real(1.0f), Vec3<SceneFrame>{5.0f, 0, 0});
    Craft c("test");
    c.register_field(wind);
    auto A = std::array<Mat3<PartFrame>, 1>{ diag_mat(-2.0f, -2.0f, -2.0f) };
    auto B = std::array<Mat3<PartFrame>, 1>{ diag_mat( 0.0f,  0.0f,  0.0f) };
    c.root().add<Surface1>("surf", A, B);
    c.root().compute_params();
    c.update();
    auto F = c.root().net_wrench().force();
    CHECK(F.x() == doctest::Approx(-10.0f).epsilon(1e-4f));
    CHECK(F.y() == doctest::Approx(  0.0f).epsilon(1e-4f));
    CHECK(F.z() == doctest::Approx(  0.0f).epsilon(1e-4f));
}

TEST_CASE("Surface2: linear + quadratic — A*v + B*v² accumulate") {
    auto wind = uniform_water(Real(1.0f), Vec3<SceneFrame>{3.0f, 0, 0});
    Craft c("test");
    c.register_field(wind);
    std::array<Mat3<PartFrame>, 2> A{ diag_mat(-1.0f, -1.0f, -1.0f),
                                      diag_mat(-0.5f, -0.5f, -0.5f) };
    std::array<Mat3<PartFrame>, 2> B{ diag_mat( 0.0f,  0.0f,  0.0f),
                                      diag_mat( 0.0f,  0.0f,  0.0f) };
    c.root().add<Surface2>("surf", A, B);
    c.root().compute_params();
    c.update();
    CHECK(c.root().net_wrench().force().x() == doctest::Approx(-7.5f).epsilon(1e-4f));
}

TEST_CASE("Surface2: sign of v_rel is preserved by the quadratic term") {
    // A positive diagonal A_2 should produce force ALONG v_rel, regardless
    // of sign on each axis. Componentwise rule: F_i = A_2.ii · sign(v_i) · v_i².
    auto fluid = uniform_water(Real(1.0f));
    std::array<Mat3<PartFrame>, 2> A{ diag_mat(0.0f, 0.0f, 0.0f),
                                      diag_mat(1.0f, 1.0f, 1.0f) };   // pure quadratic
    std::array<Mat3<PartFrame>, 2> B{ diag_mat(0.0f, 0.0f, 0.0f),
                                      diag_mat(0.0f, 0.0f, 0.0f) };

    // Body moving +z at 3 m/s in still fluid → v_rel_part = (0, 0, -3).
    {
        Craft c("test_pos");
        c.register_field(fluid);
        c.root().add<Surface2>("surf", A, B);
        c.root().compute_params();
        c.set_vel_linear(Vec3<SceneFrame>{0, 0, 3});
        c.update();
        auto F = c.root().net_wrench().force();
        CHECK(F.z() == doctest::Approx(-9.0f).epsilon(1e-4f));   // sign(-3) · 9 = -9
        CHECK(F.x() == doctest::Approx( 0.0f).epsilon(1e-4f));
        CHECK(F.y() == doctest::Approx( 0.0f).epsilon(1e-4f));
    }
    // Same body moving -z at 3 m/s → v_rel_part = (0, 0, +3).
    {
        Craft c("test_neg");
        c.register_field(fluid);
        c.root().add<Surface2>("surf", A, B);
        c.root().compute_params();
        c.set_vel_linear(Vec3<SceneFrame>{0, 0, -3});
        c.update();
        auto F = c.root().net_wrench().force();
        CHECK(F.z() == doctest::Approx(+9.0f).epsilon(1e-4f));   // sign(+3) · 9 = +9
    }
}

TEST_CASE("Surface1: torque tensor produces torque from velocity") {
    auto fluid = uniform_water(Real(1.0f));
    Craft c("test");
    c.register_field(fluid);
    std::array<Mat3<PartFrame>, 1> A{ diag_mat(0.0f, 0.0f, 0.0f) };
    std::array<Mat3<PartFrame>, 1> B{ diag_mat(1.0f, 1.0f, 1.0f) };
    c.root().add<Surface1>("surf", A, B);
    c.root().compute_params();
    c.set_vel_linear(Vec3<SceneFrame>{2, 0, 0});
    c.update();
    auto T = c.root().net_wrench().torque();
    CHECK(T.x() == doctest::Approx(-2.0f).epsilon(1e-4f));
}

// ---- DVL ----

TEST_CASE("DVL: at rest → zero velocity reading (no noise)") {
    Craft c("test");
    auto& dvl = c.root().add<DVL>("dvl");
    c.root().compute_params();
    c.update(0.01f);
    c.update(0.01f);
    CHECK(test::approx_equal(dvl.last_velocity(), Vec3<PartFrame>::zero()));
}

TEST_CASE("DVL: reads body velocity in part frame") {
    Craft c("test");
    auto& t = c.root().add<Thruster>("t", 1.0f);
    auto& dvl = c.root().add<DVL>("dvl");
    t.set_throttle(1.0f);
    t.set_mass(1.0f);
    c.root().compute_params();

    c.update(1.0f);
    c.update(0.0f);

    CHECK(dvl.last_velocity().z() == doctest::Approx(1.0f).epsilon(1e-2f));
}

#include "manta/parts/sensor/imu.hpp"

TEST_CASE("DVL: set_measurement injects external reading, bypasses update()") {
    Craft c("test");
    auto& dvl = c.root().add<DVL>("dvl");
    c.root().compute_params();

    dvl.set_measurement(Vec3<PartFrame>{1.5f, -0.5f, 0.25f});

    CHECK(dvl.last_velocity().x() == doctest::Approx( 1.5f));
    CHECK(dvl.last_velocity().y() == doctest::Approx(-0.5f));
    CHECK(dvl.last_velocity().z() == doctest::Approx( 0.25f));
}

TEST_CASE("IMU: set_measurement injects external accel + gyro") {
    Craft c("test");
    auto& imu = c.root().add<IMU>("imu");
    c.root().compute_params();

    imu.set_measurement(
        Vec3<PartFrame>{0.1f, 0.2f, 9.81f},
        Vec3<PartFrame>{0.0f, 0.0f, 0.5f});

    CHECK(imu.last_accel().x() == doctest::Approx(0.1f));
    CHECK(imu.last_accel().y() == doctest::Approx(0.2f));
    CHECK(imu.last_accel().z() == doctest::Approx(9.81f));
    CHECK(imu.last_gyro().z()  == doctest::Approx(0.5f));
}

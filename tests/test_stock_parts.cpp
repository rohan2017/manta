// Tests for the new stock part models added 2026-04-29:
//   * Mass            — point mass with full MOI
//   * PointBuoy       — single-point buoyancy
//   * Surface1..4     — N-power velocity-driven force/torque
//   * DVL             — Doppler velocity log sensor

#include <array>
#include <cmath>
#include <doctest/doctest.h>

#include "manta/core/craft.hpp"
#include "manta/core/scene.hpp"
#include "manta/core/world.hpp"
#include "manta/fields/gravity_field.hpp"
#include "manta/fields/uniform_fluid_field.hpp"
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

// ---- Mass ----

TEST_CASE("Mass: contributes mass and MOI to root compute_params") {
    Craft c("test");
    Mat3<PartFrame> I = Mat3<PartFrame>::identity();
    I.raw()(0,0) = 0.5f; I.raw()(1,1) = 0.7f; I.raw()(2,2) = 0.9f;
    c.root().add<Mass>("body", 2.0f, I);
    c.root().compute_params();
    CHECK(c.root().get_mass() == doctest::Approx(2.0f));
    CHECK(c.root().get_moi().raw()(0,0) == doctest::Approx(0.5f));
    CHECK(c.root().get_moi().raw()(2,2) == doctest::Approx(0.9f));
}

TEST_CASE("Mass: passive — no wrench applied") {
    Craft c("test");
    Mat3<PartFrame> I = Mat3<PartFrame>::identity();
    c.root().add<Mass>("body", 1.0f, I);
    c.update();
    CHECK(test::approx_equal(c.root().net_wrench().force(),  Vec3<PartFrame>::zero()));
    CHECK(test::approx_equal(c.root().net_wrench().torque(), Vec3<PartFrame>::zero()));
}

// ---- PointBuoy ----

TEST_CASE("PointBuoy: F = -ρ·V·g, opposes gravity in fluid") {
    UniformFluidField water(1000.0f);
    GravityField gf{Vec3<SceneFrame>{0, 0, -9.81f}};

    Craft c("test");
    c.register_field<FluidField>(water);
    c.register_field(gf);
    c.root().add<Mass>("body", 0.0f, Mat3<PartFrame>::identity());  // ignore weight
    c.root().add<PointBuoy>("buoy", 0.001f);  // 1 liter
    c.root().compute_params();
    c.update();

    auto F = c.root().net_wrench().force();
    // Expected: 1000 * 0.001 * 9.81 = 9.81 N upward
    CHECK(F.z() == doctest::Approx(9.81f).epsilon(1e-4f));
    CHECK(F.x() == doctest::Approx(0.0f).epsilon(1e-4f));
}

TEST_CASE("PointBuoy: zero volume → zero force") {
    UniformFluidField water(1000.0f);
    GravityField gf;
    Craft c("test");
    c.register_field<FluidField>(water);
    c.register_field(gf);
    c.root().add<PointBuoy>("buoy", 0.0f);
    c.root().compute_params();
    c.update();
    CHECK(test::approx_equal(c.root().net_wrench().force(), Vec3<PartFrame>::zero()));
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
    // Fluid at rest, body at rest → no relative motion → no force.
    UniformFluidField fluid(1.0f);  // density doesn't enter Surface
    Craft c("test");
    c.register_field<FluidField>(fluid);
    auto A = std::array<Mat3<PartFrame>, 1>{ diag_mat(-2.0f, -2.0f, -2.0f) };
    auto B = std::array<Mat3<PartFrame>, 1>{ diag_mat( 0.0f,  0.0f,  0.0f) };
    c.root().add<Surface1>("surf", A, B);
    c.root().compute_params();

    c.update();
    CHECK(test::approx_equal(c.root().net_wrench().force(), Vec3<PartFrame>::zero()));
}

TEST_CASE("Surface1: fluid moving relative to body → force") {
    // Wind +x at 5 m/s, body at rest. v_rel_part = (5, 0, 0).
    // A = diag(-2,-2,-2). F_part = -2 * (5,0,0) = (-10, 0, 0).
    UniformFluidField wind(1.0f, Vec3<SceneFrame>{5.0f, 0, 0});
    Craft c("test");
    c.register_field<FluidField>(wind);
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
    UniformFluidField wind(1.0f, Vec3<SceneFrame>{3.0f, 0, 0});
    Craft c("test");
    c.register_field<FluidField>(wind);
    // A_1 = diag(-1,-1,-1); A_2 = diag(-0.5,-0.5,-0.5).
    // Expected F = -1*(3,0,0) + -0.5*(9,0,0) = (-3 + -4.5, 0, 0) = (-7.5, 0, 0).
    std::array<Mat3<PartFrame>, 2> A{ diag_mat(-1.0f, -1.0f, -1.0f),
                                      diag_mat(-0.5f, -0.5f, -0.5f) };
    std::array<Mat3<PartFrame>, 2> B{ diag_mat( 0.0f,  0.0f,  0.0f),
                                      diag_mat( 0.0f,  0.0f,  0.0f) };
    c.root().add<Surface2>("surf", A, B);
    c.root().compute_params();
    c.update();
    CHECK(c.root().net_wrench().force().x() == doctest::Approx(-7.5f).epsilon(1e-4f));
}

TEST_CASE("Surface1: torque tensor produces torque from velocity") {
    // Body moving +x at 2 m/s in still fluid → v_rel_part = (-2, 0, 0).
    // B = identity → torque = v_rel_part = (-2, 0, 0).
    UniformFluidField fluid(1.0f);
    Craft c("test");
    c.register_field<FluidField>(fluid);
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
    c.update(0.01f);  // two ticks: scene_to_part populated
    CHECK(test::approx_equal(dvl.last_velocity(), Vec3<PartFrame>::zero()));
}

TEST_CASE("DVL: reads body velocity in part frame") {
    // 1 kg craft, 1 N thrust → v = 1 m/s after 1s.
    Craft c("test");
    auto& t = c.root().add<Thruster>("t", 1.0f);
    auto& dvl = c.root().add<DVL>("dvl");
    t.set_throttle(1.0f);
    t.set_mass(1.0f);
    c.root().compute_params();

    c.update(1.0f);   // velocity reaches 1 m/s
    c.update(0.0f);   // refresh kinematic cache, dvl reads it

    CHECK(dvl.last_velocity().z() == doctest::Approx(1.0f).epsilon(1e-2f));
}

// ---- Sensor measurement-input hooks (estimator path) ----

#include "manta/parts/sensor/imu.hpp"

TEST_CASE("DVL: set_measurement injects external reading, bypasses update()") {
    Craft c("test");
    auto& dvl = c.root().add<DVL>("dvl");
    c.root().compute_params();

    // Inject an external measurement — what an estimator-side DVL would do
    // when fed by a sim sensor reading or a real driver.
    dvl.set_measurement(Vec3<PartFrame>{1.5f, -0.5f, 0.25f});

    CHECK(dvl.last_velocity().x() == doctest::Approx( 1.5f));
    CHECK(dvl.last_velocity().y() == doctest::Approx(-0.5f));
    CHECK(dvl.last_velocity().z() == doctest::Approx( 0.25f));

    // No call to c.update() — the estimator path doesn't run kinematic_pass.
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

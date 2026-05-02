#include <cmath>
#include <doctest/doctest.h>

#include "../include/manta/core/craft.hpp"
#include "../include/manta/core/noise.hpp"
#include "../include/manta/fields/gravity_field.hpp"
#include "../include/manta/parts/sensor/imu.hpp"
#include "../include/manta/parts/structure/mass.hpp"
#include "../include/manta/parts/actuator/thruster.hpp"
#include "test_helpers.hpp"

using namespace manta;
using namespace manta::geom;
using namespace manta::parts;
using namespace manta::fields;

// ---- Mass (point-mass shorthand replaces deleted PointMass) ----

TEST_CASE("Mass(name, mass): point-mass shorthand contributes mass with zero MOI") {
    Craft c("test");
    auto& m = c.root().add<Mass>("body", 2.5f);
    (void)m;
    c.root().compute_params();
    CHECK(c.root().get_mass() == doctest::Approx(2.5f));
    // MOI defaults to zero for the point-mass overload.
    CHECK(c.root().get_moi().raw()(0, 0) == doctest::Approx(0.0f));
}

TEST_CASE("Mass: no wrench when no GravityField is registered") {
    Craft c("test");
    c.root().add<Mass>("body", 1.0f);
    c.update();
    auto w = c.root().net_wrench();
    CHECK(test::approx_equal(w.force(),  Vec3<PartFrame>::zero()));
    CHECK(test::approx_equal(w.torque(), Vec3<PartFrame>::zero()));
}

// ---- Thruster (1st-order linear, backward-compat constructor) ----

TEST_CASE("Thruster: zero throttle → no wrench") {
    Craft c("test");
    auto& t = c.root().add<Thruster>("t", 100.0f);
    t.set_throttle(0.0f);
    c.update();
    CHECK(test::approx_equal(c.root().net_wrench().force(), Vec3<PartFrame>::zero()));
}

TEST_CASE("Thruster: full throttle → max thrust along +z") {
    Craft c("test");
    auto& t = c.root().add<Thruster>("t", 10.0f);
    t.set_throttle(1.0f);
    c.update();
    CHECK(test::approx_equal(c.root().net_wrench().force(), Vec3<PartFrame>{0, 0, 10}));
}

TEST_CASE("Thruster: half throttle → half force (linear in throttle)") {
    Craft c("test");
    auto& t = c.root().add<Thruster>("t", 8.0f);
    t.set_throttle(0.5f);
    c.update();
    CHECK(test::approx_equal(c.root().net_wrench().force(), Vec3<PartFrame>{0, 0, 4}));
}

TEST_CASE("Thruster: throttle clamped to [0, 1]") {
    Craft c("test");
    auto& t = c.root().add<Thruster>("t", 10.0f);
    t.set_throttle(2.0f);
    CHECK(t.throttle() == doctest::Approx(1.0f));
    t.set_throttle(-0.5f);
    CHECK(t.throttle() == doctest::Approx(0.0f));
}

TEST_CASE("Thruster: custom direction") {
    Craft c("test");
    auto& t = c.root().add<Thruster>("t", 5.0f, Vec3<PartFrame>{1, 0, 0});
    t.set_throttle(1.0f);
    c.update();
    CHECK(test::approx_equal(c.root().net_wrench().force(), Vec3<PartFrame>{5, 0, 0}));
}

TEST_CASE("Thruster: tensor-style construction with reaction torque") {
    // Equivalent of the old PropThruster: F_1 = +z·max_thrust, τ_1 = -z·k_t·max_thrust.
    // 10 N thrust, 0.02 torque coefficient, CCW (negative τ_z).
    std::array<Vec3<PartFrame>, 1> F{Vec3<PartFrame>{0, 0, 10.0f}};
    std::array<Vec3<PartFrame>, 1> T{Vec3<PartFrame>{0, 0, -0.2f}};
    Craft c("test");
    auto& th = c.root().add<Thruster1>("th", F, T);
    th.set_throttle(1.0f);
    c.update();
    auto w = c.root().net_wrench();
    CHECK(test::approx_equal(w.force(),  Vec3<PartFrame>{0, 0, 10.0f}));
    CHECK(test::approx_equal(w.torque(), Vec3<PartFrame>{0, 0, -0.2f}));
}

TEST_CASE("Thruster2: quadratic-in-throttle force composition") {
    // F = F_1 · t + F_2 · t². At t = 0.5, F_z = 10·0.5 + 4·0.25 = 6.0.
    std::array<Vec3<PartFrame>, 2> F{
        Vec3<PartFrame>{0, 0, 10.0f},
        Vec3<PartFrame>{0, 0,  4.0f}};
    std::array<Vec3<PartFrame>, 2> T{
        Vec3<PartFrame>{0, 0, 0},
        Vec3<PartFrame>{0, 0, 0}};
    Craft c("test");
    auto& th = c.root().add<Thruster2>("th", F, T);
    th.set_throttle(0.5f);
    c.update();
    CHECK(c.root().net_wrench().force().z() == doctest::Approx(6.0f).epsilon(1e-4f));
}

TEST_CASE("Thruster: thrust drives craft dynamics") {
    Craft c("test");
    auto& t = c.root().add<Thruster>("t", 1.0f);
    t.set_throttle(1.0f);
    t.set_mass(1.0f);
    c.root().compute_params();

    c.update(1.0f);

    auto vel = c.scene_to_craft().vel_linear();
    CHECK(test::approx_equal(vel, Vec3<SceneFrame>{0, 0, 1}));
}

// ---- GravityField (new disturbance API) ----

TEST_CASE("GravityField: convenience constructor adds a uniform PERSISTENT disturbance") {
    GravityField gf{Vec3<SceneFrame>{0, 0, -9.81f}};
    auto g_at_origin = gf.state_at(Vec3<SceneFrame>::zero());
    CHECK(g_at_origin.z() == doctest::Approx(-9.81f));
    CHECK(g_at_origin.x() == doctest::Approx(0.0f));
    CHECK(gf.disturbance_count() == 1u);
}

TEST_CASE("GravityField: default-constructed field has no gravity until you add one") {
    GravityField gf;
    CHECK(gf.disturbance_count() == 0u);
    auto g = gf.state_at(Vec3<SceneFrame>::zero());
    CHECK(g.raw().norm() == doctest::Approx(0.0f));
}

// ---- IMU ----

TEST_CASE("IMU: reports zero acceleration and angular velocity at rest") {
    Craft c("test");
    auto& imu = c.root().add<IMU>("imu");
    imu.set_mass(0.1f);
    c.root().compute_params();

    c.update(0.01f);
    c.update(0.01f);

    CHECK(test::approx_equal(imu.last_accel(), Vec3<PartFrame>::zero()));
    CHECK(test::approx_equal(imu.last_gyro(),  Vec3<PartFrame>::zero()));
}

TEST_CASE("IMU: reports non-zero acceleration under thrust (no noise)") {
    Craft c("test");
    auto& t   = c.root().add<Thruster>("t",   2.0f);
    auto& imu = c.root().add<IMU>("imu");
    t.set_throttle(1.0f);
    t.set_mass(1.0f);
    c.root().compute_params();

    c.update(0.01f);
    c.update(0.01f);

    CHECK(imu.last_accel().z() == doctest::Approx(2.0f).epsilon(1e-3f));
}

TEST_CASE("IMU: noise zero-sigma leaves measurement unchanged") {
    noise_seed(0);
    Craft c("test");
    auto& imu = c.root().add<IMU>("imu", ImuNoiseParams{.accel_sigma=0.0f, .gyro_sigma=0.0f});
    c.root().compute_params();
    c.update(0.01f);
    c.update(0.01f);

    CHECK(test::approx_equal(imu.last_accel(), Vec3<PartFrame>::zero()));
    CHECK(test::approx_equal(imu.last_gyro(),  Vec3<PartFrame>::zero()));
}

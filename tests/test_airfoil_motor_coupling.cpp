// Regression: ex2 quadcopter diverges to NaN within ~70 ms of startup
// purely from the airfoil + motor coupling (verified by isolation —
// vacuum or no-airfoil configurations are stable). This test reproduces
// the failure mode minimally:
//
//   Craft = Mass(body) + Motor(z-axis, passive) + Naca00xx(blade)
//   World = GravityField(z-down) + FluidField(uniform air at rest)
//   Init  = small ω_y to seed the loop the ex2 sim grew on its own
//
// If the airfoil + articulated kinematics are correct, body state should
// stay bounded — drag may slow the body, but |v|, |ω|, and the motor
// rate must not run away to ±10^9+ in 1 s.

#include <doctest/doctest.h>

#include <cmath>

#include "../include/manta/core/craft.hpp"
#include "../include/manta/core/scene.hpp"
#include "../include/manta/core/world.hpp"
#include "../include/manta/fields/fluid_field.hpp"
#include "../include/manta/fields/gravity_field.hpp"
#include "../include/manta/parts/aero/naca00xx.hpp"
#include "../include/manta/parts/articulation/motor.hpp"
#include "../include/manta/parts/structure/mass.hpp"

using namespace manta;
using namespace manta::geom;
using namespace manta::fields;
using namespace manta::parts;

namespace {

// Soft check: every component of the craft state is a finite number with
// absolute value below `cap`. Returns the offending value (or 0 if
// everything is fine) so doctest's INFO macro can pinpoint blow-up.
template <class Craft>
double bounded(const Craft& c, double cap) {
    auto check = [cap](double v) -> double {
        if (!std::isfinite(v) || std::abs(v) > cap) return v;
        return 0.0;
    };
    const auto p = c.scene_to_craft().position().raw();
    const auto v = c.scene_to_craft().vel_linear().raw();
    const auto w = c.scene_to_craft().vel_angular().raw();
    for (int i = 0; i < 3; ++i) {
        if (auto x = check(p[i]); x != 0.0) return x;
        if (auto x = check(v[i]); x != 0.0) return x;
        if (auto x = check(w[i]); x != 0.0) return x;
    }
    return 0.0;
}

}  // namespace

// Baseline: no airfoil, just Motor + hub mass under gravity in air.
// Confirms the motor-only craft falls cleanly (sanity for the test
// scaffolding before we add the aero part).
TEST_CASE("Airfoil regression: motor-only craft is stable in free fall") {
    constexpr MFloat dt = MFloat(0.001);
    constexpr int    N  = 1000;   // 1 s

    World w;
    w.clock().set_dt(dt);
    GravityField grav(GravityField::Vec{MFloat(0), MFloat(0), MFloat(-9.81)});
    FluidField   air;
    air.add(FluidDisturbance::uniform_gas(
        /*R=*/MFloat(287.05), /*T=*/MFloat(288.15), /*p=*/MFloat(101325)),
        PERSISTENT);
    w.register_field(grav);
    w.register_field(air);

    Craft c("motor_only");
    c.root().add<Mass>("body", MFloat(1.0), [] {
        Mat3<PartFrame, PartFrame> m = Mat3<PartFrame, PartFrame>::identity();
        m.raw()(0,0) = MFloat(0.01);
        m.raw()(1,1) = MFloat(0.01);
        m.raw()(2,2) = MFloat(0.02);
        return m;
    }());
    auto& motor = c.root().add<Motor>("m", Vec3<PartFrame>{0, 0, 1});
    motor.add<Mass>("hub", MFloat(0.05), [] {
        Mat3<PartFrame, PartFrame> m = Mat3<PartFrame, PartFrame>::identity();
        m.raw()(0,0) = MFloat(1e-3);
        m.raw()(1,1) = MFloat(1e-3);
        m.raw()(2,2) = MFloat(1e-3);
        return m;
    }());
    c.root().compute_params();

    auto& scene = w.create_scene();
    InitialState init;
    init.position = Vec3<SceneFrame>{0, 0, 1};       // 1 m above origin
    init.vel_angular = Vec3<CraftFrame>{0, MFloat(1e-3), 0};   // small ω_y seed
    scene.add_craft(c, init);

    for (int i = 0; i < N; ++i) w.step();

    double bad = bounded(c, /*cap=*/1e3);
    INFO("offending value: ", bad);
    CHECK(bad == 0.0);
}

// The actual repro: add one Naca00xx blade under the motor. If the
// airfoil + articulated-kinematics coupling is correct, this should
// stay bounded just like the motor-only case — the airfoil produces
// some drag-like force that may translate or slow the body, but
// nothing should run away.
TEST_CASE("Airfoil regression: motor + one airfoil blade is stable") {
    constexpr MFloat dt = MFloat(0.001);
    constexpr int    N  = 1000;   // 1 s

    World w;
    w.clock().set_dt(dt);
    GravityField grav(GravityField::Vec{MFloat(0), MFloat(0), MFloat(-9.81)});
    FluidField   air;
    air.add(FluidDisturbance::uniform_gas(
        MFloat(287.05), MFloat(288.15), MFloat(101325)),
        PERSISTENT);
    w.register_field(grav);
    w.register_field(air);

    Craft c("airfoil_repro");
    c.root().add<Mass>("body", MFloat(1.0), [] {
        Mat3<PartFrame, PartFrame> m = Mat3<PartFrame, PartFrame>::identity();
        m.raw()(0,0) = MFloat(0.01);
        m.raw()(1,1) = MFloat(0.01);
        m.raw()(2,2) = MFloat(0.02);
        return m;
    }());
    auto& motor = c.root().add<Motor>("m", Vec3<PartFrame>{0, 0, 1});
    motor.add<Mass>("hub", MFloat(0.05), [] {
        Mat3<PartFrame, PartFrame> m = Mat3<PartFrame, PartFrame>::identity();
        m.raw()(0,0) = MFloat(1e-3);
        m.raw()(1,1) = MFloat(1e-3);
        m.raw()(2,2) = MFloat(1e-3);
        return m;
    }());

    // One blade, mounted exactly like ex2's CCW blade-0: at +0.10 m on
    // motor +x, rotated −90° about motor +z so the chord lies along
    // motor ±y, then pitched −12° about the blade's +y so the LE tilts
    // up by 12°.
    auto& blade = motor.add<Naca00xx>("blade",
        /*chord=*/MFloat(0.03), /*span=*/MFloat(0.20),
        /*t/c=*/MFloat(0.12), /*N=*/4);
    Eigen::Quaternion<MFloat> q_rot_z;
    q_rot_z = Eigen::AngleAxis<MFloat>(MFloat(-M_PI/2),
                                       Eigen::Matrix<MFloat,3,1>::UnitZ());
    Eigen::Quaternion<MFloat> q_pitch;
    q_pitch = Eigen::AngleAxis<MFloat>(MFloat(-12.0 * M_PI / 180.0),
                                       Eigen::Matrix<MFloat,3,1>::UnitY());
    Eigen::Quaternion<MFloat> q_install = q_rot_z * q_pitch;
    blade.set_transform(StaticLink<ParentFrame, PartFrame>{
        Vec3<ParentFrame>{MFloat(0.10), 0, 0},
        Ori<ParentFrame>{q_install}});
    c.root().compute_params();

    auto& scene = w.create_scene();
    InitialState init;
    init.position = Vec3<SceneFrame>{0, 0, 1};
    init.vel_angular = Vec3<CraftFrame>{0, MFloat(1e-3), 0};
    scene.add_craft(c, init);

    for (int i = 0; i < N; ++i) w.step();

    double bad = bounded(c, /*cap=*/1e3);
    INFO("offending value: ", bad);
    INFO("motor rate: ", motor.rate());
    INFO("body |v|: ", c.scene_to_craft().vel_linear().raw().norm());
    INFO("body |ω|: ", c.scene_to_craft().vel_angular().raw().norm());
    CHECK(bad == 0.0);
}

// Two-blade prop (ex2's blade pair: 180° apart, chirality-matched).
// Drag should be symmetric → no net moment on motor. Lift is also
// symmetric if AOA is symmetric.
TEST_CASE("Airfoil regression: motor + 2 opposed blades (one prop) is stable") {
    constexpr MFloat dt = MFloat(0.001);
    constexpr int    N  = 1000;

    World w;
    w.clock().set_dt(dt);
    GravityField grav(GravityField::Vec{MFloat(0), MFloat(0), MFloat(-9.81)});
    FluidField   air;
    air.add(FluidDisturbance::uniform_gas(
        MFloat(287.05), MFloat(288.15), MFloat(101325)),
        PERSISTENT);
    w.register_field(grav);
    w.register_field(air);

    Craft c("two_blade_prop");
    c.root().add<Mass>("body", MFloat(1.0), [] {
        Mat3<PartFrame, PartFrame> m = Mat3<PartFrame, PartFrame>::identity();
        m.raw()(0,0) = MFloat(0.01);
        m.raw()(1,1) = MFloat(0.01);
        m.raw()(2,2) = MFloat(0.02);
        return m;
    }());
    auto& motor = c.root().add<Motor>("m", Vec3<PartFrame>{0, 0, 1});
    motor.add<Mass>("hub", MFloat(0.05), [] {
        Mat3<PartFrame, PartFrame> m = Mat3<PartFrame, PartFrame>::identity();
        m.raw()(0,0) = MFloat(1e-3);
        m.raw()(1,1) = MFloat(1e-3);
        m.raw()(2,2) = MFloat(1e-3);
        return m;
    }());

    // Full ex2 install: chord rotation (±90° about z) + pitch (−12° about y).
    Eigen::Quaternion<MFloat> q_pitch;
    q_pitch = Eigen::AngleAxis<MFloat>(MFloat(-12.0 * M_PI / 180.0),
                                       Eigen::Matrix<MFloat,3,1>::UnitY());
    Eigen::Quaternion<MFloat> q_neg90;
    q_neg90 = Eigen::AngleAxis<MFloat>(MFloat(-M_PI/2),
                                       Eigen::Matrix<MFloat,3,1>::UnitZ());
    Eigen::Quaternion<MFloat> q_pos90;
    q_pos90 = Eigen::AngleAxis<MFloat>(MFloat(+M_PI/2),
                                       Eigen::Matrix<MFloat,3,1>::UnitZ());
    auto& blade_a = motor.add<Naca00xx>("blade_a",
        MFloat(0.03), MFloat(0.20), MFloat(0.12), 4);
    blade_a.set_transform(StaticLink<ParentFrame, PartFrame>{
        Vec3<ParentFrame>{MFloat(0.10), 0, 0},
        Ori<ParentFrame>{Eigen::Quaternion<MFloat>(q_neg90 * q_pitch)}});

    auto& blade_b = motor.add<Naca00xx>("blade_b",
        MFloat(0.03), MFloat(0.20), MFloat(0.12), 4);
    blade_b.set_transform(StaticLink<ParentFrame, PartFrame>{
        Vec3<ParentFrame>{MFloat(-0.10), 0, 0},
        Ori<ParentFrame>{Eigen::Quaternion<MFloat>(q_pos90 * q_pitch)}});
    c.root().compute_params();

    auto& scene = w.create_scene();
    InitialState init;
    init.position = Vec3<SceneFrame>{0, 0, 1};
    init.vel_angular = Vec3<CraftFrame>{0, MFloat(1e-3), 0};
    scene.add_craft(c, init);

    for (int i = 0; i < N; ++i) w.step();

    double bad = bounded(c, /*cap=*/1e3);
    INFO("offending value: ", bad);
    INFO("motor rate: ", motor.rate());
    INFO("body |v|: ", c.scene_to_craft().vel_linear().raw().norm());
    INFO("body |ω|: ", c.scene_to_craft().vel_angular().raw().norm());
    CHECK(bad == 0.0);
}

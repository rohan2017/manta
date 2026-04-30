#include <cmath>
#include <doctest/doctest.h>

#include "../include/manta/core/craft.hpp"
#include "../include/manta/core/scene.hpp"
#include "../include/manta/core/world.hpp"
#include "../include/manta/fields/gravity_field.hpp"
#include "../include/manta/fields/ocean_atmos_field.hpp"
#include "../include/manta/fields/uniform_fluid_field.hpp"
#include "../include/manta/parts/field_src/gravity_part.hpp"
#include "../include/manta/parts/structure/hull.hpp"
#include "../include/manta/parts/structure/point_mass.hpp"
#include "test_helpers.hpp"

using namespace manta;
using namespace manta::geom;
using namespace manta::parts;
using namespace manta::fields;

// ---- FluidField unit tests ----

TEST_CASE("UniformFluidField: returns constant density and velocity") {
    UniformFluidField uf(1000.0f, Vec3<SceneFrame>{1, 2, 3});
    auto s1 = uf.state_at(Vec3<SceneFrame>::zero());
    auto s2 = uf.state_at(Vec3<SceneFrame>{99, -50, 1000});
    CHECK(s1.density == doctest::Approx(1000.0f));
    CHECK(s2.density == doctest::Approx(1000.0f));
    CHECK(test::approx_equal(s1.velocity, Vec3<SceneFrame>{1, 2, 3}));
    CHECK(test::approx_equal(s2.velocity, s1.velocity));
}

TEST_CASE("OceanAtmosField: dense water below surface, light air above") {
    OceanAtmosField oa(/*sea=*/0.0f, /*water=*/1000.0f, /*air=*/1.225f);
    CHECK(oa.state_at(Vec3<SceneFrame>{0, 0, -5}).density == doctest::Approx(1000.0f));
    CHECK(oa.state_at(Vec3<SceneFrame>{0, 0,  5}).density == doctest::Approx(1.225f));
    CHECK(oa.state_at(Vec3<SceneFrame>{0, 0,  0}).density == doctest::Approx(1.225f));  // at sea level → air
}

TEST_CASE("OceanAtmosField: height_above_sea_level is signed altitude") {
    OceanAtmosField oa(/*sea=*/10.0f);
    CHECK(oa.height_above_sea_level(Vec3<SceneFrame>{0, 0, 25}) == doctest::Approx(15.0f));
    CHECK(oa.height_above_sea_level(Vec3<SceneFrame>{0, 0,  5}) == doctest::Approx(-5.0f));
}

TEST_CASE("OceanAtmosField: currents apply only below surface, wind only above") {
    OceanAtmosField oa(0.0f);
    oa.set_current(Vec3<SceneFrame>{0.5f, 0, 0});
    oa.set_wind   (Vec3<SceneFrame>{0, 10.0f, 0});
    auto under = oa.state_at(Vec3<SceneFrame>{0, 0, -1});
    auto over  = oa.state_at(Vec3<SceneFrame>{0, 0,  1});
    CHECK(test::approx_equal(under.velocity, Vec3<SceneFrame>{0.5f, 0, 0}));
    CHECK(test::approx_equal(over.velocity,  Vec3<SceneFrame>{0, 10.0f, 0}));
}

// ---- Hull tests ----

namespace {
// 4 sample points along the body's z-axis (gravity-aligned for an upright
// craft), spanning depth ±0.5 m around the part origin.
std::vector<Vec3<PartFrame>> z_column_4(float half_height = 0.5f) {
    return {
        Vec3<PartFrame>{0, 0,  half_height},
        Vec3<PartFrame>{0, 0,  half_height / 3},
        Vec3<PartFrame>{0, 0, -half_height / 3},
        Vec3<PartFrame>{0, 0, -half_height},
    };
}
}

TEST_CASE("Hull: fully submerged in uniform water → Archimedes buoyant force") {
    // Volume 0.001 m^3 (1 liter), water density 1000, g=9.81
    // F_buoy = ρVg = 1000 * 0.001 * 9.81 = 9.81 N upward (+z)
    UniformFluidField water(1000.0f);
    GravityField gf{Vec3<SceneFrame>{0, 0, -9.81f}};

    Craft c("test");
    c.register_field<FluidField>(water);
    c.register_field(gf);

    c.root().add<PointMass>("m", 0.0f);  // ignore body weight
    c.root().add<Hull>("hull", 0.001f, z_column_4(0.1f));
    c.root().compute_params();

    c.update();

    // Sum of net force at the root, in PartFrame == CraftFrame at identity.
    auto F = c.root().net_wrench().force();
    CHECK(F.z() == doctest::Approx(9.81f).epsilon(1e-3f));
    CHECK(F.x() == doctest::Approx(0.0f).epsilon(1e-3f));
    CHECK(F.y() == doctest::Approx(0.0f).epsilon(1e-3f));
}

TEST_CASE("Hull: fully out of water in air → buoyancy ≈ ρ_air * V * g (very small)") {
    OceanAtmosField oa(/*sea=*/0.0f);
    GravityField gf;  // default -9.81

    Craft c("test");
    c.register_field<FluidField>(oa);  // register OceanAtmosField under FluidField slot
    c.register_field(gf);

    c.root().add<PointMass>("m", 0.0f);
    // Place hull entirely above sea level.
    auto& hull = c.root().add<Hull>("hull", 0.001f, z_column_4(0.1f));
    StaticLink<ParentFrame, PartFrame> tf{
        Vec3<ParentFrame>{0, 0, 5.0f},  // 5 m up
        Ori<ParentFrame>::identity()};
    hull.set_transform(tf);
    c.root().compute_params();

    c.update();

    // F = ρ_air * V * g = 1.225 * 0.001 * 9.81 ≈ 0.012 N
    auto F = c.root().net_wrench().force();
    CHECK(F.z() == doctest::Approx(1.225f * 0.001f * 9.81f).epsilon(1e-3f));
}

TEST_CASE("Hull: half-submerged in ocean → buoyancy ~half of fully-submerged") {
    // Hull at sea level with samples spanning ±0.1 m. Two samples below, two
    // above. With volume 0.001, ρ_water 1000, ρ_air 1.225:
    //   F = (2 * 1000 + 2 * 1.225) * (0.001 / 4) * 9.81
    //     ≈ 4.911 N ≈ 0.5 * full water (9.81 N)
    OceanAtmosField oa(/*sea=*/0.0f);
    GravityField gf;

    Craft c("test");
    c.register_field<FluidField>(oa);
    c.register_field(gf);

    c.root().add<PointMass>("m", 0.0f);
    c.root().add<Hull>("hull", 0.001f, z_column_4(0.1f));
    c.root().compute_params();

    c.update();

    auto F = c.root().net_wrench().force();
    float expected = (2 * 1000.0f + 2 * 1.225f) * (0.001f / 4) * 9.81f;
    CHECK(F.z() == doctest::Approx(expected).epsilon(1e-3f));
}

TEST_CASE("Hull: floats at neutral buoyancy") {
    // Hull mass = 1 kg, volume = 0.001 m^3 displacing water (1000 kg/m^3 * 0.001 = 1 kg).
    // Hull weight = m*g = 9.81 N down; full buoyant force = 9.81 N up.
    // When fully submerged: net = 0 → no acceleration.
    // Spawn fully submerged, no initial velocity, verify it stays put.
    UniformFluidField water(1000.0f);
    GravityField gf;

    World w;
    w.register_field<FluidField>(water);
    w.register_field(gf);
    w.clock().set_dt(0.01f);

    auto& scene = w.create_scene();
    Craft c("boat");
    c.root().add<PointMass>("body", 1.0f);
    c.root().add<GravityPart>("grav");
    c.root().add<Hull>("hull", 0.001f, z_column_4(0.1f));
    c.root().compute_params();
    c.set_position(Vec3<SceneFrame>{0, 0, -1.0f});  // 1 m underwater
    scene.add_craft(c);

    for (int i = 0; i < 100; ++i) w.update();

    auto v = c.scene_to_craft().vel_linear();
    auto p = c.scene_to_craft().position();
    CHECK(v.z() == doctest::Approx(0.0f).epsilon(1e-3f));
    CHECK(p.z() == doctest::Approx(-1.0f).epsilon(1e-3f));
}

// ---- Hull compile-time augmentation ----

TEST_CASE("Hull: deep submersion ~ same as base water buoyancy") {
    // 1 m below sea level, with 0.05 m smoothing → smoothstep is saturated to 1
    // for every sample → ρ_eff == ρ_water for all → matches base behavior.
    OceanAtmosField oa(0.0f);
    GravityField gf;

    Craft c("test");
    c.register_field<FluidField>(oa);
    c.register_field(gf);

    c.root().add<PointMass>("m", 0.0f);
    auto& hull = c.root().add<Hull>("hull", 0.001f, z_column_4(0.05f));
    StaticLink<ParentFrame, PartFrame> tf{
        Vec3<ParentFrame>{0, 0, -1.0f},  // 1 m underwater
        Ori<ParentFrame>::identity()};
    hull.set_transform(tf);
    c.root().compute_params();

    c.update();

    auto F = c.root().net_wrench().force();
    // Full water buoyancy: ρVg = 1000 * 0.001 * 9.81 = 9.81 N
    CHECK(F.z() == doctest::Approx(9.81f).epsilon(1e-3f));
}

TEST_CASE("Hull: lift varies smoothly across surface (no step)") {
    // Sample buoyancy at three nearby surface positions. With a hard step
    // (base Hull), a 1mm vertical move that crosses one sample's surface
    // would produce a step of ~ ρ_water * v_per * g ≈ 2.45 N (large).
    // With smoothing 0.05 m, the step is replaced by a continuous ramp:
    // moving by a small fraction of h should produce a small force change.
    GravityField gf;

    auto sample_lift_at = [&](float z_offset, Real smoothing) {
        OceanAtmosField oa(0.0f);
        Craft c("t");
        c.register_field<FluidField>(oa);
        c.register_field(gf);
        c.root().add<PointMass>("m", 0.0f);
        auto& hull = c.root().add<Hull>("hull", 0.001f, z_column_4(0.1f));
        hull.set_surface_smoothing(smoothing);
        StaticLink<ParentFrame, PartFrame> tf{
            Vec3<ParentFrame>{0, 0, z_offset},
            Ori<ParentFrame>::identity()};
        hull.set_transform(tf);
        c.root().compute_params();
        c.update();
        return c.root().net_wrench().force().z();
    };

    // Use smoothing wider than the sample column so multiple samples sit in
    // the blend zone simultaneously and a small move actually changes ρ_eff.
    float F0  = sample_lift_at(0.000f, 0.30f);
    float F1  = sample_lift_at(0.001f, 0.30f);  // moved up 1 mm
    float F2  = sample_lift_at(0.005f, 0.30f);  // moved up 5 mm
    // Force should decrease as hull rises (less submerged).
    CHECK(F1 < F0);
    CHECK(F2 < F1);
    // The decrease per mm should be small relative to a full-sample step.
    // A full sample's contribution: 1000 * (0.001/4) * 9.81 ≈ 2.45 N.
    // With h=0.3 m, smoothstep slope at center ≈ 1.5/h → per mm change in
    // ρ_eff ≈ 0.005 * (ρ_water - ρ_air) → per-sample force change ≈ 0.012 N.
    CHECK((F0 - F1) < 0.5f);  // far less than 2.45 N (the hard-step value)
}

TEST_CASE("Hull: smoothing→0 collapses to hard step") {
    // With surface_smoothing → very small (0.001), behavior approaches the
    // hard-step base case: half-submerged at z=0 with samples at ±0.1, ±0.033
    // gives 2 below + 2 above the surface (at z=0, sample at z=0 is exactly at
    // surface and gets weight 0.5; nudge by ε so we land cleanly).
    OceanAtmosField oa(0.0f);
    GravityField gf;
    Craft c("t");
    c.register_field<FluidField>(oa);
    c.register_field(gf);
    c.root().add<PointMass>("m", 0.0f);
    auto& hull = c.root().add<Hull>("hull", 0.001f, z_column_4(0.1f));
    hull.set_surface_smoothing(1e-4f);  // effectively a step
    c.root().compute_params();

    c.update();

    auto F = c.root().net_wrench().force();
    // Same expected value as the base half-submerged test (within tighter tol):
    float expected = (2 * 1000.0f + 2 * 1.225f) * (0.001f / 4) * 9.81f;
    CHECK(F.z() == doctest::Approx(expected).epsilon(0.01f));
}

TEST_CASE("Hull: produces roll-righting torque when tilted in water") {
    // 4 samples in xy plane around origin. Tilt the hull about y-axis so half
    // the samples sit deeper than the others. Buoyant force differential about
    // the part origin yields a torque opposing the tilt.
    UniformFluidField water(1000.0f);
    GravityField gf;

    Craft c("test");
    c.register_field<FluidField>(water);
    c.register_field(gf);

    c.root().add<PointMass>("m", 1.0f);
    std::vector<Vec3<PartFrame>> ring = {
        Vec3<PartFrame>{ 0.5f, 0, 0},
        Vec3<PartFrame>{-0.5f, 0, 0},
        Vec3<PartFrame>{0,  0.5f, 0},
        Vec3<PartFrame>{0, -0.5f, 0},
    };
    c.root().add<Hull>("hull", 0.001f, ring);
    c.root().compute_params();

    // Tilt the craft 0.2 rad about scene y-axis.
    auto roll_axis = Eigen::AngleAxisf{0.2f, Eigen::Vector3f::UnitY()};
    Ori<SceneFrame> q{Eigen::Quaternionf{roll_axis}};
    c.set_orientation(q);

    c.update();

    // Equal-density fluid → all samples produce the same |F| in part frame
    // (buoyancy is along part-frame g, which itself rotates with the craft).
    // For uniform fluid, the symmetric ring gives zero NET torque even tilted.
    // Switch to ocean to get a torque from differential submersion:
    OceanAtmosField oa(/*sea=*/0.0f);
    Craft c2("test2");
    c2.register_field<FluidField>(oa);
    c2.register_field(gf);
    c2.root().add<PointMass>("m", 1.0f);
    c2.root().add<Hull>("hull", 0.001f, ring);
    c2.root().compute_params();
    c2.set_orientation(q);  // tilted 0.2 rad about y; samples at +x are deeper (negative z in scene)
    c2.update();

    auto T = c2.root().net_wrench().torque();
    // Roll about y direction: tilt is +y rotation; expect restoring torque (negative y).
    CHECK(T.y() < 0.0f);
}

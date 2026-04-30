#include <cmath>
#include <doctest/doctest.h>

#include "../include/manta/coupling/tether.hpp"
#include "../include/manta/core/craft.hpp"
#include "../include/manta/core/scene.hpp"
#include "../include/manta/core/world.hpp"
#include "../include/manta/parts/structure/point_mass.hpp"
#include "../include/manta/parts/coupling/tether_endpoint.hpp"
#include "test_helpers.hpp"

using namespace manta;
using namespace manta::geom;
using namespace manta::parts;
using namespace manta::coupling;

// ---- Tether dynamics unit tests ----

TEST_CASE("Tether: slack (d <= rest_length) → zero force") {
    Tether t(/*rest=*/10.0f, /*k=*/100.0f);
    auto F = t.force_on_self(
        Vec3<SceneFrame>{0, 0, 0},
        Vec3<SceneFrame>{5, 0, 0},  // separation 5 < 10
        Vec3<SceneFrame>::zero(),
        Vec3<SceneFrame>::zero());
    CHECK(test::approx_equal(F, Vec3<SceneFrame>::zero()));
}

TEST_CASE("Tether: at rest length exactly → zero force") {
    Tether t(10.0f, 100.0f);
    auto F = t.force_on_self(
        Vec3<SceneFrame>{0, 0, 0},
        Vec3<SceneFrame>{10, 0, 0},
        Vec3<SceneFrame>::zero(),
        Vec3<SceneFrame>::zero());
    CHECK(test::approx_equal(F, Vec3<SceneFrame>::zero()));
}

TEST_CASE("Tether: extended → Hooke's law toward sibling") {
    Tether t(/*rest=*/10.0f, /*k=*/100.0f);
    // d=12, extension=2, k=100 → |F|=200, direction +x (from self toward other)
    auto F = t.force_on_self(
        Vec3<SceneFrame>{0, 0, 0},
        Vec3<SceneFrame>{12, 0, 0},
        Vec3<SceneFrame>::zero(),
        Vec3<SceneFrame>::zero());
    CHECK(F.x() == doctest::Approx(200.0f));
    CHECK(F.y() == doctest::Approx(0.0f));
    CHECK(F.z() == doctest::Approx(0.0f));
}

TEST_CASE("Tether: equal-and-opposite forces on the two endpoints") {
    Tether t(10.0f, 100.0f);
    auto F_a = t.force_on_self(
        Vec3<SceneFrame>{0, 0, 0},
        Vec3<SceneFrame>{12, 0, 0},
        Vec3<SceneFrame>::zero(),
        Vec3<SceneFrame>::zero());
    auto F_b = t.force_on_self(
        Vec3<SceneFrame>{12, 0, 0},
        Vec3<SceneFrame>{0, 0, 0},
        Vec3<SceneFrame>::zero(),
        Vec3<SceneFrame>::zero());
    CHECK(test::approx_equal(F_a, Vec3<SceneFrame>::from_raw(-F_b.raw())));
}

TEST_CASE("Tether: damping resists separation when extended") {
    Tether t(10.0f, 100.0f, /*c=*/10.0f);
    // Stretched to 12 (extension 2), other moving away in +x at 1 m/s.
    // F_spring = 100 * 2 = 200; F_damp = 10 * 1 = 10; total = 210 (still toward +x).
    auto F = t.force_on_self(
        Vec3<SceneFrame>{0, 0, 0},
        Vec3<SceneFrame>{12, 0, 0},
        Vec3<SceneFrame>::zero(),
        Vec3<SceneFrame>{1, 0, 0});
    CHECK(F.x() == doctest::Approx(210.0f));
}

TEST_CASE("Tether: degenerate coincident points → zero force, no NaN") {
    Tether t(0.0f, 100.0f);
    auto F = t.force_on_self(
        Vec3<SceneFrame>::zero(),
        Vec3<SceneFrame>::zero(),
        Vec3<SceneFrame>::zero(),
        Vec3<SceneFrame>::zero());
    CHECK(test::approx_equal(F, Vec3<SceneFrame>::zero()));
}

// ---- TetherEndpoint integration tests ----

TEST_CASE("TetherEndpoint: stretched between two crafts → force pulls them together") {
    // Two 1-kg point-mass crafts, tether rest length 10, placed 12 apart
    // along +x, k=50 → |F| = 100 N on each. After 0.1s with 100 N on a 1kg
    // craft, |Δv| = 10 m/s. Crafts approach each other.
    Tether tether(/*rest=*/10.0f, /*k=*/50.0f, /*c=*/0.0f);

    World w;
    w.clock().set_dt(0.001f);
    auto& scene = w.create_scene();

    Craft c1("a"), c2("b");
    c1.root().add<PointMass>("body", 1.0f);
    c1.root().add<TetherEndpoint>("end", tether, /*is_a=*/true);
    c1.root().compute_params();
    c1.set_position(Vec3<SceneFrame>{0, 0, 0});
    scene.add_craft(c1);

    c2.root().add<PointMass>("body", 1.0f);
    c2.root().add<TetherEndpoint>("end", tether, /*is_a=*/false);
    c2.root().compute_params();
    c2.set_position(Vec3<SceneFrame>{12, 0, 0});
    scene.add_craft(c2);

    // Run a few ticks; verify both crafts have non-zero velocity toward each other.
    for (int i = 0; i < 50; ++i) w.update();

    auto v1 = c1.scene_to_craft().vel_linear();
    auto v2 = c2.scene_to_craft().vel_linear();

    // c1 starts at x=0, c2 at x=12. Tether pulls them together:
    // c1 should accelerate in +x, c2 in -x.
    CHECK(v1.x() > 0.0f);
    CHECK(v2.x() < 0.0f);
    // Equal masses + symmetric forces → equal-and-opposite velocities (within tol).
    CHECK(v1.x() == doctest::Approx(-v2.x()).epsilon(1e-3f));
}

TEST_CASE("TetherEndpoint: slack → no force, crafts drift independently") {
    Tether tether(20.0f, 100.0f);  // rest 20, separation will be 5

    World w;
    w.clock().set_dt(0.01f);
    auto& scene = w.create_scene();

    Craft c1("a"), c2("b");
    c1.root().add<PointMass>("body", 1.0f);
    c1.root().add<TetherEndpoint>("end", tether, true);
    c1.root().compute_params();
    c1.set_position   (Vec3<SceneFrame>{0, 0, 0});
    c1.set_vel_linear (Vec3<SceneFrame>{1, 0, 0});  // drift +x
    scene.add_craft(c1);

    c2.root().add<PointMass>("body", 1.0f);
    c2.root().add<TetherEndpoint>("end", tether, false);
    c2.root().compute_params();
    c2.set_position(Vec3<SceneFrame>{5, 0, 0});
    scene.add_craft(c2);

    for (int i = 0; i < 100; ++i) w.update();  // 1 s

    auto v1 = c1.scene_to_craft().vel_linear();
    auto v2 = c2.scene_to_craft().vel_linear();
    // Tether stays slack the whole time → c1 keeps its initial velocity, c2 stays at rest.
    CHECK(v1.x() == doctest::Approx(1.0f).epsilon(1e-4f));
    CHECK(test::approx_equal(v2, Vec3<SceneFrame>::zero(), 1e-4f));
}

TEST_CASE("TetherEndpoint: same-craft tether between two parts produces zero net force on craft") {
    // Two endpoints on the SAME craft, separated along x. Equal-and-opposite
    // forces should cancel at the craft level (no net translation). Internal
    // torque about CoM may exist, but the crafts CoM acceleration stays zero
    // because internal forces don't accelerate a closed system.
    Tether tether(1.0f, 100.0f);

    World w;
    w.clock().set_dt(0.01f);
    auto& scene = w.create_scene();

    Craft c("rigid");
    c.root().add<PointMass>("body", 1.0f);

    auto& a = c.root().add<TetherEndpoint>("a", tether, true);
    StaticLink<ParentFrame, PartFrame> tf_a{
        Vec3<ParentFrame>{ 2, 0, 0}, Ori<ParentFrame>::identity()};
    a.set_transform(tf_a);

    auto& b = c.root().add<TetherEndpoint>("b", tether, false);
    StaticLink<ParentFrame, PartFrame> tf_b{
        Vec3<ParentFrame>{-2, 0, 0}, Ori<ParentFrame>::identity()};
    b.set_transform(tf_b);

    c.root().compute_params();
    scene.add_craft(c);

    for (int i = 0; i < 50; ++i) w.update();

    // Internal force pair → no net translational motion of the craft origin.
    auto v = c.scene_to_craft().vel_linear();
    CHECK(test::approx_equal(v, Vec3<SceneFrame>::zero(), 1e-3f));
}

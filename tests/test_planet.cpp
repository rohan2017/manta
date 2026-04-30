#include <doctest/doctest.h>

#include "manta/core/craft.hpp"
#include "manta/core/planet.hpp"
#include "manta/core/scene.hpp"
#include "manta/core/world.hpp"
#include "manta/parts/structure/point_mass.hpp"

using namespace manta;
using namespace manta::geom;

namespace {

// Concrete planet for tests: tracks how many times update() was called and
// optionally rotates about z at a fixed rate.
class TestPlanet : public Planet {
public:
    explicit TestPlanet(std::string name = "test_planet",
                        Real omega_z = Real(0))
        : Planet(std::move(name)), omega_z_(omega_z) {
        // Set initial vel_angular so world_to_planet_ has a rotation rate.
        world_to_planet_.set_vel_angular(
            geom::Vec3<PlanetFrame>{Real(0), Real(0), omega_z_});
    }

    int update_count() const noexcept { return update_count_; }
    Real elapsed_t() const noexcept { return last_t_; }

    void update(Real t, Real /*dt*/) override {
        ++update_count_;
        last_t_ = t;
    }

    bool registered = false;
    void register_disturbances(World& /*w*/) override { registered = true; }

private:
    Real omega_z_;
    int  update_count_ = 0;
    Real last_t_ = Real(0);
};

} // namespace

TEST_CASE("Planet: World::add_planet calls register_disturbances") {
    World w;
    auto& p = w.add_planet<TestPlanet>("p1");
    CHECK(p.registered == true);
    CHECK(p.name() == "p1");
    CHECK(w.planets().size() == 1u);
}

TEST_CASE("Planet: World::update advances each planet via update()") {
    World w;
    w.clock().set_dt(0.01f);
    auto& p = w.add_planet<TestPlanet>("p1");

    CHECK(p.update_count() == 0);
    w.update();
    CHECK(p.update_count() == 1);
    w.update();
    CHECK(p.update_count() == 2);
}

TEST_CASE("Planet: world_to_planet defaults to identity, can carry rotation") {
    TestPlanet p("rotor", /*omega_z=*/Real(0.5));
    const auto& wtp = p.world_to_planet();
    CHECK(wtp.position().raw().norm() == doctest::Approx(0.0));
    // Default identity orientation.
    CHECK(wtp.orientation().raw().w() == doctest::Approx(1.0));
    // Angular velocity along z was set in the test planet's constructor.
    CHECK(wtp.vel_angular().z() == doctest::Approx(0.5));
}

TEST_CASE("Scene: optional planet anchor, defaults to none") {
    World w;
    auto& s = w.create_scene();
    CHECK(s.has_planet() == false);
    CHECK(s.planet() == nullptr);

    auto& p = w.add_planet<TestPlanet>("earth_test");
    s.set_planet(&p);
    CHECK(s.has_planet() == true);
    CHECK(s.planet() == &p);
}

TEST_CASE("Scene: world_to_scene with no planet — static, zero motion") {
    World w;
    w.clock().set_dt(0.01f);
    auto& s = w.create_scene();
    w.update();
    const auto& wts = s.world_to_scene();
    CHECK(wts.position().raw().norm() == doctest::Approx(0.0));
    CHECK(wts.vel_angular().raw().norm() == doctest::Approx(0.0));
    CHECK(wts.vel_linear().raw().norm() == doctest::Approx(0.0));
}

TEST_CASE("Scene: world_to_scene picks up planet's rotation rate") {
    World w;
    w.clock().set_dt(0.01f);
    auto& p = w.add_planet<TestPlanet>("rotor", /*omega_z=*/Real(0.5));
    auto& s = w.create_scene();
    s.set_planet(&p);

    w.update();

    const auto& wts = s.world_to_scene();
    // Scene origin coincides with planet origin (planet_to_scene = identity)
    // → no linear translation, but the angular velocity flows through.
    CHECK(wts.vel_angular().z() == doctest::Approx(0.5));
}

// ---- Phase 3: rotating-frame fictitious forces ----

// A craft at rest in a planet-rotating scene should accelerate outward due
// to centrifugal: a_cf = -ω × (ω × r). For ω_z = 1 rad/s and r = (1, 0, 0),
// ω × r = (0, 1, 0), ω × (ω × r) = (-1, 0, 0), so a_cf = (1, 0, 0).
TEST_CASE("Planet phase 3: centrifugal acceleration on a craft at rest in a "
          "rotating scene") {
    World w;
    w.clock().set_dt(0.001f);
    auto& p = w.add_planet<TestPlanet>("rotor", /*omega_z=*/Real(1.0));
    auto& s = w.create_scene();
    s.set_planet(&p);

    Craft c("rest_craft");
    c.root().add<manta::parts::PointMass>("body", 1.0f);
    c.root().compute_params();

    InitialState init;
    init.position = Vec3<SceneFrame>{1.0f, 0.0f, 0.0f};
    s.add_craft(c, init);

    // One tick advances world_to_scene's omega and computes fictitious forces.
    w.update();

    // a_linear in scene should be (~1, 0, 0): the centrifugal push outward.
    auto a = c.scene_to_craft().acc_linear();
    INFO("a = (", a.x(), ",", a.y(), ",", a.z(), ")");
    CHECK(a.x() == doctest::Approx(1.0).epsilon(1e-3));
    CHECK(std::abs(a.y()) < 1e-3);
    CHECK(std::abs(a.z()) < 1e-3);
}

// A craft moving radially outward in a rotating scene experiences Coriolis
// in the tangential direction. For ω_z = 1, v = (1, 0, 0), r = 0:
//   F_cor = -2 ω × v = -2 (0,0,1) × (1,0,0) = -2 (0, 1, 0) → a_cor = (0,-2,0)
// (centrifugal is zero at r=0, so the y-acceleration is purely Coriolis).
TEST_CASE("Planet phase 3: Coriolis acceleration on a radially-moving craft") {
    World w;
    w.clock().set_dt(0.001f);
    auto& p = w.add_planet<TestPlanet>("rotor", /*omega_z=*/Real(1.0));
    auto& s = w.create_scene();
    s.set_planet(&p);

    Craft c("moving_craft");
    c.root().add<manta::parts::PointMass>("body", 1.0f);
    c.root().compute_params();

    InitialState init;
    init.position   = Vec3<SceneFrame>{0.0f, 0.0f, 0.0f};
    init.vel_linear = Vec3<SceneFrame>{1.0f, 0.0f, 0.0f};
    s.add_craft(c, init);

    w.update();

    auto a = c.scene_to_craft().acc_linear();
    INFO("a = (", a.x(), ",", a.y(), ",", a.z(), ")");
    CHECK(std::abs(a.x()) < 1e-3);
    CHECK(a.y() == doctest::Approx(-2.0).epsilon(1e-3));
    CHECK(std::abs(a.z()) < 1e-3);
}

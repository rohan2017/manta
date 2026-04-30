#include <doctest/doctest.h>

#include "manta/core/planet.hpp"
#include "manta/core/scene.hpp"
#include "manta/core/world.hpp"

using namespace manta;

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

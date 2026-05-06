#include <doctest/doctest.h>

#include "manta/core/craft.hpp"
#include "manta/core/planet.hpp"
#include "manta/core/scene.hpp"
#include "manta/core/world.hpp"
#include "manta/parts/structure/mass.hpp"

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

    void set_acc_angular_z(Real alpha_z) noexcept {
        world_to_planet_.set_acc_angular(
            geom::Vec3<PlanetFrame>{Real(0), Real(0), alpha_z});
    }

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
    w.step();
    CHECK(p.update_count() == 1);
    w.step();
    CHECK(p.update_count() == 2);
}

TEST_CASE("Planet: world_to_planet defaults to identity, can carry rotation") {
    TestPlanet p("rotor", /*omega_z=*/Real(0.5));
    const auto& wtp = p.world_to_planet();
    CHECK(wtp.position().raw().norm() == doctest::Approx(0.0));
    CHECK(wtp.orientation().raw().w() == doctest::Approx(1.0));
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
    w.step();
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

    w.step();

    const auto& wts = s.world_to_scene();
    CHECK(wts.vel_angular().z() == doctest::Approx(0.5));
}

// ---- Phase 3 / 6: rotating-frame fictitious forces ----

TEST_CASE("Planet: centrifugal acceleration on a craft at rest in a rotating scene") {
    World w;
    w.clock().set_dt(0.001f);
    auto& p = w.add_planet<TestPlanet>("rotor", /*omega_z=*/Real(1.0));
    auto& s = w.create_scene();
    s.set_planet(&p);

    Craft c("rest_craft");
    c.root().add<manta::parts::Mass>("body", 1.0f, /*apply_gravity=*/false);
    c.root().compute_params();

    InitialState init;
    init.position = Vec3<SceneFrame>{1.0f, 0.0f, 0.0f};
    s.add_craft(c, init);

    w.step();

    auto a = c.scene_to_craft().acc_linear();
    INFO("a = (", a.x(), ",", a.y(), ",", a.z(), ")");
    CHECK(a.x() == doctest::Approx(1.0).epsilon(1e-3));
    CHECK(std::abs(a.y()) < 1e-3);
    CHECK(std::abs(a.z()) < 1e-3);
}

TEST_CASE("Planet: Coriolis acceleration on a radially-moving craft") {
    World w;
    w.clock().set_dt(0.001f);
    auto& p = w.add_planet<TestPlanet>("rotor", /*omega_z=*/Real(1.0));
    auto& s = w.create_scene();
    s.set_planet(&p);

    Craft c("moving_craft");
    c.root().add<manta::parts::Mass>("body", 1.0f, /*apply_gravity=*/false);
    c.root().compute_params();

    InitialState init;
    init.position   = Vec3<SceneFrame>{0.0f, 0.0f, 0.0f};
    init.vel_linear = Vec3<SceneFrame>{1.0f, 0.0f, 0.0f};
    s.add_craft(c, init);

    // evaluate() (no integrate) — read pseudo-force acc at the EXACT
    // initial state. update() would integrate first under the new
    // ordering, shifting position by v·dt before the aggregate, which
    // adds a tiny centrifugal contribution to a.x that just nips the
    // 1e-3 threshold.
    w.kinematic_and_aggregate();

    auto a = c.scene_to_craft().acc_linear();
    INFO("a = (", a.x(), ",", a.y(), ",", a.z(), ")");
    CHECK(std::abs(a.x()) < 1e-3);
    CHECK(a.y() == doctest::Approx(-2.0).epsilon(1e-3));
    CHECK(std::abs(a.z()) < 1e-3);
}

TEST_CASE("Planet: translational pseudo-force from non-inertial scene origin") {
    World w;
    w.clock().set_dt(0.001f);
    auto& p = w.add_planet<TestPlanet>("rotor", /*omega_z=*/Real(1.0));
    auto& s = w.create_scene();
    s.set_planet(&p);
    s.set_planet_to_scene(geom::StaticLink<PlanetFrame, SceneFrame>{
        Vec3<PlanetFrame>{Real(2.0), Real(0), Real(0)},
        geom::Ori<PlanetFrame>::identity()});

    Craft c("centered_craft");
    c.root().add<manta::parts::Mass>("body", 1.0f, /*apply_gravity=*/false);
    c.root().compute_params();

    InitialState init;
    s.add_craft(c, init);

    w.step();

    auto a = c.scene_to_craft().acc_linear();
    INFO("a = (", a.x(), ",", a.y(), ",", a.z(), ")");
    CHECK(a.x() == doctest::Approx(2.0).epsilon(1e-3));
    CHECK(std::abs(a.y()) < 1e-3);
    CHECK(std::abs(a.z()) < 1e-3);
}

TEST_CASE("Planet: Euler pseudo-force from angular acceleration") {
    World w;
    w.clock().set_dt(0.001f);
    auto& p = w.add_planet<TestPlanet>("spinner", /*omega_z=*/Real(0));
    p.set_acc_angular_z(Real(1.0));

    auto& s = w.create_scene();
    s.set_planet(&p);

    Craft c("offset_craft");
    c.root().add<manta::parts::Mass>("body", 1.0f, /*apply_gravity=*/false);
    c.root().compute_params();

    InitialState init;
    init.position = Vec3<SceneFrame>{1.0f, 0.0f, 0.0f};
    s.add_craft(c, init);

    w.step();

    auto a = c.scene_to_craft().acc_linear();
    INFO("a = (", a.x(), ",", a.y(), ",", a.z(), ")");
    CHECK(std::abs(a.x()) < 1e-3);
    CHECK(a.y() == doctest::Approx(-1.0).epsilon(1e-3));
    CHECK(std::abs(a.z()) < 1e-3);
}

TEST_CASE("Planet: translational + centrifugal compose to expected total") {
    World w;
    w.clock().set_dt(0.001f);
    auto& p = w.add_planet<TestPlanet>("rotor", /*omega_z=*/Real(1.0));
    auto& s = w.create_scene();
    s.set_planet(&p);
    s.set_planet_to_scene(geom::StaticLink<PlanetFrame, SceneFrame>{
        Vec3<PlanetFrame>{Real(3.0), Real(0), Real(0)},
        geom::Ori<PlanetFrame>::identity()});

    Craft c("co_rotating_craft");
    c.root().add<manta::parts::Mass>("body", 1.0f, /*apply_gravity=*/false);
    c.root().compute_params();

    InitialState init;
    init.position = Vec3<SceneFrame>{1.0f, 0.0f, 0.0f};
    s.add_craft(c, init);

    w.step();

    auto a = c.scene_to_craft().acc_linear();
    INFO("a = (", a.x(), ",", a.y(), ",", a.z(), ")");
    CHECK(a.x() == doctest::Approx(4.0).epsilon(1e-3));
    CHECK(std::abs(a.y()) < 1e-3);
    CHECK(std::abs(a.z()) < 1e-3);
}

// ---- Earth concrete planet ----

#include "manta/planets/earth.hpp"
#include "manta/fields/fluid_field.hpp"
#include "manta/fields/gravity_field.hpp"
#include "manta/fields/mag_field.hpp"

TEST_CASE("Earth: registers FluidField on add_planet (always)") {
    World w;
    auto& earth = w.add_planet<manta::planets::Earth>(/*sea_level=*/0.0f);
    auto* fluid_ptr = w.field_ptr(typeid(manta::fields::FluidField));
    REQUIRE(fluid_ptr != nullptr);
    auto& fluid = earth.fluid();
    // Two persistent disturbances: ocean below sea level + atmosphere above.
    CHECK(fluid.disturbance_count() == 2u);

    // Height helpers replace the deleted SeaSurface field.
    CHECK(earth.height_above_surface  (Vec3<SceneFrame>{0, 0,  3}) == doctest::Approx( 3));
    CHECK(earth.height_above_surface  (Vec3<SceneFrame>{0, 0, -2}) == doctest::Approx(-2));
    CHECK(earth.height_above_sea_level(Vec3<SceneFrame>{0, 0,  3}) == doctest::Approx( 3));

    CHECK(earth.rotation_rate() == doctest::Approx(0));
}

TEST_CASE("Earth: optional sidereal rotation pushes Coriolis through") {
    World w;
    w.clock().set_dt(0.001f);
    auto& earth = w.add_planet<manta::planets::Earth>(
        /*sea_level=*/0.0f,
        /*water_density=*/1000.0f,
        /*rotation_rate=*/1.0f);
    auto& s = w.create_scene();
    s.set_planet(&earth);
    w.step();

    CHECK(s.world_to_scene().vel_angular().z() == doctest::Approx(1.0));
}

TEST_CASE("Earth: gravity_mu activates a point-mass + J2 disturbance under GravityField") {
    World w;
    auto& earth = w.add_planet<manta::planets::Earth>(
        /*sea_level=*/0.0f,
        /*water_density=*/1000.0f,
        /*rotation_rate=*/0.0f,
        /*gravity_mu=*/manta::planets::Earth::kMu,
        /*include_j2=*/true);

    auto& g = earth.gravity();
    CHECK(g.disturbance_count() == 1u);
    auto* gf = w.field_ptr(typeid(manta::fields::GravityField));
    REQUIRE(gf != nullptr);

    // Sanity: surface gravity ~9.82 along -x at +R_eq.
    auto g_at_eq = g.state_at(Vec3<SceneFrame>{
        manta::planets::Earth::kEquatorialRadius, 0, 0});
    CHECK(g_at_eq.x() == doctest::Approx(-9.79).epsilon(0.01));
}

TEST_CASE("Earth: gravity_mu=0 means no gravity disturbance") {
    World w;
    auto& earth = w.add_planet<manta::planets::Earth>();
    CHECK(earth.gravity().disturbance_count() == 0u);
    auto g = earth.gravity().state_at(Vec3<SceneFrame>{1e7f, 0, 0});
    CHECK(g.raw().norm() == doctest::Approx(0.0).epsilon(1e-6));
}

TEST_CASE("Earth: dipole_moment activates a dipole MagField disturbance") {
    World w;
    auto& earth = w.add_planet<manta::planets::Earth>(
        /*sea_level=*/0.0f,
        /*water_density=*/1000.0f,
        /*rotation_rate=*/0.0f,
        /*gravity_mu=*/0.0f,
        /*include_j2=*/false,
        /*dipole_moment=*/manta::planets::Earth::kDipoleMoment);

    CHECK(earth.mag().disturbance_count() == 1u);
    auto* mf = w.field_ptr(typeid(manta::fields::MagField));
    REQUIRE(mf != nullptr);
}

// ---- Phase 5: craft().planet<P>() typed accessor ----

TEST_CASE("Craft::planet<P>: returns the registered planet by concrete type") {
    World w;
    w.clock().set_dt(0.001f);
    auto& earth = w.add_planet<manta::planets::Earth>();
    auto& s = w.create_scene();
    s.set_planet(&earth);

    Craft c("test");
    c.root().add<manta::parts::Mass>("body", 1.0f, /*apply_gravity=*/false);
    c.root().compute_params();
    s.add_craft(c);

    auto* p = c.planet<manta::planets::Earth>();
    REQUIRE(p != nullptr);
    CHECK(p == &earth);

    auto* not_real = c.planet<manta::Planet>();
    CHECK(not_real == &earth);
}

TEST_CASE("Craft::planet<P>: returns nullptr when no planet anchored") {
    World w;
    auto& s = w.create_scene();
    Craft c("test");
    c.root().add<manta::parts::Mass>("body", 1.0f, /*apply_gravity=*/false);
    c.root().compute_params();
    s.add_craft(c);

    CHECK(c.planet<manta::planets::Earth>() == nullptr);
}

// ---- Magnetometer Part ----

#include "manta/parts/sensor/magnetometer.hpp"

TEST_CASE("Magnetometer: reads MagField at part position, rotates to part frame") {
    World w;
    w.clock().set_dt(0.001f);
    manta::fields::MagField mag;
    mag.add(manta::fields::MagField::Disturbance::dipole(
                Vec3<SceneFrame>::zero(),
                Vec3<SceneFrame>{0, 0, -7.94e22f}),
            manta::fields::PERSISTENT);
    w.register_field(mag);

    auto& s = w.create_scene();
    Craft c("mag_test");
    c.root().add<manta::parts::Mass>("body", 1.0f, /*apply_gravity=*/false);
    auto& magp = c.root().add<manta::parts::Magnetometer>("mag");
    c.root().compute_params();
    c.set_position(Vec3<SceneFrame>{0, 0, 10.0f});
    s.add_craft(c);

    w.step();

    auto b = magp.last_b();
    INFO("b = (", b.x(), ",", b.y(), ",", b.z(), ")");
    CHECK(std::abs(b.x()) < 1e10f);
    CHECK(std::abs(b.y()) < 1e10f);
    CHECK(b.z() == doctest::Approx(-1.588e13f).epsilon(1e-2));
}

TEST_CASE("Magnetometer: zero output when no MagField is registered") {
    World w;
    w.clock().set_dt(0.001f);
    auto& s = w.create_scene();

    Craft c("no_mag");
    c.root().add<manta::parts::Mass>("body", 1.0f, /*apply_gravity=*/false);
    auto& mag = c.root().add<manta::parts::Magnetometer>("mag");
    c.root().compute_params();
    s.add_craft(c);

    w.step();
    auto b = mag.last_b();
    CHECK(std::abs(b.x()) < 1e-6f);
    CHECK(std::abs(b.y()) < 1e-6f);
    CHECK(std::abs(b.z()) < 1e-6f);
}

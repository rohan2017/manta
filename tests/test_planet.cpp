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

    // Test-only access to world_to_planet_'s motion fields. Used by phase 6
    // tests to exercise the translational and Euler pseudo-forces (which
    // require setting acc_linear/acc_angular on the planet directly).
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

// ---- Phase 6: Euler + translational pseudo-forces ----
//
// Phase 3 covered Coriolis and centrifugal; phase 6 adds the two missing
// terms in the non-inertial-scene fictitious-force decomposition:
//   a_translational = -a_S        (scene origin's linear accel in world)
//   a_euler         = -α × r      (scene's angular accel × craft position)

// Translational pseudo-force: a planet rotating at ω with the scene anchored
// at planet_to_scene = (R, 0, 0) puts the scene origin in centripetal motion
// at distance R from the spin axis. For ω_z = 1, R = 2 the scene origin's
// world-frame acceleration is a_S = -ω²R x̂ = (-2, 0, 0). The pseudo-force on
// any craft (per unit mass) is -a_S = (2, 0, 0). With the craft sitting at
// the scene origin (r = 0) and at rest (v = 0), centrifugal and Coriolis are
// both zero — the only fictitious force is the translational one.
TEST_CASE("Planet phase 6: translational pseudo-force from non-inertial scene origin") {
    World w;
    w.clock().set_dt(0.001f);
    auto& p = w.add_planet<TestPlanet>("rotor", /*omega_z=*/Real(1.0));
    auto& s = w.create_scene();
    s.set_planet(&p);
    s.set_planet_to_scene(geom::StaticLink<PlanetFrame, SceneFrame>{
        Vec3<PlanetFrame>{Real(2.0), Real(0), Real(0)},
        geom::Ori<PlanetFrame>::identity()});

    Craft c("centered_craft");
    c.root().add<manta::parts::PointMass>("body", 1.0f);
    c.root().compute_params();

    InitialState init;   // r=0, v=0 in scene
    s.add_craft(c, init);

    w.update();

    auto a = c.scene_to_craft().acc_linear();
    INFO("a = (", a.x(), ",", a.y(), ",", a.z(), ")");
    CHECK(a.x() == doctest::Approx(2.0).epsilon(1e-3));
    CHECK(std::abs(a.y()) < 1e-3);
    CHECK(std::abs(a.z()) < 1e-3);
}

// Euler pseudo-force: α_z × r at r = (1, 0, 0) with α_z = 1 gives
//   α × r = (0, 0, 1) × (1, 0, 0) = (0, 1, 0)
// so a_euler = -α × r = (0, -1, 0). To isolate Euler we keep ω = 0
// (no Coriolis or centrifugal) and put the scene origin at the planet
// origin (no translational pseudo-force).
TEST_CASE("Planet phase 6: Euler pseudo-force from angular acceleration") {
    World w;
    w.clock().set_dt(0.001f);
    auto& p = w.add_planet<TestPlanet>("spinner", /*omega_z=*/Real(0));
    p.set_acc_angular_z(Real(1.0));  // dω/dt = 1 rad/s² about z

    auto& s = w.create_scene();
    s.set_planet(&p);

    Craft c("offset_craft");
    c.root().add<manta::parts::PointMass>("body", 1.0f);
    c.root().compute_params();

    InitialState init;
    init.position = Vec3<SceneFrame>{1.0f, 0.0f, 0.0f};
    s.add_craft(c, init);

    w.update();

    auto a = c.scene_to_craft().acc_linear();
    INFO("a = (", a.x(), ",", a.y(), ",", a.z(), ")");
    CHECK(std::abs(a.x()) < 1e-3);
    CHECK(a.y() == doctest::Approx(-1.0).epsilon(1e-3));
    CHECK(std::abs(a.z()) < 1e-3);
}

// Integration test: translational + centrifugal compose correctly.
//
// On a rotating planet (ω_z = 1), put the scene at planet_to_scene = (3,0,0)
// and a craft at scene-frame (1,0,0). The craft is at world-frame (4,0,0).
//
// In the inertial world frame, a free particle at (4,0,0) would need a
// centripetal force toward the spin axis to stay co-rotating; in the
// rotating scene frame, this manifests as an outward pseudo-acceleration of
// ω² × 4 = 4 m/s² along scene-frame x̂. The decomposition is:
//   * translational  = -a_S        = -(- ω²·3 x̂) = +3 x̂
//   * centrifugal    = +ω² r_scene = +1 x̂
// for a sum of (4, 0, 0) — independent of how we split the planet_to_scene
// offset and the craft's scene-frame position. This is the key invariant:
// the kinematic chain has to give the same answer regardless of where the
// scene origin is anchored along the radial line.
TEST_CASE("Planet phase 6: translational + centrifugal compose to expected total") {
    World w;
    w.clock().set_dt(0.001f);
    auto& p = w.add_planet<TestPlanet>("rotor", /*omega_z=*/Real(1.0));
    auto& s = w.create_scene();
    s.set_planet(&p);
    s.set_planet_to_scene(geom::StaticLink<PlanetFrame, SceneFrame>{
        Vec3<PlanetFrame>{Real(3.0), Real(0), Real(0)},
        geom::Ori<PlanetFrame>::identity()});

    Craft c("co_rotating_craft");
    c.root().add<manta::parts::PointMass>("body", 1.0f);
    c.root().compute_params();

    InitialState init;
    init.position = Vec3<SceneFrame>{1.0f, 0.0f, 0.0f};   // world position (4,0,0) at t=0
    s.add_craft(c, init);

    w.update();

    auto a = c.scene_to_craft().acc_linear();
    INFO("a = (", a.x(), ",", a.y(), ",", a.z(), ")");
    CHECK(a.x() == doctest::Approx(4.0).epsilon(1e-3));
    CHECK(std::abs(a.y()) < 1e-3);
    CHECK(std::abs(a.z()) < 1e-3);
}

// ---- Earth concrete planet ----

#include "manta/planets/earth.hpp"

TEST_CASE("Earth: registers FluidField + SeaSurface on add_planet") {
    World w;
    auto& earth = w.add_planet<manta::planets::Earth>(/*sea_level=*/0.0f);

    // FluidField is registered (OceanAtmosField under the FluidField slot).
    auto* fluid_ptr = w.field_ptr(typeid(manta::fields::FluidField));
    REQUIRE(fluid_ptr != nullptr);

    // SeaSurface is registered too.
    auto* surf_ptr = w.field_ptr(typeid(manta::fields::SeaSurface));
    REQUIRE(surf_ptr != nullptr);

    auto* surf = dynamic_cast<manta::fields::SeaSurface*>(surf_ptr);
    REQUIRE(surf != nullptr);
    CHECK(surf->height_above_surface(Vec3<SceneFrame>{0, 0,  3}) == doctest::Approx( 3));
    CHECK(surf->height_above_surface(Vec3<SceneFrame>{0, 0, -2}) == doctest::Approx(-2));

    // Earth defaults to non-rotating (rotation_rate = 0).
    CHECK(earth.rotation_rate() == doctest::Approx(0));
}

TEST_CASE("Earth: optional sidereal rotation pushes Coriolis through") {
    World w;
    w.clock().set_dt(0.001f);
    auto& earth = w.add_planet<manta::planets::Earth>(
        /*sea_level=*/0.0f,
        /*water_density=*/1000.0f,
        /*air_density=*/1.225f,
        /*rotation_rate=*/1.0f);   // 1 rad/s for an exaggerated test
    auto& s = w.create_scene();
    s.set_planet(&earth);
    w.update();

    // Scene picks up Earth's rotation rate.
    CHECK(s.world_to_scene().vel_angular().z() == doctest::Approx(1.0));
}

// ---- Optional gravity model on Earth ----

TEST_CASE("Earth: gravity_mu activates a PointGravityField with optional J2") {
    World w;
    auto& earth = w.add_planet<manta::planets::Earth>(
        /*sea_level=*/0.0f,
        /*water_density=*/1000.0f,
        /*air_density=*/1.225f,
        /*rotation_rate=*/0.0f,
        /*gravity_mu=*/manta::planets::Earth::kMu,
        /*include_j2=*/true);

    REQUIRE(earth.gravity() != nullptr);
    CHECK(earth.gravity()->mu()  == doctest::Approx(manta::planets::Earth::kMu));
    CHECK(earth.gravity()->j2()  == doctest::Approx(manta::planets::Earth::kJ2));
    CHECK(earth.gravity()->eq_radius() == doctest::Approx(manta::planets::Earth::kEquatorialRadius));

    auto* gf = w.field_ptr(typeid(manta::fields::PointGravityField));
    REQUIRE(gf != nullptr);
}

TEST_CASE("Earth: gravity_mu=0 means no PointGravityField registered") {
    World w;
    auto& earth = w.add_planet<manta::planets::Earth>();
    CHECK(earth.gravity() == nullptr);
    CHECK(w.field_ptr(typeid(manta::fields::PointGravityField)) == nullptr);
}

// ---- Optional dipole magnetic model on Earth ----

#include "../include/manta/fields/mag_field.hpp"

TEST_CASE("Earth: dipole_moment activates a DipoleMagField under MagField slot") {
    World w;
    auto& earth = w.add_planet<manta::planets::Earth>(
        /*sea_level=*/0.0f,
        /*water_density=*/1000.0f,
        /*air_density=*/1.225f,
        /*rotation_rate=*/0.0f,
        /*gravity_mu=*/0.0f,
        /*include_j2=*/false,
        /*dipole_moment=*/manta::planets::Earth::kDipoleMoment);

    REQUIRE(earth.mag() != nullptr);
    auto* mf_base = w.field_ptr(typeid(manta::fields::MagField));
    REQUIRE(mf_base != nullptr);

    // Polymorphism: looking up MagField gives back our DipoleMagField.
    auto* dipole = dynamic_cast<manta::fields::DipoleMagField*>(mf_base);
    REQUIRE(dipole != nullptr);
    // moment is along -z by default.
    CHECK(dipole->moment().z() == doctest::Approx(-manta::planets::Earth::kDipoleMoment));
}

// ---- Phase 5: craft().planet<P>() typed accessor ----

TEST_CASE("Craft::planet<P>: returns the registered planet by concrete type") {
    World w;
    w.clock().set_dt(0.001f);
    auto& earth = w.add_planet<manta::planets::Earth>();
    auto& s = w.create_scene();
    s.set_planet(&earth);

    Craft c("test");
    c.root().add<manta::parts::PointMass>("body", 1.0f);
    c.root().compute_params();
    s.add_craft(c);

    // Typed accessor returns the planet pointer.
    auto* p = c.planet<manta::planets::Earth>();
    REQUIRE(p != nullptr);
    CHECK(p == &earth);

    // Querying a different planet type returns nullptr (dynamic_cast safety).
    auto* not_real = c.planet<manta::Planet>();
    CHECK(not_real == &earth);   // base class IS a match
}

TEST_CASE("Craft::planet<P>: returns nullptr when no planet anchored") {
    World w;
    auto& s = w.create_scene();   // no planet set
    Craft c("test");
    c.root().add<manta::parts::PointMass>("body", 1.0f);
    c.root().compute_params();
    s.add_craft(c);

    CHECK(c.planet<manta::planets::Earth>() == nullptr);
}

// ---- Magnetometer Part ----

#include "../include/manta/parts/sensor/magnetometer.hpp"

TEST_CASE("Magnetometer: reads MagField at part position, rotates to part frame") {
    // No-rotation craft at origin, MagField is a dipole at world origin with
    // moment along -z. At z=10, B is parallel to -z (see DipoleMagField pole
    // test). The craft has identity orientation, so part-frame B == scene B.
    World w;
    w.clock().set_dt(0.001f);
    manta::fields::DipoleMagField dipole{geom::Vec3<SceneFrame>{0, 0, -7.94e22f}};
    w.register_field(dipole);
    w.register_field<manta::fields::MagField>(dipole);

    auto& s = w.create_scene();

    Craft c("mag_test");
    c.root().add<manta::parts::PointMass>("body", 1.0f);
    auto& mag = c.root().add<manta::parts::Magnetometer>("mag");
    c.root().compute_params();
    c.set_position(Vec3<SceneFrame>{0, 0, 10.0f});
    s.add_craft(c);

    w.update();

    // At pole z=10, |B| = 2·(μ₀/4π)·|m|/r³ = 2·1e-7·7.94e22/1000 ≈ 1.588e13 T,
    // along -z (since moment is along -z, and at pole B is parallel to moment).
    // (The unrealistically-large value is because the dipole's moment magnitude
    // is Earth-scale but we sampled 10 m from a singular origin — a sanity
    // check on the formula, not a realistic field strength.)
    auto b = mag.last_b();
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
    c.root().add<manta::parts::PointMass>("body", 1.0f);
    auto& mag = c.root().add<manta::parts::Magnetometer>("mag");
    c.root().compute_params();
    s.add_craft(c);

    w.update();
    auto b = mag.last_b();
    CHECK(std::abs(b.x()) < 1e-6f);
    CHECK(std::abs(b.y()) < 1e-6f);
    CHECK(std::abs(b.z()) < 1e-6f);
}

#include <cmath>
#include <doctest/doctest.h>

#include "../include/manta/core/craft.hpp"
#include "../include/manta/core/scene.hpp"
#include "../include/manta/core/world.hpp"
#include "../include/manta/fields/gravity_field.hpp"
#include "../include/manta/parts/structure/mass.hpp"
#include "../include/manta/parts/actuator/thruster.hpp"
#include "test_helpers.hpp"

using namespace manta;
using namespace manta::geom;
using namespace manta::parts;
using namespace manta::fields;

// ---- SimClock ----

TEST_CASE("SimClock: advances time by dt each tick") {
    World w;
    w.clock().set_dt(0.1f);
    CHECK(w.clock().time() == doctest::Approx(0.0f));

    auto& s = w.create_scene();
    (void)s;
    w.update();
    CHECK(w.clock().time() == doctest::Approx(0.1f));

    w.update();
    CHECK(w.clock().time() == doctest::Approx(0.2f));
}

// ---- Scene / World wiring ----

TEST_CASE("World: create_scene returns a valid scene") {
    World w;
    auto& s = w.create_scene();
    CHECK(&s.world() == &w);
}

TEST_CASE("Scene: add_craft wires world pointer") {
    World w;
    auto& s = w.create_scene();
    Craft c("drone");
    s.add_craft(c);
    CHECK(&c.world() == &w);
    CHECK(&c.scene() == &s);
}

TEST_CASE("Scene: add_craft with InitialState applies position and velocity") {
    World w;
    auto& s = w.create_scene();
    Craft c("drone");
    InitialState init{
        Vec3<SceneFrame>{1, 2, 3},
        Ori<SceneFrame>::identity(),
        Vec3<SceneFrame>{0, 0, 5},
        Vec3<CraftFrame>{0, 1, 0},
    };
    s.add_craft(c, init);
    auto p = c.scene_to_craft().position();
    auto v = c.scene_to_craft().vel_linear();
    auto wv = c.scene_to_craft().vel_angular();
    CHECK(p.x() == doctest::Approx(1.0f));
    CHECK(p.y() == doctest::Approx(2.0f));
    CHECK(p.z() == doctest::Approx(3.0f));
    CHECK(v.z() == doctest::Approx(5.0f));
    CHECK(wv.y() == doctest::Approx(1.0f));
}

TEST_CASE("Scene: add_craft without InitialState preserves prior state") {
    World w;
    auto& s = w.create_scene();
    Craft c("drone");
    c.set_position(Vec3<SceneFrame>{7, 0, 0});
    s.add_craft(c);
    CHECK(c.scene_to_craft().position().x() == doctest::Approx(7.0f));
}

TEST_CASE("Scene: remove_craft clears world pointer") {
    World w;
    auto& s = w.create_scene();
    Craft c("drone");
    s.add_craft(c);
    s.remove_craft(c);
    CHECK(!c.has_world());
    CHECK(!c.has_scene());
    CHECK(s.crafts().empty());
}

TEST_CASE("World: update calls craft update (clock advances)") {
    World w;
    w.clock().set_dt(0.05f);
    auto& s = w.create_scene();
    Craft c("test");
    c.root().add<Mass>("body", 1.0f, /*apply_gravity=*/false);
    c.root().compute_params();
    s.add_craft(c);

    w.update();
    CHECK(w.clock().time() == doctest::Approx(0.05f));
}

// ---- Field lookup chain: Craft → World ----

TEST_CASE("World: field registered at World level accessible from Part") {
    struct QueryPart : public Part {
        float g_z_seen = 0.0f;
        explicit QueryPart(std::string n) : Part(std::move(n)) {}
        void update() override {
            g_z_seen = field<GravityField>().state_at(Vec3<SceneFrame>::zero()).z();
        }
    };

    GravityField gf{Vec3<SceneFrame>{0, 0, -9.81f}};
    World w;
    w.register_field(gf);

    auto& s = w.create_scene();
    Craft c("test");
    auto& qp = c.root().add<QueryPart>("q");
    s.add_craft(c);
    w.update();

    CHECK(qp.g_z_seen == doctest::Approx(-9.81f));
}

TEST_CASE("World: craft-local field shadows world field") {
    struct QueryPart : public Part {
        float g_z_seen = 0.0f;
        explicit QueryPart(std::string n) : Part(std::move(n)) {}
        void update() override {
            g_z_seen = field<GravityField>().state_at(Vec3<SceneFrame>::zero()).z();
        }
    };

    GravityField world_gf{Vec3<SceneFrame>{0, 0, -9.81f}};
    GravityField craft_gf{Vec3<SceneFrame>{0, 0, -1.62f}};

    World w;
    w.register_field(world_gf);

    auto& s = w.create_scene();
    Craft c("test");
    c.register_field(craft_gf);
    auto& qp = c.root().add<QueryPart>("q");
    s.add_craft(c);
    w.update();

    CHECK(qp.g_z_seen == doctest::Approx(-1.62f));
}

// ---- Mass auto-gravity end-to-end ----

TEST_CASE("Mass auto-gravity: free fall under -9.81 m/s^2 for 1 s") {
    GravityField gf{Vec3<SceneFrame>{0, 0, -9.81f}};
    World w;
    w.register_field(gf);
    w.clock().set_dt(0.001f);

    auto& scene = w.create_scene();
    Craft c("drone");
    c.root().add<Mass>("body", 1.0f);    // apply_gravity=true by default
    c.root().compute_params();
    scene.add_craft(c);

    const int steps = 1000;
    for (int i = 0; i < steps; ++i) w.update();

    auto vel = c.scene_to_craft().vel_linear();
    auto pos = c.scene_to_craft().position();

    CHECK(vel.z() == doctest::Approx(-9.81f).epsilon(0.01f));
    CHECK(pos.z() == doctest::Approx(-4.905f).epsilon(0.01f));
}

TEST_CASE("Mass + Thruster: craft hovers at constant velocity") {
    GravityField gf{Vec3<SceneFrame>{0, 0, -9.81f}};
    World w;
    w.register_field(gf);
    w.clock().set_dt(0.01f);

    auto& scene = w.create_scene();
    Craft c("drone");
    c.root().add<Mass>("body", 1.0f);
    auto& t = c.root().add<Thruster>("thruster", 9.81f);
    t.set_throttle(1.0f);
    c.root().compute_params();
    scene.add_craft(c);

    for (int i = 0; i < 100; ++i) w.update();

    auto vel = c.scene_to_craft().vel_linear();
    CHECK(test::approx_equal(vel, Vec3<SceneFrame>::zero(), Real(0.01f)));
}

TEST_CASE("World: multiple crafts in same scene all update under gravity") {
    GravityField gf{Vec3<SceneFrame>{0, 0, -9.81f}};
    World w;
    w.register_field(gf);
    w.clock().set_dt(1.0f);

    auto& scene = w.create_scene();

    Craft c1("c1");
    c1.root().add<Mass>("m", 1.0f);
    c1.root().compute_params();
    scene.add_craft(c1);

    Craft c2("c2");
    c2.root().add<Mass>("m", 2.0f);
    c2.root().compute_params();
    scene.add_craft(c2);

    // Prime the acceleration cache at the initial state so the first
    // integrate consumes the correct gravity force. Under the integrate-
    // then-aggregate ordering, an unprimed update() would drift first
    // (cache=0 → no kick) and only capture gravity in the aggregate
    // afterward, leaving v at zero on tick 1.
    w.evaluate();
    w.update();

    auto v1 = c1.scene_to_craft().vel_linear();
    auto v2 = c2.scene_to_craft().vel_linear();
    CHECK(v1.z() == doctest::Approx(-9.81f).epsilon(0.01f));
    CHECK(v2.z() == doctest::Approx(-9.81f).epsilon(0.01f));
}

TEST_CASE("Mass: apply_gravity=false opts out of auto-gravity") {
    GravityField gf{Vec3<SceneFrame>{0, 0, -9.81f}};
    World w;
    w.register_field(gf);
    w.clock().set_dt(0.01f);

    auto& scene = w.create_scene();
    Craft c("ballast");
    c.root().add<Mass>("body", 1.0f, /*apply_gravity=*/false);
    c.root().compute_params();
    scene.add_craft(c);

    for (int i = 0; i < 100; ++i) w.update();

    auto vel = c.scene_to_craft().vel_linear();
    CHECK(vel.z() == doctest::Approx(0.0f).epsilon(1e-3f));
}

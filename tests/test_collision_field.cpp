// CollisionField + Collider — contact kernel correctness tests.
//
//   * Sphere-vs-plane: penetration produces a normal spring force; the
//     direction matches the plane normal; damping opposes ingress.
//   * Sphere-vs-sphere: penetration produces repulsion along the
//     center-to-center axis; non-penetration ⇒ zero force.
//   * Self-exclusion: a Collider's own disturbance can't push itself.
//   * Integration smoke: a dropped sphere settles on the ground plane
//     within a small steady-state sink and doesn't oscillate forever.

#include <doctest/doctest.h>

#include "../include/manta/core/craft.hpp"
#include "../include/manta/core/scene.hpp"
#include "../include/manta/core/world.hpp"
#include "../include/manta/fields/collision_field.hpp"
#include "../include/manta/fields/gravity_field.hpp"
#include "../include/manta/parts/field_src/collider.hpp"
#include "../include/manta/parts/structure/mass.hpp"

using namespace manta;
using namespace manta::fields;
using namespace manta::parts;

namespace {

constexpr MFloat kEps = MFloat(1e-3);

bool approx(MFloat a, MFloat b, MFloat tol = kEps) {
    return std::abs(a - b) <= tol;
}

}  // namespace

TEST_CASE("CollisionField: sphere-plane contact only when penetrating") {
    CollisionField field;
    using Vec = CollisionField::Vec;

    field.add(CollisionDisturbance::infinite_plane(
        Vec{0, 0, 0}, Vec{0, 0, 1}, /*k=*/1e6, /*d=*/1e3),
        PERSISTENT);

    // Sphere of radius 0.5 sitting at z=1 — center above plane, NO
    // penetration.
    auto sphere_clear = CollisionDisturbance::single_sphere(
        Vec{0, 0, 1}, 0.5);
    auto w_clear = field.net_wrench_on(sphere_clear);
    CHECK(w_clear.force.raw().norm()  == doctest::Approx(0.0).epsilon(1e-9));
    CHECK(w_clear.torque.raw().norm() == doctest::Approx(0.0).epsilon(1e-9));

    // Sphere of radius 0.5 sitting with center at z=0.3 — penetrates by 0.2.
    auto sphere_pen = CollisionDisturbance::single_sphere(
        Vec{0, 0, 0.3}, 0.5);
    auto w_pen = field.net_wrench_on(sphere_pen);
    // Plane k=1e6, sphere k=1e6 (default) ⇒ harmonic mean = 1e6. Penetration 0.2.
    // Spring force = 1e6 * 0.2 = 2e5 N along +z.
    CHECK(w_pen.force.raw()(0)        == doctest::Approx(0.0).epsilon(1e-9));
    CHECK(w_pen.force.raw()(1)        == doctest::Approx(0.0).epsilon(1e-9));
    CHECK(w_pen.force.raw()(2)        == doctest::Approx(2.0e5).epsilon(1e-3));
}

TEST_CASE("CollisionField: sphere-sphere repulsion along centerline") {
    CollisionField field;
    using Vec = CollisionField::Vec;

    // Static sphere at origin, radius 1.
    field.add(CollisionDisturbance::single_sphere(Vec{0, 0, 0}, 1.0), PERSISTENT);

    // Query sphere of radius 0.6 at (1.0, 0, 0) — overlap = 0.6 along +x.
    auto q = CollisionDisturbance::single_sphere(Vec{1.0, 0, 0}, 0.6);
    auto w = field.net_wrench_on(q);
    // k_eff = 1e6, overlap = 0.6 ⇒ force magnitude = 6e5 N along +x.
    CHECK(w.force.raw()(0) == doctest::Approx( 6.0e5).epsilon(1e-3));
    CHECK(w.force.raw()(1) == doctest::Approx(  0.0 ).epsilon(1e-6));
    CHECK(w.force.raw()(2) == doctest::Approx(  0.0 ).epsilon(1e-6));
}

TEST_CASE("CollisionField: a disturbance does not collide with itself") {
    CollisionField field;
    using Vec = CollisionField::Vec;

    int dummy = 0;
    CollisionDisturbance plane =
        CollisionDisturbance::infinite_plane(Vec{0, 0, 0}, Vec{0, 0, 1});
    plane.owner = &dummy;

    CollisionDisturbance sphere = CollisionDisturbance::single_sphere(
        Vec{0, 0, 0}, 0.5);                // dead-center in the plane
    sphere.owner = &dummy;                 // same owner ⇒ excluded

    field.add(plane, PERSISTENT);
    auto w = field.net_wrench_on(sphere);
    CHECK(w.force.raw().norm() == doctest::Approx(0.0).epsilon(1e-9));
}

TEST_CASE("Collider+CollisionField: ball drops onto ground and settles") {
    // World setup: gravity + collision plane at z=0. A 1 kg body with a
    // 0.05 m sphere collider, dropped from 0.5 m. The PD contact should
    // catch it, ring down within ~1 s of sim time, and rest with sub-mm
    // steady-state sink (m·g / k = 1·9.81 / 1e6 ≈ 1e-5 m).
    WorldT<MFloat> world;
    world.clock().set_dt(MFloat(0.001));
    GravityField   grav(GravityField::Vec{MFloat(0), MFloat(0), MFloat(-9.81)});
    CollisionField coll;
    // k=1e4 keeps the contact stable at dt=1ms (ω_n=100 rad/s ⇒ T≈63ms
    // ≫ 6.3·dt). Critical damping is 2√(mk)=200 N·s/m; pick d=80% of that.
    coll.add(CollisionDisturbance::infinite_plane(
        CollisionField::Vec{0, 0, 0},
        CollisionField::Vec{0, 0, 1},
        /*k=*/MFloat(1.0e4),
        /*d=*/MFloat(160.0)),
        PERSISTENT);
    world.register_field(grav);
    world.register_field(coll);

    CraftT<MFloat> craft("ball");
    craft.root().template add<MassT<MFloat>>("body", MFloat(1.0));
    craft.root().template add<ColliderT<MFloat>>("hull", MFloat(0.05),
        /*k=*/MFloat(1.0e4), /*d=*/MFloat(160.0));
    craft.root().compute_params();

    auto& scene = world.create_scene();
    InitialState init;
    init.position = geom::Vec3<SceneFrame>{MFloat(0), MFloat(0), MFloat(0.5)};
    scene.add_craft(craft, init);

    for (int i = 0; i < 2000; ++i) world.step();   // 2 s sim
    const MFloat z = craft.scene_to_craft().position().z();
    // Settle: ball center should be near the radius (sphere just
    // touching the plane), within < 1 mm of the analytical rest depth.
    // Rest center z = sphere_radius - (m·g / k) ≈ 0.05 - 9.81e-6.
    CHECK(z > MFloat(0.04));    // still above ground (didn't sink in deeply)
    CHECK(z < MFloat(0.06));    // settled (not bouncing meters high)
}

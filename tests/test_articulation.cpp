#include <doctest/doctest.h>

#include "../include/manta/core/craft.hpp"
#include "../include/manta/core/scene.hpp"
#include "../include/manta/core/world.hpp"
#include "../include/manta/parts/articulation/motor.hpp"
#include "../include/manta/parts/structure/point_mass.hpp"
#include "test_helpers.hpp"

using namespace manta;
using namespace manta::geom;
using namespace manta::parts;

// ---- ArticulatedPart / Motor unit tests ----

TEST_CASE("Motor: at-rest Motor with no children has zero joint accel") {
    Craft c("test");
    auto& m = c.root().add<Motor>("m", Vec3<PartFrame>{0, 0, 1});
    c.root().compute_params();
    c.update(0.001f);
    CHECK(m.angle() == doctest::Approx(0.0f));
    CHECK(m.rate()  == doctest::Approx(0.0f));
}

TEST_CASE("Motor: passive joint with axial child torque accelerates") {
    // PointMass at +x offset from motor's joint output. Apply a torque about
    // the motor's z axis directly to the point-mass's wrench accumulator. The
    // motor's resolve() should compute axial accel = τ_axial / I_axial, and
    // the joint should advance accordingly after one tick.
    Craft c("test");
    auto& motor = c.root().add<Motor>("m", Vec3<PartFrame>{0, 0, 1});

    // Child mass: 1 kg at (0.5, 0, 0) in motor's joint-output frame.
    auto& mass = motor.add<PointMass>("p", 1.0f);
    mass.set_transform(StaticLink<ParentFrame, PartFrame>{
        Vec3<ParentFrame>{0.5f, 0.0f, 0.0f},
        Ori<ParentFrame>::identity()});

    // Add a synthetic torque on the mass directly. PointMass::update() is a
    // no-op so we apply the torque from outside via the public API on the
    // mass before calling update.
    motor.set_passive();

    // First update: compute_params runs in kinematic_pass; mass shows up at
    // 0.5 m → I_axial = 1*(0.5²) = 0.25 kg·m² about the motor z axis.
    c.update(0.0f);
    CHECK(motor.get_mass() == doctest::Approx(1.0f));

    // Now drive the joint: apply a torque-only wrench on the mass's frame
    // about the motor's z axis (which is the child's z axis too, since the
    // child has identity orientation in the motor frame).
    // Apply via apply_torque inside a custom post-update push: the cleanest
    // way is to inject it through an external "TorqueSource" stand-in, but
    // since that part type doesn't exist, we'll use a lambda subclass.
    struct Pusher : public Part {
        Vec3<PartFrame> tau;
        Pusher(std::string n, Vec3<PartFrame> t) : Part(std::move(n)), tau(t) {}
        void update() override { apply_torque(tau); }
    };
    auto& pusher = motor.add<Pusher>("pusher", Vec3<PartFrame>{0, 0, 1.0f});
    (void)pusher;

    // Mass at 0.5 m radius + the pusher (mass 0) → total joint subtree mass
    // is still 1, total Izz = 1 * 0.25 = 0.25.
    // Net axial torque on joint = 1.0 N·m. θ̈ = 1.0 / 0.25 = 4 rad/s².
    Real dt = 0.01f;
    c.update(dt);
    CHECK(motor.accel() == doctest::Approx(4.0f).epsilon(1e-3f));
    CHECK(motor.rate()  == doctest::Approx(4.0f * dt).epsilon(1e-3f));
}

TEST_CASE("Motor: saturating mode applies actuator torque, clamped to stall") {
    Craft c("test");
    auto& motor = c.root().add<Motor>("m", Vec3<PartFrame>{0, 0, 1},
                                       /*stall_torque=*/0.5f);
    auto& mass = motor.add<PointMass>("p", 1.0f);
    mass.set_transform(StaticLink<ParentFrame, PartFrame>{
        Vec3<ParentFrame>{0.5f, 0.0f, 0.0f},
        Ori<ParentFrame>::identity()});

    motor.set_torque(2.0f);   // commanded above stall → clamped to 0.5
    c.root().compute_params();
    c.update(0.01f);
    // I_axial = 1 * 0.25 = 0.25 → θ̈ = 0.5 / 0.25 = 2 rad/s²
    CHECK(motor.accel() == doctest::Approx(2.0f).epsilon(1e-3f));
}

// ---- Articulation invariant test ----
//
// Configuration: a Motor at the craft root (axis = z) with two equal point
// masses on its joint output, at (±r, 0, 0) — symmetric about the motor's
// own joint origin. The motor's joint-output COM is therefore exactly at
// the craft origin for any joint angle, so the craft's overall COM is
// always at the origin in CraftFrame. This dodges the "COM offset from
// craft origin" coupling that the rigid-body integrator does not yet
// handle (see project memory: that's the configuration the user
// originally described, and is queued behind a future origin/COM split
// in scene_to_craft_).
//
// Physics: the motor accelerates the joint subtree by +α; Newton's third
// gives the body an opposite +α. With matched MOIs (joint axial inertia ==
// total body inertia about z, both = 2 m r²), the masses end up stationary
// in scene — body rotation cancels joint rotation exactly.
//
// Invariants checked:
//   - Scene-frame COM stays at origin (no net linear motion).
//   - Total angular momentum about z stays at zero.
//   - Motor angle advances (sanity).
TEST_CASE("Articulation invariant: balanced Motor preserves COM and Lz") {
    constexpr float r   = 0.5f;
    constexpr float m   = 1.0f;
    constexpr float dt  = 0.001f;
    constexpr int   N   = 1000;
    constexpr float tau = 0.05f;

    World w;
    w.clock().set_dt(dt);
    auto& scene = w.create_scene();
    Craft c("balanced_motor");

    auto& motor = c.root().add<Motor>("motor", Vec3<PartFrame>{0, 0, 1},
                                      /*stall=*/10.0f);
    auto& mL = motor.add<PointMass>("mL", m);
    mL.set_transform(StaticLink<ParentFrame, PartFrame>{
        Vec3<ParentFrame>{-r, 0, 0},
        Ori<ParentFrame>::identity()});
    auto& mR = motor.add<PointMass>("mR", m);
    mR.set_transform(StaticLink<ParentFrame, PartFrame>{
        Vec3<ParentFrame>{+r, 0, 0},
        Ori<ParentFrame>::identity()});

    motor.set_torque(tau);
    scene.add_craft(c, InitialState{});

    for (int i = 0; i < N; ++i) {
        w.update();
    }
    c.kinematic_pass();   // refresh caches for inspection

    // --- Invariant 1: scene-frame COM is at origin. ---
    auto pL = mL.scene_to_part().position();
    auto pR = mR.scene_to_part().position();
    Vec3<SceneFrame> com_scene = Vec3<SceneFrame>::from_raw(
        m * (pL.raw() + pR.raw()) / (2 * m));
    INFO("pL=", pL.x(), ",", pL.y(), " pR=", pR.x(), ",", pR.y());
    CHECK(test::approx_equal(com_scene, Vec3<SceneFrame>::zero(), 1e-2f));

    // --- Invariant 2: total angular momentum about z stays at zero. ---
    auto vL = mL.scene_to_part().vel_linear().raw();
    auto vR = mR.scene_to_part().vel_linear().raw();
    auto rL = pL.raw();
    auto rR = pR.raw();
    float Lz = m * (rL.x() * vL.y() - rL.y() * vL.x())
             + m * (rR.x() * vR.y() - rR.y() * vR.x());
    INFO("Lz = ", Lz);
    CHECK(std::abs(Lz) < 1e-1f);

    // --- Sanity: motor has advanced. ---
    CHECK(std::abs(motor.angle()) > 0.01f);
}

// Asymmetric configuration from prompts/my_prompt.txt: m1 rigidly attached
// to the craft root at (+r, 0, 0); Motor at the root with axis = z; m2
// attached to the motor at (-r, 0, 0) in the joint-output frame. Total
// craft COM in CraftFrame is at the origin only at angle = 0; for any
// other motor angle, the COM moves in CraftFrame.
//
// This exercises the origin/COM split in the rigid-body integrator —
// previously failed (deferred) when integration assumed origin == COM.
//
// Invariants:
//   - Scene-frame COM (averaged over m1 + m2) stays at the origin
//     (no external forces; static-COM integrator should handle this
//     correctly even though COM moves in body frame).
//   - Total Lz remains zero (started at rest, internal torques only).
TEST_CASE("Articulation invariant: asymmetric Motor preserves COM and Lz") {
    constexpr float r   = 0.5f;
    constexpr float m   = 1.0f;
    constexpr float dt  = 0.001f;
    constexpr int   N   = 1000;
    constexpr float tau = 0.05f;

    World w;
    w.clock().set_dt(dt);
    auto& scene = w.create_scene();
    Craft c("asymmetric_motor");

    auto& m1 = c.root().add<PointMass>("m1", m);
    m1.set_transform(StaticLink<ParentFrame, PartFrame>{
        Vec3<ParentFrame>{+r, 0, 0},
        Ori<ParentFrame>::identity()});

    auto& motor = c.root().add<Motor>("motor", Vec3<PartFrame>{0, 0, 1},
                                       /*stall=*/10.0f);
    auto& m2 = motor.add<PointMass>("m2", m);
    m2.set_transform(StaticLink<ParentFrame, PartFrame>{
        Vec3<ParentFrame>{-r, 0, 0},
        Ori<ParentFrame>::identity()});

    motor.set_torque(tau);
    scene.add_craft(c, InitialState{});

    for (int i = 0; i < N; ++i) {
        w.update();
    }
    c.kinematic_pass();

    auto p1 = m1.scene_to_part().position();
    auto p2 = m2.scene_to_part().position();
    Vec3<SceneFrame> com_scene = Vec3<SceneFrame>::from_raw(
        m * (p1.raw() + p2.raw()) / (2 * m));
    INFO("p1=(", p1.x(), ",", p1.y(), ") p2=(", p2.x(), ",", p2.y(), ")");
    CHECK(test::approx_equal(com_scene, Vec3<SceneFrame>::zero(), 5e-2f));

    auto v1 = m1.scene_to_part().vel_linear().raw();
    auto v2 = m2.scene_to_part().vel_linear().raw();
    float Lz = m * (p1.x() * v1.y() - p1.y() * v1.x())
             + m * (p2.x() * v2.y() - p2.y() * v2.x());
    INFO("Lz = ", Lz);
    CHECK(std::abs(Lz) < 5e-2f);

    CHECK(std::abs(motor.angle()) > 0.01f);
}

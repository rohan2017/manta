#include <cmath>
#include <doctest/doctest.h>

#include "../include/manta/core/craft.hpp"
#include "../include/manta/core/composite_part.hpp"
#include "test_helpers.hpp"

using namespace manta;
using namespace manta::geom;

// ---- Minimal concrete parts used across tests ----

// A leaf part that optionally applies a force/torque during update().
struct ForcePart : public Part {
    Vec3<PartFrame> force_to_apply;
    Vec3<PartFrame> point_of_force;
    bool apply = false;

    explicit ForcePart(std::string name = "force_part") : Part(std::move(name)) {}
    void update() override {
        if (apply) apply_force_at(force_to_apply, point_of_force);
    }
};

// A leaf part that records its kinematic cache at update() time.
struct SensorPart : public Part {
    Vec3<CraftFrame> last_pos_craft;
    Vec3<CraftFrame> last_vel_craft;

    explicit SensorPart(std::string name = "sensor") : Part(std::move(name)) {}
    void update() override {
        last_pos_craft = position<CraftFrame>();
        last_vel_craft = velocity<CraftFrame>();
    }
};

// A leaf part that records scene-frame kinematic state.
struct SceneSensorPart : public Part {
    Vec3<SceneFrame> last_pos;
    Vec3<SceneFrame> last_vel;

    explicit SceneSensorPart(std::string name = "scene_sensor") : Part(std::move(name)) {}
    void update() override {
        last_pos = position<SceneFrame>();
        last_vel = velocity<SceneFrame>();
    }
};

// ---- Tests ----

TEST_CASE("Craft: construction wires root craft pointer") {
    Craft c("rover");
    CHECK(&c.root().craft() == &c);
    CHECK(c.root().parent() == nullptr);
    CHECK(c.name() == "rover");
}

TEST_CASE("Craft: add() propagates craft pointer to children") {
    Craft c("ship");
    auto& p = c.root().add<ForcePart>("thruster");
    CHECK(&p.craft() == &c);
    CHECK(p.parent() == &c.root());
}

TEST_CASE("Craft: kinematic pass — part at offset position") {
    Craft c("test");
    auto& s = c.root().add<SensorPart>("s");
    // Place the sensor at (2, 0, 0) in the craft frame.
    s.set_transform(StaticLink<ParentFrame, PartFrame>{
        Vec3<ParentFrame>{2, 0, 0},
        Ori<ParentFrame>::identity()
    });

    c.update();

    CHECK(test::approx_equal(s.last_pos_craft, Vec3<CraftFrame>{2, 0, 0}));
    CHECK(test::approx_equal(s.last_vel_craft, Vec3<CraftFrame>::zero()));
}

TEST_CASE("Craft: kinematic pass — identity transform gives zero position") {
    Craft c("test");
    auto& s = c.root().add<SensorPart>("s");
    // Default transform is identity.
    c.update();
    CHECK(test::approx_equal(s.last_pos_craft, Vec3<CraftFrame>::zero()));
}

TEST_CASE("Craft: kinematic pass — nested composites compose transforms") {
    Craft c("test");
    auto& arm = c.root().add<CompositePart>("arm");
    arm.set_transform(StaticLink<ParentFrame, PartFrame>{
        Vec3<ParentFrame>{1, 0, 0},
        Ori<ParentFrame>::identity()
    });
    auto& tip = arm.add<SensorPart>("tip");
    tip.set_transform(StaticLink<ParentFrame, PartFrame>{
        Vec3<ParentFrame>{0, 1, 0},
        Ori<ParentFrame>::identity()
    });

    c.update();

    // arm at (1,0,0), tip at arm+(0,1,0) = (1,1,0) in craft frame.
    CHECK(test::approx_equal(tip.last_pos_craft, Vec3<CraftFrame>{1, 1, 0}));
}

TEST_CASE("Craft: kinematic pass — rotation applied to child offset") {
    constexpr float kPi = 3.14159265358979f;

    Craft c("test");
    // Parent rotated 90° about z: x→y axis.
    auto rot = Ori<ParentFrame>::from_axis_angle(Vec3<ParentFrame>{0,0,1}, kPi/2.f);
    auto& arm = c.root().add<CompositePart>("arm");
    arm.set_transform(StaticLink<ParentFrame, PartFrame>{
        Vec3<ParentFrame>::zero(), rot
    });
    auto& tip = arm.add<SensorPart>("tip");
    tip.set_transform(StaticLink<ParentFrame, PartFrame>{
        Vec3<ParentFrame>{1, 0, 0},  // offset in arm frame
        Ori<ParentFrame>::identity()
    });

    c.update();

    // arm is at origin, rotated 90° about z.
    // tip offset (1,0,0) in arm frame → (0,1,0) in craft frame.
    CHECK(test::approx_equal(tip.last_pos_craft, Vec3<CraftFrame>{0, 1, 0}));
}

TEST_CASE("Craft: wrench aggregation — force at origin, no torque") {
    Craft c("test");
    auto& fp = c.root().add<ForcePart>("fp");
    fp.force_to_apply = Vec3<PartFrame>{1, 0, 0};
    fp.point_of_force = Vec3<PartFrame>::zero();
    fp.apply = true;

    c.update();

    // Root collects the child's wrench transformed to root frame (identity transform).
    auto w = c.root().net_wrench();
    CHECK(test::approx_equal(w.force(),  Vec3<PartFrame>{1, 0, 0}));
    CHECK(test::approx_equal(w.torque(), Vec3<PartFrame>::zero()));
}

TEST_CASE("Craft: wrench aggregation — force at offset point generates torque") {
    Craft c("test");
    auto& fp = c.root().add<ForcePart>("fp");
    fp.force_to_apply = Vec3<PartFrame>{0, 0, 1};  // F = (0,0,1)
    fp.point_of_force = Vec3<PartFrame>{1, 0, 0};  // r = (1,0,0)
    fp.apply = true;
    // Expected torque: r × F = (1,0,0) × (0,0,1) = (0*1-0*0, 0*0-1*1, 1*0-0*0) = (0,-1,0)

    c.update();

    auto w = c.root().net_wrench();
    CHECK(test::approx_equal(w.force(),  Vec3<PartFrame>{0, 0, 1}));
    CHECK(test::approx_equal(w.torque(), Vec3<PartFrame>{0, -1, 0}));
}

TEST_CASE("Craft: wrench aggregation — child at offset, force transforms to parent") {
    Craft c("test");
    auto& arm = c.root().add<CompositePart>("arm");
    // arm is at (2,0,0) in root frame, no rotation.
    arm.set_transform(StaticLink<ParentFrame, PartFrame>{
        Vec3<ParentFrame>{2, 0, 0}, Ori<ParentFrame>::identity()
    });
    auto& fp = arm.add<ForcePart>("fp");
    fp.force_to_apply = Vec3<PartFrame>{1, 0, 0};
    fp.point_of_force = Vec3<PartFrame>::zero();
    fp.apply = true;
    // Force in arm frame (1,0,0) → force in root frame (1,0,0) (no rotation).
    // Arm origin at (2,0,0) in root, so extra torque = (2,0,0) × (1,0,0) = (0,0,0) — nope:
    // torque = r × F = (2,0,0)×(1,0,0) = (0*0-0*0, 0*1-2*0, 2*0-0*1) = (0,0,0)
    // Still zero torque since r and F are parallel.

    c.update();

    auto w = c.root().net_wrench();
    CHECK(test::approx_equal(w.force(),  Vec3<PartFrame>{1, 0, 0}));
    CHECK(test::approx_equal(w.torque(), Vec3<PartFrame>::zero()));
}

TEST_CASE("Craft: wrench aggregation — child at offset, perpendicular force creates torque") {
    Craft c("test");
    auto& arm = c.root().add<CompositePart>("arm");
    arm.set_transform(StaticLink<ParentFrame, PartFrame>{
        Vec3<ParentFrame>{1, 0, 0}, Ori<ParentFrame>::identity()
    });
    auto& fp = arm.add<ForcePart>("fp");
    fp.force_to_apply = Vec3<PartFrame>{0, 1, 0};  // F = (0,1,0) in arm frame
    fp.point_of_force = Vec3<PartFrame>::zero();
    fp.apply = true;
    // arm origin at (1,0,0) in root.
    // torque in root = (1,0,0) × (0,1,0) = (0*0-0*1, 0*0-1*0, 1*1-0*0) = (0,0,1)

    c.update();

    auto w = c.root().net_wrench();
    CHECK(test::approx_equal(w.force(),  Vec3<PartFrame>{0, 1, 0}));
    CHECK(test::approx_equal(w.torque(), Vec3<PartFrame>{0, 0, 1}));
}

TEST_CASE("Craft: compute_params — total mass from children") {
    Craft c("test");
    auto& p1 = c.root().add<ForcePart>("p1");
    p1.set_mass(1.0f);
    auto& p2 = c.root().add<ForcePart>("p2");
    p2.set_mass(2.0f);

    c.root().compute_params();

    CHECK(c.root().get_mass() == doctest::Approx(3.0f));
}

TEST_CASE("Craft: compute_params — center of mass from two equal masses") {
    Craft c("test");
    auto& p1 = c.root().add<ForcePart>("p1");
    p1.set_mass(1.0f);
    p1.set_transform(StaticLink<ParentFrame, PartFrame>{
        Vec3<ParentFrame>{-1, 0, 0}, Ori<ParentFrame>::identity()
    });
    auto& p2 = c.root().add<ForcePart>("p2");
    p2.set_mass(1.0f);
    p2.set_transform(StaticLink<ParentFrame, PartFrame>{
        Vec3<ParentFrame>{1, 0, 0}, Ori<ParentFrame>::identity()
    });

    c.root().compute_params();

    CHECK(test::approx_equal(c.root().get_com(), Vec3<PartFrame>::zero()));
}

TEST_CASE("Craft: compute_params — MOI of two point masses on x-axis") {
    // Two 1 kg masses at (+1, 0, 0) and (-1, 0, 0).
    // Combined MOI about the composite COM (origin): I_yy = I_zz = m*d^2 each.
    // Each contributes m*(1^2) = 1 to y and z axes; total Iyy = Izz = 2.
    // Ixx = 0 (masses sit on x-axis, zero lever arm).
    Craft c("test");
    auto& p1 = c.root().add<ForcePart>("p1");
    p1.set_mass(1.0f);
    p1.set_transform(StaticLink<ParentFrame, PartFrame>{
        Vec3<ParentFrame>{1, 0, 0}, Ori<ParentFrame>::identity()
    });
    auto& p2 = c.root().add<ForcePart>("p2");
    p2.set_mass(1.0f);
    p2.set_transform(StaticLink<ParentFrame, PartFrame>{
        Vec3<ParentFrame>{-1, 0, 0}, Ori<ParentFrame>::identity()
    });

    c.root().compute_params();

    auto I = c.root().get_moi().raw();
    CHECK(I(0, 0) == doctest::Approx(0.0f).epsilon(1e-4f));
    CHECK(I(1, 1) == doctest::Approx(2.0f).epsilon(1e-4f));
    CHECK(I(2, 2) == doctest::Approx(2.0f).epsilon(1e-4f));
}

TEST_CASE("Craft: dynamics — scene sensor at origin before integration") {
    // Before any update the scene_to_part_ cache must be populated by pass 1.
    Craft c("test");
    auto& s = c.root().add<SceneSensorPart>("s");

    // Set craft position to (5, 0, 0).
    c.set_position(Vec3<SceneFrame>{5, 0, 0});
    c.update();  // dt=0: kinematic + wrench passes only

    CHECK(test::approx_equal(s.last_pos, Vec3<SceneFrame>{5, 0, 0}));
    CHECK(test::approx_equal(s.last_vel, Vec3<SceneFrame>::zero()));
}

TEST_CASE("Craft: dynamics — constant force integrates velocity") {
    // Single 1 kg mass at root, apply F=(1,0,0) in craft frame.
    // After dt=1 s: v_scene = (1,0,0), p_scene = (0.5,0,0) (exact for const accel).
    Craft c("test");
    auto& fp = c.root().add<ForcePart>("fp");
    fp.force_to_apply = Vec3<PartFrame>{1, 0, 0};
    fp.point_of_force = Vec3<PartFrame>::zero();
    fp.apply = true;
    fp.set_mass(1.0f);
    c.root().compute_params();

    const float dt = 1.0f;
    c.update(dt);

    auto& stc = c.scene_to_craft();
    CHECK(test::approx_equal(stc.vel_linear(),  Vec3<SceneFrame>{1, 0, 0}));
    CHECK(test::approx_equal(stc.position(),    Vec3<SceneFrame>{0.5f, 0, 0}));
}

TEST_CASE("Craft: dynamics — angular acceleration from pure torque") {
    // Diagonal MOI diag(2,2,2), torque (2,0,0) → alpha=(1,0,0).
    // After dt=1: omega=(1,0,0).
    Craft c("test");
    auto& fp = c.root().add<ForcePart>("fp");
    fp.force_to_apply = Vec3<PartFrame>::zero();
    fp.point_of_force = Vec3<PartFrame>::zero();
    fp.apply = true;
    fp.set_mass(1.0f);
    fp.set_moi(geom::Mat3<PartFrame>::from_diagonal(2.0f, 2.0f, 2.0f));

    // Override ForcePart::update to also apply a torque.
    struct TorquePart : public Part {
        explicit TorquePart(std::string n) : Part(std::move(n)) {}
        void update() override { apply_torque(Vec3<PartFrame>{2, 0, 0}); }
    };

    Craft c2("test2");
    auto& tp = c2.root().add<TorquePart>("tp");
    tp.set_mass(1.0f);
    tp.set_moi(geom::Mat3<PartFrame>::from_diagonal(2.0f, 2.0f, 2.0f));
    c2.root().compute_params();

    c2.update(1.0f);

    auto& stc = c2.scene_to_craft();
    CHECK(test::approx_equal(stc.vel_angular(), Vec3<CraftFrame>{1, 0, 0}));
}

TEST_CASE("Craft: dynamics — velocity drives position over multiple steps") {
    // Start with v=(1,0,0), no force. After two steps of dt=0.5: p=(1,0,0).
    Craft c("test");
    auto& fp = c.root().add<ForcePart>("fp");
    fp.set_mass(1.0f);
    fp.apply = false;
    c.root().compute_params();
    c.set_vel_linear(Vec3<SceneFrame>{1, 0, 0});

    c.update(0.5f);
    c.update(0.5f);

    CHECK(test::approx_equal(c.scene_to_craft().position(), Vec3<SceneFrame>{1, 0, 0}));
}

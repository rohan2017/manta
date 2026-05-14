#include <doctest/doctest.h>

#include "../include/manta/core/craft.hpp"
#include "../include/manta/core/scene.hpp"
#include "../include/manta/core/world.hpp"
#include "../include/manta/parts/articulation/motor.hpp"
#include "../include/manta/parts/structure/mass.hpp"
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
    // Mass at +x offset from motor's joint output. Apply a torque about
    // the motor's z axis directly to the point-mass's wrench accumulator. The
    // motor's resolve() should compute axial accel = τ_axial / I_axial, and
    // the joint should advance accordingly after one tick.
    Craft c("test");
    auto& motor = c.root().add<Motor>("m", Vec3<PartFrame>{0, 0, 1});

    // Child mass: 1 kg at (0.5, 0, 0) in motor's joint-output frame.
    auto& mass = motor.add<Mass>("p", 1.0f);
    mass.set_transform(StaticLink<ParentFrame, PartFrame>{
        Vec3<ParentFrame>{0.5f, 0.0f, 0.0f},
        Ori<ParentFrame>::identity()});

    // Add a synthetic torque on the mass directly. Mass::update() is a
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
    MFloat dt = 0.01f;
    c.update(dt);
    CHECK(motor.accel() == doctest::Approx(4.0f).epsilon(1e-3f));
    CHECK(motor.rate()  == doctest::Approx(4.0f * dt).epsilon(1e-3f));
}

TEST_CASE("Motor: saturating mode applies actuator torque, clamped to stall") {
    Craft c("test");
    auto& motor = c.root().add<Motor>("m", Vec3<PartFrame>{0, 0, 1},
                                       /*stall_torque=*/0.5f);
    auto& mass = motor.add<Mass>("p", 1.0f);
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
    auto& mL = motor.add<Mass>("mL", m);
    mL.set_transform(StaticLink<ParentFrame, PartFrame>{
        Vec3<ParentFrame>{-r, 0, 0},
        Ori<ParentFrame>::identity()});
    auto& mR = motor.add<Mass>("mR", m);
    mR.set_transform(StaticLink<ParentFrame, PartFrame>{
        Vec3<ParentFrame>{+r, 0, 0},
        Ori<ParentFrame>::identity()});

    motor.set_torque(tau);
    scene.add_craft(c, InitialState{});

    for (int i = 0; i < N; ++i) {
        w.step();
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

    auto& m1 = c.root().add<Mass>("m1", m);
    m1.set_transform(StaticLink<ParentFrame, PartFrame>{
        Vec3<ParentFrame>{+r, 0, 0},
        Ori<ParentFrame>::identity()});

    auto& motor = c.root().add<Motor>("motor", Vec3<PartFrame>{0, 0, 1},
                                       /*stall=*/10.0f);
    auto& m2 = motor.add<Mass>("m2", m);
    m2.set_transform(StaticLink<ParentFrame, PartFrame>{
        Vec3<ParentFrame>{-r, 0, 0},
        Ori<ParentFrame>::identity()});

    motor.set_torque(tau);
    scene.add_craft(c, InitialState{});

    for (int i = 0; i < N; ++i) {
        w.step();
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


// ---- Fictitious-force corrections for joint dynamics ----
//
// Today's ArticulatedPart model treats each joint as a decoupled scalar
// pendulum — it doesn't include the rotor's stored angular momentum in
// the body's Euler equation, doesn't propagate base-rotation Coriolis
// coupling onto the joint axis, and doesn't pull the mount outward when
// the rotor has an off-axis COM. Three fictitious-force injections
// recover those without going to a full Featherstone solver:
//
//   1. −ω_craft × h_rotor on the body's Euler equation, where
//      h_rotor = Σ I_axial · θ̇ · axis_in_craft over all joints. Gives
//      gyroscopic precession from spinning rotors.
//   2. −[ω_rotor × (I_joint · ω_rotor)] · axis on the joint's axial
//      torque (Motor::resolve). Captures the Coriolis coupling that
//      drives a non-axisymmetric rotor when its mount rotates.
//   3. −m · ω_rotor × (ω_rotor × r_COM) on the parent reaction force.
//      Off-axis-COM centrifugal pull on the mount.

// Helper for the gyro/centripetal tests — applies a constant torque to
// its parent's wrench accumulator each update().
struct TorquePusher : public Part {
    Vec3<PartFrame> tau;
    TorquePusher(std::string n, Vec3<PartFrame> t)
        : Part(std::move(n)), tau(t) {}
    void update() override { apply_torque(tau); }
};

// (1) Gyroscopic torque on body from a spinning rotor.
//
// Setup: heavy body with a fast-spinning reaction-wheel-style rotor about
// body z. Apply a transverse torque about body x. Without the gyro term,
// the body would accelerate in pure x rotation. With the gyro term, the
// body also picks up a y-direction angular acceleration (precession).
//
// Quantitative check (linear regime, before ω grows large):
//   α_y ≈ −ω_x · h_z / I_yy, with h_z = I_axial_wheel · θ̇_wheel.
TEST_CASE("Gyro torque: spinning rotor causes body precession under transverse torque") {
    constexpr float dt           = 0.001f;
    constexpr int   N_STEPS      = 500;     // 0.5 s
    constexpr float tau_x_input  = 0.05f;   // transverse torque on body about x
    constexpr float wheel_rate   = 200.0f;  // rad/s; spin pre-set
    constexpr float wheel_mass   = 0.5f;
    constexpr float wheel_R      = 0.1f;
    // Thin disk axial MOI: ½·m·R². Perpendicular: ¼·m·R².
    constexpr float I_axial_wheel = 0.5f * wheel_mass * wheel_R * wheel_R;
    constexpr float I_perp_wheel  = 0.25f * wheel_mass * wheel_R * wheel_R;
    // Heavy body so its bare MOI dominates the aggregate.
    constexpr float I_body       = 0.5f;

    World w;
    w.clock().set_dt(dt);
    auto& scene = w.create_scene();
    Craft c("gyro_test");

    auto& body = c.root().add<Mass>("body", 1.0f);
    body.set_moi(Mat3<PartFrame, PartFrame>{
        (Eigen::Matrix<MFloat, 3, 3>() <<
             I_body, 0,      0,
             0,      I_body, 0,
             0,      0,      I_body).finished()});

    // Reaction wheel: Motor about body z + a Mass child with disk MOI.
    auto& motor = c.root().add<Motor>("wheel_motor", Vec3<PartFrame>{0, 0, 1});
    auto& disk  = motor.add<Mass>("wheel", wheel_mass);
    disk.set_moi(Mat3<PartFrame, PartFrame>{
        (Eigen::Matrix<MFloat, 3, 3>() <<
             I_perp_wheel, 0,            0,
             0,            I_perp_wheel, 0,
             0,            0,            I_axial_wheel).finished()});

    // Pre-spin the wheel before it enters the scene (no actuator drive).
    motor.set_passive();
    motor.set_rate(wheel_rate);

    // Apply an external transverse torque on the body via a TorquePusher.
    c.root().add<TorquePusher>("ext_x",
                               Vec3<PartFrame>{tau_x_input, 0.0f, 0.0f});

    scene.add_craft(c, InitialState{});

    for (int i = 0; i < N_STEPS; ++i) w.step();
    c.kinematic_pass();   // refresh caches for inspection

    const auto omega = c.get_state().vel_angular.raw();
    INFO("ω_body = (", omega.x(), ", ", omega.y(), ", ", omega.z(), ")");

    // Without the gyro term, ω_y would be exactly 0 (no torque about y).
    // With the gyro term, the precession response gives measurable
    // positive ω_y. Sign: h_rotor = +h_z·ẑ; ω_x > 0; the corrective
    // torque is −ω × h_rotor = −(ω_x·x̂)×(h_z·ẑ) = −ω_x·h_z·(x̂×ẑ)
    //                       = −ω_x·h_z·(−ŷ) = +ω_x·h_z·ŷ
    // — a positive-y torque on the body, integrating to positive ω_y.
    CHECK(omega.y() > 1e-3f);

    // Linearized prediction:
    //   I_xx ≈ I_body + I_perp_wheel
    //   ω_x(t) ≈ τ_x · t / I_xx
    //   α_y(t) ≈ +ω_x(t) · h_z / I_yy
    //   ω_y(t) ≈ +τ_x · h_z · t² / (2 · I_xx · I_yy)
    const float h_z       = I_axial_wheel * wheel_rate;
    const float I_xx_tot  = I_body + I_perp_wheel;
    const float I_yy_tot  = I_body + I_perp_wheel;
    const float t_final   = N_STEPS * dt;
    const float expected_y =
        +tau_x_input * h_z * t_final * t_final / (2.0f * I_xx_tot * I_yy_tot);
    INFO("expected ω_y ≈ ", expected_y, "  actual ", omega.y());
    // Generous tolerance: linearized prediction drifts as ω grows.
    CHECK(omega.y() == doctest::Approx(expected_y).epsilon(0.3f));
}

// (2) Coriolis joint torque from base rotation through a non-axisymmetric
//     rotor.
//
// Setup: heavy body rotating about its x axis, with a Motor on body z
// carrying a single offset point-mass child (rotor MOI in mount frame is
// non-axisymmetric about z). At motor angle θ=π/4 the mount-frame I
// tensor has non-trivial off-diagonal elements that project a non-zero
// Coriolis torque onto the joint axis.
//
// Theory at this configuration (m at +x in joint-output frame, θ=π/4):
//   I_joint_mount(π/4) = R_z(π/4) · diag(0, mr², mr²) · R_z(π/4)^T
//                      = [[mr²/2, −mr²/2, 0],
//                         [−mr²/2, mr²/2, 0],
//                         [0,      0,     mr²]]
//   ω_rotor = (ω_x, 0, 0)  (ω_mount in mount frame; rate=0)
//   I·ω = (mr²/2·ω_x, −mr²/2·ω_x, 0)
//   ω×(I·ω) = (0, 0, −mr²·ω_x²/2)
//   τ_corio_axial = −[ω×(I·ω)]·ẑ = +mr²·ω_x²/2
//   I_axial = mr²
//   θ̈_corio = +ω_x²/2
TEST_CASE("Coriolis: non-axisymmetric rotor on yawing base drives joint") {
    constexpr float r            = 0.5f;
    constexpr float m            = 1.0f;
    constexpr float omega_x      = 10.0f;
    constexpr float dt           = 0.001f;
    constexpr float theta_init   = 0.7853981633974483f;   // π/4

    World w;
    w.clock().set_dt(dt);
    auto& scene = w.create_scene();
    Craft c("coriolis_test");

    // Heavy body so its bare MOI dominates. Keeps ω_mount nearly constant
    // during the tick — otherwise the rotor's gyro reaction would
    // significantly perturb the body's angular velocity.
    constexpr float I_body = 1000.0f;
    auto& body = c.root().add<Mass>("body", 1.0f);
    body.set_moi(Mat3<PartFrame, PartFrame>{
        (Eigen::Matrix<MFloat, 3, 3>() <<
             I_body, 0,      0,
             0,      I_body, 0,
             0,      0,      I_body).finished()});

    auto& motor = c.root().add<Motor>("m", Vec3<PartFrame>{0, 0, 1});
    auto& mass  = motor.add<Mass>("p", m);
    mass.set_transform(StaticLink<ParentFrame, PartFrame>{
        Vec3<ParentFrame>{r, 0.0f, 0.0f},
        Ori<ParentFrame>::identity()});

    motor.set_passive();
    motor.set_angle(theta_init);
    motor.set_rate(0.0f);

    InitialState init;
    init.vel_angular = Vec3<CraftFrame>{omega_x, 0.0f, 0.0f};
    scene.add_craft(c, init);

    // One step computes the joint acceleration via the Coriolis term.
    w.step();

    // Expected joint acceleration from the analytical formula above.
    const float expected_accel = +0.5f * omega_x * omega_x;   // ω_x²/2
    INFO("motor.accel() = ", motor.accel(),
         "  expected = ", expected_accel);
    CHECK(motor.accel() == doctest::Approx(expected_accel).epsilon(0.05f));
}

// (3) Moving-COM correction: free craft with off-axis spinning rotor.
//
// Conservation of linear momentum says a closed system with no external
// forces cannot accumulate net displacement. The body origin DOES move
// (it wobbles in a bounded circle around the system COM), but the
// SYSTEM COM stays exactly where it started.
//
// Setup: 1 kg body + 1 kg rotor mass at (r=0.5, 0, 0) in motor's
// joint-output frame. Pre-spin the rotor at 30 rad/s. No external
// forces, no actuator drive. The system COM at t=0 is at
//   r_C = (m_R · (r,0,0)) / m_total = (0.25, 0, 0)
// and should stay there for all time.
//
// The body origin oscillates around that COM with amplitude
// m_R/m_total · r = 0.25 m, and peak velocity θ̇·m_R/m_total·r = 7.5 m/s.
//
// Without the moving-COM correction (or with the old buggy F_centrifugal-
// to-wrench approach), the system COM would drift — adding F to the
// body's Newton equation as m_total·a_C = F_total spuriously accelerates
// the COM, which is exactly the violation conservation forbids.
TEST_CASE("Moving COM: free craft with spinning unbalanced rotor — system COM is conserved") {
    constexpr float r          = 0.5f;
    constexpr float m_body     = 1.0f;
    constexpr float m_rotor    = 1.0f;
    constexpr float m_total    = m_body + m_rotor;
    constexpr float wheel_rate = 30.0f;     // rad/s
    constexpr float dt         = 0.001f;
    constexpr int   N_STEPS    = 500;       // 0.5 s — ~2.4 rotor periods

    World w;
    w.clock().set_dt(dt);
    auto& scene = w.create_scene();
    Craft c("conservation_test");

    auto& body = c.root().add<Mass>("body", m_body);
    body.set_moi(Mat3<PartFrame, PartFrame>::from_diagonal(0.1f, 0.1f, 0.1f));

    auto& motor = c.root().add<Motor>("m", Vec3<PartFrame>{0, 0, 1});
    auto& mass  = motor.add<Mass>("p", m_rotor);
    mass.set_transform(StaticLink<ParentFrame, PartFrame>{
        Vec3<ParentFrame>{r, 0.0f, 0.0f},
        Ori<ParentFrame>::identity()});

    motor.set_passive();
    motor.set_rate(wheel_rate);

    scene.add_craft(c, InitialState{});
    c.kinematic_pass();   // refresh caches for initial COM read

    // System COM at t=0 (in scene frame).
    auto p_body_0  = c.get_state().position.raw();              // (0,0,0)
    auto p_rotor_0 = mass.scene_to_part().position().raw();      // (r,0,0)
    Eigen::Matrix<MFloat, 3, 1> com_0 =
        (m_body * p_body_0 + m_rotor * p_rotor_0) / m_total;

    // System COM velocity at t=0. At rest body + spinning rotor: only the
    // rotor mass has inertial velocity (= θ̇·axis × position), so
    //   v_C(0) = m_R/m_total · θ̇·z × (r, 0, 0) = m_R/m_total · (0, θ̇·r, 0)
    Eigen::Matrix<MFloat, 3, 1> v_C_0{
        0.0f,
        m_rotor / m_total * wheel_rate * r,
        0.0f};

    for (int i = 0; i < N_STEPS; ++i) w.step();
    c.kinematic_pass();

    auto p_body  = c.get_state().position.raw();
    auto p_rotor = mass.scene_to_part().position().raw();
    Eigen::Matrix<MFloat, 3, 1> com =
        (m_body * p_body + m_rotor * p_rotor) / m_total;

    // Linear-momentum conservation: with no external forces, the COM moves
    // at constant velocity, so
    //   COM(t) − COM(0) − v_C(0)·t  must stay ≈ 0.
    const float t_final = N_STEPS * dt;
    Eigen::Matrix<MFloat, 3, 1> com_predicted = com_0 + v_C_0 * t_final;
    Eigen::Matrix<MFloat, 3, 1> drift = com - com_predicted;

    INFO("initial COM   = (", com_0.x(),       ",", com_0.y(),       ",", com_0.z(),       ")");
    INFO("predicted COM = (", com_predicted.x(),",", com_predicted.y(),",", com_predicted.z(),")");
    INFO("actual    COM = (", com.x(),         ",", com.y(),         ",", com.z(),         ")");
    INFO("drift         = ", drift.norm());

    // Drift tolerance accounts for explicit-Euler integrator error over
    // ~15 rad of acceleration-vector rotation. With the buggy F-in-wrench
    // approach the drift would be ~3 m (the entire COM-conservation
    // violation we're testing for); 0.1 m comfortably separates the two.
    CHECK(drift.norm() < 0.1f);

    // Sanity: the body origin moved measurably (not pinned at start).
    CHECK((p_body - p_body_0).norm() > 0.01f);
}

// (4) Rigid-body regression: stationary joint + rotating body should
//     behave as a single rigid body.
//
// Setup: identical part tree to (3), but the rotor is locked (θ̇=0) and
// the body is given a non-zero initial ω_z. Since the joint isn't
// moving, the whole craft is instantaneously rigid in body frame —
// conservation of L and linear momentum applies. With no external
// forces, the system COM is exactly stationary (no drift, no wobble).
//
// This catches the bug that the old patch 3 had: it added
// F = −m·ω_rotor × (ω_rotor × r) to the parent wrench, which for
// θ̇=0 reduces to F = −m·ω_body × (ω_body × r) — a spurious centripetal
// term layered on top of the body-Euler equation's existing
// ω × (ω × r_OC) correction, causing double-counting and COM drift.
TEST_CASE("Rigid-body regression: stationary joint on rotating body conserves COM") {
    constexpr float r          = 0.5f;
    constexpr float m_body     = 1.0f;
    constexpr float m_rotor    = 1.0f;
    constexpr float m_total    = m_body + m_rotor;
    constexpr float omega_z    = 5.0f;      // body spin rad/s
    constexpr float dt         = 0.001f;
    constexpr int   N_STEPS    = 500;       // 0.5 s, several body rotations

    World w;
    w.clock().set_dt(dt);
    auto& scene = w.create_scene();
    Craft c("rigid_regression_test");

    auto& body = c.root().add<Mass>("body", m_body);
    body.set_moi(Mat3<PartFrame, PartFrame>::from_diagonal(0.5f, 0.5f, 0.5f));

    auto& motor = c.root().add<Motor>("m", Vec3<PartFrame>{0, 0, 1});
    auto& mass  = motor.add<Mass>("p", m_rotor);
    mass.set_transform(StaticLink<ParentFrame, PartFrame>{
        Vec3<ParentFrame>{r, 0.0f, 0.0f},
        Ori<ParentFrame>::identity()});

    motor.set_passive();
    motor.set_rate(0.0f);   // joint locked

    InitialState init;
    init.vel_angular = Vec3<CraftFrame>{0.0f, 0.0f, omega_z};
    scene.add_craft(c, init);
    c.kinematic_pass();

    auto p_body_0  = c.get_state().position.raw();
    auto p_rotor_0 = mass.scene_to_part().position().raw();
    Eigen::Matrix<MFloat, 3, 1> com_0 =
        (m_body * p_body_0 + m_rotor * p_rotor_0) / m_total;

    // System COM velocity at t=0: body rotates at ω_z, rotor mass is offset,
    // so rotor mass has inertial velocity ω×r in scene frame; body mass is
    // at origin (no velocity).
    //   v_C(0) = m_R/m_total · ω × (r, 0, 0) = m_R/m_total · (0, ω·r, 0)
    Eigen::Matrix<MFloat, 3, 1> v_C_0{
        0.0f,
        m_rotor / m_total * omega_z * r,
        0.0f};

    for (int i = 0; i < N_STEPS; ++i) w.step();
    c.kinematic_pass();

    auto p_body  = c.get_state().position.raw();
    auto p_rotor = mass.scene_to_part().position().raw();
    Eigen::Matrix<MFloat, 3, 1> com =
        (m_body * p_body + m_rotor * p_rotor) / m_total;

    const float t_final = N_STEPS * dt;
    Eigen::Matrix<MFloat, 3, 1> com_predicted = com_0 + v_C_0 * t_final;
    Eigen::Matrix<MFloat, 3, 1> drift = com - com_predicted;

    INFO("initial COM   = (", com_0.x(),       ",", com_0.y(),       ",", com_0.z(),       ")");
    INFO("predicted COM = (", com_predicted.x(),",", com_predicted.y(),",", com_predicted.z(),")");
    INFO("actual    COM = (", com.x(),         ",", com.y(),         ",", com.z(),         ")");
    INFO("drift         = ", drift.norm());

    // Linear-momentum conservation: COM trajectory must be straight-line.
    // This is the strict rigid-body regression check — catches the old
    // F-in-wrench bug that would add a spurious ω×(ω×r) centripetal
    // force layered on top of the body Euler equation's existing r_OC
    // correction, causing extra COM drift.
    CHECK(drift.norm() < 0.05f);
}

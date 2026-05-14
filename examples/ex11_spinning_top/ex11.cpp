// Example 11 — Spinning-top precession on a flat ground plane.
//
// Pure C++, no codegen, no Zenoh. Showcases the articulation patches that
// landed for moving-COM and gyroscopic-torque physics.
//
// Craft: a thin vertical stick (mass 0.1 kg, length 0.8 m) with a heavy
// flywheel (2 kg disk, r = 0.15 m) attached via a Motor whose joint axis
// is along body +z (parallel to the stick). The stick carries two small
// sphere Colliders, one at each end, that interact with a flat ground
// plane via the CollisionField. A small Thruster at the top of the stick
// fires laterally at t = 3 s to tip the craft over slightly.
//
// Initial conditions: stick balanced on the bottom collider, flywheel
// pre-spun to 200 rad/s about body +z. Without the flywheel's stored
// angular momentum, the small lateral kick would tip the stick straight
// over and it would simply fall in the kicked direction. With the
// flywheel, the body precesses around the vertical — `dL/dt = τ` with
// L mostly along +ẑ and τ from gravity perpendicular to ẑ produces a
// horizontal rotation of the tilt direction instead of an unbounded
// growth of the tilt itself.
//
// Theoretical precession rate (regular precession of a heavy symmetric
// top, neglecting nutation):
//     Ω = M·g·h / (I_axial · θ̇)
// with M ≈ 2.1 kg total, h ≈ 0.6 m from contact to flywheel COM,
// I_axial = ½·m_fly·r² = 0.0225 kg·m², θ̇ = 200 rad/s. So Ω ≈ 2.7 rad/s,
// period ≈ 2.3 s — visible 3-4× over the 12 s sim.

#include <cmath>
#include <cstdio>

#include "manta/core/craft.hpp"
#include "manta/core/scene.hpp"
#include "manta/core/world.hpp"
#include "manta/fields/collision_field.hpp"
#include "manta/fields/gravity_field.hpp"
#include "manta/parts/actuator/thruster.hpp"
#include "manta/parts/articulation/motor.hpp"
#include "manta/parts/field_src/collider.hpp"
#include "manta/parts/structure/mass.hpp"

using namespace manta;
using namespace manta::geom;
using namespace manta::parts;

namespace {

// ---- sim ----
constexpr float DT          = 0.001f;     // 1 kHz
constexpr float T_END       = 12.0f;
constexpr float LOG_EVERY   = 0.2f;       // print state every 0.2 s

// ---- stick ----
constexpr float L_STICK     = 0.8f;       // m, full length
constexpr float STICK_MASS  = 0.1f;       // kg
// Thin rod about a transverse axis through its center: I = m·L²/12.
constexpr float STICK_I_PERP = STICK_MASS * L_STICK * L_STICK / 12.0f;

// ---- flywheel ----
constexpr float FLY_MASS    = 2.0f;       // kg
constexpr float FLY_R       = 0.15f;      // m
// Thin disk: I_axial = ½·m·r², I_perp = ¼·m·r².
constexpr float FLY_I_AXIAL = 0.5f  * FLY_MASS * FLY_R * FLY_R;   // 0.0225
constexpr float FLY_I_PERP  = 0.25f * FLY_MASS * FLY_R * FLY_R;
constexpr float FLY_RATE_0  = 200.0f;     // rad/s pre-spin
constexpr float MOTOR_Z     = 0.20f;      // motor mount offset from body origin

// ---- thruster ----
constexpr float THRUSTER_Z       = +L_STICK / 2.0f;   // at top of stick
constexpr float THRUSTER_FORCE   = 10.0f;             // N when on
constexpr float KICK_START_T     = 3.0f;              // s
constexpr float KICK_DURATION    = 0.05f;             // 50 ms — brief impulse

// ---- ground ----
constexpr float COLLIDER_R       = 0.02f;
constexpr float GROUND_K         = 5000.0f;           // N/m   normal stiffness
constexpr float GROUND_D         = 50.0f;             // N·s/m normal damping
constexpr float MU_STATIC        = 0.9f;
constexpr float MU_KINETIC       = 0.7f;

Mat3<PartFrame> diag_moi(float ixx, float iyy, float izz) {
    Mat3<PartFrame> m;
    m.raw().setZero();
    m.raw()(0, 0) = ixx;
    m.raw()(1, 1) = iyy;
    m.raw()(2, 2) = izz;
    return m;
}

}  // namespace

int main() {
    World w;
    w.clock().set_dt(DT);

    // Gravity: uniform −9.81 m/s² along scene −z.
    fields::GravityField gravity{Vec3<SceneFrame>{0, 0, -9.81f}};
    w.register_field(gravity);

    // Ground: an infinite plane at z=0 in scene frame, normal +z.
    fields::CollisionField ground;
    ground.add(fields::CollisionField::Disturbance::infinite_plane(
                   /*point=*/Vec3<SceneFrame>{0, 0, 0},
                   /*normal=*/Vec3<SceneFrame>{0, 0, 1},
                   GROUND_K, GROUND_D, MU_STATIC, MU_KINETIC),
               fields::PERSISTENT);
    w.register_field(ground);

    auto& scene = w.create_scene();

    // ---- craft ----
    Craft craft("top");

    // Stick body: a thin rod along body z with no axial inertia and
    // m·L²/12 transverse inertia. Mass concentrated at the body origin
    // (no offset COM contribution from the stick alone).
    craft.root().add<Mass>(
        "stick",
        STICK_MASS,
        diag_moi(STICK_I_PERP, STICK_I_PERP, /*Izz=*/1e-6f),
        /*apply_gravity=*/true);

    // Two end-cap colliders: small spheres at the stick ends. They share
    // the same field-disturbance pool so collide independently with the
    // ground plane. The bottom collider provides the pivot once the
    // craft rests on the floor; the top collider only matters if the
    // stick rotates past horizontal.
    auto& bottom_col = craft.root().add<Collider>(
        "bottom_col", COLLIDER_R, GROUND_K, GROUND_D, MU_STATIC, MU_KINETIC);
    bottom_col.set_transform(StaticLink<ParentFrame, PartFrame>{
        Vec3<ParentFrame>{0, 0, -L_STICK / 2.0f},
        Ori<ParentFrame>::identity()});

    auto& top_col = craft.root().add<Collider>(
        "top_col", COLLIDER_R, GROUND_K, GROUND_D, MU_STATIC, MU_KINETIC);
    top_col.set_transform(StaticLink<ParentFrame, PartFrame>{
        Vec3<ParentFrame>{0, 0, +L_STICK / 2.0f},
        Ori<ParentFrame>::identity()});

    // Motor: revolute joint about body +z, mounted at z = MOTOR_Z above
    // body origin (= MOTOR_Z + L_STICK/2 above the bottom contact).
    // Passive mode: no actuator drive, the joint spins freely under
    // whatever axial torque it sees (zero here — pure flywheel inertia).
    auto& motor = craft.root().add<Motor>(
        "fly_motor", Vec3<PartFrame>{0, 0, 1}, /*stall=*/0.0f, /*damping=*/0.0f);
    motor.set_transform(StaticLink<ParentFrame, PartFrame>{
        Vec3<ParentFrame>{0, 0, MOTOR_Z},
        Ori<ParentFrame>::identity()});
    motor.set_passive();

    // Flywheel: heavy disk child of the motor. Thin disk about z: I_axial
    // = ½·m·r², I_perp = ¼·m·r². Sits at the motor's joint output origin.
    auto& flywheel = motor.add<Mass>(
        "flywheel",
        FLY_MASS,
        diag_moi(FLY_I_PERP, FLY_I_PERP, FLY_I_AXIAL),
        /*apply_gravity=*/true);
    (void)flywheel;

    // Thruster: lateral (+x in body frame), mounted near the top of the
    // stick. Acts via set_throttle in the tick loop; force = max_thrust
    // along direction at throttle = 1.
    auto& thruster = craft.root().add<Thruster>(
        "kick",
        THRUSTER_FORCE,
        Vec3<PartFrame>{1, 0, 0});
    thruster.set_transform(StaticLink<ParentFrame, PartFrame>{
        Vec3<ParentFrame>{0, 0, THRUSTER_Z},
        Ori<ParentFrame>::identity()});

    // ---- initial state ----
    // Place the body origin just high enough that the bottom collider
    // sphere's *bottom point* touches z=0. The sphere center is at
    // (0, 0, z_body − L_STICK/2); for its surface to be at z=0,
    //     z_body = L_STICK/2 + COLLIDER_R.
    InitialState init;
    init.position = Vec3<SceneFrame>{0, 0, L_STICK / 2.0f + COLLIDER_R};
    scene.add_craft(craft, init);

    // Pre-spin the flywheel. With passive mode and no friction the rate
    // is conserved by the joint integrator (modulo the gyroscopic
    // back-reaction from a tilting body — which only switches on after
    // the kick).
    motor.set_rate(FLY_RATE_0);

    std::printf("# t         x         y         z       tilt[deg]  azim[deg]  fly_rate\n");

    float t          = 0.0f;
    float t_next_log = 0.0f;
    while (t < T_END) {
        // Kick window: 50 ms of full-throttle +x at the top of the stick.
        // Force impulse = 10 N × 0.05 s = 0.5 N·s; torque impulse about
        // the bottom contact ≈ F · L_stick · dt = 0.5 × 0.8 ≈ 0.4 N·m·s.
        const bool kicking = (t >= KICK_START_T) &&
                             (t < KICK_START_T + KICK_DURATION);
        thruster.set_throttle(kicking ? 1.0f : 0.0f);

        w.step();
        t += DT;

        if (t >= t_next_log) {
            t_next_log = t + LOG_EVERY;
            const auto p = craft.get_state().position.raw();
            const auto q = craft.get_state().orientation.raw();

            // Body +z direction in scene frame: R · ẑ_body. The tilt
            // angle is the angle between this and scene +z. Azimuth is
            // the planar direction the tilt points.
            Eigen::Matrix<MFloat, 3, 1> z_body_in_scene =
                q.toRotationMatrix() * Eigen::Matrix<MFloat, 3, 1>{0, 0, 1};
            const float zc   = std::min(MFloat(1),
                                        std::max(MFloat(-1),
                                                 z_body_in_scene.z()));
            const float tilt = std::acos(zc);
            const float azim = std::atan2(z_body_in_scene.y(),
                                          z_body_in_scene.x());

            std::printf("%6.2f  %8.4f  %8.4f  %8.4f  %8.2f  %8.2f  %8.2f%s\n",
                        t,
                        p.x(), p.y(), p.z(),
                        tilt * 180.0f / float(M_PI),
                        azim * 180.0f / float(M_PI),
                        motor.rate(),
                        kicking ? "   [kicking]" : "");
        }
    }

    return 0;
}

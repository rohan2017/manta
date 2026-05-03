// Example 3 — TVC rocket "hopper" (post-2026-05 redesign).
//
// Library workflow: Ex3Craft is codegen-emitted from craft.py. The deleted
// `GimbaledThruster` is now a stack of two `Motor`s (yaw outer, pitch inner)
// hosting a `Thruster1` engine — same physics, expressed via composition.
//
// The old `engine.set_gimbal(pitch, yaw)` call set the gimbal angle
// instantaneously. Motors are torque-controlled, so we run a stiff position
// PD on each axis to track the desired gimbal angle. Stall torque on each
// motor (100 N·m vs ~50 N peak from the engine) is plenty of headroom.
//
// Zenoh:
//   subscribe 'manta/ex3/cmd'   = [thr, pitch_rate, yaw_rate]
//   publish   'manta/ex3/state' = standard nested telemetry JSON

#include <atomic>
#include <csignal>
#include <cstdio>
#include <mutex>
#include <string>
#include <vector>

#include <zenoh.hxx>

#include "manta/control/pid.hpp"
#include "manta/core/scene.hpp"
#include "manta/core/world.hpp"
#include "manta/fields/gravity_field.hpp"

#include "ex3_craft.hpp"
#include "ex3_telemetry.hpp"

#include "sim_loop.hpp"
#include "state_codec.hpp"

using namespace manta;
using namespace manta::geom;
using namespace manta::control;
using namespace manta::examples;

namespace {
std::atomic<bool> g_run{true};
void on_signal(int) { g_run.store(false); }

// Stiff position PD on a single Motor — emulates instantaneous angle tracking
// the way the old GimbaledThruster did. Kp/Kd tuned for the 5 kg rocket
// inertia: the gimbal converges to a step setpoint in ~10 ms.
inline float gimbal_pd(parts::Motor& m, float angle_setpoint,
                       float Kp = 50.0f, float Kd = 5.0f) {
    float err     = angle_setpoint - m.angle();
    float err_dot = -m.rate();
    return Kp * err + Kd * err_dot;
}

constexpr float MAX_GIMBAL = 0.15f;   // rad — same envelope as old design

inline float clamp(float v, float lim) { return v < -lim ? -lim : (v > lim ? lim : v); }
}

int main() {
    std::signal(SIGINT,  on_signal);
    std::signal(SIGTERM, on_signal);

    constexpr float DT = 0.001f;
    constexpr float HOVER_FRACTION = (5.0f * 9.81f) / (1.5f * 5.0f * 9.81f);

    fields::GravityField gf{Vec3<SceneFrame>{0, 0, -9.81f}};
    World w;
    w.register_field(gf);
    w.clock().set_dt(DT);
    auto& scene = w.create_scene();

    Ex3Craft craft;
    scene.add_craft(craft);

    // Outer-loop rate PIDs (rate-error → desired gimbal angle).
    PID<float> pitch_pid(0.20f, 0.10f, 0.02f, /*ilim=*/0.5f);
    PID<float> yaw_pid  (0.20f, 0.10f, 0.02f, 0.5f);

    zenoh::Config cfg = zenoh::Config::create_default();
    auto session = zenoh::Session::open(std::move(cfg));

    std::mutex cmd_mtx;
    float thr_sp = 0, pitch_sp = 0, yaw_sp = 0;

    auto cmd_sub = session.declare_subscriber(
        zenoh::KeyExpr("manta/ex3/cmd"),
        [&](const zenoh::Sample& s) {
            std::vector<float> v;
            std::string payload(s.get_payload().as_string());
            if (parse_float_array(payload, v, 3)) {
                std::lock_guard<std::mutex> lk(cmd_mtx);
                thr_sp = v[0]; pitch_sp = v[1]; yaw_sp = v[2];
            } else {
                std::fprintf(stderr, "ex3: bad cmd: %s\n", payload.c_str());
            }
        },
        zenoh::closures::none);

    auto state_pub = session.declare_publisher(zenoh::KeyExpr("manta/ex3/state"));

    std::printf("ex3: ready. Hover throttle ≈ %.2f. Publish [thr, pitch_rate, yaw_rate] on manta/ex3/cmd.\n",
                HOVER_FRACTION);

    RealTimePacer pacer(DT);
    int pub_decim = 0;
    const int pub_every = 20;

    while (g_run.load()) {
        float thr, ps, ys;
        {
            std::lock_guard<std::mutex> lk(cmd_mtx);
            thr = thr_sp; ps = pitch_sp; ys = yaw_sp;
        }

        // Outer loop: rate error → desired gimbal-angle setpoint, clamped.
        // Yaw rate is about body x; pitch rate about body y.
        auto gyro = craft.imu().last_gyro();
        float pitch_rate_err = ps - gyro.y();
        float yaw_rate_err   = ys - gyro.x();
        float pitch_angle_sp = clamp(pitch_pid.update(pitch_rate_err, DT), MAX_GIMBAL);
        float yaw_angle_sp   = clamp(yaw_pid  .update(yaw_rate_err,   DT), MAX_GIMBAL);

        // Engine sits 1 m below CoM. With the old sign convention, +pitch
        // gimbal pushed +x at z<0 → -y body torque. To increase +y body rate
        // (positive PID output), we want NEGATIVE pitch gimbal — same as the
        // old `set_gimbal(-u_pitch, +u_yaw)` mapping.
        craft.pitch_motor().set_torque(gimbal_pd(craft.pitch_motor(), -pitch_angle_sp));
        craft.yaw_motor()  .set_torque(gimbal_pd(craft.yaw_motor(),   +yaw_angle_sp));
        craft.engine().set_throttle(thr);

        w.update();

        if (++pub_decim >= pub_every) {
            pub_decim = 0;
            Ex3CraftTelemetry telem;
            capture_ex3_telemetry(craft, w.clock().time(), telem);
            state_pub.put(zenoh::Bytes(telem.to_json()));
        }

        pacer.wait_for_next_tick();
    }

    std::printf("ex3: shutting down.\n");
    return 0;
}

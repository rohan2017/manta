// Example 2 — Quadcopter with rate-control PIDs.
//
// Library workflow: the Craft type (Ex2Craft) is codegen-emitted from
// `craft.py`. This file is the user-written main: it does the
// world/scene/Zenoh wiring, runs the rate PIDs, and applies X-config mixing
// to the four PropThrusters that the codegen instantiated.
//
// Zenoh:
//   subscribe 'manta/ex2/cmd'   = [thr, roll_rate, pitch_rate, yaw_rate]
//   publish   'manta/ex2/state' = standard nested telemetry JSON

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

#include "ex2_craft.hpp"
#include "ex2_telemetry.hpp"

#include "sim_loop.hpp"
#include "state_codec.hpp"

using namespace manta;
using namespace manta::geom;
using namespace manta::control;
using namespace manta::examples;

namespace {
std::atomic<bool> g_run{true};
void on_signal(int) { g_run.store(false); }
}

int main() {
    std::signal(SIGINT,  on_signal);
    std::signal(SIGTERM, on_signal);

    constexpr float DT = 0.001f;  // 1 kHz sim

    // World, scene, gravity. The Craft type comes from the codegen.
    fields::GravityField gf;
    World w;
    w.register_field(gf);
    w.clock().set_dt(DT);
    auto& scene = w.create_scene();

    Ex2Craft craft;
    scene.add_craft(craft);

    // Rate controllers — gains live in user code (not in the descriptor).
    PID<float> roll_pid (0.10f, 0.05f, 0.005f, /*ilim=*/1.0f);
    PID<float> pitch_pid(0.10f, 0.05f, 0.005f, 1.0f);
    PID<float> yaw_pid  (0.20f, 0.10f, 0.000f, 1.0f);

    // ---- zenoh ----
    zenoh::Config cfg = zenoh::Config::create_default();
    auto session = zenoh::Session::open(std::move(cfg));

    std::mutex cmd_mtx;
    float thr_sp = 0, roll_sp = 0, pitch_sp = 0, yaw_sp = 0;

    auto cmd_sub = session.declare_subscriber(
        zenoh::KeyExpr("manta/ex2/cmd"),
        [&](const zenoh::Sample& s) {
            std::vector<float> v;
            std::string payload(s.get_payload().as_string());
            if (parse_float_array(payload, v, 4)) {
                std::lock_guard<std::mutex> lk(cmd_mtx);
                thr_sp = v[0]; roll_sp = v[1]; pitch_sp = v[2]; yaw_sp = v[3];
            } else {
                std::fprintf(stderr, "ex2: bad cmd: %s\n", payload.c_str());
            }
        },
        zenoh::closures::none);

    auto state_pub = session.declare_publisher(zenoh::KeyExpr("manta/ex2/state"));

    std::printf("ex2: ready. Publish [thr, roll_rate, pitch_rate, yaw_rate] on manta/ex2/cmd.\n");

    RealTimePacer pacer(DT);
    int pub_decim = 0;
    const int pub_every = 20;  // ~50 Hz

    while (g_run.load()) {
        float thr, rs, ps, ys;
        {
            std::lock_guard<std::mutex> lk(cmd_mtx);
            thr = thr_sp; rs = roll_sp; ps = pitch_sp; ys = yaw_sp;
        }

        // gyro in part/body frame (IMU sits at root → body == part).
        auto gyro = craft.imu().last_gyro();
        float roll_err  = rs - gyro.x();
        float pitch_err = ps - gyro.y();
        float yaw_err   = ys - gyro.z();

        float u_roll  = roll_pid .update(roll_err , DT);
        float u_pitch = pitch_pid.update(pitch_err, DT);
        float u_yaw   = yaw_pid  .update(yaw_err  , DT);

        // X-config mixing.
        // CCW props: fr, bl. CW props: fl, br. PropThruster CCW yields -z body torque.
        // For +yaw command we want +z body torque → boost CW (fl, br), dim CCW (fr, bl).
        craft.fr().set_throttle(thr - u_roll + u_pitch - u_yaw);
        craft.fl().set_throttle(thr + u_roll + u_pitch + u_yaw);
        craft.bl().set_throttle(thr + u_roll - u_pitch - u_yaw);
        craft.br().set_throttle(thr - u_roll - u_pitch + u_yaw);

        w.update();

        if (++pub_decim >= pub_every) {
            pub_decim = 0;
            Ex2CraftTelemetry telem;
            capture_ex2_telemetry(craft, w.clock().time(), telem);
            state_pub.put(zenoh::Bytes(telem.to_json()));
        }

        pacer.wait_for_next_tick();
    }

    std::printf("ex2: shutting down.\n");
    return 0;
}

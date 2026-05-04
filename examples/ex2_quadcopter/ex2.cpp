// Example 2 — Quadcopter with rate-control PIDs.
//
// Library-workflow example under the harness architecture: the codegen
// emits `manta_gen::ex2::{w, scene, craft, setup, tick, shutdown}` (in
// generated/ex2/ex2.hpp). This file is a thin user-written main: it calls
// the harness's setup, runs rate PIDs + X-config mixing on top of the
// craft, ticks the harness, and adds its own Zenoh cmd subscriber +
// state publisher (the cmd payload is a 4-float controller struct, not a
// part-defined signal — out of scope for the binding system).
//
// Zenoh:
//   subscribe 'manta/ex2/cmd'   = [thr, roll_rate, pitch_rate, yaw_rate]
//   publish   'manta/ex2/state' = standard nested telemetry JSON
//
// To regenerate the harness:
//   PYTHONPATH=python/manta_codegen/src \
//       python -m manta_codegen.cli examples/ex2_quadcopter/craft.py \
//           --workflow library

#include <atomic>
#include <csignal>
#include <cstdio>
#include <mutex>
#include <string>
#include <vector>

#include <zenoh.hxx>

#include "manta/control/pid.hpp"

#include "ex2.hpp"             // manta_gen::ex2 harness
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

    // Build world + scene + Ex2Craft via the harness.
    manta_gen::ex2::setup();
    auto& craft = manta_gen::ex2::craft;
    auto& world = manta_gen::ex2::w;

    constexpr float DT = manta_gen::ex2::DT;

    // Rate controllers — gains live in user code (not in the descriptor).
    PID<float> roll_pid (0.10f, 0.05f, 0.005f, /*ilim=*/1.0f);
    PID<float> pitch_pid(0.10f, 0.05f, 0.005f, 1.0f);
    PID<float> yaw_pid  (0.20f, 0.10f, 0.000f, 1.0f);

    // ---- zenoh ----
    // Separate session from the harness's internal one — the cmd payload
    // is a user-defined 4-float struct, so the binding system can't
    // express it. Two sessions in one process work fine.
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
        // CCW props: fr, bl. CW props: fl, br. Thruster1's CCW yields -z body torque.
        // For +yaw command we want +z body torque → boost CW (fl, br), dim CCW (fr, bl).
        craft.fr().set_throttle(thr - u_roll + u_pitch - u_yaw);
        craft.fl().set_throttle(thr + u_roll + u_pitch + u_yaw);
        craft.bl().set_throttle(thr + u_roll - u_pitch - u_yaw);
        craft.br().set_throttle(thr - u_roll - u_pitch + u_yaw);

        manta_gen::ex2::tick();

        if (++pub_decim >= pub_every) {
            pub_decim = 0;
            Ex2CraftTelemetry telem;
            capture_ex2_telemetry(craft, world.clock().time(), telem);
            state_pub.put(zenoh::Bytes(telem.to_json()));
        }

        pacer.wait_for_next_tick();
    }

    std::printf("ex2: shutting down.\n");
    manta_gen::ex2::shutdown();
    return 0;
}

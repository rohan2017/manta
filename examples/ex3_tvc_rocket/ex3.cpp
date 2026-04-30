// Example 3 — TVC rocket "hopper".
//
// Library workflow: Ex3Craft is codegen-emitted from craft.py. This file is
// the user-written main: world/scene/Zenoh wiring, rate PIDs, gimbal mapping.
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

#include "ex3.hpp"
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
}

int main() {
    std::signal(SIGINT,  on_signal);
    std::signal(SIGTERM, on_signal);

    constexpr float DT = 0.001f;  // 1 kHz
    constexpr float HOVER_FRACTION = (5.0f * 9.81f) / (1.5f * 5.0f * 9.81f);  // = 0.667

    fields::GravityField gf;
    World w;
    w.register_field(gf);
    w.clock().set_dt(DT);
    auto& scene = w.create_scene();

    Ex3Craft craft;
    scene.add_craft(craft);

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

        auto gyro = craft.imu().last_gyro();
        float pitch_err = ps - gyro.y();
        float yaw_err   = ys - gyro.x();

        float u_pitch = pitch_pid.update(pitch_err, DT);
        float u_yaw   = yaw_pid  .update(yaw_err,   DT);

        // Engine is below CoM: +pitch gimbal → +x thrust at z<0 → -y body torque.
        // To increase +y body rate (positive u_pitch), need NEGATIVE gimbal pitch.
        craft.engine().set_gimbal(/*pitch=*/-u_pitch, /*yaw=*/+u_yaw);
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

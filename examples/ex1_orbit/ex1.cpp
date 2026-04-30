// Example 1 — Orbit.
//
// Same 6-thruster craft as ex0, this time orbiting Earth at 1 km altitude
// (modeled as inverse-square central gravity, no atmosphere or rotation).
// Initial state: tangential velocity = sqrt(mu/r) for a circular orbit.
//
// Zenoh:
//   subscribe 'manta/ex1/cmd'   = [tx+, tx-, ty+, ty-, tz+, tz-]
//   publish   'manta/ex1/state' = standard craft state JSON
//
// Sim runs faster than realtime to make orbital mechanics observable in a
// reasonable wall-clock time. SIM_RATE_MULT scales how many sim seconds we
// advance per wall second.

#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <mutex>
#include <string>
#include <vector>

#include <zenoh.hxx>

#include "manta/core/craft.hpp"
#include "manta/core/scene.hpp"
#include "manta/core/world.hpp"
#include "manta/fields/point_gravity_field.hpp"
#include "manta/parts/field_src/point_gravity_part.hpp"
#include "manta/parts/structure/point_mass.hpp"
#include "manta/parts/actuator/thruster.hpp"

#include "sim_loop.hpp"
#include "state_codec.hpp"

using namespace manta;
using namespace manta::geom;
using namespace manta::parts;
using namespace manta::fields;
using namespace manta::examples;

namespace {
std::atomic<bool> g_run{true};
void on_signal(int) { g_run.store(false); }
}

int main() {
    std::signal(SIGINT,  on_signal);
    std::signal(SIGTERM, on_signal);

    // ---- physics constants ----
    constexpr float MU              = 3.986004418e14f;  // Earth GM, m^3/s^2
    constexpr float EARTH_RADIUS    = 6.371e6f;         // m
    constexpr float ALTITUDE        = 1.0e3f;           // 1 km
    constexpr float ORBIT_R         = EARTH_RADIUS + ALTITUDE;
    const     float V_CIRC          = std::sqrt(MU / ORBIT_R);  // ~7910 m/s

    // ---- sim parameters ----
    constexpr float SIM_DT          = 0.005f;  // 5 ms (200 Hz)
    constexpr int   SIM_RATE_MULT   = 200;     // 200 sim-s per wall-s (full orbit ≈ 28 wall-s)
    const float     WALL_PERIOD     = SIM_DT / float(SIM_RATE_MULT);

    // ---- world ----
    PointGravityField pgf{MU};
    World w;
    w.register_field(pgf);
    w.clock().set_dt(SIM_DT);
    auto& scene = w.create_scene();

    Craft c("orbiter");
    c.root().add<PointMass>("body", 1.0f);

    auto& tx_p = c.root().add<Thruster>("tx_p", 5.0f, Vec3<PartFrame>{ 1, 0, 0});
    auto& tx_n = c.root().add<Thruster>("tx_n", 5.0f, Vec3<PartFrame>{-1, 0, 0});
    auto& ty_p = c.root().add<Thruster>("ty_p", 5.0f, Vec3<PartFrame>{ 0, 1, 0});
    auto& ty_n = c.root().add<Thruster>("ty_n", 5.0f, Vec3<PartFrame>{ 0,-1, 0});
    auto& tz_p = c.root().add<Thruster>("tz_p", 5.0f, Vec3<PartFrame>{ 0, 0, 1});
    auto& tz_n = c.root().add<Thruster>("tz_n", 5.0f, Vec3<PartFrame>{ 0, 0,-1});
    Thruster* thrusters[6] = {&tx_p, &tx_n, &ty_p, &ty_n, &tz_p, &tz_n};

    c.root().add<PointGravityPart>("grav");
    c.root().compute_params();

    // Initial state: at (R, 0, 0), velocity (0, V_circ, 0) → CCW in xy plane.
    c.set_position   (Vec3<SceneFrame>{ORBIT_R, 0, 0});
    c.set_vel_linear (Vec3<SceneFrame>{0, V_CIRC, 0});
    scene.add_craft(c);

    // ---- zenoh ----
    zenoh::Config cfg = zenoh::Config::create_default();
    auto session = zenoh::Session::open(std::move(cfg));

    std::mutex cmd_mtx;
    float cmd[6] = {0,0,0,0,0,0};

    auto cmd_sub = session.declare_subscriber(
        zenoh::KeyExpr("manta/ex1/cmd"),
        [&](const zenoh::Sample& s) {
            std::vector<float> v;
            std::string payload(s.get_payload().as_string());
            if (parse_float_array(payload, v, 6)) {
                std::lock_guard<std::mutex> lk(cmd_mtx);
                for (int i = 0; i < 6; ++i) cmd[i] = v[i];
            } else {
                std::fprintf(stderr, "ex1: bad cmd: %s\n", payload.c_str());
            }
        },
        zenoh::closures::none);

    auto state_pub = session.declare_publisher(zenoh::KeyExpr("manta/ex1/state"));

    std::printf("ex1: orbit at %.1f km altitude, v_circ = %.1f m/s. Sim runs %dx realtime.\n",
                ALTITUDE / 1000.0f, V_CIRC, SIM_RATE_MULT);

    // ---- sim loop ----
    RealTimePacer pacer(WALL_PERIOD);
    int  pub_decim = 0;
    const int pub_every = 10;  // ~20 Hz wall-time publish

    while (g_run.load()) {
        float local[6];
        {
            std::lock_guard<std::mutex> lk(cmd_mtx);
            std::copy(std::begin(cmd), std::end(cmd), std::begin(local));
        }
        for (int i = 0; i < 6; ++i) thrusters[i]->set_throttle(local[i]);

        w.update();

        if (++pub_decim >= pub_every) {
            pub_decim = 0;
            std::string s = encode_craft_state(w.clock().time(), c);
            state_pub.put(zenoh::Bytes(s));
        }

        pacer.wait_for_next_tick();
    }

    std::printf("ex1: shutting down.\n");
    return 0;
}

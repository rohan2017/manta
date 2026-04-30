// ex8 — multi-craft swarm demo.
//
// Three identical Ex8Craft drones in one Scene, connected in a chain by
// two Tethers:
//
//     drone0 ── tether01 ── drone1 ── tether12 ── drone2
//
// drone0 is the "leader" — its thruster receives commands on
// `manta/ex8/leader/cmd`. The tethers (springs with mild damping) drag
// drones 1 and 2 along whenever the chain stretches past rest length.
//
// Each drone publishes its own state on `manta/ex8/<idx>/state`, where
// idx ∈ {0, 1, 2}. This exercises Scene's three-phase update across
// multiple sibling crafts: each tether-endpoint Part reads its sibling's
// scene_to_part cache to compute the spring force, which is only safe
// because Scene runs the kinematic pass for ALL crafts before any of
// them runs sense_and_aggregate.
//
// Also serves as a stand-in demo for future Goal 4 work: scale this to
// many crafts and the same code, with a Zenoh-backed Field, becomes a
// distributed swarm.

#include <atomic>
#include <csignal>
#include <cstdio>
#include <mutex>
#include <string>
#include <vector>

#include <zenoh.hxx>

#include "manta/core/scene.hpp"
#include "manta/core/world.hpp"
#include "manta/coupling/tether.hpp"
#include "manta/parts/coupling/tether_endpoint.hpp"

#include "ex8.hpp"
#include "sim_loop.hpp"
#include "state_codec.hpp"

namespace {
std::atomic<bool> g_run{true};
void on_signal(int) { g_run.store(false); }
}

int main() {
    std::signal(SIGINT,  on_signal);
    std::signal(SIGTERM, on_signal);

    constexpr double DT     = 0.001;
    constexpr int    PUB_DECIM = 20;
    constexpr float  REST_LENGTH = 1.0f;   // m
    constexpr float  STIFFNESS   = 50.0f;  // N/m
    constexpr float  DAMPING     = 5.0f;   // N·s/m

    manta::World w;
    w.clock().set_dt(static_cast<float>(DT));
    auto& scene = w.create_scene();

    // Three drones. Codegen-emitted craft type — same shape for each
    // instance; tether endpoints are added per-instance below.
    Ex8Craft drone0, drone1, drone2;
    Ex8Craft* drones[3] = {&drone0, &drone1, &drone2};

    // Two tethers connecting them in a chain.
    manta::coupling::Tether tether01(REST_LENGTH, STIFFNESS, DAMPING);
    manta::coupling::Tether tether12(REST_LENGTH, STIFFNESS, DAMPING);

    // Wire the endpoints. Each Tether has 2 endpoints living on different
    // crafts. The boolean (is_a) just disambiguates which slot the endpoint
    // registers in; the dynamics are symmetric.
    drone0.root().add<manta::parts::TetherEndpoint>("hook_a", tether01, /*is_a=*/true);
    drone1.root().add<manta::parts::TetherEndpoint>("hook_a", tether01, /*is_a=*/false);
    drone1.root().add<manta::parts::TetherEndpoint>("hook_b", tether12, /*is_a=*/true);
    drone2.root().add<manta::parts::TetherEndpoint>("hook_b", tether12, /*is_a=*/false);

    // compute_params re-derives mass/MOI/COM after the new endpoints landed.
    for (auto* d : drones) d->root().compute_params();

    // Initial positions: drones strung along x at rest length.
    manta::InitialState init0; init0.position = {0.0f, 0.0f, 0.0f};
    manta::InitialState init1; init1.position = {1.0f, 0.0f, 0.0f};
    manta::InitialState init2; init2.position = {2.0f, 0.0f, 0.0f};
    scene.add_craft(drone0, init0);
    scene.add_craft(drone1, init1);
    scene.add_craft(drone2, init2);

    // Zenoh: subscribe leader command, publish all 3 states.
    zenoh::Config cfg = zenoh::Config::create_default();
    auto session = zenoh::Session::open(std::move(cfg));

    std::mutex cmd_mtx;
    std::vector<float> leader_cmd;
    auto cmd_sub = session.declare_subscriber(
        zenoh::KeyExpr("manta/ex8/leader/cmd"),
        [&](const zenoh::Sample& s) {
            std::vector<float> v;
            std::string payload(s.get_payload().as_string());
            if (manta::examples::parse_float_array(payload, v, /*expected=*/1)) {
                std::lock_guard<std::mutex> lk(cmd_mtx);
                leader_cmd = std::move(v);
            }
        }, zenoh::closures::none);

    auto pub0 = session.declare_publisher(zenoh::KeyExpr("manta/ex8/0/state"));
    auto pub1 = session.declare_publisher(zenoh::KeyExpr("manta/ex8/1/state"));
    auto pub2 = session.declare_publisher(zenoh::KeyExpr("manta/ex8/2/state"));

    std::printf("ex8: 3-drone swarm ready. Leader cmd on 'manta/ex8/leader/cmd'. "
                "States on 'manta/ex8/{0,1,2}/state'.\n");

    manta::examples::RealTimePacer pacer(DT);
    int pub_decim = 0;

    while (g_run.load()) {
        {
            std::lock_guard<std::mutex> lk(cmd_mtx);
            if (!leader_cmd.empty()) drone0.forward().set_throttle(leader_cmd[0]);
        }

        w.update();

        if (++pub_decim >= PUB_DECIM) {
            pub_decim = 0;
            double t = double(w.clock().time());
            pub0.put(zenoh::Bytes(manta::examples::encode_craft_state(t, drone0)));
            pub1.put(zenoh::Bytes(manta::examples::encode_craft_state(t, drone1)));
            pub2.put(zenoh::Bytes(manta::examples::encode_craft_state(t, drone2)));
        }

        pacer.wait_for_next_tick();
    }

    std::printf("ex8: shutting down.\n");
    return 0;
}

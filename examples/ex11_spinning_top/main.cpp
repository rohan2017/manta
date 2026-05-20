// Example 11 — Spinning-top precession on a flat ground plane.
//
// Library-workflow example. The codegen emits the Ex11Craft type, a
// telemetry struct, and a harness `manta_gen::ex11::{w, scene, craft,
// setup, tick, shutdown}` (in generated/ex11/ex11.hpp). This file is
// the thin user-written main: it pre-spins the flywheel, sequences the
// brief lateral kick at t=3 s, and publishes JSON telemetry on the
// Zenoh topic `manta/ex11/state` for the Rerun viewer.
//
// To regenerate the harness:
//   .venv/bin/python -m manta_codegen.cli \
//       examples/ex11_spinning_top/config.py --workflow library

#include <atomic>
#include <csignal>
#include <cstdio>

#include <zenoh.hxx>

#include "ex11.hpp"            // manta_gen::ex11 harness
#include "ex11_telemetry.hpp"  // Ex11CraftTelemetry + capture helper

#include "sim_loop.hpp"

using namespace manta;
using namespace manta::examples;

namespace {

constexpr float KICK_START_T  = 3.0f;    // s — fire the lateral thruster
constexpr float KICK_DURATION = 0.05f;   // s — 50 ms impulse
// Flywheel RPM trades off precession period vs. tilt-amplitude
// dampening. At θ̇ = 350 rad/s:
//   h_rotor       = I_axial · θ̇         = 0.0225 · 350  =  7.9 kg·m²/s
//   precession Ω  = M·g·h / h_rotor     ≈ 12.4 / 7.9   ≈ 1.6 rad/s
//   precession T  = 2π/Ω                ≈ 4.0 s        → visible cycles
// Combined with the 4 N kick (config.py), the tilt sweeps a circle at
// ≈ 15° opening half-angle before contact friction slowly tips it.
constexpr float FLY_RATE_0    = 350.0f;  // rad/s — initial flywheel spin

std::atomic<bool> g_run{true};
void on_signal(int) { g_run.store(false); }

}  // namespace

int main() {
    std::signal(SIGINT,  on_signal);
    std::signal(SIGTERM, on_signal);

    // Build world + scene + Ex11Craft via the harness.
    manta_gen::ex11::setup();
    auto& craft = manta_gen::ex11::craft;
    auto& world = manta_gen::ex11::w;

    constexpr float DT = manta_gen::ex11::DT;

    // Pre-spin the flywheel before the first tick. The motor is in
    // passive mode (no actuator drive), so the flywheel coasts at this
    // rate forever — modulo gyroscopic back-reaction once the body
    // starts moving.
    craft.fly_motor().set_passive();
    craft.fly_motor().set_rate(FLY_RATE_0);

    // ---- Zenoh: state publisher on manta/ex11/state ----
    // The harness also opens its own session for binding-managed topics
    // (none here — we only publish the bundled telemetry JSON below).
    // Two sessions in one process are fine.
    zenoh::Config cfg = zenoh::Config::create_default();
    auto session = zenoh::Session::open(std::move(cfg));
    auto state_pub = session.declare_publisher(
        zenoh::KeyExpr("manta/ex11/state"));

    std::printf("ex11: ready. Pre-spin %.0f rad/s, kick at t=%.1f s. "
                "Ctrl-C to stop.\n",
                double(FLY_RATE_0), double(KICK_START_T));

    RealTimePacer pacer(DT);
    int   pub_decim = 0;
    const int pub_every = 20;     // ~50 Hz state publish

    float t = 0.0f;
    while (g_run.load()) {
        // Brief lateral kick: 50 ms at full throttle at t = KICK_START_T.
        // Force = 10 N along body +x at the top of the stick → torque
        // ≈ 10 N × 0.8 m × 0.05 s = 0.4 N·m·s impulse about the bottom
        // contact. Small enough to tip the craft a few degrees; the
        // resulting gravity-driven torque-on-tilted-top then precesses
        // around the vertical instead of falling straight in +x.
        const bool kicking = (t >= KICK_START_T) &&
                             (t < KICK_START_T + KICK_DURATION);
        craft.kick().set_throttle(kicking ? 1.0f : 0.0f);

        manta_gen::ex11::tick();
        t += DT;

        if (++pub_decim >= pub_every) {
            pub_decim = 0;
            Ex11CraftTelemetry telem;
            capture_ex11_telemetry(craft, world.clock().time(), telem);
            state_pub.put(zenoh::Bytes(telem.to_json()));
        }

        pacer.wait_for_next_tick();
    }

    std::printf("ex11: shutting down.\n");
    manta_gen::ex11::shutdown();
    return 0;
}

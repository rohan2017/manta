// Example 0 — drag-comparison freefall.
//
// Pure C++, no codegen, no Zenoh, no Python. The minimal "manta in a
// terminal" demo: build, run, watch text output, see drag at work.
//
// World: uniform 1g downward, uniform air at sea level (R=287, T=288.15 K,
// p=101325 Pa). Two crafts share the world: both 1 kg, both have a
// straight-up Thruster, but one carries a `Surface1` part with a diagonal
// linear-drag tensor.
//
// Run: thruster pulse for `BURN` seconds (the user can override at the
// prompt), then both crafts coast under gravity (and drag, on the drag
// craft). The sim records each craft's peak altitude and the time it
// returns to z=0. Stops when both have come back down (or 60 s safety
// timeout).
//
// Drag: Surface2 with a diagonal positive tensor. The k-th order term
// scales as sign(v) · |v|^k, so a positive A_2 always points along
// v_rel — i.e. drag opposes motion regardless of which axis or which
// direction the body is moving. F_drag = A_1·v + A_2·sign(v)·v² in
// each axis.

#include <array>
#include <cstdio>
#include <iostream>
#include <string>

#include "manta/core/craft.hpp"
#include "manta/core/scene.hpp"
#include "manta/core/world.hpp"
#include "manta/fields/fluid_field.hpp"
#include "manta/fields/gravity_field.hpp"
#include "manta/parts/actuator/thruster.hpp"
#include "manta/parts/structure/mass.hpp"
#include "manta/parts/structure/surface.hpp"

using namespace manta;
using namespace manta::geom;
using namespace manta::parts;

namespace {

constexpr float DT     = 0.001f;        // 1 kHz sim
constexpr float MASS   = 1.0f;          // kg
constexpr float THRUST = 20.0f;         // N — about 2g, plenty of altitude
constexpr float DRAG_K1 = 0.10f;        // N·s/m   — linear drag coef
constexpr float DRAG_K2 = 0.05f;        // N·s²/m² — quadratic drag coef

float prompt_burn_seconds(float fallback) {
    std::printf("Thrust burn duration in seconds [default %.1f]: ", fallback);
    std::fflush(stdout);
    std::string line;
    if (!std::getline(std::cin, line) || line.empty()) return fallback;
    try { return std::stof(line); }
    catch (...) {
        std::printf("(couldn't parse, using %.1f)\n", fallback);
        return fallback;
    }
}

Mat3<PartFrame> diag(float v) {
    Mat3<PartFrame> m;
    m.raw().setZero();
    m.raw()(0, 0) = v;
    m.raw()(1, 1) = v;
    m.raw()(2, 2) = v;
    return m;
}

}  // namespace

int main() {
    const float burn = prompt_burn_seconds(2.0f);

    World w;
    w.clock().set_dt(DT);

    fields::GravityField gravity{Vec3<SceneFrame>{0, 0, -9.81f}};
    w.register_field(gravity);

    fields::FluidField air;
    air.add(fields::FluidField::Disturbance::uniform_gas(
                /*R=*/287.0f, /*T=*/288.15f, /*p=*/101325.0f),
            fields::PERSISTENT);
    w.register_field(air);

    auto& scene = w.create_scene();

    // Craft A — clean (no drag).
    Craft clean("clean");
    clean.root().add<Mass>("body", MASS);
    auto& thr_clean = clean.root().add<Thruster>("up", THRUST);   // +z by default
    clean.root().compute_params();
    scene.add_craft(clean);

    // Craft B — same as A plus a Surface2 part with diagonal linear +
    // quadratic drag tensors. Both A_1 and A_2 are positive, so each
    // tensor produces force along v_rel = v_fluid − v_self (i.e.
    // opposing the craft's motion through still air).
    Craft drag("drag");
    drag.root().add<Mass>("body", MASS);
    auto& thr_drag = drag.root().add<Thruster>("up", THRUST);
    drag.root().add<Surface2>(
        "drag_plate",
        std::array<Mat3<PartFrame>, 2>{diag(DRAG_K1), diag(DRAG_K2)},
        std::array<Mat3<PartFrame>, 2>{diag(0.0f),    diag(0.0f)});
    drag.root().compute_params();
    scene.add_craft(drag);

    thr_clean.set_throttle(1.0f);
    thr_drag .set_throttle(1.0f);

    float t = 0.0f;
    float peak_clean = 0.0f, peak_drag = 0.0f;
    float t_return_clean = -1.0f, t_return_drag = -1.0f;
    bool done_clean = false, done_drag = false;

    constexpr float SAFETY_TIMEOUT = 60.0f;

    while (!(done_clean && done_drag) && t < SAFETY_TIMEOUT) {
        if (t >= burn) {
            thr_clean.set_throttle(0.0f);
            thr_drag .set_throttle(0.0f);
        }
        w.step();
        t += DT;

        const float z_clean = clean.scene_to_craft().position().z();
        const float z_drag  = drag .scene_to_craft().position().z();
        const float vz_clean = clean.scene_to_craft().vel_linear().z();
        const float vz_drag  = drag .scene_to_craft().vel_linear().z();

        if (z_clean > peak_clean) peak_clean = z_clean;
        if (z_drag  > peak_drag)  peak_drag  = z_drag;

        if (!done_clean && t > burn && z_clean <= 0.0f && vz_clean < 0.0f) {
            t_return_clean = t;
            done_clean     = true;
        }
        if (!done_drag && t > burn && z_drag <= 0.0f && vz_drag < 0.0f) {
            t_return_drag = t;
            done_drag     = true;
        }
    }

    std::printf("\n");
    std::printf("burn duration : %.2f s @ %.1f N (each craft, 1 kg)\n", burn, THRUST);
    std::printf("clean craft   : peak z = %7.2f m,  t_return = %5.2f s\n",
                peak_clean, t_return_clean);
    std::printf("drag craft    : peak z = %7.2f m,  t_return = %5.2f s   "
                "(k1 = %.2f N·s/m, k2 = %.2f N·s²/m²)\n",
                peak_drag, t_return_drag, DRAG_K1, DRAG_K2);
    std::printf("drag/clean    : peak ratio = %.1f%%\n",
                (peak_clean > 0.0f) ? 100.0f * peak_drag / peak_clean : 0.0f);
    return 0;
}

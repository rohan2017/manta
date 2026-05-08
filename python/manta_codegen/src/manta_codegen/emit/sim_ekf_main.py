"""Emit <name>_main.cpp for a Target containing one sim World + one EKF.

This is the in-process two-world shape (Pattern A from the original
estimation_workflow.md). Tick layout:

    1. Apply input bindings  (Zenoh sub → sim parts | est sensors)
    2. sim_w.step()          (run sim physics)
    3. Cross-world connect() (sim sensor outputs → est set_measurement,
                              sim throttle → est throttle for predict)
    4. ekf.predict(dt, Q)
    5. Per-sensor consume_fresh + update_n<N>
    6. Publish out bindings  (decimated)

The sim world keeps its full setup (planets, fields, scene, crafts,
tethers, field-sync). The EKF holds its own internal craft-pair via
manta::estimation::EKF, with `register_field` propagating any field
instances the est-side parts query during predict.

Naming convention to keep the two worlds non-colliding:
  * Sim manta::World variable     → `sim_w`
  * Sim scene variable             → `sim_scene`
  * Sim craft (single-craft only)  → `sim_<craft_name>`
  * EKF variable                   → `ekf_<id>` (from EKF.cpp_var_name())

This covers ex5. Generalizing to multiple sim worlds or multiple EKFs is
deferred.
"""

from __future__ import annotations

from .._format import cpp_float as _f
from ..signal import is_craft_signal
from ._util import GENERATED_BANNER


def emit_composed_hpp(target, sim_world, filter_obj) -> str:
    """Emit `<target>.hpp` for a sim+filter Target."""
    sim_name    = sim_world.name
    filter_name = filter_obj.world.name
    name        = target.name

    return "\n".join([
        GENERATED_BANNER, "#pragma once", "",
        '#include "manta/core/harness.hpp"',
        f'#include "{sim_name}.hpp"',
        f'#include "{filter_name}.hpp"',
        "",
        f"namespace manta_gen::{name} {{",
        "",
        f"// Sim-tick parameters, frozen at codegen time. Drives the pacer",
        f"// in <target>_main.cpp; both sub-harnesses read DT through their",
        f"// own namespaces.",
        f"inline constexpr float DT             = {_f(target.dt)};",
        f"inline constexpr float SIM_RATE_MULT  = {_f(target.sim_rate_mult)};",
        "",
        f"// Composed setup: calls manta_gen::{sim_name}::setup() then",
        f"// manta_gen::{filter_name}::setup().",
        "void setup();",
        "",
        f"// Composed tick: runs the sim world, then the cross-world",
        f"// connect() steps, then the filter. Sub-harness ticks handle",
        f"// their own bindings and intra-world connects.",
        "void tick();",
        "",
        f"// Composed shutdown: tears down the filter then the sim, in",
        f"// reverse-init order, before main() returns.",
        "void shutdown();",
        "",
        "// Polymorphic adapter — see manta/core/harness.hpp.",
        "struct Harness : public manta::Harness {",
        "    void setup()    override;",
        "    void tick()     override;",
        "    void shutdown() override;",
        "};",
        "extern Harness harness;",
        "",
        f"}}  // namespace manta_gen::{name}",
        "",
    ])


def emit_composed_cpp(target, sim_world, filter_obj, kind: str = "ekf") -> str:
    """Emit `<target>.cpp` — composed setup/tick/shutdown that delegate to
    each sub-harness in order. The EKF/UKF's own setup() owns all
    cross-world wiring (measurements via `ekf.measure(...)` + actuator
    mirroring inside the filter tick), so the orchestrator does nothing
    between sim_tick() and filter_tick()."""
    sim_name    = sim_world.name
    filter_name = filter_obj.world.name
    name        = target.name
    sim_ns      = f"manta_gen::{sim_name}"
    filter_ns   = f"manta_gen::{filter_name}"

    lines: list[str] = [
        GENERATED_BANNER, "",
        f'#include "{name}.hpp"',
        "",
        f"namespace manta_gen::{name} {{",
        "",
        "void setup() {",
        f"    {sim_ns}::setup();",
        f"    {filter_ns}::setup();",
        "}",
        "",
        "void tick() {",
        f"    {sim_ns}::tick();",
        f"    {filter_ns}::tick();",
        "}",
        "",
        "void shutdown() {",
        f"    {filter_ns}::shutdown();",
        f"    {sim_ns}::shutdown();",
        "}",
        "",
        "// ---- Polymorphic Harness adapter ----",
        "void Harness::setup()    { ::manta_gen::" + name + "::setup();    }",
        "void Harness::tick()     { ::manta_gen::" + name + "::tick();     }",
        "void Harness::shutdown() { ::manta_gen::" + name + "::shutdown(); }",
        "Harness harness;",
        "",
        f"}}  // namespace manta_gen::{name}",
        "",
    ]
    return "\n".join(lines)


def emit_composed_main_cpp(target) -> str:
    """Emit `<target>_main.cpp` — thin pacer + signal handler."""
    name = target.name
    return "\n".join([
        GENERATED_BANNER, "",
        "#include <atomic>",
        "#include <chrono>",
        "#include <csignal>",
        "#include <cstdint>",
        "#include <cstdio>",
        "#include <thread>",
        "",
        f'#include "{name}.hpp"',
        "",
        "namespace {",
        "std::atomic<bool> g_run{true};",
        "void on_signal(int) { g_run.store(false); }",
        "}",
        "",
        "int main() {",
        "    std::signal(SIGINT,  on_signal);",
        "    std::signal(SIGTERM, on_signal);",
        "",
        f"    manta_gen::{name}::setup();",
        f'    std::printf("{name}: ready (sim+filter).\\n");',
        "",
        f"    constexpr float WALL_PERIOD = manta_gen::{name}::DT "
        f"/ manta_gen::{name}::SIM_RATE_MULT;",
        "    auto next = std::chrono::steady_clock::now();",
        "    const auto period = std::chrono::microseconds(int64_t(WALL_PERIOD * 1e6));",
        "",
        "    while (g_run.load()) {",
        f"        manta_gen::{name}::tick();",
        "        next += period;",
        "        std::this_thread::sleep_until(next);",
        "    }",
        "",
        f'    std::printf("{name}: shutting down.\\n");',
        f"    manta_gen::{name}::shutdown();",
        "    return 0;",
        "}",
        "",
    ])


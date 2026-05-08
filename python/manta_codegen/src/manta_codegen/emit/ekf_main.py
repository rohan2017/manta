"""EKF/UKF target emit — top-level entry points.

After Phase 6 deleted the legacy `manta::estimation::EKF<NumCrafts,...>`
and `UKF<NumCrafts,...>` C++ wrappers, this module is a thin dispatcher:

  * `emit_filter_hpp(target, filter, kind)` → header for the EKF/UKF
    sub-namespace. Routes to `ekf_state_spec.emit_filter_hpp` (kind="ekf")
    or `ukf_state_spec.emit_filter_hpp` (kind="ukf").

  * `emit_filter_cpp(target, filter, kind)` → implementation. Same
    dispatch.

  * `emit_filter_main_cpp(target, filter, kind)` → the boilerplate
    `int main()` that wires SIGINT/SIGTERM, calls setup(), runs the
    tick loop, calls shutdown(). Identical for both filter kinds.
"""

from __future__ import annotations

from ._util import GENERATED_BANNER


def emit_filter_hpp(target, filter_obj, kind: str = "ekf") -> str:
    if kind == "ekf":
        from . import ekf_state_spec
        return ekf_state_spec.emit_filter_hpp(target, filter_obj)
    if kind == "ukf":
        from . import ukf_state_spec
        return ukf_state_spec.emit_filter_hpp(target, filter_obj)
    raise ValueError(f"unknown filter kind {kind!r}")


def emit_filter_cpp(target, filter_obj, kind: str = "ekf") -> str:
    if kind == "ekf":
        from . import ekf_state_spec
        return ekf_state_spec.emit_filter_cpp(target, filter_obj)
    if kind == "ukf":
        from . import ukf_state_spec
        return ukf_state_spec.emit_filter_cpp(target, filter_obj)
    raise ValueError(f"unknown filter kind {kind!r}")


def emit_filter_main_cpp(target, filter_obj, kind: str = "ekf") -> str:
    name        = filter_obj.world.name
    n_crafts    = len(filter_obj.world.crafts)
    n_meas      = len(filter_obj.measurements)
    n_bindings  = len(filter_obj.world.bindings)
    kind_label  = kind.upper()

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
        f'    std::printf("{target.name}: ready ({kind_label}). {n_crafts} craft(s), '
        f'{n_bindings} binding(s), {n_meas} measurement sensor(s).\\n");',
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
        f'    std::printf("{target.name}: shutting down.\\n");',
        f"    manta_gen::{name}::shutdown();",
        "    return 0;",
        "}",
        "",
    ])

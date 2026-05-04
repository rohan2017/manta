"""Emit <name>_main.cpp for a Target containing one sim World + one EKF.

This is the in-process two-world shape (Pattern A from the original
estimation_workflow.md). Tick layout:

    1. Apply input bindings  (Zenoh sub → sim parts | est sensors)
    2. sim_w.update()        (run sim physics)
    3. Cross-world connect() (sim sensor outputs → est set_measurement,
                              sim throttle → est throttle for predict)
    4. ekf.predict(dt, Q)
    5. Per-sensor consume_fresh + update_n<N>
    6. Publish out bindings  (decimated)

The sim world keeps its full setup (planets, fields, scene, crafts,
tethers, field-sync). The EKF holds its own internal craft-pair via
WorldEKF, with `register_field` propagating any field instances the
est-side parts query during predict.

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
from .ekf_main import _double_qualify_partframe, _is_filter_signal




def _emit_composed_connect_step(
        lines: list[str], conn,
        sim_world, sim_ns: str,
        filter_obj, filter_ns: str,
        templated_filter_craft: bool,
        indent: str = "    ") -> None:
    """Cross-world connect: source in one world's namespace, sink in the
    other's. Resolves accessors with full `manta_gen::<world>::` prefixes
    so the composed harness can reach into both sub-namespaces."""
    src = conn.source
    snk = conn.sink

    sim_ids = {id(e.craft) for e in sim_world.crafts}
    filter_ids = {id(e.craft) for e in filter_obj.world.crafts}

    def _accessor(bs, scope_namespace: str, in_filter_world: bool) -> tuple[str, bool]:
        """Return (accessor_expression, is_templated_filter_craft)."""
        if _is_filter_signal(bs):
            # ekf_<id> / ukf_<id> sentinel — name reaches in via the filter ns.
            return f"{filter_ns}::{filter_obj.cpp_var_name()}", False
        if in_filter_world:
            # The filter's Real-side craft is exposed at namespace scope
            # (`manta_gen::<filter_world>::craft`). Reach in directly —
            # `<filter_var>.craft()` is the same instance but the namespace-
            # direct path keeps the templated CraftT<double>'s part accessors
            # in scope.
            if len(filter_obj.world.crafts) == 1:
                base = f"{filter_ns}::craft"
            else:
                idx = next(i for i, e in enumerate(filter_obj.world.crafts)
                           if id(e.craft) == id(bs.craft_ref))
                base = f"{filter_ns}::craft_{idx}"
            tplated = templated_filter_craft
        else:
            # Sim-world side. Single-craft worlds use `craft`; multi uses
            # craft_<i> indexed in sim_world.crafts.
            if len(sim_world.crafts) == 1:
                base = f"{sim_ns}::craft"
            else:
                idx = next(i for i, e in enumerate(sim_world.crafts)
                           if id(e.craft) == id(bs.craft_ref))
                base = f"{sim_ns}::craft_{idx}"
            tplated = False
        if is_craft_signal(bs):
            return base, tplated
        return f"{base}.{bs.part_name}()", tplated

    src_in_filter = (id(src.craft_ref) in filter_ids
                     or _is_filter_signal(src))
    snk_in_filter = (id(snk.craft_ref) in filter_ids
                     or _is_filter_signal(snk))
    src_acc, _   = _accessor(src, sim_ns, src_in_filter)
    snk_acc, snk_templated = _accessor(snk, sim_ns, snk_in_filter)

    n = src.signal.n_floats
    lines.append(f"{indent}// connect: {src.part_name}.{src.name} → "
                 f"{snk.part_name}.{snk.name}")
    lines.append(f"{indent}{{")
    for i in range(n):
        expr = src.signal.cpp_read_exprs[i].format(accessor=src_acc)
        lines.append(f"{indent}    const double v{i} = double({expr});")
    fmt = {"accessor": snk_acc}
    for i in range(n):
        fmt[f"v{i}"] = f"v{i}"
    stmt = snk.signal.cpp_write_stmt.format(**fmt)
    if snk_templated:
        stmt = _double_qualify_partframe(stmt)
    lines.append(f"{indent}    {stmt}")
    lines.append(f"{indent}}}")


def emit_composed_hpp(target, sim_world, filter_obj) -> str:
    """Emit `<target>.hpp` for a sim+filter Target."""
    sim_name    = sim_world.name
    filter_name = filter_obj.world.name
    name        = target.name

    return "\n".join([
        GENERATED_BANNER, "#pragma once", "",
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
        f"}}  // namespace manta_gen::{name}",
        "",
    ])


def emit_composed_cpp(target, sim_world, filter_obj, kind: str = "ekf") -> str:
    """Emit `<target>.cpp` — composed setup/tick/shutdown that delegate to
    each sub-harness with cross-world connects between them."""
    sim_name    = sim_world.name
    filter_name = filter_obj.world.name
    name        = target.name
    sim_ns      = f"manta_gen::{sim_name}"
    filter_ns   = f"manta_gen::{filter_name}"

    est_craft = filter_obj.world.crafts[0].craft
    templated_filter_craft = bool(getattr(est_craft, "scalar_templated", False))

    # Cross-world connections live on whichever world's `connections` list
    # the source signal resolves to (via _world_for_signal). Walk both
    # worlds and pick the connections whose sink is in the OTHER world.
    sim_ids    = {id(e.craft) for e in sim_world.crafts}
    filter_ids = {id(e.craft) for e in filter_obj.world.crafts}

    def _is_cross(conn):
        src_in_filter = (id(conn.source.craft_ref) in filter_ids
                         or _is_filter_signal(conn.source))
        snk_in_filter = (id(conn.sink.craft_ref) in filter_ids
                         or _is_filter_signal(conn.sink))
        src_in_sim    = id(conn.source.craft_ref) in sim_ids and not _is_filter_signal(conn.source)
        snk_in_sim    = id(conn.sink.craft_ref)   in sim_ids and not _is_filter_signal(conn.sink)
        # Cross-world iff one endpoint is in sim and the other is in the
        # filter's est world.
        return ((src_in_sim and snk_in_filter)
                or (src_in_filter and snk_in_sim))

    cross_conns = [c for c in (list(sim_world.connections) +
                               list(filter_obj.world.connections))
                   if _is_cross(c)]

    lines: list[str] = [
        GENERATED_BANNER, "",
        f'#include "{name}.hpp"',
        "",
        "#include <Eigen/Core>",
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
        "",
    ]

    if cross_conns:
        lines.append("    // ---- Cross-world connect() steps ----")
        for conn in cross_conns:
            _emit_composed_connect_step(
                lines, conn,
                sim_world, sim_ns,
                filter_obj, filter_ns,
                templated_filter_craft, indent="    ")
        lines.append("")

    lines += [
        f"    {filter_ns}::tick();",
        "}",
        "",
        "void shutdown() {",
        f"    {filter_ns}::shutdown();",
        f"    {sim_ns}::shutdown();",
        "}",
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


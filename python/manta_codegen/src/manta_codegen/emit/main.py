"""Emit <name>_main.cpp — sim binary main with Zenoh I/O. (workflow="binary")

Wires the typed Craft into a 1 kHz sim loop. Two pub/sub paths coexist:

  * Legacy flag-based path — active when `craft.bindings` is empty. Each
    part with `publish_state=True` contributes to a bundled state topic;
    each part with `subscribe_command=True` gets a `<topic>/cmd` topic.
    Used by all pre-migration examples.

  * New Binding-based path (phase 2 of the API redesign) — active when
    `craft.bindings` is non-empty. Each Binding becomes one Zenoh
    publisher/subscriber on its specified topic. Per-binding wire format
    is determined by `binding.encoding` (json today, binary later). When
    a craft has any bindings, it must declare ALL of its pub/sub via
    bindings (the legacy bundled-state topic is not auto-emitted).
"""

from __future__ import annotations

from .._format import cpp_float as _f
from ..core import Craft, World, world_from_craft
from ..signal import Binding, accessor_for
from ._util import GENERATED_BANNER, class_name_for_craft


def _quote(s: str) -> str:
    return '"' + s.replace('\\', '\\\\').replace('"', '\\"') + '"'


def _initial_state_literal(entry) -> str:
    """C++ literal for `manta::InitialState{...}` built from a _CraftEntry.
    Returns `manta::InitialState{}` when the state is identity, matching the
    legacy behavior."""
    px, py, pz = entry.position
    ow, ox, oy, oz = entry.orientation
    vx, vy, vz = entry.vel_linear
    wx, wy, wz = entry.vel_angular
    if (entry.position == (0.0, 0.0, 0.0)
            and entry.orientation == (1.0, 0.0, 0.0, 0.0)
            and entry.vel_linear == (0.0, 0.0, 0.0)
            and entry.vel_angular == (0.0, 0.0, 0.0)):
        return "manta::InitialState{}"
    return (
        "manta::InitialState{"
        f"manta::geom::Vec3<manta::SceneFrame>{{{_f(px)}, {_f(py)}, {_f(pz)}}}, "
        f"manta::geom::Ori<manta::SceneFrame>{{Eigen::Quaternionf{{{_f(ow)}, {_f(ox)}, {_f(oy)}, {_f(oz)}}}}}, "
        f"manta::geom::Vec3<manta::SceneFrame>{{{_f(vx)}, {_f(vy)}, {_f(vz)}}}, "
        f"manta::geom::Vec3<manta::CraftFrame>{{{_f(wx)}, {_f(wy)}, {_f(wz)}}}"
        "}"
    )


def emit_main_cpp(craft: Craft, world: World | None = None) -> str:
    cls = class_name_for_craft(craft.name)
    use_bindings = bool(craft.bindings)
    publishing_parts = [] if use_bindings else [p for p in craft.all_parts() if p.publish_state]
    subscribing_parts = [] if use_bindings else [p for p in craft.all_parts() if p.subscribe_command]

    # World drives the sim loop config (dt, sim_rate_mult) and per-craft
    # initial state. Tests / older callers that pass only a Craft get a
    # synthesized World built from the Craft's deprecated sim_config /
    # initial_state fields.
    if world is None:
        world = world_from_craft(craft)
    entry = next((e for e in world.crafts if e.craft is craft), None)
    if entry is None:
        raise RuntimeError(
            f"emit_main_cpp: craft {craft.name!r} not registered with the World")

    state_topic = f"{craft.topic_prefix}/state"

    lines: list[str] = [
        GENERATED_BANNER,
        "",
        "#include <atomic>",
        "#include <chrono>",
        "#include <csignal>",
        "#include <cstdio>",
        "#include <cstdlib>",
        "#include <mutex>",
        "#include <string>",
        "#include <thread>",
        "#include <vector>",
        "",
        "#include <zenoh.hxx>",
        "",
        "#include \"manta/core/scene.hpp\"",
        "#include \"manta/core/world.hpp\"",
        f'#include "{craft.name}.hpp"',
        f'#include "{craft.name}_telemetry.hpp"',
    ]
    for f in world.fields:
        lines.append(f'#include "{f.cpp_header}"')
    for p in world.planets:
        lines.append(f'#include "{p.cpp_header}"')
    lines += [
        "",
        "namespace {",
        "std::atomic<bool> g_run{true};",
        "void on_signal(int) { g_run.store(false); }",
        "",
        "// Tiny float-array parser for command payloads.",
        "bool parse_float_array(std::string_view s, std::vector<float>& out) {",
        "    out.clear();",
        "    auto lb = s.find('['); auto rb = s.rfind(']');",
        "    if (lb == std::string_view::npos || rb == std::string_view::npos || rb <= lb) return false;",
        "    std::string body(s.substr(lb + 1, rb - lb - 1));",
        "    char* p = body.data(); char* end = body.data() + body.size();",
        "    while (p < end) {",
        "        while (p < end && (*p == ' ' || *p == ',' || *p == '\\t' || *p == '\\n')) ++p;",
        "        if (p >= end) break;",
        "        char* next = nullptr;",
        "        float v = std::strtof(p, &next);",
        "        if (next == p) return false;",
        "        out.push_back(v);",
        "        p = next;",
        "    }",
        "    return true;",
        "}",
        "}",
        "",
        "int main() {",
        "    std::signal(SIGINT,  on_signal);",
        "    std::signal(SIGTERM, on_signal);",
        "",
        f"    constexpr float DT             = {_f(world.dt)};",
        f"    constexpr float SIM_RATE_MULT  = {_f(world.sim_rate_mult)};",
        f"    const     float WALL_PERIOD    = DT / SIM_RATE_MULT;",
        "",
        "    manta::World w;",
        "    w.clock().set_dt(DT);",
        "    auto& scene = w.create_scene();",
    ]

    # Planets — added to World, then the (single) scene anchors to the first.
    # Each planet contributes its own field disturbances inside add_planet's
    # call to register_disturbances, so user-level field registration only
    # needs to cover whatever the user explicitly listed in world.fields.
    for i, p in enumerate(world.planets):
        var = f"planet_{i}"
        lines.append(f"    auto& {var} = w.add_planet<{p.cpp_class}>("
                     f"{p.emit_constructor_args()});")
    if world.planets:
        lines.append(f"    scene.set_planet(&planet_0);")

    # Field instances live with static storage in main; register with the World.
    # Concrete-type registration is implicit via template deduction; additional
    # base slots from `register_as` get explicit `register_field<Base>(...)`.
    for i, f in enumerate(world.fields):
        var = f"field_{i}"
        lines.append(f"    {f.emit_construction(var)}")
        lines.append(f"    w.register_field({var});")
        for base in getattr(f, "register_as", []) or []:
            lines.append(f"    w.register_field<{base}>({var});")

    lines += [
        f"    {cls} craft;",
        f"    scene.add_craft(craft, {_initial_state_literal(entry)});",
        "",
        "    zenoh::Config cfg = zenoh::Config::create_default();",
        "    auto session = zenoh::Session::open(std::move(cfg));",
        "",
    ]

    # ---- Subscribers ----
    if use_bindings:
        for i, b in enumerate(craft.bindings):
            if b.direction != "in":
                continue
            _emit_binding_subscriber(lines, i, b)
    else:
        # Legacy flag-based subscribers — each holds a small mutex-guarded payload buffer.
        for p in subscribing_parts:
            topic = p.command_topic or f"{craft.topic_prefix}/{p.name}/cmd"
            lines += [
                f"    std::mutex {p.name}_cmd_mtx;",
                f"    std::vector<float> {p.name}_cmd;",
                f"    auto {p.name}_sub = session.declare_subscriber(",
                f"        zenoh::KeyExpr({_quote(topic)}),",
                f"        [&](const zenoh::Sample& s) {{",
                f"            std::vector<float> v;",
                f"            std::string payload(s.get_payload().as_string());",
                f"            if (parse_float_array(payload, v)) {{",
                f"                std::lock_guard<std::mutex> lk({p.name}_cmd_mtx);",
                f"                {p.name}_cmd = std::move(v);",
                f"            }}",
                f"        }}, zenoh::closures::none);",
            ]

    # ---- Publishers ----
    if use_bindings:
        for i, b in enumerate(craft.bindings):
            if b.direction != "out":
                continue
            lines.append(
                f"    auto pub_{i} = session.declare_publisher(zenoh::KeyExpr({_quote(b.topic)}));")
    else:
        # Legacy flag-based: top-level bundled state + per-part publishers.
        lines.append(f"    auto state_pub = session.declare_publisher(zenoh::KeyExpr({_quote(state_topic)}));")
        for p in publishing_parts:
            topic = p.state_topic or f"{craft.topic_prefix}/{p.name}/state"
            lines.append(
                f"    auto {p.name}_pub = session.declare_publisher(zenoh::KeyExpr({_quote(topic)}));"
            )

    ready_msg = (
        f"{craft.name}: ready. {len(craft.bindings)} explicit binding(s)."
        if use_bindings
        else f"{craft.name}: ready. State on '{state_topic}'."
    )
    lines += [
        "",
        f'    std::printf("{ready_msg}\\n");',
        "",
        "    auto next = std::chrono::steady_clock::now();",
        "    const auto period = std::chrono::microseconds(int64_t(WALL_PERIOD * 1e6));",
        "    int pub_decim = 0;",
        "    const int pub_every = 20;  // ~50 Hz publish",
        "",
        "    while (g_run.load()) {",
    ]

    # Apply commands.
    if use_bindings:
        for i, b in enumerate(craft.bindings):
            if b.direction != "in":
                continue
            _emit_binding_apply(lines, i, b)
    else:
        for p in subscribing_parts:
            apply_stmt = p.emit_command_apply(f"craft.{p.name}()", f"{p.name}_cmd")
            lines += [
                f"        {{ std::lock_guard<std::mutex> lk({p.name}_cmd_mtx);",
                f"          if (!{p.name}_cmd.empty()) {{",
                f"              {apply_stmt}",
                f"          }} }}",
            ]

    lines += [
        "",
        "        w.update();",
        "",
        "        if (++pub_decim >= pub_every) {",
        "            pub_decim = 0;",
    ]

    if use_bindings:
        for i, b in enumerate(craft.bindings):
            if b.direction != "out":
                continue
            _emit_binding_publish(lines, i, b)
    else:
        lines += [
            f"            {cls}Telemetry telem;",
            f"            capture_{craft.name}_telemetry(craft, w.clock().time(), telem);",
            "            state_pub.put(zenoh::Bytes(telem.to_json()));",
        ]

    lines += [
        "        }",
        "",
        "        next += period;",
        "        std::this_thread::sleep_until(next);",
        "    }",
        "",
        f"    std::printf(\"{craft.name}: shutting down.\\n\");",
        "    return 0;",
        "}",
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Binding emit helpers (phase 2)

def _emit_binding_subscriber(lines: list[str], i: int, b: Binding) -> None:
    """Declare a Zenoh subscriber for an in-Binding. Stores the most recent
    parsed payload (size = b.total_floats) under a mutex; the main loop
    consumes it via _emit_binding_apply()."""
    lines += [
        f"    std::mutex bind_{i}_mtx;",
        f"    std::vector<float> bind_{i}_payload;",
        f"    auto bind_{i}_sub = session.declare_subscriber(",
        f"        zenoh::KeyExpr({_quote(b.topic)}),",
        f"        [&](const zenoh::Sample& s) {{",
        f"            std::vector<float> v;",
        f"            std::string payload(s.get_payload().as_string());",
        f"            if (parse_float_array(payload, v) && v.size() >= {b.total_floats}) {{",
        f"                std::lock_guard<std::mutex> lk(bind_{i}_mtx);",
        f"                bind_{i}_payload = std::move(v);",
        f"            }}",
        f"        }}, zenoh::closures::none);",
    ]


def _emit_binding_apply(lines: list[str], i: int, b: Binding) -> None:
    """Inside the main loop: under the mutex, if the payload is fresh (matches
    expected width), substitute v0..v_{n-1} into each member's cpp_write_stmt
    and call. After consumption clears the buffer so re-runs don't re-apply."""
    lines += [
        f"        {{ std::lock_guard<std::mutex> lk(bind_{i}_mtx);",
        f"          if (bind_{i}_payload.size() >= {b.total_floats}) {{",
    ]
    cursor = 0
    for member_name, sig in b.members.items():
        n = sig.signal.n_floats
        accessor = accessor_for(sig)
        v_subs = {f"v{k}": f"bind_{i}_payload[{cursor + k}]" for k in range(n)}
        v_subs["accessor"] = accessor
        stmt = sig.signal.cpp_write_stmt.format(**v_subs)
        lines.append(f"              {stmt}    // member: {member_name}")
        cursor += n
    lines += [
        f"              bind_{i}_payload.clear();",
        f"          }} }}",
    ]


def _emit_binding_publish(lines: list[str], i: int, b: Binding) -> None:
    """Inside the decimated publish block: build the wire payload from the
    binding's read expressions and put() it on its publisher. JSON encoding
    today; binary will be a separate branch on b.encoding."""
    if b.encoding != "json":
        raise NotImplementedError(
            f"Binding {b.topic!r}: encoding {b.encoding!r} not yet supported "
            f"(only 'json' is implemented)")

    # Build a JSON object: {"member_name":[v0,...],...}.
    # Use snprintf into a fixed-size buffer; per-member width is sum of n_floats.
    # For simplicity we emit a stream of std::string concatenations — readable
    # in the generated code, fast enough for typical 50 Hz publish rates.
    lines.append(f"            {{ std::string _json = \"{{\";")
    first_member = True
    for member_name, sig in b.members.items():
        n = sig.signal.n_floats
        accessor = accessor_for(sig)
        if not first_member:
            lines.append('              _json += ",";')
        first_member = False
        lines.append(f'              _json += "\\"{member_name}\\":[";')
        for k, expr in enumerate(sig.signal.cpp_read_exprs):
            cpp_expr = expr.format(accessor=accessor)
            sep = '","' if k > 0 else '""'
            lines.append(
                f"              {{ char _b[32]; std::snprintf(_b, sizeof(_b), \"%s%g\", "
                f"{sep}, double({cpp_expr})); _json += _b; }}"
            )
        lines.append('              _json += "]";')
    lines += [
        '              _json += "}";',
        f"              pub_{i}.put(zenoh::Bytes(_json));",
        "            }",
    ]

"""Emit <name>_main.cpp — sim binary main with Zenoh I/O. (workflow="binary")

Wires the typed Craft into a 1 kHz sim loop. Each part with `publish_state=True`
gets a Zenoh publisher; each part with `subscribe_command=True` gets a Zenoh
subscriber that calls the part's `emit_command_apply` snippet.

Top-level kinematic state is published on `<topic_prefix>/state`.
"""

from __future__ import annotations

from .._format import cpp_float as _f
from ..core import Craft
from ._util import GENERATED_BANNER, class_name_for_craft


def _quote(s: str) -> str:
    return '"' + s.replace('\\', '\\\\').replace('"', '\\"') + '"'


def emit_main_cpp(craft: Craft) -> str:
    cls = class_name_for_craft(craft.name)
    publishing_parts = [p for p in craft.all_parts() if p.publish_state]
    subscribing_parts = [p for p in craft.all_parts() if p.subscribe_command]

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
    for f in craft.fields:
        lines.append(f'#include "{f.cpp_header}"')
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
        f"    constexpr float DT             = {_f(craft.dt)};",
        f"    constexpr float SIM_RATE_MULT  = {_f(craft.sim_rate_mult)};",
        f"    const     float WALL_PERIOD    = DT / SIM_RATE_MULT;",
        "",
        "    manta::World w;",
        "    w.clock().set_dt(DT);",
        "    auto& scene = w.create_scene();",
    ]

    # Field instances live with static storage in main; register with the World.
    # Concrete-type registration is implicit via template deduction; additional
    # base slots from `register_as` get explicit `register_field<Base>(...)`.
    for i, f in enumerate(craft.fields):
        var = f"field_{i}"
        lines.append(f"    {f.emit_construction(var)}")
        lines.append(f"    w.register_field({var});")
        for base in getattr(f, "register_as", []) or []:
            lines.append(f"    w.register_field<{base}>({var});")

    lines += [
        f"    {cls} craft;",
        f"    scene.add_craft(craft, {craft.emit_initial_state_cpp()});",
        "",
        "    zenoh::Config cfg = zenoh::Config::create_default();",
        "    auto session = zenoh::Session::open(std::move(cfg));",
        "",
    ]

    # Per-part subscribers — each holds a small mutex-guarded payload buffer.
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

    # Publishers — one per publishing part + one for top-level state.
    lines.append(f"    auto state_pub = session.declare_publisher(zenoh::KeyExpr({_quote(state_topic)}));")
    for p in publishing_parts:
        topic = p.state_topic or f"{craft.topic_prefix}/{p.name}/state"
        lines.append(
            f"    auto {p.name}_pub = session.declare_publisher(zenoh::KeyExpr({_quote(topic)}));"
        )

    lines += [
        "",
        f"    std::printf(\"{craft.name}: ready. State on '{state_topic}'.\\n\");",
        "",
        "    auto next = std::chrono::steady_clock::now();",
        "    const auto period = std::chrono::microseconds(int64_t(WALL_PERIOD * 1e6));",
        "    int pub_decim = 0;",
        "    const int pub_every = 20;  // ~50 Hz publish",
        "",
        "    while (g_run.load()) {",
    ]

    # Apply commands to subscribed parts.
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
        f"            {cls}Telemetry telem;",
        f"            capture_{craft.name}_telemetry(craft, w.clock().time(), telem);",
        "            state_pub.put(zenoh::Bytes(telem.to_json()));",
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

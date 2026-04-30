"""Emit a runnable C++ main that subscribes to Zenoh topics and feeds the
estimator's sensor parts via set_measurement(). Used by the `real_data`
workflow.

The user supplies a topic mapping `{part_name: zenoh_topic}`. For each entry,
the emitter declares a Zenoh subscriber on that topic, parses the payload
via the part's `emit_measurement_decode()` snippet, and calls the part's
`set_measurement(...)`. No sim Craft, no World, no Scene — just an
estimator-side Craft, sensor inputs, and a subscriber loop.

The emitted main does NOT include the EKF wiring — that's a user concern
(the user picks an Estimator + state layout + tuning). This main exists to
move the data plumbing off the user's plate.
"""

from __future__ import annotations

from ..core import Craft
from ._util import GENERATED_BANNER, class_name_for_craft


def _quote(s: str) -> str:
    return '"' + s.replace('\\', '\\\\').replace('"', '\\"') + '"'


def emit_real_data_main_cpp(craft: Craft, topic_map: dict[str, str]) -> str:
    cls = class_name_for_craft(craft.name)

    # Find the parts named in `topic_map` — error if any name doesn't exist
    # in the craft tree (catches typos at codegen time).
    parts_by_name = {p.name: p for p in craft.all_parts()}
    bound: list[tuple[str, str, object]] = []
    for part_name, topic in topic_map.items():
        if part_name not in parts_by_name:
            raise ValueError(
                f"--topics: part {part_name!r} is not in craft {craft.name!r}. "
                f"Available parts: {sorted(parts_by_name)}")
        bound.append((part_name, topic, parts_by_name[part_name]))

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
        f'#include "{craft.name}.hpp"',
        "",
        "namespace {",
        "std::atomic<bool> g_run{true};",
        "void on_signal(int) { g_run.store(false); }",
        "",
        "// Tiny float-array parser for sensor payloads.",
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
        f"    {cls} est;",
        "",
        "    zenoh::Config cfg = zenoh::Config::create_default();",
        "    auto session = zenoh::Session::open(std::move(cfg));",
        "",
    ]

    # One subscriber per bound part.
    for part_name, topic, part in bound:
        decode = part.emit_measurement_decode(f"est.{part_name}()", "v")
        if not decode:
            raise ValueError(
                f"Part {part_name!r} (type {part.cpp_class!r}) does not "
                "implement emit_measurement_decode — cannot bind to a real-data "
                "topic. Add a measurement decoder on the descriptor.")
        sub_var = f"{part_name}_sub"
        lines += [
            f"    auto {sub_var} = session.declare_subscriber(",
            f"        zenoh::KeyExpr({_quote(topic)}),",
            "        [&](const zenoh::Sample& s) {",
            "            std::vector<float> v;",
            "            std::string payload(s.get_payload().as_string());",
            "            if (!parse_float_array(payload, v)) return;",
            f"            {decode}",
            "        }, zenoh::closures::none);",
        ]

    sub_topics = ", ".join(t for _, t, _ in bound)
    lines += [
        "",
        f'    std::printf("{craft.name}: subscribed to: {sub_topics}\\n");',
        "    while (g_run.load()) {",
        "        std::this_thread::sleep_for(std::chrono::milliseconds(100));",
        "    }",
        "    std::printf(\"" + craft.name + ": shutting down.\\n\");",
        "    return 0;",
        "}",
        "",
    ]
    return "\n".join(lines)

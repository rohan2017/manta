"""Emit <name>_telemetry.hpp — per-craft telemetry struct + JSON encoder.

Each part that opts into `publish_state=True` contributes a sub-struct of named
members (whatever its `telemetry_fields()` returns). The codegen aggregates all
of them into a single `<Cls>Telemetry` struct.

Capture is a free function `capture_<name>_telemetry(craft, telemetry)` that
reads each part's accessor methods (e.g. `craft.motor_0().throttle()`).

JSON encoding is hand-rolled (no external dep), matching the existing
state_codec.hpp style.
"""

from __future__ import annotations

from ..core import Craft
from ._util import GENERATED_BANNER, CPP_INCLUDE_GUARD, class_name_for_craft


def emit_telemetry_hpp(craft: Craft) -> str:
    cls = class_name_for_craft(craft.name)
    publishing_parts = [p for p in craft.all_parts() if p.publish_state]

    lines: list[str] = [
        GENERATED_BANNER,
        CPP_INCLUDE_GUARD,
        "",
        "#include <cstdio>",
        "#include <string>",
        "",
        f'#include "{craft.name}.hpp"',
        "",
        f"struct {cls}Telemetry {{",
        "    double t_sec = 0.0;",
        "    // Top-level kinematic state (always populated).",
        "    float p[3] = {0,0,0};",
        "    float q[4] = {1,0,0,0};   // (w, x, y, z)",
        "    float v[3] = {0,0,0};",
        "    float w[3] = {0,0,0};",
    ]

    for p in publishing_parts:
        fields = p.telemetry_fields()
        if not fields:
            continue
        lines.append(f"    struct {p.name.capitalize()}T {{")
        for member, ctype in fields:
            lines.append(f"        {ctype} {member}{{}};")
        lines.append(f"    }} {p.name};")

    lines += [
        "",
        "    std::string to_json() const;",
        "};",
        "",
        f"void capture_{craft.name}_telemetry(const {cls}& craft, double t_sec, {cls}Telemetry& telem);",
        "",
    ]

    # Inline implementation — header-only is convenient for the small workflow-binary case.
    # JSON is nested per-part: { "t":..., "p":[...], ..., "<part>": {"member": value, ...}, ... }
    lines += [
        f"inline std::string {cls}Telemetry::to_json() const {{",
        "    char buf[256];",
        "    std::string s = \"{\";",
        "    std::snprintf(buf, sizeof(buf),",
        "        \"\\\"t\\\":%.6f,\\\"p\\\":[%.6f,%.6f,%.6f],\\\"q\\\":[%.6f,%.6f,%.6f,%.6f],\"",
        "        \"\\\"v\\\":[%.6f,%.6f,%.6f],\\\"w\\\":[%.6f,%.6f,%.6f]\",",
        "        t_sec,",
        "        (double)p[0], (double)p[1], (double)p[2],",
        "        (double)q[0], (double)q[1], (double)q[2], (double)q[3],",
        "        (double)v[0], (double)v[1], (double)v[2],",
        "        (double)w[0], (double)w[1], (double)w[2]);",
        "    s += buf;",
    ]

    # Per-part nested object: ,"<part>":{"<member>":value, ...}
    for p in publishing_parts:
        fields = p.telemetry_fields()
        if not fields:
            continue
        # Open the nested object.
        lines.append(f'    s += ",\\"{p.name}\\":{{";')
        for i, (member, ctype) in enumerate(fields):
            sep = "" if i == 0 else ","
            t = ctype.strip()
            if t in ("float", "double", "manta::Real") or t.startswith("int"):
                lines.append(
                    f'    {{ char b[64]; std::snprintf(b, sizeof(b), '
                    f'"{sep}\\"{member}\\":%.6f", (double){p.name}.{member}); s += b; }}'
                )
            else:
                # Unsupported type — skip silently for now. Future: dispatch to
                # a free function `to_json_member(&) -> std::string` per type.
                lines.append(
                    f'    // TODO: telemetry serializer for type "{ctype}" '
                    f'(member {p.name}.{member})'
                )
        lines.append('    s += "}";')

    lines.append('    s += "}";')
    lines.append("    return s;")
    lines.append("}")
    lines.append("")

    # Capture function: reads from the typed Craft accessors.
    lines += [
        f"inline void capture_{craft.name}_telemetry(const {cls}& craft, double t_sec, {cls}Telemetry& telem) {{",
        "    telem.t_sec = t_sec;",
        "    auto p = craft.scene_to_craft().position();",
        "    auto q = craft.scene_to_craft().orientation().raw();",
        "    auto v = craft.scene_to_craft().vel_linear();",
        "    auto w = craft.scene_to_craft().vel_angular();",
        "    telem.p[0]=p.x(); telem.p[1]=p.y(); telem.p[2]=p.z();",
        "    telem.q[0]=q.w(); telem.q[1]=q.x(); telem.q[2]=q.y(); telem.q[3]=q.z();",
        "    telem.v[0]=v.x(); telem.v[1]=v.y(); telem.v[2]=v.z();",
        "    telem.w[0]=w.x(); telem.w[1]=w.y(); telem.w[2]=w.z();",
    ]
    for p in publishing_parts:
        for member, rhs in p.emit_telemetry_reads():
            lines.append(f"    telem.{p.name}.{member} = {rhs};")
    lines.append("}")
    lines.append("")

    return "\n".join(lines)

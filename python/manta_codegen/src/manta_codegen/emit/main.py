"""Emit <name>_main.cpp — sim binary main with Zenoh I/O. (workflow="binary")

Wires the typed Craft into a sim loop driven by World.dt. Each Binding
on the Craft becomes one Zenoh publisher/subscriber on its specified
topic. Per-binding wire format is determined by `binding.encoding`
(json today, binary later). A craft must declare ALL of its pub/sub
via bindings; there is no auto-bundled state topic.
"""

from __future__ import annotations

from .._format import cpp_float as _f
from ..core import Craft, World
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


def _world_unique_crafts(world: World) -> list[Craft]:
    """Walk a World's craft entries returning each unique Craft object once
    (by Python identity), preserving insertion order. A multi-craft world
    that adds the same Craft instance multiple times shares one set of
    generated files, so codegen emits one #include per unique Craft."""
    seen: set[int] = set()
    out: list[Craft] = []
    for entry in world.crafts:
        if id(entry.craft) not in seen:
            seen.add(id(entry.craft))
            out.append(entry.craft)
    return out


def emit_main_cpp(world: World) -> str:
    if not world.crafts:
        raise RuntimeError("emit_main_cpp: World has no crafts")

    unique_crafts = _world_unique_crafts(world)
    primary = unique_crafts[0]

    lines: list[str] = [
        GENERATED_BANNER,
        "",
        "#include <atomic>",
        "#include <chrono>",
        "#include <csignal>",
        "#include <cstdint>",
        "#include <cstdio>",
        "#include <cstdlib>",
        "#include <cstring>",
        "#include <mutex>",
        "#include <string>",
        "#include <thread>",
        "#include <vector>",
        "",
        "#include <zenoh.hxx>",
        "",
        "#include \"manta/core/scene.hpp\"",
        "#include \"manta/core/world.hpp\"",
    ]
    # One #include per unique Craft type.
    for c in unique_crafts:
        lines.append(f'#include "{c.name}.hpp"')
    for f in world.fields:
        lines.append(f'#include "{f.cpp_header}"')
    for p in world.planets:
        lines.append(f'#include "{p.cpp_header}"')
    if world.tethers:
        lines.append('#include "manta/coupling/tether.hpp"')
        lines.append('#include "manta/parts/coupling/tether_endpoint.hpp"')
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
    sync_field_idxs: list[int] = []
    for i, f in enumerate(world.fields):
        var = f"field_{i}"
        lines.append(f"    {f.emit_construction(var)}")
        for stmt in (f.emit_extra_setup(var) if hasattr(f, "emit_extra_setup") else []):
            lines.append(f"    {stmt}")
        lines.append(f"    w.register_field({var});")
        for base in getattr(f, "register_as", []) or []:
            lines.append(f"    w.register_field<{base}>({var});")
        if getattr(f, "synchronized", False):
            sync_field_idxs.append(i)

    # Instantiate one C++ object per World.crafts entry. For a single-craft
    # world the variable is named `craft`; for multi-craft it's
    # craft_0, craft_1, .... Each is added to the scene with its own
    # initial state literal.
    multi = len(world.crafts) > 1
    craft_var = lambda idx: "craft" if not multi else f"craft_{idx}"
    for idx, entry in enumerate(world.crafts):
        cls = class_name_for_craft(entry.craft.name)
        var = craft_var(idx)
        lines.append(f"    {cls} {var};")

    # Tethers + endpoints. Each endpoint is added to the craft's root
    # AFTER craft construction (the per-Craft .cpp doesn't know about
    # cross-craft tethers). compute_params() must re-run on each affected
    # craft to fold the new endpoint's mass/MOI into the rigid-body params.
    if world.tethers:
        lines.append("")
        lines.append("    // ---- Tethers ----")
        # Need <coupling/tether.hpp> + tether endpoint. They're already
        # included transitively via craft.hpp -> part.hpp; explicit include
        # for clarity.
        # Find the craft index for a given Craft Python object.
        craft_idx_of = {}
        for ci, e in enumerate(world.crafts):
            if id(e.craft) not in craft_idx_of:
                craft_idx_of[id(e.craft)] = ci
        affected_crafts: set[int] = set()
        for ti, te in enumerate(world.tethers):
            tname = f"tether_{ti}"
            t = te.tether
            lines.append(
                f"    manta::coupling::Tether {tname}("
                f"{_f(t.rest_length)}, {_f(t.stiffness)}, {_f(t.damping)});"
            )
            for ep, is_a in ((te.endpoint_a, True), (te.endpoint_b, False)):
                ep_craft, ep_name = ep
                ci = craft_idx_of[id(ep_craft)]
                affected_crafts.add(ci)
                lines.append(
                    f"    {craft_var(ci)}.root().add<manta::parts::TetherEndpoint>("
                    f'"{ep_name}", {tname}, {"true" if is_a else "false"});'
                )
        # Re-run compute_params on each craft we touched so the new
        # endpoint's mass/MOI/COM contribution lands in the cached
        # rigid-body params.
        for ci in sorted(affected_crafts):
            lines.append(f"    {craft_var(ci)}.root().compute_params();")
        lines.append("")

    for idx, entry in enumerate(world.crafts):
        var = craft_var(idx)
        lines.append(f"    scene.add_craft({var}, {_initial_state_literal(entry)});")

    lines += [
        "    zenoh::Config cfg = zenoh::Config::create_default();",
        "    auto session = zenoh::Session::open(std::move(cfg));",
        "",
    ]

    # Subscribers / publishers: per-craft, per-binding. Bindings are
    # numbered globally across the World (bind_0 / bind_1 / ...) so each
    # gets its own Zenoh handle and mutex.
    bind_idx = 0
    bind_assignments: list[tuple[int, int, "Binding"]] = []  # (bind_id, craft_idx, binding)
    # Bindings live on the World now. Each binding carries a struct of
    # BoundSignals; we look at the first member's craft_ref to pick the
    # craft index for accessor emission. Bundled bindings whose members
    # span multiple crafts aren't supported (a binding's payload represents
    # one logical value tied to one craft); the validator below catches
    # that case.
    craft_index_by_id: dict[int, int] = {
        id(entry.craft): cidx for cidx, entry in enumerate(world.crafts)
    }
    for b in world.bindings:
        member_crafts = {id(m.craft_ref) for m in b.members.values()}
        if len(member_crafts) != 1:
            raise RuntimeError(
                f"emit_main_cpp: binding on topic {b.topic!r} mixes signals "
                f"from multiple crafts; split into one binding per craft.")
        craft_id = next(iter(member_crafts))
        if craft_id not in craft_index_by_id:
            raise RuntimeError(
                f"emit_main_cpp: binding on topic {b.topic!r} references a "
                f"craft not registered with this world.")
        bind_assignments.append((bind_idx, craft_index_by_id[craft_id], b))
        bind_idx += 1

    for bid, cidx, b in bind_assignments:
        if b.direction == "in":
            _emit_binding_subscriber(lines, bid, b)
    for bid, cidx, b in bind_assignments:
        if b.direction == "out":
            lines.append(
                f"    auto pub_{bid} = session.declare_publisher("
                f"zenoh::KeyExpr({_quote(b.topic)}));")

    # Field sync: each field with `synchronized=True` gets a Zenoh pub +
    # sub on `manta/<world>/field_<i>/disturbance`. The publisher fires
    # from the field's tx_hook on every (non-USER) add(); the subscriber
    # decodes the wire bytes and calls field_<i>.receive(...). The Field's
    # internal recursion guard suppresses echo on the receiving side.
    for i in sync_field_idxs:
        topic = world.fields[i].sync_topic or f"manta/{world.name}/field_{i}/disturbance"
        _emit_field_sync(lines, i, topic, world.fields[i].cpp_class)

    total_bindings = len(bind_assignments)
    lines += [
        "",
        f'    std::printf("{world.name}: ready. {len(world.crafts)} craft(s),'
        f' {total_bindings} binding(s).\\n");',
        "",
        "    auto next = std::chrono::steady_clock::now();",
        "    const auto period = std::chrono::microseconds(int64_t(WALL_PERIOD * 1e6));",
        "    int pub_decim = 0;",
        "    const int pub_every = 20;  // ~50 Hz publish",
        "",
        "    while (g_run.load()) {",
    ]

    # Apply commands (in-bindings).
    for bid, cidx, b in bind_assignments:
        if b.direction == "in":
            _emit_binding_apply(lines, bid, b, craft_var=craft_var(cidx))

    lines += [
        "",
        "        w.update();",
        "",
    ]

    # In-process connect() steps run right after w.update() so the values
    # they propagate are the just-computed ones. Bindings (Zenoh pub) read
    # the same values on the publish-decimation tick below.
    for conn in world.connections:
        _emit_connection_step(lines, conn, craft_index_by_id, multi)

    lines += [
        "        if (++pub_decim >= pub_every) {",
        "            pub_decim = 0;",
    ]

    for bid, cidx, b in bind_assignments:
        if b.direction == "out":
            _emit_binding_publish(lines, bid, b, craft_var=craft_var(cidx))

    lines += [
        "        }",
        "",
        "        next += period;",
        "        std::this_thread::sleep_until(next);",
        "    }",
        "",
        f"    std::printf(\"{world.name}: shutting down.\\n\");",
        "    return 0;",
        "}",
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Binding emit helpers (phase 2)

def _emit_connection_step(lines: list[str],
                          conn,
                          craft_index_by_id: dict[int, int],
                          multi: bool) -> None:
    """Emit one tick's worth of in-process signal-to-signal copy. Reads the
    source's per-component cpp_read_exprs into temporaries, then runs the
    sink's cpp_write_stmt with v0..v{n-1} bound to those temps."""
    src = conn.source
    snk = conn.sink
    src_var = "craft" if not multi else f"craft_{craft_index_by_id[id(src.craft_ref)]}"
    snk_var = "craft" if not multi else f"craft_{craft_index_by_id[id(snk.craft_ref)]}"
    src_acc = _accessor_for_with_var(src, src_var)
    snk_acc = _accessor_for_with_var(snk, snk_var)
    n = src.signal.n_floats
    lines.append(f"        // connect: {src.part_name}.{src.name} → "
                 f"{snk.part_name}.{snk.name}")
    lines.append("        {")
    for i in range(n):
        expr = src.signal.cpp_read_exprs[i].format(accessor=src_acc)
        lines.append(f"            const float v{i} = {expr};")
    fmt = {"accessor": snk_acc}
    for i in range(n):
        fmt[f"v{i}"] = f"v{i}"
    lines.append(f"            {snk.signal.cpp_write_stmt.format(**fmt)}")
    lines.append("        }")


def _emit_field_sync(lines: list[str], i: int, topic: str, cpp_class: str) -> None:
    """Emit Zenoh tx/rx wiring for a synchronized Field instance.

    Wire layout (binary, little-endian, schema_version=1):
        uint16  schema_version
        uint16  tag
        int32   lifetime
        uint8[K] params         (K = field's kParamsBytes; currently 96)
    """
    lines += [
        f"    auto pub_field_{i} = session.declare_publisher("
        f"zenoh::KeyExpr({_quote(topic)}));",
        f"    field_{i}.set_tx_hook("
        f"[&pub_field_{i}](std::uint16_t tag, "
        f"const {cpp_class}::Params& params, int lifetime) {{",
        f"        std::vector<std::uint8_t> buf;",
        f"        buf.resize(2 + 2 + 4 + params.size());",
        f"        std::uint16_t ver = 1;",
        f"        std::memcpy(buf.data() + 0, &ver,      2);",
        f"        std::memcpy(buf.data() + 2, &tag,      2);",
        f"        std::memcpy(buf.data() + 4, &lifetime, 4);",
        f"        std::memcpy(buf.data() + 8, params.data(), params.size());",
        f"        pub_field_{i}.put(zenoh::Bytes(std::move(buf)));",
        f"    }});",
        f"    auto sub_field_{i} = session.declare_subscriber(",
        f"        zenoh::KeyExpr({_quote(topic)}),",
        f"        [&](const zenoh::Sample& s) {{",
        f"            auto payload = s.get_payload().as_vector();",
        f"            if (payload.size() < 8 + {cpp_class}::kParamsBytes) return;",
        f"            std::uint16_t ver = 0, tag = 0;",
        f"            std::int32_t  lifetime = 0;",
        f"            std::memcpy(&ver,      payload.data() + 0, 2);",
        f"            std::memcpy(&tag,      payload.data() + 2, 2);",
        f"            std::memcpy(&lifetime, payload.data() + 4, 4);",
        f"            if (ver != 1) return;",
        f"            {cpp_class}::Params p{{}};",
        f"            std::memcpy(p.data(), payload.data() + 8, p.size());",
        f"            field_{i}.receive(tag, p, lifetime);",
        f"        }}, zenoh::closures::none);",
    ]


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


def _accessor_for_with_var(sig, craft_var: str) -> str:
    """Like accessor_for() but with a configurable C++ craft variable name.
    Replaces the hard-coded `craft` with `craft_var` so multi-craft mains
    can use craft_0, craft_1, ... for distinct instances."""
    base = accessor_for(sig)   # "craft" or "craft.<part>()"
    if base == "craft":
        return craft_var
    assert base.startswith("craft.")
    return craft_var + base[len("craft"):]


def _emit_binding_apply(lines: list[str], i: int, b: Binding,
                        craft_var: str = "craft") -> None:
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
        accessor = _accessor_for_with_var(sig, craft_var)
        v_subs = {f"v{k}": f"bind_{i}_payload[{cursor + k}]" for k in range(n)}
        v_subs["accessor"] = accessor
        stmt = sig.signal.cpp_write_stmt.format(**v_subs)
        lines.append(f"              {stmt}    // member: {member_name}")
        cursor += n
    lines += [
        f"              bind_{i}_payload.clear();",
        f"          }} }}",
    ]


def _emit_binding_publish(lines: list[str], i: int, b: Binding,
                          craft_var: str = "craft") -> None:
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
        accessor = _accessor_for_with_var(sig, craft_var)
        if not first_member:
            lines.append('              _json += ",";')
        first_member = False
        # Scalar (n=1) members emit bare numbers; multi-component members
        # emit JSON arrays. Both shapes are idiomatic for downstream parsers.
        if n == 1:
            cpp_expr = sig.signal.cpp_read_exprs[0].format(accessor=accessor)
            lines.append(f'              _json += "\\"{member_name}\\":";')
            lines.append(
                f"              {{ char _b[32]; std::snprintf(_b, sizeof(_b), \"%g\", "
                f"double({cpp_expr})); _json += _b; }}"
            )
        else:
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

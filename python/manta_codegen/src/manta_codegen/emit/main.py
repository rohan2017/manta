"""Emit the World harness — `<world>.hpp` + `<world>.cpp` + `<world>_main.cpp`.

Three artifacts:

  * `<world>.hpp` — `namespace manta_gen::<world>` declarations: the
    `manta::World` instance, scene pointer, planet pointers, field
    instances, craft instances, plus `void setup()` / `void tick()`.
    Includes the per-craft headers (`<craft>_craft.hpp`) so symbols
    like the craft class and accessor methods are reachable from any
    TU that includes the world header.

  * `<world>.cpp` — storage definitions for everything declared in the
    header, plus `setup()` (build the scene, register planets/fields,
    add crafts, declare Zenoh pubs/subs) and `tick()` (apply
    in-bindings, `w.step()`, in-process `connect()` steps,
    decimated publish). Zenoh session + per-binding state live in
    file-private anonymous-namespace storage.

  * `<world>_main.cpp` — the binary `main()`: signal handlers, calls
    `setup()` once, runs `tick()` in a paced loop. Only emitted when
    workflow == "binary".

The library workflow emits the first two; the user provides their
own main and reuses `manta_gen::<world>::setup`/`tick`.
"""

from __future__ import annotations

from .._format import cpp_float as _f
from ..core import Craft, World
from ..signal import Binding, accessor_for
from ._util import GENERATED_BANNER, CPP_INCLUDE_GUARD, class_name_for_craft


def _quote(s: str) -> str:
    return '"' + s.replace('\\', '\\\\').replace('"', '\\"') + '"'


def _initial_state_literal(entry) -> str:
    """C++ literal for `manta::InitialState{...}` built from a _CraftEntry."""
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
    (by Python identity), preserving insertion order."""
    seen: set[int] = set()
    out: list[Craft] = []
    for entry in world.crafts:
        if id(entry.craft) not in seen:
            seen.add(id(entry.craft))
            out.append(entry.craft)
    return out


def _craft_var_name(world: World, idx: int) -> str:
    """Single-craft worlds use `craft`; multi-craft use craft_0/1/..."""
    return "craft" if len(world.crafts) == 1 else f"craft_{idx}"


def _world_includes(world: World) -> list[str]:
    """Standard set of `#include` lines a World harness needs."""
    out = [
        '#include "manta/core/harness.hpp"',
        '#include "manta/core/scene.hpp"',
        '#include "manta/core/world.hpp"',
    ]
    for c in _world_unique_crafts(world):
        out.append(f'#include "{c.name}_craft.hpp"')
    seen: set[str] = set()
    for f in world.fields:
        if f.cpp_header not in seen:
            seen.add(f.cpp_header)
            out.append(f'#include "{f.cpp_header}"')
    for p in world.planets:
        if p.cpp_header not in seen:
            seen.add(p.cpp_header)
            out.append(f'#include "{p.cpp_header}"')
    if world.tethers:
        out.append('#include "manta/coupling/tether.hpp"')
        out.append('#include "manta/parts/coupling/tether_endpoint.hpp"')
    return out


def _build_bind_assignments(world: World) -> list[tuple[int, int, "Binding"]]:
    """Walk world.bindings, attach each to its source craft index, give it a
    globally-unique bind_<id>. Returns (bind_id, craft_idx, binding) tuples."""
    craft_index_by_id: dict[int, int] = {
        id(entry.craft): cidx for cidx, entry in enumerate(world.crafts)
    }
    out: list[tuple[int, int, "Binding"]] = []
    for bid, b in enumerate(world.bindings):
        member_crafts = {id(m.craft_ref) for m in b.members.values()}
        if len(member_crafts) != 1:
            raise RuntimeError(
                f"emit_world: binding on topic {b.topic!r} mixes signals "
                f"from multiple crafts; split into one binding per craft.")
        craft_id = next(iter(member_crafts))
        if craft_id not in craft_index_by_id:
            raise RuntimeError(
                f"emit_world: binding on topic {b.topic!r} references a "
                f"craft not registered with this world.")
        out.append((bid, craft_index_by_id[craft_id], b))
    return out


# ---------------------------------------------------------------------------
# .hpp emit

def emit_world_hpp(world: World) -> str:
    """Emit `<world>.hpp` — the public-facing header for the World harness."""
    if not world.crafts:
        raise RuntimeError("emit_world_hpp: World has no crafts")

    name = world.name
    lines: list[str] = [
        GENERATED_BANNER, CPP_INCLUDE_GUARD, "",
    ]
    lines += _world_includes(world)
    lines += [
        "",
        f"namespace manta_gen::{name} {{",
        "",
        f"// Sim-tick parameters, frozen at codegen time.",
        f"inline constexpr float DT             = {_f(world.dt)};",
        f"inline constexpr float SIM_RATE_MULT  = {_f(world.sim_rate_mult)};",
        "",
        f"// World + scene. `scene` is null until `setup()` runs.",
        f"extern manta::World    w;",
        f"extern manta::Scene*   scene;",
        "",
    ]

    # Planet pointers — populated by setup() since the World owns the storage.
    if world.planets:
        lines.append("// Planet handles (point into w's planet storage; valid after setup()).")
        for i, p in enumerate(world.planets):
            lines.append(f"extern {p.cpp_class}* planet_{i};")
        lines.append("")

    # Field instances — owned at namespace scope.
    if world.fields:
        lines.append("// Registered fields. setup() populates disturbances + registers with w.")
        for i, f in enumerate(world.fields):
            lines.append(f"extern {f.cpp_class} field_{i};")
        lines.append("")

    # Craft instances — owned at namespace scope; setup() adds them to scene.
    lines.append("// Craft instance(s). setup() adds them to the scene.")
    for idx, entry in enumerate(world.crafts):
        cls = class_name_for_craft(entry.craft.name)
        var = _craft_var_name(world, idx)
        lines.append(f"extern {cls} {var};")
    lines.append("")

    # Tether storage — at namespace scope so endpoints can reference them.
    if world.tethers:
        lines.append("// Tethers (the C++ Tether storage; endpoints reference these).")
        for ti, te in enumerate(world.tethers):
            lines.append(f"extern manta::coupling::Tether tether_{ti};")
        lines.append("")

    lines += [
        "// One-time initialization. Builds the scene, registers planets +",
        "// fields, adds crafts, declares Zenoh pubs/subs. Call once before",
        "// the tick loop.",
        "void setup();",
        "",
        "// One simulation step: applies pending in-bindings, runs",
        "// w.step() to advance physics, executes in-process connect()",
        "// links, and (every ~50 Hz) publishes out-bindings.",
        "void tick();",
        "",
        "// Tear down Zenoh state. Must be called before the program exits;",
        "// destroying the Zenoh session at static-destruction time blows up",
        "// inside the Tokio runtime. The codegen-emitted main calls this",
        "// before returning from main().",
        "void shutdown();",
        "",
        "// Polymorphic adapter: implements `manta::Harness` by delegating",
        "// to the free functions above. Use this when passing the harness",
        "// through a `manta::Harness*` interface (composition, plugin",
        "// dispatch, generic test rigs). Hot-path code should call the",
        "// free functions directly to avoid virtual dispatch.",
        "struct Harness : public manta::Harness {",
        "    void setup()    override;",
        "    void tick()     override;",
        "    void shutdown() override;",
        "};",
        "extern Harness harness;",
        "",
        f"}}  // namespace manta_gen::{name}",
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# .cpp emit

def emit_world_cpp(world: World) -> str:
    """Emit `<world>.cpp` — definitions + setup()/tick() bodies."""
    if not world.crafts:
        raise RuntimeError("emit_world_cpp: World has no crafts")

    name = world.name
    bind_assignments = _build_bind_assignments(world)
    sync_field_idxs = [i for i, f in enumerate(world.fields)
                       if getattr(f, "synchronized", False)]

    lines: list[str] = [
        GENERATED_BANNER, "",
        f'#include "{name}.hpp"',
        "",
        "#include <cstdint>",
        "#include <cstdio>",
        "#include <cstdlib>",
        "#include <cstring>",
        "#include <mutex>",
        "#include <optional>",
        "#include <string>",
        "#include <string_view>",
        "#include <vector>",
        "",
        "#include <zenoh.hxx>",
        "",
        f"namespace manta_gen::{name} {{",
        "",
        "manta::World    w{};",
        "manta::Scene*   scene = nullptr;",
    ]

    for i, p in enumerate(world.planets):
        lines.append(f"{p.cpp_class}* planet_{i} = nullptr;")
    for i, f in enumerate(world.fields):
        # Default-construct at namespace scope; setup() populates disturbances.
        # Field descriptors emit a single-line construction stmt — for
        # namespace-scope use it as the definition's initializer.
        construction = f.emit_construction(f"field_{i}")
        # construction looks like:  manta::fields::FooField field_<i>{ctor};
        # Prepend nothing — it's a complete definition statement.
        lines.append(construction)

    for idx, entry in enumerate(world.crafts):
        cls = class_name_for_craft(entry.craft.name)
        var = _craft_var_name(world, idx)
        lines.append(f"{cls} {var}{{}};")

    for ti, te in enumerate(world.tethers):
        t = te.tether
        lines.append(
            f"manta::coupling::Tether tether_{ti}("
            f"{_f(t.rest_length)}, {_f(t.stiffness)}, {_f(t.damping)});")

    lines += [
        "",
        f"}}  // namespace manta_gen::{name}",
        "",
    ]

    # ---- Anonymous-namespace file-private state for Zenoh + bindings ----
    lines += [
        "namespace {",
        "",
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
        "",
        "std::optional<zenoh::Session> g_session;",
        "",
    ]

    # Per-binding state.
    for bid, _, b in bind_assignments:
        if b.direction == "in":
            lines += [
                f"std::mutex bind_{bid}_mtx;",
                f"std::vector<float> bind_{bid}_payload;",
                f"std::optional<zenoh::Subscriber<void>> bind_{bid}_sub;",
            ]
        else:  # out
            lines.append(f"std::optional<zenoh::Publisher> pub_{bid};")
    if bind_assignments:
        lines.append("")

    # Field-sync state.
    for i in sync_field_idxs:
        lines += [
            f"std::optional<zenoh::Publisher>          pub_field_{i};",
            f"std::optional<zenoh::Subscriber<void>>   sub_field_{i};",
        ]
    if sync_field_idxs:
        lines.append("")

    # Publish decimation counter.
    lines += [
        "int g_pub_decim = 0;",
        "constexpr int kPubEvery = 20;  // ~50 Hz publish",
        "",
        "}  // anonymous namespace",
        "",
    ]

    # ---- setup() ----
    lines += [
        f"namespace manta_gen::{name} {{",
        "",
        "void setup() {",
        f"    w.clock().set_dt(DT);",
        f"    scene = &w.create_scene();",
    ]

    for i, p in enumerate(world.planets):
        lines.append(
            f"    planet_{i} = &w.add_planet<{p.cpp_class}>("
            f"{p.emit_constructor_args()});")
    if world.planets:
        lines.append("    scene->set_planet(planet_0);")

    # Field setup: extras + register.
    for i, f in enumerate(world.fields):
        var = f"field_{i}"
        for stmt in (f.emit_extra_setup(var) if hasattr(f, "emit_extra_setup") else []):
            lines.append(f"    {stmt}")
        lines.append(f"    w.register_field({var});")
        for base in getattr(f, "register_as", []) or []:
            lines.append(f"    w.register_field<{base}>({var});")

    # Tether endpoints. Endpoints attach to the craft's root post-construction.
    if world.tethers:
        craft_idx_of = {id(e.craft): ci for ci, e in enumerate(world.crafts)}
        affected_crafts: set[int] = set()
        for ti, te in enumerate(world.tethers):
            for ep, is_a in ((te.endpoint_a, True), (te.endpoint_b, False)):
                ep_craft, ep_name = ep
                ci = craft_idx_of[id(ep_craft)]
                affected_crafts.add(ci)
                cv = _craft_var_name(world, ci)
                lines.append(
                    f"    {cv}.root().add<manta::parts::TetherEndpoint>("
                    f'"{ep_name}", tether_{ti}, {"true" if is_a else "false"});')
        for ci in sorted(affected_crafts):
            cv = _craft_var_name(world, ci)
            lines.append(f"    {cv}.root().compute_params();")

    # Add crafts to scene.
    for idx, entry in enumerate(world.crafts):
        cv = _craft_var_name(world, idx)
        lines.append(
            f"    scene->add_craft({cv}, {_initial_state_literal(entry)});")

    # Zenoh session.
    lines += [
        "",
        "    g_session.emplace(zenoh::Session::open(zenoh::Config::create_default()));",
        "",
    ]

    # Subscriber / publisher declarations.
    for bid, _, b in bind_assignments:
        if b.direction == "in":
            _emit_subscriber_setup(lines, bid, b)
        else:
            lines.append(
                f"    pub_{bid}.emplace(g_session->declare_publisher("
                f"zenoh::KeyExpr({_quote(b.topic)})));")

    for i in sync_field_idxs:
        topic = world.fields[i].sync_topic or f"manta/{name}/field_{i}/disturbance"
        _emit_field_sync_setup(lines, i, topic, world.fields[i].cpp_class)

    lines += [
        "}",
        "",
    ]

    # ---- tick() ----
    multi = len(world.crafts) > 1
    craft_index_by_id = {id(e.craft): ci for ci, e in enumerate(world.crafts)}

    lines += ["void tick() {"]

    # Apply in-bindings.
    for bid, cidx, b in bind_assignments:
        if b.direction == "in":
            _emit_binding_apply(lines, bid, b,
                                craft_var=_craft_var_name(world, cidx))

    lines += [
        "",
        "    w.step();",
        "",
    ]

    # Connect steps run after w.step() so propagated values are fresh.
    # Only intra-world connects belong here — cross-world ones (e.g. ex5's
    # sim.dvl → est.dvl) defer to a containing composed harness, which
    # walks both worlds' connection lists.
    intra_world_ids = {id(e.craft) for e in world.crafts}
    for conn in world.connections:
        if id(conn.sink.craft_ref) not in intra_world_ids:
            continue   # cross-world; defer to outer composed harness
        _emit_connection_step(lines, conn, craft_index_by_id, world)

    # Decimated publish.
    if any(b.direction == "out" for _, _, b in bind_assignments):
        lines += [
            "    if (++g_pub_decim >= kPubEvery) {",
            "        g_pub_decim = 0;",
        ]
        for bid, cidx, b in bind_assignments:
            if b.direction == "out":
                _emit_binding_publish(lines, bid, b,
                                      craft_var=_craft_var_name(world, cidx))
        lines.append("    }")

    lines += [
        "}",
        "",
        "void shutdown() {",
        "    // Reset Zenoh handles in reverse-init order: per-binding pubs",
        "    // and subs first, then field-sync handles, then the session.",
        "    // The thin main calls this before returning so destruction",
        "    // happens while the runtime is still live.",
    ]
    for bid, _, b in bind_assignments:
        if b.direction == "in":
            lines.append(f"    bind_{bid}_sub.reset();")
        else:
            lines.append(f"    pub_{bid}.reset();")
    for i in sync_field_idxs:
        lines.append(f"    sub_field_{i}.reset();")
        lines.append(f"    pub_field_{i}.reset();")
    lines += [
        "    g_session.reset();",
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


# ---------------------------------------------------------------------------
# Thin main emit

def emit_world_main_cpp(world: World) -> str:
    """Emit `<world>_main.cpp` — thin binary main: sets up signal handlers,
    calls `setup()` once, ticks in a paced loop. All real work lives in the
    harness's `<world>.cpp`."""
    name = world.name
    n_crafts = len(world.crafts)
    n_bindings = len(world.bindings)

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
        f'    std::printf("{name}: ready. {n_crafts} craft(s), '
        f'{n_bindings} binding(s).\\n");',
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


# Back-compat alias — emit_config still calls emit_main_cpp.
def emit_main_cpp(world: World) -> str:
    """Legacy entry point. Use emit_world_hpp/cpp/main instead."""
    return emit_world_main_cpp(world)


# ---------------------------------------------------------------------------
# Per-binding apply / publish helpers — used by tick() body emission.

def _emit_subscriber_setup(lines: list[str], i: int, b: Binding) -> None:
    """Inside setup(): declare a Zenoh subscriber that writes the parsed
    payload into the matching anonymous-namespace `bind_<i>_payload` slot."""
    lines += [
        f"    bind_{i}_sub.emplace(g_session->declare_subscriber(",
        f"        zenoh::KeyExpr({_quote(b.topic)}),",
        f"        [](const zenoh::Sample& s) {{",
        f"            std::vector<float> v;",
        f"            std::string payload(s.get_payload().as_string());",
        f"            if (parse_float_array(payload, v) && v.size() >= {b.total_floats}) {{",
        f"                std::lock_guard<std::mutex> lk(bind_{i}_mtx);",
        f"                bind_{i}_payload = std::move(v);",
        f"            }}",
        f"        }}, zenoh::closures::none));",
    ]


def _emit_field_sync_setup(lines: list[str], i: int, topic: str, cpp_class: str) -> None:
    """Inside setup(): declare the Zenoh tx/rx pair for a synchronized field
    + install the field's tx_hook so disturbance adds replicate over Zenoh."""
    lines += [
        f"    pub_field_{i}.emplace(g_session->declare_publisher("
        f"zenoh::KeyExpr({_quote(topic)})));",
        f"    field_{i}.set_tx_hook(",
        f"        [](std::uint16_t tag, const manta::fields::Params& params, int lifetime) {{",
        f"            std::vector<std::uint8_t> buf;",
        f"            buf.resize(2 + 2 + 4 + params.size());",
        f"            std::uint16_t ver = 1;",
        f"            std::memcpy(buf.data() + 0, &ver,      2);",
        f"            std::memcpy(buf.data() + 2, &tag,      2);",
        f"            std::memcpy(buf.data() + 4, &lifetime, 4);",
        f"            std::memcpy(buf.data() + 8, params.data(), params.size());",
        f"            pub_field_{i}->put(zenoh::Bytes(std::move(buf)));",
        f"        }});",
        f"    sub_field_{i}.emplace(g_session->declare_subscriber(",
        f"        zenoh::KeyExpr({_quote(topic)}),",
        f"        [](const zenoh::Sample& s) {{",
        f"            auto payload = s.get_payload().as_vector();",
        f"            if (payload.size() < 8 + manta::fields::kParamsBytes) return;",
        f"            std::uint16_t ver = 0, tag = 0;",
        f"            std::int32_t  lifetime = 0;",
        f"            std::memcpy(&ver,      payload.data() + 0, 2);",
        f"            std::memcpy(&tag,      payload.data() + 2, 2);",
        f"            std::memcpy(&lifetime, payload.data() + 4, 4);",
        f"            if (ver != 1) return;",
        f"            manta::fields::Params p{{}};",
        f"            std::memcpy(p.data(), payload.data() + 8, p.size());",
        f"            field_{i}.receive(tag, p, lifetime);",
        f"        }}, zenoh::closures::none));",
    ]


def _emit_connection_step(lines: list[str], conn,
                          craft_index_by_id: dict[int, int],
                          world: World) -> None:
    """Emit one tick's worth of in-process signal-to-signal copy."""
    src = conn.source
    snk = conn.sink
    src_var = _craft_var_name(world, craft_index_by_id[id(src.craft_ref)])
    snk_var = _craft_var_name(world, craft_index_by_id[id(snk.craft_ref)])
    src_acc = _accessor_for_with_var(src, src_var)
    snk_acc = _accessor_for_with_var(snk, snk_var)
    n = src.signal.n_floats
    lines.append(f"    // connect: {src.part_name}.{src.name} → "
                 f"{snk.part_name}.{snk.name}")
    lines.append("    {")
    for i in range(n):
        expr = src.signal.cpp_read_exprs[i].format(accessor=src_acc)
        lines.append(f"        const float v{i} = {expr};")
    fmt = {"accessor": snk_acc}
    for i in range(n):
        fmt[f"v{i}"] = f"v{i}"
    lines.append(f"        {snk.signal.cpp_write_stmt.format(**fmt)}")
    lines.append("    }")


def _accessor_for_with_var(sig, craft_var: str) -> str:
    """Like accessor_for() but with a configurable C++ craft variable name."""
    base = accessor_for(sig)   # "craft" or "craft.<part>()"
    if base == "craft":
        return craft_var
    assert base.startswith("craft.")
    return craft_var + base[len("craft"):]


def _emit_binding_apply(lines: list[str], i: int, b: Binding,
                        craft_var: str = "craft") -> None:
    """Inside tick(): under the binding's mutex, apply each member's
    cpp_write_stmt with payload values bound to v0..v{n-1}."""
    lines += [
        f"    {{ std::lock_guard<std::mutex> lk(bind_{i}_mtx);",
        f"      if (bind_{i}_payload.size() >= {b.total_floats}) {{",
    ]
    cursor = 0
    for member_name, sig in b.members.items():
        n = sig.signal.n_floats
        accessor = _accessor_for_with_var(sig, craft_var)
        v_subs = {f"v{k}": f"bind_{i}_payload[{cursor + k}]" for k in range(n)}
        v_subs["accessor"] = accessor
        stmt = sig.signal.cpp_write_stmt.format(**v_subs)
        lines.append(f"          {stmt}    // member: {member_name}")
        cursor += n
    lines += [
        f"          bind_{i}_payload.clear();",
        f"      }} }}",
    ]


def _emit_binding_publish(lines: list[str], i: int, b: Binding,
                          craft_var: str = "craft") -> None:
    """Inside the decimated publish block: serialize the binding's read
    expressions and put() onto its publisher."""
    if b.encoding != "json":
        raise NotImplementedError(
            f"Binding {b.topic!r}: encoding {b.encoding!r} not yet supported.")

    lines.append(f"        {{ std::string _json = \"{{\";")
    first_member = True
    for member_name, sig in b.members.items():
        n = sig.signal.n_floats
        accessor = _accessor_for_with_var(sig, craft_var)
        if not first_member:
            lines.append('          _json += ",";')
        first_member = False
        if n == 1:
            cpp_expr = sig.signal.cpp_read_exprs[0].format(accessor=accessor)
            lines.append(f'          _json += "\\"{member_name}\\":";')
            lines.append(
                f"          {{ char _b[32]; std::snprintf(_b, sizeof(_b), \"%g\", "
                f"double({cpp_expr})); _json += _b; }}"
            )
        else:
            lines.append(f'          _json += "\\"{member_name}\\":[";')
            for k, expr in enumerate(sig.signal.cpp_read_exprs):
                cpp_expr = expr.format(accessor=accessor)
                sep = '","' if k > 0 else '""'
                lines.append(
                    f"          {{ char _b[32]; std::snprintf(_b, sizeof(_b), \"%s%g\", "
                    f"{sep}, double({cpp_expr})); _json += _b; }}"
                )
            lines.append('          _json += "]";')
    lines += [
        '          _json += "}";',
        f"          pub_{i}->put(zenoh::Bytes(_json));",
        "        }",
    ]


# Public re-exports for sim_ekf_main / ekf_main, which still pull in some
# helpers (subscriber emit, field-sync emit, the quoting helper, the unique-
# craft walker, `_initial_state_literal`).
_emit_field_sync = _emit_field_sync_setup       # back-compat name
_emit_binding_subscriber = _emit_subscriber_setup

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
CraftEKF, with `register_field` propagating any field instances the
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
from ..signal import Binding, accessor_for, is_craft_signal
from ._util import GENERATED_BANNER, class_name_for_craft
from .ekf_main import (_MeasFunctor, _double_qualify_partframe,
                        _filter_concrete_craft_type, _filter_construction,
                        _first_mag_field_var, _is_filter_signal)
from .main import _emit_binding_subscriber, _emit_field_sync, _quote


def _initial_state_literal(entry) -> str:
    """C++ literal for `manta::InitialState{...}` from a _CraftEntry."""
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


class _AccessorCtx:
    """Tracks per-target accessor mappings: craft → variable name and
    which mapping needs the EKF-double scalar substitution applied."""

    def __init__(self):
        self.craft_id_to_var: dict[int, str] = {}
        self.craft_id_is_templated: dict[int, bool] = {}
        self.ekf_var: str | None = None

    def add_sim_craft(self, craft, var: str) -> None:
        self.craft_id_to_var[id(craft)] = var
        self.craft_id_is_templated[id(craft)] = False

    def add_ekf(self, ekf, ekf_var: str) -> None:
        """Register an EKF/UKF wrapper. Bindings targeting the wrapped
        craft route through `<filter_var>.craft().<part>()`. The
        `Vec3<PartFrame> -> Vec3<PartFrame, double>` patch only applies
        when the wrapped craft is scalar-templated (the EKF case, or a
        UKF wrapping a templated craft); plain non-templated UKF crafts
        already use Real-typed signals."""
        self.ekf_var = ekf_var
        wrapped = ekf.world.crafts[0].craft
        self.craft_id_to_var[id(wrapped)] = f"{ekf_var}.craft()"
        self.craft_id_is_templated[id(wrapped)] = bool(
            getattr(wrapped, "scalar_templated", False))

    def accessor_for(self, sig) -> tuple[str, bool]:
        """Returns (accessor_expr, is_templated). `is_templated` is True
        when the underlying craft is the est (double-instantiated) one
        — signals targeting it need the Vec3<PartFrame, double> patch."""
        if _is_filter_signal(sig):
            return self.ekf_var, False
        var = self.craft_id_to_var[id(sig.craft_ref)]
        templated = self.craft_id_is_templated[id(sig.craft_ref)]
        if is_craft_signal(sig):
            return var, templated
        # Craft-rooted accessor — sig.part_name is the C++ accessor method.
        return f"{var}.{sig.part_name}()", templated

    def write_stmt(self, sig, fmt_args: dict) -> str:
        """Format a sig.cpp_write_stmt and apply the templated patch when
        the receiver is the est-side (double-typed) craft."""
        accessor, templated = self.accessor_for(sig)
        fmt_args = dict(fmt_args)
        fmt_args["accessor"] = accessor
        stmt = sig.signal.cpp_write_stmt.format(**fmt_args)
        if templated:
            stmt = _double_qualify_partframe(stmt)
        return stmt


def emit_sim_plus_ekf_main_cpp(target, sim_world, filter_obj,
                                kind: str = "ekf") -> str:
    """Emit main.cpp for a Target whose drives are [sim_world, filter].

    `kind` selects EKF (CraftEKF, requires scalar_templated craft) vs UKF
    (CraftUKFOf, takes any craft). Most of the body is filter-agnostic;
    the wrapper-type/ctor differences come from
    `_filter_construction` and `_filter_concrete_craft_type`.
    """

    if len(sim_world.crafts) != 1:
        raise NotImplementedError(
            "emit_sim_plus_filter_main_cpp: only single-craft sim worlds are "
            "supported in this build (multi-craft sim + filter lands later).")
    if len(filter_obj.world.crafts) != 1:
        raise NotImplementedError(
            "emit_sim_plus_filter_main_cpp: only single-craft est worlds are "
            "supported.")

    sim_craft = sim_world.crafts[0].craft
    sim_craft_var = f"sim_{sim_craft.name}"
    est_craft_obj = filter_obj.world.crafts[0].craft
    est_craft_type = _filter_concrete_craft_type(kind, est_craft_obj)
    filter_var = filter_obj.cpp_var_name()

    ctx = _AccessorCtx()
    ctx.add_sim_craft(sim_craft, sim_craft_var)
    ctx.add_ekf(filter_obj, filter_var)

    # Field instances are constructed in main(); the same descriptor object
    # may appear in both worlds, in which case it shares one C++ var.
    field_var_for_id = _plan_field_vars(sim_world, filter_obj.world)
    mag_field_var = (_first_mag_field_var(sim_world, field_var_for_id)
                     or _first_mag_field_var(filter_obj.world, field_var_for_id))

    # Per-sensor measurement functors (one per part in filter.measurements).
    meas_specs = []
    for m in filter_obj.measurements:
        part_var = f"{filter_var}.craft().{m.name}()"
        spec = _MeasFunctor.for_part(m, filter_var, part_var,
                                     mag_field_var=mag_field_var)
        meas_specs.append((m, spec))

    meas_dim = sum(s.n_floats for _, s in meas_specs) if meas_specs else 1

    filter_header, filter_ctor = _filter_construction(
        kind, filter_obj, est_craft_type, filter_var, meas_dim)

    # Build the file.
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
        "#include <Eigen/Core>",
        "#include <Eigen/Geometry>",
        "#include <zenoh.hxx>",
        "",
        "#include \"manta/core/scene.hpp\"",
        "#include \"manta/core/world.hpp\"",
        filter_header,
    ]
    lines.append(f'#include "{sim_craft.name}.hpp"')
    if est_craft_obj.name != sim_craft.name:
        lines.append(f'#include "{est_craft_obj.name}.hpp"')
    seen_headers: set[str] = set()
    for f in list(sim_world.fields) + list(filter_obj.world.fields):
        if f.cpp_header not in seen_headers:
            seen_headers.add(f.cpp_header)
            lines.append(f'#include "{f.cpp_header}"')
    for p in sim_world.planets:
        lines.append(f'#include "{p.cpp_header}"')
    if sim_world.tethers or filter_obj.world.tethers:
        lines.append('#include "manta/coupling/tether.hpp"')
        lines.append('#include "manta/parts/coupling/tether_endpoint.hpp"')
    if filter_obj.world.planets:
        raise NotImplementedError(
            "emit_sim_plus_filter_main_cpp: planets in the filter's wrapped "
            "world aren't supported (CraftEKF/UKF don't own a Scene). "
            "Register fields directly via World.fields on the est-side world.")

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
    ]

    # Measurement-functor decls at file scope.
    for m, spec in meas_specs:
        for ln in spec.body.rstrip("\n").split("\n"):
            lines.append(ln)
        lines.append("")

    lines += [
        "int main() {",
        "    std::signal(SIGINT,  on_signal);",
        "    std::signal(SIGTERM, on_signal);",
        "",
        f"    constexpr float DT             = {_f(target.dt)};",
        f"    constexpr float SIM_RATE_MULT  = {_f(target.sim_rate_mult)};",
        f"    const     float WALL_PERIOD    = DT / SIM_RATE_MULT;",
        "",
        "    // ---- Sim world ----",
        "    manta::World sim_w;",
        "    sim_w.clock().set_dt(DT);",
        "    auto& sim_scene = sim_w.create_scene();",
    ]

    # Sim planets.
    for i, p in enumerate(sim_world.planets):
        var = f"sim_planet_{i}"
        lines.append(
            f"    auto& {var} = sim_w.add_planet<{p.cpp_class}>("
            f"{p.emit_constructor_args()});")
    if sim_world.planets:
        lines.append("    sim_scene.set_planet(&sim_planet_0);")

    # Sim fields. The same descriptor object may also appear in the est
    # world; reuse the sim var via the precomputed `field_var_for_id` dict
    # rather than re-constructing.
    sync_field_idxs: list[int] = []
    for i, f in enumerate(sim_world.fields):
        var = field_var_for_id[id(f)]
        lines.append(f"    {f.emit_construction(var)}")
        for stmt in (f.emit_extra_setup(var) if hasattr(f, "emit_extra_setup") else []):
            lines.append(f"    {stmt}")
        lines.append(f"    sim_w.register_field({var});")
        for base in getattr(f, "register_as", []) or []:
            lines.append(f"    sim_w.register_field<{base}>({var});")
        if getattr(f, "synchronized", False):
            sync_field_idxs.append(i)

    # Sim craft instantiation + add to scene.
    cls = class_name_for_craft(sim_craft.name)
    lines.append(f"    {cls} {sim_craft_var};")
    lines.append(
        f"    sim_scene.add_craft({sim_craft_var}, "
        f"{_initial_state_literal(sim_world.crafts[0])});")

    lines += [
        "",
        f"    // ---- {kind.upper()} ----",
        f"    {filter_ctor}",
        f"    using EkfT = decltype({filter_var});",
        "    EkfT::StateVec x0 = EkfT::StateVec::Zero();",
        "    x0(3) = 1.0;     // identity quaternion w",
        f"    EkfT::StateCov P0 = EkfT::StateCov::Identity() * "
        f"{_f(filter_obj.initial_covariance)};",
        f"    EkfT::StateCov Q  = EkfT::StateCov::Identity() * "
        f"{_f(filter_obj.process_noise)};",
        f"    {filter_var}.set_state(x0);",
        f"    {filter_var}.set_covariance(P0);",
        "",
    ]

    # Construct est-only fields (those NOT in sim) and register all est-
    # needed fields on the filter. Var names come from `field_var_for_id`
    # so any MagField found by `_first_mag_field_var` resolves to the same
    # instance the codegen emits.
    sim_field_ids = {id(f) for f in sim_world.fields}
    for i, f in enumerate(filter_obj.world.fields):
        var = field_var_for_id[id(f)]
        if id(f) not in sim_field_ids:
            lines.append(f"    {f.emit_construction(var)}")
            for stmt in (f.emit_extra_setup(var) if hasattr(f, "emit_extra_setup") else []):
                lines.append(f"    {stmt}")
        lines.append(f"    {filter_var}.register_field({var});")
        for base in getattr(f, "register_as", []) or []:
            lines.append(f"    {filter_var}.template register_field<{base}>({var});")

    # Per-sensor R blocks.
    for m, spec in meas_specs:
        diag = spec.sigma_squared_diag(m)
        n = spec.n_floats
        rname = f"R_{m.name}"
        lines.append(f"    Eigen::Matrix<double, {n}, {n}> {rname} = "
                     f"Eigen::Matrix<double, {n}, {n}>::Zero();")
        for i, v in enumerate(diag):
            lines.append(f"    {rname}({i}, {i}) = {_f(v)};")
        lines.append("")

    lines += [
        "    zenoh::Config cfg = zenoh::Config::create_default();",
        "    auto session = zenoh::Session::open(std::move(cfg));",
        "",
    ]

    # Bindings: union of sim_world.bindings + filter.world.bindings.
    bind_assignments: list[tuple[int, "Binding"]] = []
    bid = 0
    for w in (sim_world, filter_obj.world):
        for b in w.bindings:
            bind_assignments.append((bid, b))
            bid += 1

    for bid, b in bind_assignments:
        if b.direction == "in":
            _emit_binding_subscriber(lines, bid, b)
    for bid, b in bind_assignments:
        if b.direction == "out":
            lines.append(
                f"    auto pub_{bid} = session.declare_publisher("
                f"zenoh::KeyExpr({_quote(b.topic)}));")

    # Field sync — only sim-side synchronized fields. The est world mirrors
    # the sim's field instances when they're identity-shared, so the rx
    # path on sim_w naturally updates the est-visible field state.
    for i in sync_field_idxs:
        topic = (sim_world.fields[i].sync_topic
                 or f"manta/{sim_world.name}/field_{i}/disturbance")
        _emit_field_sync(lines, i, topic, sim_world.fields[i].cpp_class)

    total_bindings = len(bind_assignments)
    kind_label = kind.upper()
    lines += [
        "",
        f'    std::printf("{target.name}: ready (sim+{kind_label}). 1 sim craft, '
        f'1 est craft, {total_bindings} binding(s), {len(meas_specs)} '
        f'measurement sensor(s).\\n");',
        "",
        "    auto next = std::chrono::steady_clock::now();",
        "    const auto period = std::chrono::microseconds(int64_t(WALL_PERIOD * 1e6));",
        "    int pub_decim = 0;",
        "    const int pub_every = 20;  // ~50 Hz publish",
        "",
        "    while (g_run.load()) {",
    ]

    # Apply input bindings (both worlds).
    for bid, b in bind_assignments:
        if b.direction == "in":
            _emit_input_binding_apply(lines, bid, b, ctx)

    lines += [
        "",
        "        sim_w.update();",
        "",
    ]

    # Cross-world + within-world connect() steps. Run AFTER sim_w.update()
    # so sensor outputs are fresh, BEFORE filter.predict() so the
    # set_measurement updates land first.
    all_connections = list(sim_world.connections) + list(filter_obj.world.connections)
    for conn in all_connections:
        _emit_connection_step(lines, conn, ctx)

    lines += [
        "",
        f"        {filter_var}.predict(DT, Q);",
        "",
    ]

    # Per-sensor consume_fresh + update_n. The spec emits its own block
    # (Magnetometer adds a pre-update MagField lookup; others use the
    # default form).
    for m, spec in meas_specs:
        rname = f"R_{m.name}"
        part_acc = f"{filter_var}.craft().{m.name}()"
        spec.emit_update_block(lines, filter_var, part_acc, rname)
    if meas_specs:
        lines.append("")

    lines += [
        "        if (++pub_decim >= pub_every) {",
        "            pub_decim = 0;",
    ]
    for bid, b in bind_assignments:
        if b.direction == "out":
            _emit_output_binding_publish(lines, bid, b, ctx)

    lines += [
        "        }",
        "",
        "        next += period;",
        "        std::this_thread::sleep_until(next);",
        "    }",
        "",
        f"    std::printf(\"{target.name}: shutting down.\\n\");",
        "    return 0;",
        "}",
        "",
    ]
    return "\n".join(lines)


def _emit_connection_step(lines: list[str], conn, ctx: _AccessorCtx) -> None:
    """Emit one tick's worth of in-process signal-to-signal copy. Reads
    `n_floats` source-component temps, writes them through the sink's
    cpp_write_stmt. Endpoints can live in different worlds (e.g.
    sim.dvl.last_velocity → est.dvl.set_measurement)."""
    src = conn.source
    snk = conn.sink
    src_acc, _ = ctx.accessor_for(src)
    n = src.signal.n_floats
    lines.append(f"        // connect: {src.part_name}.{src.name} → "
                 f"{snk.part_name}.{snk.name}")
    lines.append("        {")
    for i in range(n):
        expr = src.signal.cpp_read_exprs[i].format(accessor=src_acc)
        lines.append(f"            const double v{i} = double({expr});")
    fmt = {f"v{i}": f"v{i}" for i in range(n)}
    stmt = ctx.write_stmt(snk, fmt)
    lines.append(f"            {stmt}")
    lines.append("        }")


def _emit_input_binding_apply(lines: list[str], i: int, b: Binding,
                              ctx: _AccessorCtx) -> None:
    """Sub callback wrote into bind_<i>_payload; this lifts the payload into
    each member's cpp_write_stmt. Accessor resolves through ctx so signals
    targeting the est craft pick up the templated double form."""
    lines += [
        f"        {{ std::lock_guard<std::mutex> lk(bind_{i}_mtx);",
        f"          if (bind_{i}_payload.size() >= {b.total_floats}) {{",
    ]
    cursor = 0
    for member_name, sig in b.members.items():
        n = sig.signal.n_floats
        v_subs = {f"v{k}": f"bind_{i}_payload[{cursor + k}]" for k in range(n)}
        stmt = ctx.write_stmt(sig, v_subs)
        lines.append(f"              {stmt}    // member: {member_name}")
        cursor += n
    lines += [
        f"              bind_{i}_payload.clear();",
        f"          }} }}",
    ]


def _emit_output_binding_publish(lines: list[str], i: int, b: Binding,
                                 ctx: _AccessorCtx) -> None:
    if b.encoding != "json":
        raise NotImplementedError(
            f"Binding {b.topic!r}: encoding {b.encoding!r} not yet supported.")

    lines.append(f"            {{ std::string _json = \"{{\";")
    first_member = True
    for member_name, sig in b.members.items():
        n = sig.signal.n_floats
        accessor, _ = ctx.accessor_for(sig)
        if not first_member:
            lines.append('              _json += ",";')
        first_member = False
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


def _plan_field_vars(sim_world, est_world) -> dict[int, str]:
    """Plan C++ var names for every field across both worlds.

    Sim fields get `sim_field_<i>`. Est-only fields get `est_field_<i>`.
    A field descriptor that appears in BOTH worlds (Python identity)
    shares the sim var, so `register_field` on the EKF refers to the
    same C++ instance the sim is updating.
    """
    out: dict[int, str] = {}
    for i, f in enumerate(sim_world.fields):
        out[id(f)] = f"sim_field_{i}"
    for i, f in enumerate(est_world.fields):
        if id(f) in out:
            continue
        out[id(f)] = f"est_field_{i}"
    return out

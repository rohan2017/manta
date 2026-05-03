"""Emit <name>_main.cpp for an EKF Target.

Shape mirrors emit_main_cpp() but the tick loop is driven by a
`manta::estimation::CraftEKF<EstCraftT, MeasDim>` instance:

    ekf.predict(dt, Q);
    if (ekf.craft().dvl().consume_fresh()) {
        Eigen::Matrix<double, 3, 1> z = ...;
        ekf.template update_n<3>(DvlBodyVelMeas{}, z, R_dvl);
    }
    // ...for each measurement sensor

Subscribers and publishers reuse the same Zenoh wiring as emit_main_cpp:
input bindings write into `ekf.craft().<sensor>().set_measurement(...)`,
output bindings read from `ekf.craft().<part>().<getter>()` for craft-
rooted signals or from `ekf.<accessor>()(i)` for EKF-rooted signals.

Phase-A scope: only DVL is supported as a measurement. IMU and
Magnetometer raise NotImplementedError when listed in `ekf.measurements`
(they can still be wired as inputs via subscribe(...)).
"""

from __future__ import annotations

from .._format import cpp_float as _f
from ..signal import Binding, accessor_for
from ._util import GENERATED_BANNER, class_name_for_craft
from .main import (
    _emit_binding_subscriber,
    _emit_field_sync,
    _quote,
    _world_unique_crafts,
)


def _is_ekf_signal(b) -> bool:
    """A BoundSignal whose craft_ref is the EKF descriptor itself (not a Craft).
    Detected via the `$ekf/` part_name sentinel; reading craft_ref is also
    sufficient but the prefix check avoids the import cycle."""
    from ..estimation.ekf import EKF_SENTINEL_PREFIX
    return getattr(b, "part_name", "").startswith(EKF_SENTINEL_PREFIX)


def _accessor_for_ekf_or_craft(sig, craft_var: str, ekf_var: str) -> str:
    """Resolve {accessor} for a BoundSignal in an EKF target.

    Three cases:
      * EKF-rooted (sig.craft_ref is the EKF):   accessor = `ekf_var`.
      * Craft-rooted, $craft sentinel:           accessor = `ekf_var.craft()`.
      * Craft-rooted, named part:                accessor = `ekf_var.craft().<part>()`.
    """
    if _is_ekf_signal(sig):
        return ekf_var
    base = accessor_for(sig)   # "craft" or "craft.<part>()"
    if base == "craft":
        return craft_var
    assert base.startswith("craft.")
    return craft_var + base[len("craft"):]


# ---------------------------------------------------------------------------
# Per-sensor measurement-functor codegen.
#
# Each entry maps a sensor's PartDescriptor.cpp_class_template → (n_floats,
# functor C++ definition, code that reads `z` from the part). The functor
# is templated on the scalar so the EKF can run it through both double and
# Jet evaluations.

class _MeasFunctor:
    """Bundle of (n_floats, functor decl, z-read) for one sensor type.
    Functor name is f"{ekf_var}_{part_name}_meas".
    """
    def __init__(self, n_floats: int, body: str, z_read: list[str]):
        self.n_floats = n_floats
        self.body = body
        self.z_read = z_read   # list of n_floats C++ expressions returning double

    @classmethod
    def for_part(cls, part, ekf_var: str, part_var: str) -> "_MeasFunctor":
        """Dispatch by part type. Phase-A only supports DVL.

        `part_var` is the C++ accessor for the sensor (e.g. `ekf_0.craft().dvl()`).
        """
        cct = getattr(type(part), "cpp_class_template", "")
        if cct == "manta::parts::DVLT":
            functor_name = f"_{ekf_var}_{part.name}_meas"
            body = (
                f"// DVL: predicted body-frame velocity = R(q)^T * v_scene.\n"
                f"struct {functor_name} {{\n"
                f"    template <class S>\n"
                f"    Eigen::Matrix<S, 3, 1> operator()(const Eigen::Matrix<S, 13, 1>& x) const {{\n"
                f"        Eigen::Quaternion<S> q(x(3), x(4), x(5), x(6));\n"
                f"        Eigen::Matrix<S, 3, 1> v_scene(x(7), x(8), x(9));\n"
                f"        return q.conjugate() * v_scene;\n"
                f"    }}\n"
                f"}};\n"
            )
            z_read = [
                f"{part_var}.last_velocity().raw()(0)",
                f"{part_var}.last_velocity().raw()(1)",
                f"{part_var}.last_velocity().raw()(2)",
            ]
            return cls(n_floats=3, body=body, z_read=z_read)
        raise NotImplementedError(
            f"EKF codegen phase A: only DVL is supported as a measurement; "
            f"got {type(part).__name__} ({part.name!r}). "
            f"IMU/Magnetometer measurement-update support lands in a follow-up.")

    def functor_typename(self, ekf_var: str, part_name: str) -> str:
        return f"_{ekf_var}_{part_name}_meas"

    def sigma_squared_diag(self, part) -> list[float]:
        """Diagonal of R for this sensor — reads the part's noise sigma fields."""
        cct = getattr(type(part), "cpp_class_template", "")
        if cct == "manta::parts::DVLT":
            s = float(getattr(part, "velocity_sigma", 0.0))
            return [s * s, s * s, s * s]
        raise NotImplementedError(cct)


def emit_ekf_main_cpp(target, ekf) -> str:
    """Emit the main.cpp for a Target whose driveable is an EKF.

    `target` is a `manifest.Target`. `ekf` is the EKF descriptor (the sole
    drive in target.drives).
    """
    world = ekf.world
    if not world.crafts:
        raise RuntimeError("emit_ekf_main_cpp: EKF's wrapped world has no crafts")
    if len(world.crafts) > 1:
        raise NotImplementedError(
            "emit_ekf_main_cpp: only single-craft worlds are supported in v1.")

    unique_crafts = _world_unique_crafts(world)
    primary = unique_crafts[0]
    primary_class_tmpl = class_name_for_craft(primary.name) + "T"

    ekf_var  = ekf.cpp_var_name()                # "ekf_<id>"
    craft_var = f"{ekf_var}.craft()"             # accessor for the wrapped craft
    meas_dim = sum(_MeasFunctor.for_part(m, ekf_var,
                                         f"{craft_var}.{m.name}()").n_floats
                   for m in ekf.measurements) if ekf.measurements else 1
    # CraftEKF<...> template still requires a positive MeasDim; if the user
    # only listed inputs (no updates), default to 1 — update_n<N> is what
    # actually drives every measurement, MeasDim is just a placeholder.
    if meas_dim < 1:
        meas_dim = 1

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
        "#include \"manta/estimation/craft_ekf.hpp\"",
    ]
    for c in unique_crafts:
        lines.append(f'#include "{c.name}.hpp"')
    for f in world.fields:
        lines.append(f'#include "{f.cpp_header}"')
    if world.planets:
        raise NotImplementedError(
            "emit_ekf_main_cpp: planets in an EKF-wrapped world aren't "
            "supported yet (CraftEKF doesn't own a Scene). Register the "
            "needed fields directly via World.fields for now.")

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

    # Per-sensor measurement functors emitted at file scope, keyed by ekf_var
    # + part name so multiple EKFs in one binary don't collide.
    meas_specs = []
    for m in ekf.measurements:
        part_var = f"{craft_var}.{m.name}()"
        spec = _MeasFunctor.for_part(m, ekf_var, part_var)
        meas_specs.append((m, spec))
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
        f"    manta::estimation::CraftEKF<{primary_class_tmpl}, {meas_dim}> {ekf_var};",
        "",
        f"    using EkfT = decltype({ekf_var});",
        "    EkfT::StateVec x0 = EkfT::StateVec::Zero();",
        "    x0(3) = 1.0;     // identity quaternion w",
        f"    EkfT::StateCov P0 = EkfT::StateCov::Identity() * {_f(ekf.initial_covariance)};",
        f"    EkfT::StateCov Q  = EkfT::StateCov::Identity() * {_f(ekf.process_noise)};",
        f"    {ekf_var}.set_state(x0);",
        f"    {ekf_var}.set_covariance(P0);",
        "",
    ]

    # Per-sensor R matrices.
    for m, spec in meas_specs:
        diag = spec.sigma_squared_diag(m)
        n = spec.n_floats
        rname = f"R_{m.name}"
        lines.append(f"    Eigen::Matrix<double, {n}, {n}> {rname} = "
                     f"Eigen::Matrix<double, {n}, {n}>::Zero();")
        for i, v in enumerate(diag):
            lines.append(f"    {rname}({i}, {i}) = {_f(v)};")
        lines.append("")

    # Field instances live with static storage in main; register on the EKF
    # (which forwards to both internal crafts).
    sync_field_idxs: list[int] = []
    for i, f in enumerate(world.fields):
        var = f"field_{i}"
        lines.append(f"    {f.emit_construction(var)}")
        for stmt in (f.emit_extra_setup(var) if hasattr(f, "emit_extra_setup") else []):
            lines.append(f"    {stmt}")
        lines.append(f"    {ekf_var}.register_field({var});")
        for base in getattr(f, "register_as", []) or []:
            lines.append(f"    {ekf_var}.template register_field<{base}>({var});")
        if getattr(f, "synchronized", False):
            sync_field_idxs.append(i)

    lines += [
        "    zenoh::Config cfg = zenoh::Config::create_default();",
        "    auto session = zenoh::Session::open(std::move(cfg));",
        "",
    ]

    # Bindings: subscribers + publishers, identical wiring to emit_main_cpp
    # but the craft variable for input/output accessor expansion is now
    # `ekf_<id>.craft()` (or `ekf_<id>` for EKF-rooted output signals).
    bind_assignments: list[tuple[int, "Binding"]] = []
    for i, b in enumerate(world.bindings):
        bind_assignments.append((i, b))

    for bid, b in bind_assignments:
        if b.direction == "in":
            _emit_binding_subscriber(lines, bid, b)
    for bid, b in bind_assignments:
        if b.direction == "out":
            lines.append(
                f"    auto pub_{bid} = session.declare_publisher("
                f"zenoh::KeyExpr({_quote(b.topic)}));")

    for i in sync_field_idxs:
        topic = world.fields[i].sync_topic or f"manta/{world.name}/field_{i}/disturbance"
        _emit_field_sync(lines, i, topic, world.fields[i].cpp_class)

    total_bindings = len(bind_assignments)
    lines += [
        "",
        f'    std::printf("{target.name}: ready (EKF). 1 craft, '
        f'{total_bindings} binding(s), {len(meas_specs)} measurement sensor(s).\\n");',
        "",
        "    auto next = std::chrono::steady_clock::now();",
        "    const auto period = std::chrono::microseconds(int64_t(WALL_PERIOD * 1e6));",
        "    int pub_decim = 0;",
        "    const int pub_every = 20;  // ~50 Hz publish",
        "",
        "    while (g_run.load()) {",
    ]

    # Apply input bindings — write into ekf.craft().<sensor>().set_measurement(...).
    for bid, b in bind_assignments:
        if b.direction == "in":
            _emit_input_binding_apply(lines, bid, b, craft_var, ekf_var)

    # Predict step.
    lines += [
        "",
        f"        {ekf_var}.predict(DT, Q);",
        "",
    ]

    # Per-sensor consume_fresh + update_n.
    for m, spec in meas_specs:
        n = spec.n_floats
        functor_t = spec.functor_typename(ekf_var, m.name)
        rname = f"R_{m.name}"
        part_acc = f"{craft_var}.{m.name}()"
        lines.append(f"        if ({part_acc}.consume_fresh()) {{")
        lines.append(f"            Eigen::Matrix<double, {n}, 1> z;")
        for i, expr in enumerate(spec.z_read):
            lines.append(f"            z({i}) = {expr};")
        lines.append(
            f"            {ekf_var}.template update_n<{n}>({functor_t}{{}}, z, {rname});"
        )
        lines.append("        }")
    if meas_specs:
        lines.append("")

    lines += [
        "        if (++pub_decim >= pub_every) {",
        "            pub_decim = 0;",
    ]

    for bid, b in bind_assignments:
        if b.direction == "out":
            _emit_output_binding_publish(lines, bid, b, craft_var, ekf_var)

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


# ---------------------------------------------------------------------------
# Binding apply/publish for EKF targets — accessor resolution differs from
# the world-target path (uses `ekf_<id>.craft()` and detects EKF-rooted
# output signals).

def _emit_input_binding_apply(lines: list[str], i: int, b: Binding,
                              craft_var: str, ekf_var: str) -> None:
    lines += [
        f"        {{ std::lock_guard<std::mutex> lk(bind_{i}_mtx);",
        f"          if (bind_{i}_payload.size() >= {b.total_floats}) {{",
    ]
    cursor = 0
    for member_name, sig in b.members.items():
        n = sig.signal.n_floats
        accessor = _accessor_for_ekf_or_craft(sig, craft_var, ekf_var)
        v_subs = {f"v{k}": f"bind_{i}_payload[{cursor + k}]" for k in range(n)}
        v_subs["accessor"] = accessor
        stmt = sig.signal.cpp_write_stmt.format(**v_subs)
        # The EKF wraps a scalar-templated craft instantiated with double.
        # Sensor signal cpp_write_stmts hardcode `Vec3<PartFrame>` (which
        # defaults to Real=float) — patch in the explicit double scalar so
        # the call resolves against the double-instantiated overload.
        stmt = _double_qualify_partframe(stmt)
        lines.append(f"              {stmt}    // member: {member_name}")
        cursor += n
    lines += [
        f"              bind_{i}_payload.clear();",
        f"          }} }}",
    ]


def _double_qualify_partframe(stmt: str) -> str:
    """Replace `Vec3<manta::PartFrame>` with `Vec3<manta::PartFrame, double>`
    so that sensor `set_measurement(...)` calls resolve against the
    double-instantiated estimator craft. Sensor signal cpp_write_stmts
    were authored for Real-scalared crafts; the EKF path needs the
    explicit scalar."""
    return stmt.replace(
        "manta::geom::Vec3<manta::PartFrame>",
        "manta::geom::Vec3<manta::PartFrame, double>")


def _emit_output_binding_publish(lines: list[str], i: int, b: Binding,
                                 craft_var: str, ekf_var: str) -> None:
    if b.encoding != "json":
        raise NotImplementedError(
            f"Binding {b.topic!r}: encoding {b.encoding!r} not yet supported "
            f"(only 'json' is implemented)")

    lines.append(f"            {{ std::string _json = \"{{\";")
    first_member = True
    for member_name, sig in b.members.items():
        n = sig.signal.n_floats
        accessor = _accessor_for_ekf_or_craft(sig, craft_var, ekf_var)
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

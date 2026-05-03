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

Phase-A scope: DVL, IMU, and Magnetometer measurement updates are
supported. Magnetometer uses a locally-constant-B approximation (B
captured from the registered MagField at update time; ∂h/∂q is exact,
∂h/∂p is dropped — fine when |∇B|·|δp| << |B| over a tick).
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


def _is_filter_signal(b) -> bool:
    """A BoundSignal whose craft_ref is an EKF or UKF descriptor (not a
    Craft). Detected via the part_name sentinel prefix; works for both
    filter kinds without import cycles."""
    from ..estimation.ekf import EKF_SENTINEL_PREFIX
    from ..estimation.ukf import UKF_SENTINEL_PREFIX
    pn = getattr(b, "part_name", "")
    return pn.startswith(EKF_SENTINEL_PREFIX) or pn.startswith(UKF_SENTINEL_PREFIX)


# Back-compat alias — historical callsites named this _is_ekf_signal.
_is_ekf_signal = _is_filter_signal


def _accessor_for_ekf_or_craft(sig, craft_var: str, ekf_var: str) -> str:
    """Resolve {accessor} for a BoundSignal in a filter target.

    Three cases:
      * Filter-rooted (sig.craft_ref is the EKF/UKF): accessor = `ekf_var`.
      * Craft-rooted, $craft sentinel:                accessor = `ekf_var.craft()`.
      * Craft-rooted, named part:                     accessor = `ekf_var.craft().<part>()`.

    `ekf_var` is the C++ var name for the filter (`ekf_<id>` or `ukf_<id>`);
    the parameter name is historical.
    """
    if _is_filter_signal(sig):
        return ekf_var
    base = accessor_for(sig)   # "craft" or "craft.<part>()"
    if base == "craft":
        return craft_var
    assert base.startswith("craft.")
    return craft_var + base[len("craft"):]


# ---------------------------------------------------------------------------
# Per-sensor measurement-functor codegen.

class _MeasFunctor:
    """Per-sensor measurement-functor codegen bundle.

    Carries (n_floats, file-scope functor decl, z-vector reads) and emits
    its own `if (consume_fresh()) { ... update_n<N>(...); }` block via
    `emit_update_block`. Most sensors use the simple form; Magnetometer
    captures the registered MagField's value at the current state position
    before each update (locally-constant-B approximation).
    """
    def __init__(self, n_floats: int, body: str, z_read: list[str],
                 functor_name: str,
                 mag_field_var: str | None = None):
        self.n_floats = n_floats
        self.body = body
        self.z_read = z_read
        self.functor_name = functor_name
        self.mag_field_var = mag_field_var   # set for magnetometer specs only

    @classmethod
    def for_part(cls, part, ekf_var: str, part_var: str,
                 mag_field_var: str | None = None) -> "_MeasFunctor":
        """Dispatch by part type. Supported: DVL, IMU, Magnetometer.

        `part_var` is the sensor's C++ accessor (e.g. `ekf_0.craft().dvl()`).
        `mag_field_var` is the C++ var name of the first registered MagField
        in the surrounding scope — required if Magnetometer is among the
        measurements; ignored otherwise.
        """
        cct = getattr(type(part), "cpp_class_template", "")
        functor_name = f"_{ekf_var}_{part.name}_meas"
        if cct == "manta::parts::DVLT":
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
            return cls(3, body, z_read, functor_name)
        if cct == "manta::parts::IMUT":
            body = (
                f"// IMU: predicted [accel_body=0; gyro_body=ω_body] under the\n"
                f"// no-net-force / free-flight assumption (est dynamics has\n"
                f"// no thrusters or gravity). gyro_body is a direct state\n"
                f"// slice; accel_body is zero. For est crafts with active\n"
                f"// forces, replace this functor.\n"
                f"struct {functor_name} {{\n"
                f"    template <class S>\n"
                f"    Eigen::Matrix<S, 6, 1> operator()(const Eigen::Matrix<S, 13, 1>& x) const {{\n"
                f"        Eigen::Matrix<S, 6, 1> z;\n"
                f"        z(0) = S(0); z(1) = S(0); z(2) = S(0);\n"
                f"        z(3) = x(10); z(4) = x(11); z(5) = x(12);\n"
                f"        return z;\n"
                f"    }}\n"
                f"}};\n"
            )
            z_read = [
                f"{part_var}.last_accel().raw()(0)",
                f"{part_var}.last_accel().raw()(1)",
                f"{part_var}.last_accel().raw()(2)",
                f"{part_var}.last_gyro().raw()(0)",
                f"{part_var}.last_gyro().raw()(1)",
                f"{part_var}.last_gyro().raw()(2)",
            ]
            return cls(6, body, z_read, functor_name)
        if cct == "manta::parts::MagnetometerT":
            if mag_field_var is None:
                raise RuntimeError(
                    f"EKF codegen: Magnetometer {part.name!r} requires a "
                    f"`MagField` registered on the EKF's wrapped world. Add "
                    f"`world.add_field(MagField()...)` (or wire one via "
                    f"`Earth(dipole_moment=...)`) so the codegen can query "
                    f"B at the current state position each update.")
            # Locally-constant-B approximation: B(p) is treated as constant
            # at update time — we read it once from the registered field at
            # the current best-estimate position, then h(x) just rotates by
            # R(q)^T. The Jacobian is exact in q (the dominant term) and
            # zero in p (the small-position-gradient simplification). For
            # most magnetometer applications |grad B|*|dp| << |B| over a
            # single tick, so the loss is in the noise.
            body = (
                f"// Magnetometer: predicted body-frame B = R(q)^T * B(p_now).\n"
                f"// B is captured at update-time from the registered MagField\n"
                f"// (locally-constant-B approximation: dh/dq is exact,\n"
                f"// dh/dp is dropped). Set b_scene_now before passing the\n"
                f"// functor instance to update_n<3>.\n"
                f"struct {functor_name} {{\n"
                f"    Eigen::Matrix<double, 3, 1> b_scene_now;\n"
                f"    template <class S>\n"
                f"    Eigen::Matrix<S, 3, 1> operator()(const Eigen::Matrix<S, 13, 1>& x) const {{\n"
                f"        Eigen::Quaternion<S> q(x(3), x(4), x(5), x(6));\n"
                f"        Eigen::Matrix<S, 3, 1> b;\n"
                f"        b(0) = S(b_scene_now(0));\n"
                f"        b(1) = S(b_scene_now(1));\n"
                f"        b(2) = S(b_scene_now(2));\n"
                f"        return q.conjugate() * b;\n"
                f"    }}\n"
                f"}};\n"
            )
            z_read = [
                f"{part_var}.last_b().raw()(0)",
                f"{part_var}.last_b().raw()(1)",
                f"{part_var}.last_b().raw()(2)",
            ]
            return cls(3, body, z_read, functor_name,
                       mag_field_var=mag_field_var)
        raise NotImplementedError(
            f"EKF codegen: no measurement-functor template for "
            f"{type(part).__name__} ({part.name!r}, cpp_class_template={cct!r}).")

    def functor_typename(self, ekf_var: str, part_name: str) -> str:
        return self.functor_name

    def sigma_squared_diag(self, part) -> list[float]:
        """Diagonal of R for this sensor — reads the part's noise sigma fields."""
        cct = getattr(type(part), "cpp_class_template", "")
        if cct == "manta::parts::DVLT":
            s = float(getattr(part, "velocity_sigma", 0.0))
            return [s * s, s * s, s * s]
        if cct == "manta::parts::IMUT":
            sa = float(getattr(part, "accel_sigma", 0.0))
            sg = float(getattr(part, "gyro_sigma", 0.0))
            return [sa * sa, sa * sa, sa * sa, sg * sg, sg * sg, sg * sg]
        if cct == "manta::parts::MagnetometerT":
            s = float(getattr(part, "sigma", 0.0))
            return [s * s, s * s, s * s]
        raise NotImplementedError(cct)

    def emit_update_block(self, lines: list[str], ekf_var: str,
                          part_var: str, rname: str,
                          indent: str = "        ") -> None:
        """Append the `if (consume_fresh()) { ... update_n<N>(...); }` block
        for this sensor. Magnetometer adds a pre-update position lookup +
        B capture step; everything else uses the default form."""
        n = self.n_floats
        lines.append(f"{indent}if ({part_var}.consume_fresh()) {{")
        if self.mag_field_var is not None:
            lines.append(f"{indent}    {self.functor_name} _h;")
            lines.append(f"{indent}    auto _p_d = {ekf_var}.position();")
            lines.append(f"{indent}    Eigen::Matrix<float, 3, 1> _p_f("
                         f"float(_p_d(0)), float(_p_d(1)), float(_p_d(2)));")
            lines.append(
                f"{indent}    auto _b_now = {self.mag_field_var}.state_at("
                f"manta::geom::Vec3<manta::SceneFrame>::from_raw(_p_f));"
            )
            lines.append(f"{indent}    _h.b_scene_now << "
                         f"double(_b_now.x()), double(_b_now.y()), double(_b_now.z());")
            lines.append(f"{indent}    Eigen::Matrix<double, {n}, 1> z;")
            for i, expr in enumerate(self.z_read):
                lines.append(f"{indent}    z({i}) = {expr};")
            lines.append(
                f"{indent}    {ekf_var}.template update_n<{n}>(_h, z, {rname});"
            )
        else:
            lines.append(f"{indent}    Eigen::Matrix<double, {n}, 1> z;")
            for i, expr in enumerate(self.z_read):
                lines.append(f"{indent}    z({i}) = {expr};")
            lines.append(
                f"{indent}    {ekf_var}.template update_n<{n}>("
                f"{self.functor_name}{{}}, z, {rname});"
            )
        lines.append(f"{indent}}}")


def _first_mag_field_var(world, var_for_id: dict[int, str] | None = None) -> str | None:
    """Return the C++ var name of the first MagField registered on `world`.

    `var_for_id` maps `id(field_descriptor) -> var_name` (used by sim+ekf
    paths where field vars are precomputed). When None, the EKF-only path
    naming `field_<i>` is assumed."""
    for i, f in enumerate(world.fields):
        if getattr(f, "cpp_class", "") == "manta::fields::MagField":
            if var_for_id is not None:
                return var_for_id.get(id(f), f"field_{i}")
            return f"field_{i}"
    return None


def _filter_construction(kind: str, filter_obj, primary_class_tmpl: str,
                         filter_var: str, meas_dim: int) -> tuple[str, str]:
    """Return (header_include_line, ctor_line) for the filter wrapper.

    EKF instantiates `CraftEKF<EstCraftT, MeasDim>` (template-template
    parameter — the templated craft class is passed verbatim). UKF
    uses `CraftUKFOf<EstCraft<double>, MeasDim>` so non-templated
    crafts work too — the codegen always picks the `<double>` form
    for a templated craft.
    """
    if kind == "ekf":
        return (
            "#include \"manta/estimation/craft_ekf.hpp\"",
            f"manta::estimation::CraftEKF<{primary_class_tmpl}, {meas_dim}> "
            f"{filter_var};",
        )
    if kind == "ukf":
        # CraftUKFOf takes a concrete type. Use Tmpl<double> when the craft
        # is scalar-templated; otherwise the class name is the concrete
        # craft itself (no <Scalar>).
        # We pass the templated form here because `class_name_for_craft + 'T'`
        # refers to the templated alias. For non-templated crafts the
        # caller passes the plain class name without the trailing 'T'.
        a, b, k = filter_obj.alpha, filter_obj.beta, filter_obj.kappa
        return (
            "#include \"manta/estimation/craft_ukf.hpp\"",
            f"manta::estimation::CraftUKFOf<{primary_class_tmpl}, {meas_dim}> "
            f"{filter_var}({_f(a)}, {_f(b)}, {_f(k)});",
        )
    raise ValueError(f"unknown filter kind {kind!r}")


def _filter_concrete_craft_type(kind: str, craft) -> str:
    """C++ type for the wrapped craft as seen by the filter wrapper.

    EKF wants the templated alias (`Ex5EstCraftT` — the Tpl<class>),
    UKF wants the concrete instantiation (`Ex5EstCraftT<double>` for
    templated crafts, plain `MyCraft` for non-templated).
    """
    base = class_name_for_craft(craft.name)
    if kind == "ekf":
        return base + "T"
    # UKF: concrete class.
    if getattr(craft, "scalar_templated", False):
        return base + "T<double>"
    return base


def emit_ekf_main_cpp(target, filter_obj, kind: str = "ekf") -> str:
    """Emit the main.cpp for a Target whose driveable is an EKF or UKF.

    `target` is a `manifest.Target`. `filter_obj` is the EKF/UKF
    descriptor (the sole drive in target.drives). `kind` selects the
    C++ wrapper type — kept as a string to keep emit_config's dispatch
    simple. Most of the body is filter-agnostic; the differences are
    confined to `_filter_construction` and `_filter_concrete_craft_type`.
    """
    world = filter_obj.world
    if not world.crafts:
        raise RuntimeError(
            f"emit_{kind}_main_cpp: filter's wrapped world has no crafts")
    if len(world.crafts) > 1:
        raise NotImplementedError(
            f"emit_{kind}_main_cpp: only single-craft worlds are supported in v1.")

    unique_crafts = _world_unique_crafts(world)
    primary = unique_crafts[0]
    primary_class = _filter_concrete_craft_type(kind, primary)
    templated_craft = bool(getattr(primary, "scalar_templated", False))

    filter_var = filter_obj.cpp_var_name()           # "ekf_<id>" or "ukf_<id>"
    craft_var = f"{filter_var}.craft()"

    mag_field_var = _first_mag_field_var(world)

    meas_specs: list[tuple[object, _MeasFunctor]] = []
    for m in filter_obj.measurements:
        part_var = f"{craft_var}.{m.name}()"
        spec = _MeasFunctor.for_part(m, filter_var, part_var,
                                     mag_field_var=mag_field_var)
        meas_specs.append((m, spec))

    meas_dim = sum(s.n_floats for _, s in meas_specs) if meas_specs else 1
    if meas_dim < 1:
        meas_dim = 1

    filter_header, filter_ctor = _filter_construction(
        kind, filter_obj, primary_class, filter_var, meas_dim)

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
        filter_header,
    ]
    for c in unique_crafts:
        lines.append(f'#include "{c.name}_craft.hpp"')
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

    # Per-sensor measurement functors emitted at file scope.
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
        f"    {filter_ctor}",
        "",
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

    sync_field_idxs: list[int] = []
    for i, f in enumerate(world.fields):
        var = f"field_{i}"
        lines.append(f"    {f.emit_construction(var)}")
        for stmt in (f.emit_extra_setup(var) if hasattr(f, "emit_extra_setup") else []):
            lines.append(f"    {stmt}")
        lines.append(f"    {filter_var}.register_field({var});")
        for base in getattr(f, "register_as", []) or []:
            lines.append(f"    {filter_var}.template register_field<{base}>({var});")
        if getattr(f, "synchronized", False):
            sync_field_idxs.append(i)

    lines += [
        "    zenoh::Config cfg = zenoh::Config::create_default();",
        "    auto session = zenoh::Session::open(std::move(cfg));",
        "",
    ]

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
    kind_label = kind.upper()
    lines += [
        "",
        f'    std::printf("{target.name}: ready ({kind_label}). 1 craft, '
        f'{total_bindings} binding(s), {len(meas_specs)} measurement sensor(s).\\n");',
        "",
        "    auto next = std::chrono::steady_clock::now();",
        "    const auto period = std::chrono::microseconds(int64_t(WALL_PERIOD * 1e6));",
        "    int pub_decim = 0;",
        "    const int pub_every = 20;  // ~50 Hz publish",
        "",
        "    while (g_run.load()) {",
    ]

    for bid, b in bind_assignments:
        if b.direction == "in":
            _emit_input_binding_apply(lines, bid, b, craft_var, filter_var,
                                      templated_craft=templated_craft)

    lines += [
        "",
        f"        {filter_var}.predict(DT, Q);",
        "",
    ]

    for m, spec in meas_specs:
        rname = f"R_{m.name}"
        part_acc = f"{craft_var}.{m.name}()"
        spec.emit_update_block(lines, filter_var, part_acc, rname)
    if meas_specs:
        lines.append("")

    lines += [
        "        if (++pub_decim >= pub_every) {",
        "            pub_decim = 0;",
    ]

    for bid, b in bind_assignments:
        if b.direction == "out":
            _emit_output_binding_publish(lines, bid, b, craft_var, filter_var)

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


def emit_ukf_main_cpp(target, ukf) -> str:
    """Emit main.cpp for a Target whose driveable is a UKF descriptor.
    Thin wrapper that forwards to the shared filter emitter."""
    return emit_ekf_main_cpp(target, ukf, kind="ukf")


# ---------------------------------------------------------------------------
# Binding apply/publish for EKF targets — accessor resolution differs from
# the world-target path (uses `ekf_<id>.craft()` and detects EKF-rooted
# output signals).

def _emit_input_binding_apply(lines: list[str], i: int, b: Binding,
                              craft_var: str, ekf_var: str,
                              templated_craft: bool = True) -> None:
    """Apply an in-binding's payload through each member's cpp_write_stmt.

    `templated_craft` controls whether `Vec3<PartFrame> ->
    Vec3<PartFrame, double>` rewriting is applied. EKF always wraps a
    scalar-templated craft (instantiated with double), so the patch
    runs. UKF can wrap either a templated craft (also instantiated with
    double — patch runs) or a plain non-templated craft (Scalar=Real,
    no patch needed)."""
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
        if templated_craft:
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

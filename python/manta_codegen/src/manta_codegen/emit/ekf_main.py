"""Emit <name>_main.cpp for an EKF Target.

Shape mirrors emit_main_cpp() but the tick loop is driven by a
`manta::estimation::WorldEKF<EstCraftT, MeasDim>` instance:

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
from ._util import GENERATED_BANNER, CPP_INCLUDE_GUARD, class_name_for_craft
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


def _filter_construction(kind: str, filter_obj, num_crafts: int,
                         filter_var: str, meas_dim: int) -> tuple[str, str]:
    """Return (header_include_line, ctor_line) for the filter wrapper.

    Both EKF and UKF now take `<NumCrafts, MeasDim>` template args. The
    crafts are bound at runtime via `ekf.bind(...)`, so the wrapper type
    no longer mentions a specific craft class.
    """
    if kind == "ekf":
        return (
            "#include \"manta/estimation/world_ekf.hpp\"",
            f"manta::estimation::WorldEKF<{num_crafts}, {meas_dim}> "
            f"{filter_var};",
        )
    if kind == "ukf":
        a, b, k = filter_obj.alpha, filter_obj.beta, filter_obj.kappa
        return (
            "#include \"manta/estimation/world_ukf.hpp\"",
            f"manta::estimation::WorldUKF<{num_crafts}, {meas_dim}> "
            f"{filter_var}({_f(a)}, {_f(b)}, {_f(k)});",
        )
    raise ValueError(f"unknown filter kind {kind!r}")


def _filter_real_craft_type(craft) -> str:
    """C++ type for the Real-side craft instance owned by the filter
    harness. Filter targets always require scalar_templated crafts; the
    Real instance is `<name>T<double>` (not the Real=float alias)."""
    if not getattr(craft, "scalar_templated", False):
        raise ValueError(
            f"Filter targets require scalar_templated=True on craft "
            f"{craft.name!r}. Set `c.scalar_templated = True` in the "
            f"craft descriptor.")
    return class_name_for_craft(craft.name) + "T<double>"


def _filter_jet_craft_type(craft) -> str:
    """C++ type for the Jet-shadow craft. Same templated class, instantiated
    on the EKF's Jet scalar (the user's harness <world>.cpp drops the
    `Ex<...>CraftT<JetType>` symbol; we just emit the class template name
    here and the caller appends `<JetType>`)."""
    return class_name_for_craft(craft.name) + "T"




# ---------------------------------------------------------------------------
# Filter-harness split emit (Phase 2 of the harness redesign).
#
# A Filter Target now produces three artifacts:
#
#   <world>.hpp  — public surface in `namespace manta_gen::<world>`:
#                  filter wrapper instance + field instances + DT
#                  constants + setup/tick/shutdown declarations.
#
#   <world>.cpp  — definitions + setup/tick/shutdown bodies.
#                  Anonymous namespace owns the Zenoh session, per-
#                  binding state, R blocks, publish-decimation counter.
#                  Per-sensor measurement functors live at file scope
#                  (above the anonymous namespace) so `tick()` can name
#                  their types without exposing them in the header.
#
#   <world>_main.cpp — thin pacing loop on top of setup/tick/shutdown.

def _filter_collect(target, filter_obj, kind):
    """Gather the per-target metadata both emit_filter_hpp and
    emit_filter_cpp need: unique crafts, the C++ wrapper-class string,
    a list of (part, _MeasFunctor) measurement specs, the meas_dim,
    the bind-id assignments, and the field-sync indices."""
    world = filter_obj.world
    if not world.crafts:
        raise RuntimeError(
            f"emit_{kind}_main_cpp: filter's wrapped world has no crafts")
    if len(world.crafts) > 1:
        raise NotImplementedError(
            f"emit_{kind}_main_cpp: only single-craft worlds are supported in v1.")
    if world.planets:
        raise NotImplementedError(
            f"emit_{kind}_main_cpp: planets in a filter-wrapped world aren't "
            f"supported yet (WorldEKF/WorldUKFOf don't own a Scene). "
            f"Register fields directly via World.fields for now.")

    unique_crafts = _world_unique_crafts(world)
    primary = unique_crafts[0]
    real_craft_type = _filter_real_craft_type(primary)
    jet_class_tmpl  = _filter_jet_craft_type(primary)
    num_crafts      = len(world.crafts)

    filter_var = filter_obj.cpp_var_name()
    # The Real craft is a namespace-scope variable named `craft` (or
    # `craft_<i>` for multi-craft worlds — single-craft is the only
    # case currently supported).
    craft_var = "craft"
    mag_field_var = _first_mag_field_var(world)

    meas_specs: list[tuple[object, _MeasFunctor]] = []
    for m in filter_obj.measurements:
        part_var = f"{craft_var}.{m.name}()"
        spec = _MeasFunctor.for_part(m, filter_var, part_var,
                                     mag_field_var=mag_field_var)
        meas_specs.append((m, spec))

    meas_dim = sum(s.n_floats for _, s in meas_specs) if meas_specs else 1
    meas_dim = max(meas_dim, 1)

    filter_header_inc, filter_ctor = _filter_construction(
        kind, filter_obj, num_crafts, filter_var, meas_dim)

    bind_assignments = list(enumerate(world.bindings))
    sync_field_idxs = [i for i, f in enumerate(world.fields)
                       if getattr(f, "synchronized", False)]

    return {
        "world":             world,
        "name":              world.name,
        "kind":              kind,
        "unique_crafts":     unique_crafts,
        "real_craft_type":   real_craft_type,
        "jet_class_tmpl":    jet_class_tmpl,
        "num_crafts":        num_crafts,
        "filter_var":        filter_var,
        "craft_var":         craft_var,
        "filter_header_inc": filter_header_inc,
        "filter_ctor":       filter_ctor,
        "meas_specs":        meas_specs,
        "meas_dim":          meas_dim,
        "bind_assignments":  bind_assignments,
        "sync_field_idxs":   sync_field_idxs,
    }


def emit_filter_hpp(target, filter_obj, kind: str = "ekf") -> str:
    ctx = _filter_collect(target, filter_obj, kind)
    name             = ctx["name"]
    unique_crafts    = ctx["unique_crafts"]
    filter_var       = ctx["filter_var"]
    filter_inc       = ctx["filter_header_inc"]
    filter_ctor      = ctx["filter_ctor"]
    real_craft_type  = ctx["real_craft_type"]
    meas_dim         = ctx["meas_dim"]
    world            = ctx["world"]

    # Strip ctor args off the line to recover the bare wrapper type for
    # the extern decl in the header.
    filter_type = filter_ctor.split(" " + filter_var, 1)[0].strip()

    lines: list[str] = [
        GENERATED_BANNER, CPP_INCLUDE_GUARD, "",
        '#include "manta/core/harness.hpp"',
        '#include "manta/core/scene.hpp"',
        '#include "manta/core/world.hpp"',
        filter_inc,
    ]
    for c in unique_crafts:
        lines.append(f'#include "{c.name}_craft.hpp"')
    seen: set[str] = set()
    for f in world.fields:
        if f.cpp_header not in seen:
            seen.add(f.cpp_header)
            lines.append(f'#include "{f.cpp_header}"')
    lines += [
        "",
        f"namespace manta_gen::{name} {{",
        "",
        f"// Sim-tick parameters, frozen at codegen time.",
        f"inline constexpr float DT             = {_f(target.dt)};",
        f"inline constexpr float SIM_RATE_MULT  = {_f(target.sim_rate_mult)};",
        "",
        f"// Real-side simulation infrastructure. The filter holds its own",
        f"// `manta::WorldT<double>` (estimator state lives in double, not the",
        f"// sim's `Real`=float, for filter conditioning). The Jet shadow",
        f"// `WorldT<Jet>` used for the Jacobian step lives file-private in",
        f"// the .cpp.",
        f"extern manta::WorldT<double>          w;",
        f"extern manta::SceneT<double>*         scene;          // valid after setup()",
    ]
    if world.fields:
        for i, f in enumerate(world.fields):
            lines.append(f"extern {f.cpp_class} field_{i};")
    lines.append(f"extern {real_craft_type}    craft;")
    lines += [
        "",
        f"// {kind.upper()} wrapper. Bound to `w` (Real) + the Jet shadow + the",
        f"// craft pointer(s) inside setup(). State dim = 13 * num_crafts.",
        f"extern {filter_type} {filter_var};",
        "",
        "// One-time initialization. Builds both worlds (Real + Jet shadow),",
        "// registers fields, instantiates the filter wrapper + binds it to",
        "// the worlds, opens Zenoh + declares pubs/subs.",
        "void setup();",
        "",
        f"// One step: applies in-bindings, runs predict(), then for each",
        f"// measurement sensor with consume_fresh()==true runs update_n<N>().",
        f"// On every kPubEvery=20 ticks, publishes out-bindings.",
        "void tick();",
        "",
        "// Tear down Zenoh state before main() returns.",
        "void shutdown();",
        "",
        "// Polymorphic adapter — see manta/core/harness.hpp.",
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


def emit_filter_cpp(target, filter_obj, kind: str = "ekf") -> str:
    """Emit the harness body for an EKF/UKF Target.

    Layout:
      file scope:
        - per-sensor measurement functor structs (templated on Scalar)

      namespace manta_gen::<name>:
        - manta::WorldT<double>  w;
        - manta::SceneT<double>* scene = nullptr;
        - <Field> field_<i> instances
        - <Craft>T<double>       craft;
        - filter wrapper instance (WorldEKF<NumCrafts, MeasDim> or
          WorldUKF<NumCrafts, MeasDim>)

      anonymous namespace:
        - parse_float_array
        - For EKF: manta::WorldT<JetType> w_jet; SceneT<JetType>* scene_jet
                   + Jet-instantiated craft instance
        - g_Q, R_<sensor> blocks
        - per-binding mutex + payload + Subscriber/Publisher
        - field-sync handles
        - g_session, g_pub_decim/kPubEvery

      namespace manta_gen::<name>:
        - setup()  builds both worlds (Real always, Jet only for EKF),
                   registers fields on each, adds crafts to scenes,
                   binds the filter wrapper, declares Zenoh subs/pubs
        - tick()   applies in-bindings, runs predict() + per-sensor
                   update_n<N>(), decimated publish
        - shutdown() resets Zenoh handles in reverse-init order
    """
    ctx = _filter_collect(target, filter_obj, kind)
    name             = ctx["name"]
    filter_var       = ctx["filter_var"]
    craft_var        = ctx["craft_var"]
    filter_ctor      = ctx["filter_ctor"]
    real_craft_type  = ctx["real_craft_type"]
    jet_class_tmpl   = ctx["jet_class_tmpl"]
    num_crafts       = ctx["num_crafts"]
    meas_specs       = ctx["meas_specs"]
    bind_assignments = ctx["bind_assignments"]
    sync_field_idxs  = ctx["sync_field_idxs"]
    world            = ctx["world"]

    needs_jet = (kind == "ekf")

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
        "#include <Eigen/Core>",
        "#include <Eigen/Geometry>",
        "#include <zenoh.hxx>",
        "",
    ]

    # Per-sensor measurement functors at file scope (above any namespace).
    for _, spec in meas_specs:
        for ln in spec.body.rstrip("\n").split("\n"):
            lines.append(ln)
        lines.append("")

    # Public namespace: Real-side storage + filter wrapper.
    lines += [
        f"namespace manta_gen::{name} {{",
        "",
        "manta::WorldT<double>  w{};",
        "manta::SceneT<double>* scene = nullptr;",
    ]
    for i, f in enumerate(world.fields):
        lines.append(f.emit_construction(f"field_{i}"))
    lines.append(f"{real_craft_type} {craft_var}{{}};")
    lines.append(filter_ctor)
    lines += [
        "",
        f"}}  // namespace manta_gen::{name}",
        "",
    ]

    # Anonymous-namespace file-private state.
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
        f"using EkfT = decltype(manta_gen::{name}::{filter_var});",
        f"EkfT::StateCov g_Q = EkfT::StateCov::Identity() * "
        f"{_f(filter_obj.process_noise)};",
        "",
    ]

    if needs_jet:
        # Jet shadow World — required by WorldEKF for the Jacobian step.
        lines += [
            f"// Jet shadow world. Built identically to the Real side in",
            f"// setup(); WorldEKF::predict drives this through autodiff to",
            f"// extract the state-transition Jacobian.",
            f"using JetType = EkfT::Jet;",
            f"manta::WorldT<JetType>   w_jet{{}};",
            f"manta::SceneT<JetType>*  scene_jet = nullptr;",
        ]
        for i, f in enumerate(world.fields):
            # The same field instance is shared between Real and Jet worlds
            # (registration is by-pointer; field state is non-templated).
            # No separate Jet field decl needed.
            pass
        lines.append(f"{jet_class_tmpl}<JetType> {craft_var}_jet{{}};")
        lines.append("")

    for m, spec in meas_specs:
        n = spec.n_floats
        rname = f"R_{m.name}"
        lines.append(f"Eigen::Matrix<double, {n}, {n}> {rname} = "
                     f"Eigen::Matrix<double, {n}, {n}>::Zero();")
    if meas_specs:
        lines.append("")

    for bid, b in bind_assignments:
        if b.direction == "in":
            lines += [
                f"std::mutex bind_{bid}_mtx;",
                f"std::vector<float> bind_{bid}_payload;",
                f"std::optional<zenoh::Subscriber<void>> bind_{bid}_sub;",
            ]
        else:
            lines.append(f"std::optional<zenoh::Publisher> pub_{bid};")
    if bind_assignments:
        lines.append("")

    for i in sync_field_idxs:
        lines += [
            f"std::optional<zenoh::Publisher>          pub_field_{i};",
            f"std::optional<zenoh::Subscriber<void>>   sub_field_{i};",
        ]
    if sync_field_idxs:
        lines.append("")

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
        "    // ---- Real world ----",
        "    w.clock().set_dt(DT);",
        "    scene = &w.create_scene();",
    ]
    for i, f in enumerate(world.fields):
        var = f"field_{i}"
        for stmt in (f.emit_extra_setup(var) if hasattr(f, "emit_extra_setup") else []):
            lines.append(f"    {stmt}")
        lines.append(f"    w.register_field({var});")
        for base in getattr(f, "register_as", []) or []:
            lines.append(f"    w.register_field<{base}>({var});")
    lines.append(f"    scene->add_craft({craft_var});")
    lines.append("")

    if needs_jet:
        lines += [
            "    // ---- Jet shadow world (built identically) ----",
            "    w_jet.clock().set_dt(DT);",
            "    scene_jet = &w_jet.create_scene();",
        ]
        for i, f in enumerate(world.fields):
            # Register the SAME field instance on the Jet world. Field state
            # is shared between Real + Jet; the Real side adds disturbances
            # in setup, the Jet side just reads them via state_at_templated.
            var = f"field_{i}"
            lines.append(f"    w_jet.register_field({var});")
            for base in getattr(f, "register_as", []) or []:
                lines.append(f"    w_jet.register_field<{base}>({var});")
        lines.append(f"    scene_jet->add_craft({craft_var}_jet);")
        lines.append("")

    # Filter init + bind.
    lines += [
        "    // ---- Filter init ----",
        f"    EkfT::StateVec x0 = EkfT::StateVec::Zero();",
    ]
    for k in range(num_crafts):
        lines.append(f"    x0({k * 13 + 3}) = 1.0;     // craft {k}: identity quaternion w")
    lines += [
        f"    EkfT::StateCov P0 = EkfT::StateCov::Identity() * "
        f"{_f(filter_obj.initial_covariance)};",
        f"    {filter_var}.set_state(x0);",
        f"    {filter_var}.set_covariance(P0);",
    ]
    if needs_jet:
        lines.append(f"    {filter_var}.bind(w_jet, "
                     f"{{&{craft_var}}}, {{&{craft_var}_jet}});")
    else:
        lines.append(f"    {filter_var}.bind(w, {{&{craft_var}}});")
    lines.append("")

    # Initialize R diag entries.
    for m, spec in meas_specs:
        diag = spec.sigma_squared_diag(m)
        rname = f"R_{m.name}"
        for i, v in enumerate(diag):
            lines.append(f"    {rname}({i}, {i}) = {_f(v)};")
    if meas_specs:
        lines.append("")

    # Zenoh.
    lines += [
        "    // ---- Zenoh ----",
        "    g_session.emplace(zenoh::Session::open(zenoh::Config::create_default()));",
        "",
    ]
    for bid, b in bind_assignments:
        if b.direction == "in":
            _emit_subscriber_setup(lines, bid, b)
        else:
            lines.append(
                f"    pub_{bid}.emplace(g_session->declare_publisher("
                f"zenoh::KeyExpr({_quote(b.topic)})));")
    for i in sync_field_idxs:
        topic = world.fields[i].sync_topic or f"manta/{name}/field_{i}/disturbance"
        _emit_field_sync_setup(lines, i, topic, world.fields[i].cpp_class)
    lines += ["}", ""]

    # ---- tick() ----
    lines += ["void tick() {"]

    # Apply in-bindings.
    for bid, b in bind_assignments:
        if b.direction == "in":
            _emit_input_binding_apply_indented(
                lines, bid, b, craft_var, filter_var,
                templated_craft=True, indent="    ")

    lines += [
        "",
        f"    {filter_var}.predict(DT, g_Q);",
        "",
    ]

    for m, spec in meas_specs:
        rname = f"R_{m.name}"
        part_acc = f"{craft_var}.{m.name}()"
        spec.emit_update_block(lines, filter_var, part_acc, rname, indent="    ")
    if meas_specs:
        lines.append("")

    if any(b.direction == "out" for _, b in bind_assignments):
        lines += [
            "    if (++g_pub_decim >= kPubEvery) {",
            "        g_pub_decim = 0;",
        ]
        for bid, b in bind_assignments:
            if b.direction == "out":
                _emit_output_binding_publish_indented(
                    lines, bid, b, craft_var, filter_var, indent="        ")
        lines.append("    }")

    lines += ["}", ""]

    # ---- shutdown() ----
    lines += [
        "void shutdown() {",
    ]
    for bid, b in bind_assignments:
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


def emit_filter_main_cpp(target, filter_obj, kind: str = "ekf") -> str:
    name = filter_obj.world.name
    n_meas = len(filter_obj.measurements)
    n_bindings = len(filter_obj.world.bindings)
    kind_label = kind.upper()

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
        f'    std::printf("{target.name}: ready ({kind_label}). 1 craft, '
        f'{n_bindings} binding(s), {n_meas} measurement sensor(s).\\n");',
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
        f'    std::printf("{target.name}: shutting down.\\n");',
        f"    manta_gen::{name}::shutdown();",
        "    return 0;",
        "}",
        "",
    ])


# ---------------------------------------------------------------------------
# Indented-form binding helpers — the harness's setup()/tick() bodies are at
# 4-space indentation rather than the 8-space the legacy main-in-main path
# used. Existing _emit_input_binding_apply / _emit_output_binding_publish
# stay unchanged for back-compat with ekf_main_cpp; these wrappers re-emit
# at the new indentation.

def _emit_input_binding_apply_indented(lines: list[str], i: int, b: Binding,
                                       craft_var: str, ekf_var: str,
                                       templated_craft: bool,
                                       indent: str) -> None:
    lines += [
        f"{indent}{{ std::lock_guard<std::mutex> lk(bind_{i}_mtx);",
        f"{indent}  if (bind_{i}_payload.size() >= {b.total_floats}) {{",
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
        lines.append(f"{indent}      {stmt}    // member: {member_name}")
        cursor += n
    lines += [
        f"{indent}      bind_{i}_payload.clear();",
        f"{indent}  }} }}",
    ]


def _emit_output_binding_publish_indented(lines: list[str], i: int, b: Binding,
                                          craft_var: str, ekf_var: str,
                                          indent: str) -> None:
    if b.encoding != "json":
        raise NotImplementedError(
            f"Binding {b.topic!r}: encoding {b.encoding!r} not supported.")
    lines.append(f"{indent}{{ std::string _json = \"{{\";")
    first_member = True
    for member_name, sig in b.members.items():
        n = sig.signal.n_floats
        accessor = _accessor_for_ekf_or_craft(sig, craft_var, ekf_var)
        if not first_member:
            lines.append(f'{indent}  _json += ",";')
        first_member = False
        if n == 1:
            cpp_expr = sig.signal.cpp_read_exprs[0].format(accessor=accessor)
            lines.append(f'{indent}  _json += "\\"{member_name}\\":";')
            lines.append(
                f"{indent}  {{ char _b[32]; std::snprintf(_b, sizeof(_b), \"%g\", "
                f"double({cpp_expr})); _json += _b; }}"
            )
        else:
            lines.append(f'{indent}  _json += "\\"{member_name}\\":[";')
            for k, expr in enumerate(sig.signal.cpp_read_exprs):
                cpp_expr = expr.format(accessor=accessor)
                sep = '","' if k > 0 else '""'
                lines.append(
                    f"{indent}  {{ char _b[32]; std::snprintf(_b, sizeof(_b), \"%s%g\", "
                    f"{sep}, double({cpp_expr})); _json += _b; }}"
                )
            lines.append(f'{indent}  _json += "]";')
    lines += [
        f'{indent}  _json += "}}";',
        f"{indent}  pub_{i}->put(zenoh::Bytes(_json));",
        f"{indent}}}",
    ]


# Helpers shared with the World harness emit (reused via cross-module import
# in __init__.py). Defined locally here too so this module is self-contained.
def _emit_subscriber_setup(lines: list[str], i: int, b: Binding) -> None:
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
    lines += [
        f"    pub_field_{i}.emplace(g_session->declare_publisher("
        f"zenoh::KeyExpr({_quote(topic)})));",
        f"    field_{i}.set_tx_hook(",
        f"        [](std::uint16_t tag, const {cpp_class}::Params& params, int lifetime) {{",
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
        f"        }}, zenoh::closures::none));",
    ]


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

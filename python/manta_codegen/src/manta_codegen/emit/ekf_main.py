"""Emit <name>_main.cpp for an EKF Target.

Shape mirrors emit_main_cpp() but the tick loop is driven by a
`manta::estimation::EKF<EstCraftT, MeasDim>` instance:

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
                 mag_field_var: str | None = None,
                 craft_idx: int = 0,
                 scope: str = "file",
                 update_method: str = "update_n"):
        self.n_floats = n_floats
        self.body = body
        self.z_read = z_read
        self.functor_name = functor_name
        self.mag_field_var = mag_field_var   # set for magnetometer specs only
        self.craft_idx = craft_idx           # which craft slot in the joint state
        # "file": functor is purely a pure function of x (DVL, Mag) and
        # gets emitted at file scope above the public namespace.
        # "anon": functor reaches into the Jet shadow world (e.g. IMU,
        # which calls `w_jet.kinematic_and_aggregate()`) and must live inside the
        # anonymous namespace alongside `w_jet` / `craft_jet`.
        self.scope = scope
        # Which filter API method `emit_update_block` should call:
        # "update_n" for UKF (legacy single-shot path) or "add_update"
        # for the EKF's fused begin_step / add_update / end_step bracket.
        self.update_method = update_method

    @classmethod
    def for_part(cls, part, ekf_var: str, part_var: str,
                 craft_idx: int = 0,
                 state_dim: int = 13,
                 mag_field_var: str | None = None,
                 craft_jet_vars: list[str] | None = None,
                 kind: str = "ekf") -> "_MeasFunctor":
        """Dispatch by part type. Supported: DVL, IMU, Magnetometer.

        `part_var` is the sensor's C++ accessor (e.g.
            `manta_gen::ex9::craft_0.dvl()`).
        `craft_idx` selects which 13-DOF block of the joint state vector
            this sensor's craft occupies. For single-craft worlds it's 0;
            for multi-craft, sensor-on-craft-1 reads from state segment
            [13, 26).
        `state_dim` is the full filter state width (13 * num_crafts);
            used for the templated functor input matrix size hint.
        `mag_field_var` is the C++ var name of the first registered
            MagField in the surrounding scope — required if Magnetometer
            is among the measurements; ignored otherwise.
        """
        cct = getattr(type(part), "cpp_class_template", "")
        # Functor name includes craft_idx — multi-craft worlds typically
        # have parts with the same name (e.g. each drone's `imu`).
        functor_name = f"_{ekf_var}_c{craft_idx}_{part.name}_meas"
        # Per-craft state offset into the joint state vector.
        s0 = craft_idx * 13
        # EKF uses the fused begin_step / add_update / end_step API;
        # UKF uses the legacy update_n (one Jet pass per measurement
        # is irrelevant for UKF since it doesn't autodiff anyway).
        update_method = "add_update" if kind == "ekf" else "update_n"
        if cct == "manta::parts::DVLT":
            z_read = [
                f"{part_var}.last_velocity().raw()(0)",
                f"{part_var}.last_velocity().raw()(1)",
                f"{part_var}.last_velocity().raw()(2)",
            ]
            if kind == "ekf":
                # Read from the Jet sensor directly. After begin_step's
                # `evaluate()` the Jet craft's scene_to_part is populated
                # at x_pre with identity-seeded derivatives, so
                # `velocity_body()` gives both h(x_pre) and ∂h/∂x_pre in
                # one read — no separate state-vector parsing. Reaches
                # into the anon-namespace Jet craft global by name (the
                # EKF::craft_jet() accessor returns the base type
                # which doesn't expose the templated part accessors).
                if craft_jet_vars is None:
                    raise RuntimeError(
                        "EKF DVL codegen: craft_jet_vars must be provided.")
                jet_var = craft_jet_vars[craft_idx]
                body = (
                    f"// DVL (EKF): h = R(q)^T * v_scene, read from Jet sensor.\n"
                    f"struct {functor_name} {{\n"
                    f"    Eigen::Matrix<JetType, 3, 1> operator()(EkfT&) const {{\n"
                    f"        return {jet_var}.{part.name}()"
                    f".velocity_body().raw();\n"
                    f"    }}\n"
                    f"}};\n"
                )
                return cls(3, body, z_read, functor_name,
                           craft_idx=craft_idx, scope="anon",
                           update_method=update_method)
            # UKF path keeps the closed-form sigma-point form (h(x) on a
            # plain state vector, no Jet seeds, no Jet world).
            body = (
                f"// DVL (UKF): closed-form h(x) on the joint state vector.\n"
                f"// Reads craft-{craft_idx} slice (offset {s0}).\n"
                f"struct {functor_name} {{\n"
                f"    template <class S>\n"
                f"    Eigen::Matrix<S, 3, 1> operator()(const Eigen::Matrix<S, {state_dim}, 1>& x) const {{\n"
                f"        Eigen::Quaternion<S> q(x({s0+3}), x({s0+4}), x({s0+5}), x({s0+6}));\n"
                f"        Eigen::Matrix<S, 3, 1> v_scene(x({s0+7}), x({s0+8}), x({s0+9}));\n"
                f"        return q.conjugate() * v_scene;\n"
                f"    }}\n"
                f"}};\n"
            )
            return cls(3, body, z_read, functor_name, craft_idx=craft_idx,
                       update_method=update_method)
        if cct == "manta::parts::IMUT":
            z_read = [
                f"{part_var}.last_accel().raw()(0)",
                f"{part_var}.last_accel().raw()(1)",
                f"{part_var}.last_accel().raw()(2)",
                f"{part_var}.last_gyro().raw()(0)",
                f"{part_var}.last_gyro().raw()(1)",
                f"{part_var}.last_gyro().raw()(2)",
            ]
            if kind != "ekf":
                # UKF path keeps the analytic free-fall placeholder for
                # now — the UKF doesn't autodiff h, so a dynamics-driven
                # h(x) would mutate the (single, user-facing) value world's
                # crafts inside each sigma-point evaluation. Wiring a
                # save/restore is out of scope for this fix.
                body = (
                    f"// IMU (UKF): placeholder no-net-force prediction.\n"
                    f"// TODO: dynamics-driven h(x) once UKF has a state-\n"
                    f"// preserving evaluate path. Reads craft-{craft_idx}\n"
                    f"// slice (offset {s0}).\n"
                    f"struct {functor_name} {{\n"
                    f"    template <class S>\n"
                    f"    Eigen::Matrix<S, 6, 1> operator()(const Eigen::Matrix<S, {state_dim}, 1>& x) const {{\n"
                    f"        Eigen::Matrix<S, 6, 1> z;\n"
                    f"        z(0) = S(0); z(1) = S(0); z(2) = S(0);\n"
                    f"        z(3) = x({s0+10}); z(4) = x({s0+11}); z(5) = x({s0+12});\n"
                    f"        return z;\n"
                    f"    }}\n"
                    f"}};\n"
                )
                return cls(6, body, z_read, functor_name,
                           craft_idx=craft_idx, scope="file",
                           update_method=update_method)

            # EKF path: read specific_force_body + body angular velocity
            # straight from the Jet sensor. begin_step has already seeded
            # the Jet craft at x_pre with identity derivatives and run
            # w_jet.kinematic_and_aggregate(), so the cached acc_linear (which feeds
            # specific_force_body) and scene_to_part vel_angular both
            # carry the right Jet derivatives — one read gives both h
            # and H.
            #
            # Specific force = (kinematic body accel − gravity_body):
            # what a real accelerometer reports. Free fall reads zero,
            # stationary craft on the ground reads −g_body (≈ +9.81 ẑ
            # at q=identity). Thrust, drag, contact, cross-craft
            # coupling all flow through via the dynamics; gravity is
            # queried from the registered GravityField and subtracted
            # out. Reads craft-{craft_idx} slice (offset {s0}).
            if craft_jet_vars is None:
                raise RuntimeError(
                    "EKF IMU codegen: craft_jet_vars must be provided.")
            jet_var = craft_jet_vars[craft_idx]
            body = (
                f"// IMU (EKF): h(x) = [specific_force_body; ω_body].\n"
                f"// Reads the Jet sensor directly; values + H come\n"
                f"// from the begin_step evaluate at x_pre.\n"
                f"struct {functor_name} {{\n"
                f"    Eigen::Matrix<JetType, 6, 1> operator()(EkfT&) const {{\n"
                f"        Eigen::Matrix<JetType, 6, 1> z;\n"
                f"        const auto _a = {jet_var}.{part.name}().specific_force_body();\n"
                f"        const auto _w = {jet_var}.{part.name}().angular_velocity_body();\n"
                f"        z(0) = _a.raw()(0);\n"
                f"        z(1) = _a.raw()(1);\n"
                f"        z(2) = _a.raw()(2);\n"
                f"        z(3) = _w.raw()(0);\n"
                f"        z(4) = _w.raw()(1);\n"
                f"        z(5) = _w.raw()(2);\n"
                f"        return z;\n"
                f"    }}\n"
                f"}};\n"
            )
            return cls(6, body, z_read, functor_name,
                       craft_idx=craft_idx, scope="anon",
                       update_method=update_method)
        if cct == "manta::parts::MagnetometerT":
            if mag_field_var is None:
                raise RuntimeError(
                    f"EKF codegen: Magnetometer {part.name!r} requires a "
                    f"`MagField` registered on the EKF's wrapped world.")
            z_read = [
                f"{part_var}.last_b().raw()(0)",
                f"{part_var}.last_b().raw()(1)",
                f"{part_var}.last_b().raw()(2)",
            ]
            if kind == "ekf":
                # Locally-constant-B: capture B at the value-side position
                # before each update, then h(x_pre) = R(q_pre)^T · B. q
                # comes from the Jet craft post-evaluate. ∂h/∂q is exact
                # through autodiff; ∂h/∂p ≈ 0 from the const-B approx.
                if craft_jet_vars is None:
                    raise RuntimeError(
                        "EKF Magnetometer codegen: craft_jet_vars must be provided.")
                jet_var = craft_jet_vars[craft_idx]
                body = (
                    f"// Magnetometer (EKF): h = R(q_pre)^T · B_captured.\n"
                    f"struct {functor_name} {{\n"
                    f"    Eigen::Matrix<double, 3, 1> b_scene_now;\n"
                    f"    Eigen::Matrix<JetType, 3, 1> operator()(EkfT&) const {{\n"
                    f"        const auto& q = {jet_var}"
                    f".scene_to_craft().orientation().raw();\n"
                    f"        Eigen::Matrix<JetType, 3, 1> b;\n"
                    f"        b(0) = JetType(b_scene_now(0));\n"
                    f"        b(1) = JetType(b_scene_now(1));\n"
                    f"        b(2) = JetType(b_scene_now(2));\n"
                    f"        return q.conjugate() * b;\n"
                    f"    }}\n"
                    f"}};\n"
                )
                return cls(3, body, z_read, functor_name,
                           mag_field_var=mag_field_var, craft_idx=craft_idx,
                           scope="anon", update_method=update_method)
            # UKF path: closed-form on state vector.
            body = (
                f"// Magnetometer (UKF): closed-form h(x) on state vector.\n"
                f"struct {functor_name} {{\n"
                f"    Eigen::Matrix<double, 3, 1> b_scene_now;\n"
                f"    template <class S>\n"
                f"    Eigen::Matrix<S, 3, 1> operator()(const Eigen::Matrix<S, {state_dim}, 1>& x) const {{\n"
                f"        Eigen::Quaternion<S> q(x({s0+3}), x({s0+4}), x({s0+5}), x({s0+6}));\n"
                f"        Eigen::Matrix<S, 3, 1> b;\n"
                f"        b(0) = S(b_scene_now(0));\n"
                f"        b(1) = S(b_scene_now(1));\n"
                f"        b(2) = S(b_scene_now(2));\n"
                f"        return q.conjugate() * b;\n"
                f"    }}\n"
                f"}};\n"
            )
            return cls(3, body, z_read, functor_name,
                       mag_field_var=mag_field_var, craft_idx=craft_idx,
                       update_method=update_method)
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
        """Append the `if (consume_fresh()) { ... add_update<N>(...); }`
        block for this sensor. Magnetometer adds a pre-update B-capture
        step (locally-constant-B approximation); everything else just
        constructs the functor and queues the update.

        Lives inside a `begin_step / end_step` bracket emitted by the
        caller — `add_update` reads h + H from the Jet sensors that the
        begin_step's evaluate already populated, so no extra Jet world
        pass is needed per measurement.
        """
        n = self.n_floats
        lines.append(f"{indent}if ({part_var}.consume_fresh()) {{")
        if self.mag_field_var is not None:
            lines.append(f"{indent}    {self.functor_name} _h;")
            lines.append(f"{indent}    auto _p_d = {ekf_var}.position({self.craft_idx});")
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
                f"{indent}    {ekf_var}.template {self.update_method}<{n}>(_h, z, {rname});"
            )
        else:
            lines.append(f"{indent}    Eigen::Matrix<double, {n}, 1> z;")
            for i, expr in enumerate(self.z_read):
                lines.append(f"{indent}    z({i}) = {expr};")
            lines.append(
                f"{indent}    {ekf_var}.template {self.update_method}<{n}>("
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
        # Block-decomposed variant for decoupled-craft swarms: per-craft
        # Jet width = 13, NumCrafts passes per tick. Cost scales linearly
        # in NumCrafts instead of quadratically; only valid when crafts
        # don't physically couple (no tether/contact/fluid coupling).
        if getattr(filter_obj, "block_decomposed", False):
            return (
                "#include \"manta/estimation/block_decomposed_ekf.hpp\"",
                f"manta::estimation::BlockDecomposedEKF<"
                f"{num_crafts}, {meas_dim}> {filter_var};",
            )
        return (
            "#include \"manta/estimation/ekf.hpp\"",
            f"manta::estimation::EKF<{num_crafts}, {meas_dim}> "
            f"{filter_var};",
        )
    if kind == "ukf":
        a, b, k = filter_obj.alpha, filter_obj.beta, filter_obj.kappa
        return (
            "#include \"manta/estimation/ukf.hpp\"",
            f"manta::estimation::UKF<{num_crafts}, {meas_dim}> "
            f"{filter_var}({_f(a)}, {_f(b)}, {_f(k)});",
        )
    raise ValueError(f"unknown filter kind {kind!r}")


def _filter_real_craft_type(craft) -> str:
    """C++ type for the value-side craft instance owned by the filter
    harness. Filter targets always require scalar_templated crafts; the
    MFloat instance is `<name>T<double>` (not the value=float alias)."""
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
    emit_filter_cpp need: per-craft type info + var names, measurement
    specs (each tagged with its craft_idx), bind-id assignments,
    field-sync indices.

    Multi-craft filter targets work: each Craft becomes craft_0/1/...
    in the namespace, the EKF state is the concat of every craft's
    13-DOF rigid-body state, and per-sensor measurement functors
    offset their state-segment reads by `13 * craft_idx`.
    """
    world = filter_obj.world
    if not world.crafts:
        raise RuntimeError(
            f"emit_{kind}_main_cpp: filter's wrapped world has no crafts")
    if world.planets:
        raise NotImplementedError(
            f"emit_{kind}_main_cpp: planets in a filter-wrapped world aren't "
            f"supported yet. Register fields directly via World.fields for now.")

    unique_crafts = _world_unique_crafts(world)
    num_crafts    = len(unique_crafts)
    state_dim     = 13 * num_crafts

    # Per-craft naming + types.
    def craft_var_for(idx: int) -> str:
        return "craft" if num_crafts == 1 else f"craft_{idx}"

    def craft_jet_var_for(idx: int) -> str:
        return f"{craft_var_for(idx)}_jet"

    craft_id_to_idx: dict[int, int] = {
        id(c): i for i, c in enumerate(unique_crafts)
    }

    real_craft_types = [_filter_real_craft_type(c) for c in unique_crafts]
    jet_class_tmpls  = [_filter_jet_craft_type(c)  for c in unique_crafts]

    filter_var = filter_obj.cpp_var_name()
    mag_field_var = _first_mag_field_var(world)

    meas_specs: list[tuple[object, _MeasFunctor]] = []
    for m in filter_obj.measurements:
        # Resolve the craft this part is attached to.
        m_craft = getattr(m, "_craft", None)
        if m_craft is None:
            raise RuntimeError(
                f"EKF codegen: measurement part {m.name!r} not attached "
                f"to a craft (call craft.add(part) before adding to EKF).")
        if id(m_craft) not in craft_id_to_idx:
            raise RuntimeError(
                f"EKF codegen: measurement part {m.name!r}'s craft "
                f"{m_craft.name!r} is not in the filter's wrapped world.")
        c_idx = craft_id_to_idx[id(m_craft)]
        part_var = f"{craft_var_for(c_idx)}.{m.name}()"
        spec = _MeasFunctor.for_part(
            m, filter_var, part_var,
            craft_idx=c_idx,
            state_dim=state_dim,
            mag_field_var=mag_field_var,
            craft_jet_vars=[craft_jet_var_for(k) for k in range(num_crafts)],
            kind=kind)
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
        "real_craft_types":  real_craft_types,
        "jet_class_tmpls":   jet_class_tmpls,
        "num_crafts":        num_crafts,
        "state_dim":         state_dim,
        "craft_id_to_idx":   craft_id_to_idx,
        "craft_var_for":     craft_var_for,
        "craft_jet_var_for": craft_jet_var_for,
        "filter_var":        filter_var,
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
    real_craft_types = ctx["real_craft_types"]
    num_crafts       = ctx["num_crafts"]
    craft_var_for    = ctx["craft_var_for"]
    filter_var       = ctx["filter_var"]
    filter_inc       = ctx["filter_header_inc"]
    filter_ctor      = ctx["filter_ctor"]
    meas_dim         = ctx["meas_dim"]
    world            = ctx["world"]

    # Strip ctor args off the line to recover the bare wrapper type.
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
        f"// value-side simulation infrastructure. The filter holds its own",
        f"// `manta::WorldT<double>` (estimator state lives in double for",
        f"// filter conditioning). The Jet shadow `WorldT<Jet>` for the",
        f"// Jacobian step lives file-private in the .cpp.",
        f"extern manta::WorldT<double>          w;",
        f"extern manta::SceneT<double>*         scene;          // valid after setup()",
    ]
    if world.fields:
        for i, f in enumerate(world.fields):
            lines.append(f"extern {f.cpp_class} field_{i};")
    if num_crafts == 1:
        lines.append(f"extern {real_craft_types[0]} craft;")
    else:
        lines.append(f"// Crafts are concatenated in the EKF state vector in the order")
        lines.append(f"// they're declared here: state[0..12]=craft_0, state[13..25]=craft_1, ...")
        for i, t in enumerate(real_craft_types):
            lines.append(f"extern {t} {craft_var_for(i)};")
    lines += [
        "",
        f"// {kind.upper()} wrapper. State dim = 13 * {num_crafts} = {13*num_crafts}.",
        f"// Bound inside setup() to the value world + (for EKF) Jet shadow +",
        f"// per-craft pointer arrays.",
        f"extern {filter_type} {filter_var};",
        "",
        "// One-time initialization. Builds both worlds (MFloat + Jet shadow),",
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
        - filter wrapper instance (EKF<NumCrafts, MeasDim> or
          UKF<NumCrafts, MeasDim>)

      anonymous namespace:
        - parse_float_array
        - For EKF: manta::WorldT<JetType> w_jet; SceneT<JetType>* scene_jet
                   + Jet-instantiated craft instance
        - g_Q, R_<sensor> blocks
        - per-binding mutex + payload + Subscriber/Publisher
        - field-sync handles
        - g_session, g_pub_decim/kPubEvery

      namespace manta_gen::<name>:
        - setup()  builds both worlds (MFloat always, Jet only for EKF),
                   registers fields on each, adds crafts to scenes,
                   binds the filter wrapper, declares Zenoh subs/pubs
        - tick()   applies in-bindings, runs predict() + per-sensor
                   update_n<N>(), decimated publish
        - shutdown() resets Zenoh handles in reverse-init order
    """
    ctx = _filter_collect(target, filter_obj, kind)
    name              = ctx["name"]
    filter_var        = ctx["filter_var"]
    filter_ctor       = ctx["filter_ctor"]
    real_craft_types  = ctx["real_craft_types"]
    jet_class_tmpls   = ctx["jet_class_tmpls"]
    num_crafts        = ctx["num_crafts"]
    craft_var_for     = ctx["craft_var_for"]
    craft_jet_var_for = ctx["craft_jet_var_for"]
    craft_id_to_idx   = ctx["craft_id_to_idx"]
    meas_specs        = ctx["meas_specs"]
    bind_assignments  = ctx["bind_assignments"]
    sync_field_idxs   = ctx["sync_field_idxs"]
    world             = ctx["world"]
    unique_crafts     = ctx["unique_crafts"]

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

    # File-scope measurement functors (pure functions of x — DVL, Mag).
    # Anon-scope ones (IMU's dynamics-driven h, which calls into w_jet)
    # are emitted later, inside the anonymous namespace alongside the
    # Jet shadow declarations.
    for _, spec in meas_specs:
        if spec.scope != "file":
            continue
        for ln in spec.body.rstrip("\n").split("\n"):
            lines.append(ln)
        lines.append("")

    # Public namespace: value-side storage + filter wrapper.
    lines += [
        f"namespace manta_gen::{name} {{",
        "",
        "manta::WorldT<double>  w{};",
        "manta::SceneT<double>* scene = nullptr;",
    ]
    for i, f in enumerate(world.fields):
        lines.append(f.emit_construction(f"field_{i}"))
    for i in range(num_crafts):
        lines.append(f"{real_craft_types[i]} {craft_var_for(i)}{{}};")
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
        # Jet shadow World — required by EKF for the Jacobian step.
        lines += [
            f"// Jet shadow world. Built identically to the value side in",
            f"// setup(); EKF::predict drives this through autodiff to",
            f"// extract the state-transition Jacobian.",
            f"using JetType = EkfT::Jet;",
            f"manta::WorldT<JetType>   w_jet{{}};",
            f"manta::SceneT<JetType>*  scene_jet = nullptr;",
        ]
        for i in range(num_crafts):
            lines.append(
                f"{jet_class_tmpls[i]}<JetType> {craft_jet_var_for(i)}{{}};")
        lines.append("")

        # Anon-scope measurement functors (the dynamics-driven IMU h(x)
        # reaches into `w_jet` and `craft_jet[*]`, which are only
        # accessible from inside this anonymous namespace).
        anon_specs = [(m, s) for m, s in meas_specs if s.scope == "anon"]
        for _, spec in anon_specs:
            for ln in spec.body.rstrip("\n").split("\n"):
                lines.append(ln)
            lines.append("")

    for m, spec in meas_specs:
        n = spec.n_floats
        rname = f"R_c{spec.craft_idx}_{m.name}"
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
        "    // ---- value world ----",
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
    for i in range(num_crafts):
        lines.append(f"    scene->add_craft({craft_var_for(i)});")
    lines.append("")

    if needs_jet:
        lines += [
            "    // ---- Jet shadow world (built identically) ----",
            "    w_jet.clock().set_dt(DT);",
            "    scene_jet = &w_jet.create_scene();",
        ]
        for i, f in enumerate(world.fields):
            # Register the SAME field instance on the Jet world. Field state
            # is shared between MFloat + Jet; the value side adds disturbances
            # in setup, the Jet side just reads them via state_at_templated.
            var = f"field_{i}"
            lines.append(f"    w_jet.register_field({var});")
            for base in getattr(f, "register_as", []) or []:
                lines.append(f"    w_jet.register_field<{base}>({var});")
        for i in range(num_crafts):
            lines.append(f"    scene_jet->add_craft({craft_jet_var_for(i)});")
        lines.append("")

    # Filter init + bind.
    #
    # Per-craft initial-state and per-block initial-variance knobs each
    # accept four shapes — None / scalar-or-tuple (broadcast) / list
    # (per-craft positional) / dict (per-craft by name). Both helpers
    # resolve to a per-craft list of length num_crafts at emit time.
    craft_names = [world.crafts[k].craft.name for k in range(num_crafts)]

    def _resolve_state(attr: str, default_extractor, n_components: int):
        """Per-craft state vector (tuple of n_components floats per
        craft). None → world.add_craft default; scalar tuple → broadcast;
        list of tuples → positional per-craft; dict → by craft name."""
        override = getattr(filter_obj, attr, None)
        def _from_world(k):
            return tuple(float(v) for v in default_extractor(world.crafts[k]))
        if override is None:
            return [_from_world(k) for k in range(num_crafts)]
        if isinstance(override, dict):
            out = []
            for k in range(num_crafts):
                v = override.get(craft_names[k])
                out.append(tuple(float(x) for x in v) if v is not None else _from_world(k))
            return out
        # Single tuple (broadcast) — heuristic: length matches and
        # contents are scalars (not nested tuples/lists).
        if (len(override) == n_components
                and not isinstance(override[0], (list, tuple))):
            return [tuple(float(v) for v in override) for _ in range(num_crafts)]
        # List of tuples (one per craft).
        if len(override) != num_crafts:
            raise ValueError(
                f"EKF.{attr}: expected {num_crafts} per-craft tuples (or a "
                f"single broadcast tuple, or a dict by craft name); "
                f"got list of length {len(override)}.")
        return [tuple(float(v) for v in override[k]) for k in range(num_crafts)]

    def _resolve_var(attr: str):
        """Per-craft scalar (or None) for a variance block. None →
        leave at initial_covariance; scalar → broadcast; list → per-
        craft positional; dict → by craft name (others stay default)."""
        override = getattr(filter_obj, attr, None)
        if override is None:
            return [None] * num_crafts
        if isinstance(override, dict):
            return [
                float(override[craft_names[k]]) if craft_names[k] in override else None
                for k in range(num_crafts)
            ]
        if isinstance(override, (list, tuple)):
            if len(override) != num_crafts:
                raise ValueError(
                    f"EKF.{attr}: expected {num_crafts} per-craft scalars "
                    f"(or a single broadcast scalar, or a dict by craft "
                    f"name); got list of length {len(override)}.")
            return [float(v) for v in override]
        # Plain scalar → broadcast.
        return [float(override)] * num_crafts

    pos_per_craft = _resolve_state("initial_position",         lambda e: e.position,    3)
    ori_per_craft = _resolve_state("initial_orientation",      lambda e: e.orientation, 4)
    vel_per_craft = _resolve_state("initial_velocity",         lambda e: e.vel_linear,  3)
    ang_per_craft = _resolve_state("initial_angular_velocity", lambda e: e.vel_angular, 3)

    lines += [
        "    // ---- Filter init ----",
        f"    EkfT::StateVec x0 = EkfT::StateVec::Zero();",
    ]
    for k in range(num_crafts):
        s0 = k * 13
        p, q, v, w = pos_per_craft[k], ori_per_craft[k], vel_per_craft[k], ang_per_craft[k]
        lines.append(f"    // craft {k} initial state")
        lines.append(f"    x0({s0+0}) = {_f(p[0])}; x0({s0+1}) = {_f(p[1])}; x0({s0+2}) = {_f(p[2])};")
        lines.append(f"    x0({s0+3}) = {_f(q[0])}; x0({s0+4}) = {_f(q[1])}; "
                     f"x0({s0+5}) = {_f(q[2])}; x0({s0+6}) = {_f(q[3])};")
        lines.append(f"    x0({s0+7}) = {_f(v[0])}; x0({s0+8}) = {_f(v[1])}; x0({s0+9}) = {_f(v[2])};")
        lines.append(f"    x0({s0+10}) = {_f(w[0])}; x0({s0+11}) = {_f(w[1])}; x0({s0+12}) = {_f(w[2])};")

    lines.append(
        f"    EkfT::StateCov P0 = EkfT::StateCov::Identity() * "
        f"{_f(filter_obj.initial_covariance)};")

    # Per-block + per-craft variance overrides. Each block (position,
    # attitude, velocity, angular velocity) has its own *_var field on
    # the EKF descriptor; each accepts None (use initial_covariance),
    # scalar (broadcast), list (per-craft positional), or dict (per-
    # craft by name). _resolve_var returns a length-num_crafts list of
    # (float | None); we emit a P0 override line only for crafts where
    # it's set.
    var_blocks = [
        ("initial_position_var",         0, 3),
        ("initial_attitude_var",         3, 4),
        ("initial_velocity_var",         7, 3),
        ("initial_angular_velocity_var", 10, 3),
    ]
    for attr, off, n in var_blocks:
        per_craft = _resolve_var(attr)
        for k in range(num_crafts):
            v = per_craft[k]
            if v is None:
                continue
            base = k * 13 + off
            for j in range(n):
                lines.append(f"    P0({base+j}, {base+j}) = {_f(v)};")

    lines += [
        f"    {filter_var}.set_state(x0);",
        f"    {filter_var}.set_covariance(P0);",
    ]
    real_arr = "{" + ", ".join(f"&{craft_var_for(i)}" for i in range(num_crafts)) + "}"
    if needs_jet:
        jet_arr = "{" + ", ".join(f"&{craft_jet_var_for(i)}" for i in range(num_crafts)) + "}"
        lines.append(f"    {filter_var}.bind(w_jet, {real_arr}, {jet_arr});")
    else:
        lines.append(f"    {filter_var}.bind(w, {real_arr});")
    lines.append("")

    # Initialize R diag entries.
    for m, spec in meas_specs:
        diag = spec.sigma_squared_diag(m)
        rname = f"R_c{spec.craft_idx}_{m.name}"
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

    # Resolve which craft each binding member belongs to. For single-craft
    # worlds the lookup collapses to "craft" / `filter_var`; for multi-craft
    # we look up by craft_ref's id().
    def _craft_var_for_binding(b):
        # All members of one binding must share a craft (the bindings system
        # already enforces this for non-EKF-rooted signals).
        m0 = next(iter(b.members.values()))
        if _is_filter_signal(m0):
            return craft_var_for(0)   # not used for filter-rooted signals
        cidx = craft_id_to_idx.get(id(m0.craft_ref), 0)
        return craft_var_for(cidx)

    # Apply in-bindings.
    for bid, b in bind_assignments:
        if b.direction == "in":
            _emit_input_binding_apply_indented(
                lines, bid, b, _craft_var_for_binding(b), filter_var,
                templated_craft=True, indent="    ")

    # Mirror per-tick actuator command state from each value-side craft to
    # its Jet shadow before predict(). Without this, predict()'s Jacobian
    # step (which runs `w_jet.step()` with Jet-typed scalars) would see
    # default-valued actuators — zero throttle, zero motor torque, etc. —
    # regardless of what the user just commanded on the value craft via
    # cross-world `connect()` or external Zenoh inputs. The value craft
    # holds the user-facing command state; the Jet shadow gets it copied
    # one-way each tick. Sensor measurement state (last_accel etc.) is
    # NOT mirrored: the Jet sensors recompute from the Jet kinematic
    # state inside `w_jet.step()` and we never read from them.
    if needs_jet:
        mirror_lines: list[str] = []
        for c_idx, c in enumerate(unique_crafts):
            real_var = craft_var_for(c_idx)
            jet_var  = craft_jet_var_for(c_idx)
            for part in c.all_parts():
                pairs = getattr(type(part), "actuator_state", None) or []
                for setter, getter in pairs:
                    # Wrap in JetType(...) so scalar getters' `double`
                    # returns become Jets with zero gradient (the actuator
                    # command is an external input, not a state variable
                    # being differentiated).
                    mirror_lines.append(
                        f"    {jet_var}.{part.name}().{setter}("
                        f"JetType({real_var}.{part.name}().{getter}()));")
        if mirror_lines:
            lines.append("")
            lines += mirror_lines

    # Fused single-pass predict + update (PyPose-style). begin_step seeds
    # Jets at x_pre with identity, runs `w_jet.kinematic_and_aggregate()` to populate
    # the Jet sensor caches; each `add_update` reads h(x_pre) + H from
    # those caches and queues a sequential update; end_step advances
    # the Jet world to x_post (no re-aggregate), reads F, computes
    # P_pre, applies queued updates, and mirrors posterior to value.
    #
    # Block-decomposed variant runs NumCrafts smaller Jet passes
    # bracketed by begin_craft/end_craft. Each pass seeds one craft
    # with identity-deriv Jets, others with zero-deriv values, evaluates,
    # collects that craft's measurements, then advances. All sensors
    # for craft k must be added between `begin_craft(k)` and
    # `end_craft()` so the H block lands in the right column slice.
    if needs_jet:
        block_decomposed = getattr(filter_obj, "block_decomposed", False)
        lines += [
            "",
            f"    {filter_var}.begin_step(DT, g_Q);",
            "",
        ]
        if block_decomposed:
            # Group measurement specs by craft_idx so each pass picks
            # up exactly one craft's sensors. Stable ordering keeps
            # codegen deterministic.
            specs_by_craft: list[list[tuple[object, _MeasFunctor]]] = [
                [] for _ in range(num_crafts)
            ]
            for m, spec in meas_specs:
                specs_by_craft[spec.craft_idx].append((m, spec))
            for k in range(num_crafts):
                lines.append(f"    // ---- craft {k}: per-craft Jet pass ----")
                lines.append(f"    {filter_var}.begin_craft({k});")
                for m, spec in specs_by_craft[k]:
                    rname = f"R_c{spec.craft_idx}_{m.name}"
                    part_acc = f"{craft_var_for(spec.craft_idx)}.{m.name}()"
                    spec.emit_update_block(
                        lines, filter_var, part_acc, rname, indent="    ")
                lines.append(f"    {filter_var}.end_craft();")
                lines.append("")
        else:
            for m, spec in meas_specs:
                rname = f"R_c{spec.craft_idx}_{m.name}"
                part_acc = f"{craft_var_for(spec.craft_idx)}.{m.name}()"
                spec.emit_update_block(lines, filter_var, part_acc, rname, indent="    ")
        lines += [
            "",
            f"    {filter_var}.end_step();",
            "",
        ]
    else:
        # UKF path keeps the legacy predict + per-sensor update_n flow
        # (no Jet world, no fused step).
        lines += [
            "",
            f"    {filter_var}.predict(DT, g_Q);",
            "",
        ]
        for m, spec in meas_specs:
            rname = f"R_c{spec.craft_idx}_{m.name}"
            part_acc = f"{craft_var_for(spec.craft_idx)}.{m.name}()"
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
                    lines, bid, b, _craft_var_for_binding(b), filter_var,
                    indent="        ")
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
    n_crafts = len(filter_obj.world.crafts)
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
        f'    std::printf("{target.name}: ready ({kind_label}). {n_crafts} craft(s), '
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
    double — patch runs) or a plain non-templated craft (Scalar=MFloat,
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
    were authored for MFloat-scalared crafts; the EKF path needs the
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

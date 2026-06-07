"""World-tick compilation: one shared CasADi function over every craft.

`Sim(world)` always routes through `compile_world_tick` —
there's no per-component partitioning. The resulting CompiledGraph
takes the concatenated state of every craft (with `<craft_name>.`
prefixed slot names) plus every state-bearing disturbance and
advances all of them by one tick. Coupling wrenches (e.g. Tether)
are computed symbolically inside the graph and injected into each
craft's net wrench BEFORE Newton-Euler, so the per-craft dynamics see
the full inter-craft coupling.

Field-mediated craft-to-craft coupling (e.g. a `CraftWindBubble`
anchored to craft A whose contribution another craft B's DragSurface
samples) also Just Works — both crafts' symbolic state is in scope
during the single tick compile.

The per-craft physics pipeline is: inertials → state inputs →
TickContext → part wrench aggregation → Newton-Euler → symplectic
integration → outputs. The multi-craft wrapper around it (a)
prefixes all per-craft IO names with the craft name, (b) shares one
`dt` + `t` input across all crafts, and (c) splices in coupling
wrenches between the part-aggregation and N-E steps. The N=1 case is
not special-cased — a single craft is just a one-element component.
"""

from __future__ import annotations

from typing import Any

import casadi as ca
import numpy as np

from .. import ir
from ..craft import (
    TickContext, _aggregate_inertials,
    _wrench_rotate_to_craft, _shift_wrench,
)
from .inertia import symbolic_inertia_rollup
from .joint_space import DEGENERATE_INERTIA_EPS, build_joint_space
from .kinematics import kinematic_pass
from ..ir.frames import WorldFrame, CraftFrame, PartFrame
from ..ir.manifold import SO3Manifold
from ..parts.base import CompositePart, Part, PartUpdate, TraceBindings
from ..ir.wrench import Wrench


def compile_world_tick(crafts: list,
                       couplings: list,
                       *,
                       gravity_field=None,
                       fluid_field=None,
                       mag_field=None,
                       collision_field=None,
                       world=None,
                       ) -> "ir.graph.CompiledGraph":
    """Compile a CasADi-MX tick over multiple coupled crafts.

    Args:
        crafts    — list of Craft instances. Must contain every craft
                    referenced by `couplings`.
        couplings — list of Coupling instances (subclasses with
                    compute_wrenches_sym(ctx_a, ctx_b) → (Wrench, Wrench)).
        gravity_field, fluid_field, mag_field, collision_field —
                    the world's shared fields, queried per craft.
                    Shared across all crafts in this component. Any
                    unset field defaults
                    to its empty form (zero gravity / zero density /
                    zero B / zero penetration).
        world     — the owning World, if any; handed to each TickContext
                    so parts can introspect optional registrations
                    (`ctx.has_field`, `ctx.get_field`, `ctx.planet`).

    State layout (input + output):
        <craft_name>.position
        <craft_name>.orientation
        <craft_name>.velocity
        <craft_name>.angular_velocity
        <craft_name>.<part>.<state>      (per-part State slots)
        <craft_name>.<part>.<input>      (per-part Input slots, input-only)
    Plus a shared `dt` input. Sensor outputs use the same prefix.
    """
    # Resolve field defaults.
    from ..fields import (
        CollisionField as _CollisionField,
        FluidField as _FluidField,
        GravityField as _GravityField,
        MagField as _MagField,
    )
    if gravity_field is None:
        gravity_field = _GravityField()
    if fluid_field is None:
        fluid_field = _FluidField()
    if mag_field is None:
        mag_field = _MagField()
    if collision_field is None:
        collision_field = _CollisionField()

    if not crafts:
        raise ValueError("compile_world_tick: needs at least one craft.")

    # Validate couplings reference only crafts in this component.
    craft_set = {id(c) for c in crafts}
    for cp in couplings:
        if id(cp.craft_a) not in craft_set or id(cp.craft_b) not in craft_set:
            raise ValueError(
                "compile_world_tick: coupling references a craft not in "
                "the given crafts list.")

    # Quick numpy snapshot per craft (mass-positivity guard only). The
    # actual COM / I_com used in Newton-Euler are symbolic and built
    # inside the ir.Graph block below — they pick up joint-angle
    # dependence when a joint reorients a rotor.
    for craft in crafts:
        ai = _aggregate_inertials(craft.parts)
        if ai["m_total"] <= 0.0:
            raise ValueError(
                f"Craft '{craft.name}': total mass is "
                f"{ai['m_total']}; need m > 0.")

    name = "_".join(c.name for c in crafts) + "_world_tick"
    # All State/Input/Noise attribute reads (and the per-craft rigid-body
    # state symbols) inside the trace resolve through `trace` (thread-local
    # TraceBindings) — the part, disturbance, and craft instances are never
    # mutated, so there is nothing to restore, even when a part's update()
    # raises mid-trace.
    with ir.Graph(name=name) as g, TraceBindings() as trace:
        dt = ir.Scalar.input("dt")
        t  = ir.Scalar.input("t")

        # Pass 0a: rigid-body state inputs for every craft, bound on the
        # trace so disturbances (and the per-craft trace below) can
        # reference any craft's symbolic state. This must precede
        # disturbance plumbing because a disturbance might close over a
        # craft's position (e.g. CraftWindBubble).
        for craft in crafts:
            prefix = f"{craft.name}."
            trace.bind_craft_state(craft, {
                "position":         ir.Vec3[WorldFrame].input(prefix + "position"),
                "orientation":      ir.Quat[WorldFrame, CraftFrame].input(prefix + "orientation"),
                "velocity":         ir.Vec3[WorldFrame].input(prefix + "velocity"),
                "angular_velocity": ir.Vec3[CraftFrame].input(prefix + "angular_velocity"),
            })

        # Pass 0b: plumb State/Noise declarations on field disturbances.
        # These bindings need to be in scope BEFORE any part's update()
        # queries a field.
        all_fields = [gravity_field, fluid_field, mag_field,
                      collision_field]
        dist_state_outputs = _plumb_field_disturbances(
            all_fields, dt, trace)

        # Pass 1: per-craft trace. The rigid-body state symbols are
        # already created (Pass 0a); the per-craft helper picks them up
        # from the trace.
        per_craft: dict[int, dict[str, Any]] = {}
        for craft in crafts:
            per_craft[id(craft)] = _trace_craft_pass1(
                craft, gravity_field, fluid_field, mag_field, dt, t,
                trace, collision_field=collision_field, world=world)

        # Pass 2: coupling wrench injection.
        for cp in couplings:
            pc_a = per_craft[id(cp.craft_a)]
            pc_b = per_craft[id(cp.craft_b)]
            w_a, w_b = cp.compute_wrenches_sym(pc_a["ctx"], pc_b["ctx"])
            pc_a["net"] = pc_a["net"] + w_a
            pc_b["net"] = pc_b["net"] + w_b

        # Pass 3: finalize per-craft dynamics.
        for craft in crafts:
            _emit_per_craft_dynamics(g, craft, per_craft[id(craft)], dt)

        # Disturbance state outputs (deterministic passthrough for plain
        # State; bias_next = bias + sqrt(dt)·driver for RW Noise).
        for out_name, out_val in dist_state_outputs:
            g.output(out_val, out_name)

    cg = g.compile(defaults={"t": 0.0})
    # Aggregate per-part rate declarations (ctx.sample / ctx.hold) onto
    # the compiled tick. Approach A: this is pure metadata — the kernel
    # is unchanged; the Sim/EKF runtimes read it to gate their I/O ports.
    rates: dict[str, float] = {}
    for craft in crafts:
        rates.update(per_craft[id(craft)].get("sample_rates", {}))
    cg.sample_rates = rates
    return cg


# ---------------------------------------------------------------------------
# Field-disturbance state plumbing
# ---------------------------------------------------------------------------

def _plumb_field_disturbances(fields, dt, trace) -> list[tuple[str, Any]]:
    """Walk every disturbance on each field, binding its declared
    State / Noise attributes to symbolic graph inputs (via `trace`; the
    disturbance instance is never mutated). Returns `state_outputs` —
    `(output_name, output_value)` pairs the caller emits as graph outputs
    after the part-loop completes."""
    from ..fields.base import Disturbance

    state_outputs: list[tuple[str, Any]] = []
    seen_names: set[str] = set()

    for field in fields:
        if field is None:
            continue
        for dist in field._disturbances:
            if not isinstance(dist, Disturbance):
                continue
            sdecls = dist.state_declarations()
            ndecls = dist.noise_declarations()
            if not sdecls and not ndecls:
                continue
            if dist.name in seen_names:
                raise ValueError(
                    f"compile_world_tick: duplicate disturbance name "
                    f"{dist.name!r}. Disturbance names must be unique "
                    f"within a world.")
            seen_names.add(dist.name)

            prefix = f"{dist.name}."

            # User-declared State slots: input → identity passthrough.
            # (Disturbance state advances only via paired RW Noise.)
            for sname, sdecl in sdecls.items():
                if sdecl.manifold.kind != "scalar":
                    raise NotImplementedError(
                        f"{type(dist).__name__}('{dist.name}'): State "
                        f"manifold kind {sdecl.manifold.kind!r} not yet "
                        f"supported on disturbance-declared state.")
                sym = sdecl.manifold.ir_input(
                    prefix + sname, default_frame=WorldFrame)
                trace.bind(dist, sname, sym)
                state_outputs.append((prefix + sname, sym))

            # Noise channels — synthesis is polymorphic on the Noise
            # subclass; the compiler just binds the returned symbol and
            # optionally records the synthesized state update.
            for nname, ndecl in ndecls.items():
                synth = ndecl.synthesize(
                    base_name=prefix + nname,
                    name=nname,
                    dt=dt,
                    default_frame=WorldFrame,
                    owner=dist,
                )
                trace.bind(dist, nname, synth.signal_sym)
                if synth.state_update is not None:
                    state_outputs.append(synth.state_update)

    return state_outputs


# ---------------------------------------------------------------------------
# Per-craft helpers
# ---------------------------------------------------------------------------

def _typed_views(fv_raw: dict) -> dict:
    """Wrap the kinematic pass's raw-MX frame views in typed `Vec3[F]`
    values — the single place the typed frame tags are applied. Each
    entry X[F] is the quantity relative to frame F, in F's coords, so
    the tag is exactly the view's frame."""
    return {qty: {F: ir.Vec3[F].from_mx(mx) for F, mx in by_frame.items()}
            for qty, by_frame in fv_raw.items()}


def _trace_craft_pass1(craft,
                       gravity_field,
                       fluid_field,
                       mag_field,
                       dt,
                       t,
                       trace,
                       collision_field=None,
                       world=None) -> dict[str, Any]:
    """Set up one craft's state inputs, part bindings (via `trace`),
    TickContext, run all parts' update(), and collect their wrench
    contributions + state outputs + sensor outputs. Returns a dict
    carrying everything the later passes need."""
    prefix = f"{craft.name}."

    # Rigid-body state symbols were created in `compile_world_tick`'s
    # Pass 0a and bound on the trace (so disturbances anchored to other
    # crafts can reference them). Pick them up here.
    sym_state   = trace.craft_sym_state(craft)
    position    = sym_state["position"]
    orientation = sym_state["orientation"]
    velocity    = sym_state["velocity"]
    ang_vel     = sym_state["angular_velocity"]
    # Placeholder MX symbols for current-tick body acceleration / α
    # (substituted with the real Newton-Euler outputs after the wrench
    # sum is known — see the TickContext docstring in craft.py for why
    # the a/α a part reads must be a compile-time placeholder).
    a_world_sym = ca.MX.sym(f"{craft.name}_a_world", 3, 1)
    alpha_sym    = ca.MX.sym(f"{craft.name}_alpha", 3, 1)

    if collision_field is None:
        from ..fields import CollisionField as _CF
        collision_field = _CF()

    # State + Input bindings on each part — must happen BEFORE the
    # kinematic pass, so it sees the joint angle/rate symbols when
    # building per-part kinematic states.
    state_input_nodes: dict[Part, dict[str, Any]] = {}
    for part in craft.parts:
        decls = part.state_declarations()
        if decls:
            part_states: dict[str, Any] = {}
            for sname, sdecl in decls.items():
                input_name = prefix + f"{part.name}.{sname}"
                sym = sdecl.manifold.ir_input(
                    input_name, default_frame=CraftFrame)
                part_states[sname] = sym
                trace.bind(part, sname, sym)
            state_input_nodes[part] = part_states

        for iname in part.input_declarations():
            trace.bind(part, iname,
                       ir.Scalar.input(prefix + f"{part.name}.{iname}"))

    # Per-part Noise plumbing — synthesis is polymorphic on the Noise
    # subclass (see manta/parts/base.py). The compiler binds the
    # returned symbol (via `trace`) and records any synthesized state
    # update.
    rw_bias_updates: list[tuple[str, Any]] = []   # (state_name, bias_next)
    for part in craft.parts:
        for nname, ndecl in part.noise_declarations().items():
            input_name = prefix + f"{part.name}.{nname}"
            synth = ndecl.synthesize(
                base_name=input_name,
                name=nname,
                dt=dt,
                default_frame=CraftFrame,
                owner=part,
            )
            trace.bind(part, nname, synth.signal_sym)
            if synth.state_update is not None:
                rw_bias_updates.append(synth.state_update)

    # Per-joint DOF-acceleration placeholders. A joint's θ̈/d̈ is not a
    # state slot — it's an output of the joint-space solve (a pure
    # function of state) — so, exactly like the body's a/α, the kinematic
    # pass takes a placeholder symbol now and the framework substitutes
    # the solved q̈ after Newton-Euler. These feed the joint-relative
    # acceleration of every part on the rotor (and hence rotor-mounted
    # sensors + the moving-COM origin recoil).
    from ..parts.articulation.joint import ArticulatedJoint
    joint_accel_syms: dict[Any, Any] = {}
    for part in craft.parts:
        if isinstance(part, ArticulatedJoint):
            joint_accel_syms[part] = ca.MX.sym(
                f"{craft.name}_{part.name}_jaccel", 1, 1)

    # Symbolic kinematic + inertia passes over the part tree. The
    # kinematic pass is frame-blind raw MX (see kinematics.py); the typed
    # frame tags are applied HERE, once, where each ctx is built.
    kin_states = kinematic_pass(
        craft.root, position._mx, orientation._mx, velocity._mx,
        ang_vel._mx,
        body_acceleration_world=a_world_sym,
        body_angular_acceleration=alpha_sym,
        joint_angular_accels=joint_accel_syms)
    inertia = symbolic_inertia_rollup(craft.root)

    fields_tuple = (gravity_field, fluid_field, mag_field, collision_field)

    # Per-craft TickContext (root view) for couplings to read. The root's
    # own frame IS CraftFrame, so its frame-indexed views + body attitude
    # come straight from the root kinematic state.
    root_kin = kin_states[craft.root]
    fv_root = _typed_views(root_kin.frame_views)
    ctx = TickContext(
        t=t,
        dt=dt,
        fields=fields_tuple,
        world=world,
        orientation=orientation,                 # Quat[WorldFrame, CraftFrame]
        position=fv_root["position"],
        velocity=fv_root["velocity"],
        acceleration=fv_root["acceleration"],
        angular_velocity=fv_root["angular_velocity"],
        angular_acceleration=fv_root["angular_acceleration"],
        R_craft_from_part=ir.Mat3[CraftFrame, CraftFrame].from_mx(
            root_kin.R_craft_from_input),
    )

    # Per-part external wrench, rotated to body coords about the part's
    # own origin (NOT lifted). The cascade below reduces these into the
    # craft net wrench; the joint-space assembly projects them onto each
    # joint's motion subspace.
    own_wrench: dict[Part, Wrench] = {}
    new_state_outputs: list[tuple[str, Any]] = []
    sensor_outputs:    list[tuple[str, Any]] = []
    # Rate declarations (ctx.sample / ctx.hold) → {full_io_name: rate_hz}.
    sample_rates:      dict[str, float] = {}
    for part in craft.parts:
        kin = kin_states[part]
        # A part's update() works entirely in its OWN frame. ctx exposes
        # each kinematic quantity as a frame-indexed view (X[Frame] = the
        # value relative to Frame, in Frame coords), the part's own world
        # attitude, and the Craft←Part rotation. The wrench the part returns
        # is in PartFrame; `_wrench_rotate_to_craft` maps it to body coords.
        orientation_part = ir.Quat[WorldFrame, PartFrame].from_mx(
            kin.q_world_from_input)
        R_craft_from_part = ir.Mat3[CraftFrame, PartFrame].from_mx(
            kin.R_craft_from_input)
        fv = _typed_views(kin.frame_views)
        ctx_part = TickContext(
            t=t,
            dt=dt,
            fields=fields_tuple,
            world=world,
            orientation=orientation_part,
            position=fv["position"],
            velocity=fv["velocity"],
            acceleration=fv["acceleration"],
            angular_velocity=fv["angular_velocity"],
            angular_acceleration=fv["angular_acceleration"],
            R_craft_from_part=R_craft_from_part,
        )
        result = part.update(ctx_part)
        if isinstance(result, Wrench):
            w_part = result; new_state = {}; outputs = {}
        elif isinstance(result, PartUpdate):
            w_part, new_state, outputs = result.wrench, result.new_state, result.outputs
        else:
            raise TypeError(
                f"{type(part).__name__}('{part.name}').update(): must "
                f"return Wrench or PartUpdate, got {type(result).__name__}")
        if w_part.frame is not PartFrame:
            from ..ir.frames import FrameError, _capture_user_source
            raise FrameError(
                f"{type(part).__name__}.update",
                expected="Wrench in PartFrame",
                got=f"frame={w_part.frame.__name__}",
                source=_capture_user_source(),
            )

        # State validation + queue.
        decls = part.state_declarations()
        unknown = set(new_state) - set(decls)
        if unknown:
            raise KeyError(
                f"{type(part).__name__}('{part.name}'): unknown state "
                f"slot(s): {sorted(unknown)}.")
        # A joint's DOF/rate are emitted by the central integrator in the
        # cascade below (its update() is a no-op), so skip them here to
        # avoid emitting stale (un-integrated) values.
        if not isinstance(part, ArticulatedJoint):
            for sname in decls:
                val = new_state.get(sname, state_input_nodes[part][sname])
                # SO(3) state: defensively renormalize the quaternion so
                # multi-tick integration doesn't drift off the unit sphere
                # — mirrors the rigid-body orientation handling below. A
                # Part integrates such a slot via SO3Manifold().boxplus,
                # which already returns a (frame-tagged) Quat.
                if decls[sname].manifold.kind == "quat":
                    val = val.normalize()
                new_state_outputs.append(
                    (prefix + f"{part.name}.{sname}", val))

        # Output validation + queue.
        out_decls = part.output_declarations()
        unknown_out = set(outputs) - set(out_decls)
        missing_out = set(out_decls) - set(outputs)
        if unknown_out:
            raise KeyError(
                f"{type(part).__name__}('{part.name}'): unknown output "
                f"slot(s): {sorted(unknown_out)}.")
        if missing_out:
            raise KeyError(
                f"{type(part).__name__}('{part.name}'): output slot(s) "
                f"declared but not written: {sorted(missing_out)}.")
        for oname, oval in outputs.items():
            sensor_outputs.append((prefix + f"{part.name}.{oname}", oval))

        # Rate declarations: match the values passed to ctx.sample()/
        # ctx.hold() (by identity) against this part's emitted outputs and
        # its bound Input symbols, recording {full_io_name: rate_hz}.
        if ctx_part._sample_records:
            rate_by_id = {sid: r for sid, r, _ in ctx_part._sample_records}
            for oname, oval in outputs.items():
                r = rate_by_id.get(id(oval))
                if r is not None:
                    sample_rates[prefix + f"{part.name}.{oname}"] = r
            for iname in part.input_declarations():
                r = rate_by_id.get(id(getattr(part, iname)))
                if r is not None:
                    sample_rates[prefix + f"{part.name}.{iname}"] = r

        # Part-frame wrench → rotate to body coords (NO lift). The cascade
        # below lifts + sums it up the joint tree.
        own_wrench[part] = _wrench_rotate_to_craft(w_part, R_craft_from_part)

    # --- Bottom-up wrench cascade --------------------------------------
    # Reduce the per-part wrenches up the part tree: every composite —
    # joints included — sums its children (each shifted to the
    # composite's own origin). NOTHING is split or absorbed at a joint:
    # the net wrench is the craft's full external wrench about the
    # origin, which both the system-COM linear theorem (a_com = F/m) and
    # the body rows of the joint-space solve need whole. The per-hop
    # shifts telescope to the single r×F lift of the flat path.
    zero_w = Wrench.zero(CraftFrame)

    def _cascade(part) -> Wrench:
        accum = own_wrench.get(part, zero_w)
        r_part = kin_states[part].r_in_craft
        if isinstance(part, CompositePart):
            for child in part.children:
                child_up = _cascade(child)
                delta_r = ir.Vec3[CraftFrame].from_mx(
                    kin_states[child].r_in_craft - r_part)
                accum = accum + _shift_wrench(child_up, delta_r)
        return accum

    net = _cascade(craft.root)

    # Body-relative COM motion for the moving-COM origin recoil — the
    # mass-weighted reduction of the per-part relative kinematics the
    # kinematic pass just produced (same quantities a rotor-mounted sensor
    # reads). Zero when no joint shifts the COM. Carries the θ̈/d̈
    # placeholders via a_rel; resolved in `_emit_per_craft_dynamics`.
    m_total = inertia["m_total"]
    v_com_rel_mx = ca.MX.zeros(3, 1)
    a_com_rel_mx = ca.MX.zeros(3, 1)
    for part in craft.parts:
        m = float(getattr(part, "mass", 0.0) or 0.0)
        if m <= 0.0:
            continue
        kin = kin_states[part]
        v_com_rel_mx = v_com_rel_mx + m * kin.velocity_rel_body
        a_com_rel_mx = a_com_rel_mx + m * kin.acceleration_rel_body
    v_com_rel_mx = v_com_rel_mx / m_total
    a_com_rel_mx = a_com_rel_mx / m_total

    # --- Joint-space block assembly --------------------------------------
    # The coupled body ⊕ joint dynamics: generalized mass matrix A(q)
    # over [ω; q̇], Hamel bias, and the joints' virtual-work generalized
    # forces (see tick/joint_space.py). None for a craft without live
    # articulated DOFs — `_emit_per_craft_dynamics` then keeps the plain
    # 3×3 body solve, bit-identical to the flat path.
    joint_space = build_joint_space(
        craft,
        kin_states=kin_states,
        inertia=inertia,
        state_input_nodes=state_input_nodes,
        own_wrench=own_wrench,
        om_mx=ang_vel._mx,
        v_com_rel_mx=v_com_rel_mx,
        joint_accel_syms=joint_accel_syms)

    return {
        "position":          position,
        "orientation":       orientation,
        "velocity":          velocity,
        "ang_vel":           ang_vel,
        "ctx":               ctx,
        "net":               net,
        "inertia":           inertia,
        "new_state_outputs": new_state_outputs,
        "sensor_outputs":    sensor_outputs,
        "sample_rates":      sample_rates,
        "rw_bias_updates":   rw_bias_updates,
        "joint_space":       joint_space,
        "joint_accel_syms":  joint_accel_syms,
        "state_input_nodes": state_input_nodes,
        "com_rel_motion":    (v_com_rel_mx, a_com_rel_mx),
        "a_world_sym":      a_world_sym,
        "alpha_sym":         alpha_sym,
    }


def _emit_per_craft_dynamics(g_ctx, craft, pc, dt) -> None:
    """Newton-Euler + symplectic integration + emit state/sensor outputs
    for one craft. Reads pc["net"] (which by this point includes any
    coupling-injected wrenches) and pc["inertia"] (symbolic rollup)."""
    prefix = f"{craft.name}."

    position    = pc["position"]
    orientation = pc["orientation"]
    velocity    = pc["velocity"]
    ang_vel     = pc["ang_vel"]
    net         = pc["net"]
    inertia     = pc["inertia"]

    m_total          = inertia["m_total"]
    com_mx           = inertia["com_in_craft_mx"]
    I_com_mx         = inertia["I_com_in_craft_mx"]
    I_com_at_zero_np = inertia["I_com_at_zero"]

    F_craft    = net.force
    tau_origin = net.torque
    r_com = ir.Vec3[CraftFrame].from_mx(com_mx)
    tau_com = tau_origin - r_com.cross(F_craft)

    f_world = orientation.apply(F_craft / m_total)
    a_com_world = f_world

    I_com   = ir.Mat3[CraftFrame, CraftFrame].from_mx(I_com_mx)
    I_omega = I_com @ ang_vel
    tau_eff = tau_com - ang_vel.cross(I_omega)

    js = pc["joint_space"]
    joint_accel_syms  = pc["joint_accel_syms"]
    state_input_nodes = pc["state_input_nodes"]
    dt_mx = dt._mx

    # --- Coupled body ⊕ joint solve ---------------------------------------
    # With live articulated DOFs, body α and every joint q̈ come from ONE
    # block system A(q)·[α; q̈] = [τ_com; Q_q] − bias (see
    # tick/joint_space.py): nested-gimbal coupling, configuration-
    # dependent generalized inertia, gyroscopic couples, Coriolis joint
    # torques, recoil, and the mount's full (angular AND linear)
    # acceleration feedback are all exact — uniform gravity cancels in
    # the joint rows, so a free-falling pendulum no longer "swings".
    # Without joints: the plain 3×3 body solve, bit-identical to before.
    qdd_by_part: dict[Any, Any] = {}
    if js is None:
        if np.linalg.det(I_com_at_zero_np) > DEGENERATE_INERTIA_EPS:
            alpha_mx = ca.solve(I_com_mx, tau_eff._mx)
            alpha = ir.Vec3[CraftFrame].from_mx(alpha_mx)
        else:
            alpha = ir.Vec3[CraftFrame].constant((0.0, 0.0, 0.0))
    else:
        rhs_mx = ca.vertcat(tau_com._mx, js["force_q"]) - js["bias"]
        acc_mx = ca.solve(js["A"], rhs_mx)
        alpha = ir.Vec3[CraftFrame].from_mx(acc_mx[0:3])
        for jdx, j_part in enumerate(js["joints"]):
            qdd_by_part[j_part] = acc_mx[3 + jdx]

    # --- Central integration of the joint DOF states ----------------------
    # Same semi-implicit scheme as the body integrator. Locked joints
    # (zero generalized inertia) hold q̈ = 0. The rate output is later
    # overridden by the momentum-form value for live joints.
    dof_state_outputs: list[tuple[str, Any]] = []
    for j_part in joint_accel_syms:
        qdd_mx = qdd_by_part.get(j_part, ca.MX(0.0))
        pos_name, rate_name = j_part.dof_state_names()
        rate_mx = state_input_nodes[j_part][rate_name]._mx
        pos_mx  = state_input_nodes[j_part][pos_name]._mx
        new_pos_mx = (pos_mx + rate_mx * dt_mx
                      + 0.5 * qdd_mx * dt_mx * dt_mx)
        dof_state_outputs.append(
            (prefix + f"{j_part.name}.{pos_name}", ir.Scalar(new_pos_mx)))
        dof_state_outputs.append(
            (prefix + f"{j_part.name}.{rate_name}",
             ir.Scalar(rate_mx + qdd_mx * dt_mx)))

    r_OC = -r_com
    offset_term = alpha.cross(r_OC) + ang_vel.cross(ang_vel.cross(r_OC))
    a_origin_world = a_com_world + orientation.apply(offset_term)

    # --- Moving-COM correction (articulated craft) -----------------------
    # When a joint actuates, the COM translates within the body, so r_OC is
    # time-varying and the rigid origin-transfer above is incomplete. Add the
    # missing 2ω×ṙ_OC + r̈_OC = −2ω·v_com_body − a_com_body, where the
    # body-relative COM motion (v_com_body, a_com_body) is the mass-weighted
    # reduction of the per-part relative kinematics from the kinematic pass
    # (built in pass-1; see `com_rel_motion`). It carries the joint θ̈
    # placeholders (via a_rel), resolved below alongside the body a/α.
    # Identically zero when no joint shifts the COM (on-axis symmetric
    # rotors), and what keeps a free-floating articulated craft's system COM
    # from drifting (linear-momentum conservation).
    v_com_rel_mx, a_com_rel_mx = pc["com_rel_motion"]
    v_com_body = ir.Vec3[CraftFrame].from_mx(v_com_rel_mx)
    a_com_body = ir.Vec3[CraftFrame].from_mx(a_com_rel_mx)
    moving_term = ang_vel.cross(v_com_body) * (-2.0) - a_com_body
    a_origin_world = a_origin_world + orientation.apply(moving_term)

    # --- Validate wrench independence from placeholder dynamics ----------
    # Wrenches must be functions of state only: forbid dependence on the body
    # a/α placeholders AND on any joint q̈ placeholder (each would otherwise
    # leave an unresolved symbol in `net`, which we never substitute).
    a_world_sym = pc["a_world_sym"]
    alpha_sym    = pc["alpha_sym"]
    joint_syms  = []
    joint_reals = []
    for j_part, sym in joint_accel_syms.items():
        joint_syms.append(sym)
        joint_reals.append(qdd_by_part.get(j_part, ca.MX(0.0)))
    # The solved q̈ are pure functions of state by construction (A and the
    # generalized forces contain no placeholders) — assert it, so a
    # future feedback channel can't silently emit an unresolved symbol.
    for real in joint_reals:
        if (ca.depends_on(real, a_world_sym)
                or ca.depends_on(real, alpha_sym)):
            raise ValueError(
                f"Craft '{craft.name}': a joint DOF acceleration depends "
                f"on a body-dynamics placeholder — the joint-space solve "
                f"must be a function of state only.")
    checks = [("acceleration_world", a_world_sym),
              ("acceleration_body", a_world_sym),
              ("angular_acceleration", alpha_sym)]
    checks += [("a joint's angular acceleration", s) for s in joint_syms]
    for sym_name, sym_mx in checks:
        if (ca.depends_on(net.force._mx, sym_mx)
                or ca.depends_on(net.torque._mx, sym_mx)):
            raise ValueError(
                f"Craft '{craft.name}': a part's wrench or a coupling "
                f"depends on ctx.{sym_name}. Wrenches must be a function "
                f"of state only.")

    # --- Substitute placeholders → real dynamics in emitted outputs -----
    # First resolve the joint θ̈ placeholders inside a_origin_world itself
    # (they entered via the moving-COM term) so the integrated body state is
    # placeholder-free; then a_world_sym maps to that resolved expression.
    if joint_syms:
        a_origin_world = ir.Vec3[WorldFrame].from_mx(
            ca.substitute(a_origin_world._mx,
                          ca.vertcat(*joint_syms),
                          ca.vertcat(*joint_reals)))
    placeholders = ca.vertcat(a_world_sym, alpha_sym, *joint_syms)
    real_values  = ca.vertcat(a_origin_world._mx, alpha._mx, *joint_reals)
    from ..ir.types import _IRValue

    def _resolve(val):
        if not isinstance(val, _IRValue):
            return val
        new_mx = ca.substitute(val._mx, placeholders, real_values)
        if isinstance(val, ir.Vec3):
            return type(val)._from_mx(new_mx, frame=val._frame)
        if isinstance(val, (ir.Mat3, ir.Quat)):
            return type(val)._from_mx(new_mx,
                                       from_frame=val._from_frame,
                                       to_frame=val._to_frame)
        return type(val)._from_mx(new_mx)

    new_state_outputs = [(n, _resolve(v))
                          for n, v in pc["new_state_outputs"]]
    new_state_outputs += dof_state_outputs   # solved q̈ — placeholder-free
    sensor_outputs    = [(n, _resolve(v))
                          for n, v in pc["sensor_outputs"]]

    new_velocity = velocity + a_origin_world * dt
    new_position = position + velocity * dt + a_origin_world * (0.5 * dt * dt)

    # --- Structure-preserving angular update ----------------------------
    # Explicit Euler on ω (ω += α·dt) pumps energy into gyroscopic
    # oscillations at λ ≈ Ω²·dt/2 (Ω = nutation rate) — fatal for fast
    # rotors on precessing mounts, and a slow leak for any tumbling body.
    # Instead, integrate the body-frame GENERALIZED momentum p = A·[ω; q̇]
    # (p_ω is the total angular momentum about the COM, p_q the joint
    # momenta):
    #   p_ω⁺ = Exp(−ω·dt)·(p_ω + τ_com·dt)      (world-frame H exactly
    #                                            conserved at zero torque)
    #   p_q⁺ = p_q + (Q_q + ∂T/∂q)·dt           (Hamel joint-row impulse)
    #   [ω⁺; q̇⁺] = A⁻¹·p⁺
    # — the joint rates come from the SAME momentum solve; an Euler rate
    # update alongside the momentum ω update would break the axial-
    # momentum bookkeeping (e.g. a reaction wheel's exact axial balance).
    # Without joints this is the plain H⁺ = Exp(−ω·dt)(H + τ·dt) update.
    # First-order identical to ω += α·dt, so flat-craft behaviour shifts
    # only at O(dt²).
    om_mx = ang_vel._mx
    # Counter-rotation Exp(−ω·dt) via Rodrigues (branch-free at ω→0).
    # The MIDPOINT rate (ω + ½α·dt) sets the rotation: a first-order
    # (old-ω) rotation leaves an O(dt) phase error that pumps kinetic
    # energy into nutation at λ ≈ Ω²·dt/2; the midpoint drops the pump
    # to O(dt²) — negligible at any practical rate.
    rv = -(om_mx + 0.5 * alpha._mx * dt_mx) * dt_mx
    th2 = ca.dot(rv, rv)
    th = ca.sqrt(th2 + 1e-30)
    K = ca.skew(rv)
    R_neg = (ca.MX.eye(3) + (ca.sin(th) / th) * K
             + ((1.0 - ca.cos(th)) / (th2 + 1e-30)) * (K @ K))
    rate_overrides: dict[str, Any] = {}
    if js is None:
        H_mx = I_com_mx @ om_mx
        H_new_mx = R_neg @ (H_mx + tau_com._mx * dt_mx)
        if np.linalg.det(I_com_at_zero_np) > DEGENERATE_INERTIA_EPS:
            new_om_mx = ca.solve(I_com_mx, H_new_mx)
        else:
            new_om_mx = om_mx + alpha._mx * dt_mx
    else:
        p_mx = js["p"]
        p_w_new = R_neg @ (p_mx[0:3] + tau_com._mx * dt_mx)
        p_q_new = p_mx[3:] + (js["force_q"] + js["dTdq"]) * dt_mx
        # Recover the new velocities from the ADVANCED configuration's
        # mass matrix: next tick rebuilds p = A(q⁺)·u⁺, so solving with
        # the advanced A makes the momentum flow consistent — solving
        # with the stale A(q) silently drops the (∂A/∂q·q̇)·u drift and
        # leaks angular momentum whenever the mass matrix is
        # configuration-dependent (offset hinges, gimbals, sliders).
        # Advance with q + q̇·dt (NOT the full q⁺ = q + q̇·dt + ½q̈·dt²):
        # the ½q̈dt² term only changes A at O(dt²) per step — the same
        # tier as the rest of the integrator — while substituting q̈ (the
        # whole block-solve expression) into every entry of A would
        # square the graph size and blow up code generation.
        q_pos_syms, q_pos_adv = [], []
        for j_part in js["joints"]:
            pos_name, rate_name = j_part.dof_state_names()
            pos_mx  = state_input_nodes[j_part][pos_name]._mx
            rate_mx = state_input_nodes[j_part][rate_name]._mx
            q_pos_syms.append(pos_mx)
            q_pos_adv.append(pos_mx + rate_mx * dt_mx)
        A_new = ca.substitute(js["A"],
                              ca.vertcat(*q_pos_syms),
                              ca.vertcat(*q_pos_adv))
        u_new = ca.solve(A_new, ca.vertcat(p_w_new, p_q_new))
        new_om_mx = u_new[0:3]
        for jdx, j_part in enumerate(js["joints"]):
            rate_overrides[prefix + f"{j_part.name}.rate"] = ir.Scalar(
                u_new[3 + jdx])
    new_ang_vel = ir.Vec3[CraftFrame].from_mx(new_om_mx)
    new_state_outputs = [(n, rate_overrides.get(n, v))
                         for n, v in new_state_outputs]
    omega_dt_world = orientation.apply(ang_vel * dt)
    new_orientation = SO3Manifold().boxplus(orientation, omega_dt_world).normalize()

    g_ctx.output(new_position,    prefix + "position")
    g_ctx.output(new_orientation, prefix + "orientation")
    g_ctx.output(new_velocity,    prefix + "velocity")
    g_ctx.output(new_ang_vel,     prefix + "angular_velocity")
    for out_name, out_val in new_state_outputs:
        g_ctx.output(out_val, out_name)
    # RW bias state outputs (per-craft prefix already baked into the
    # `input_name` recorded during pass-1 plumbing).
    for bias_name, bias_next in pc.get("rw_bias_updates", []):
        g_ctx.output(bias_next, bias_name)
    for out_name, out_val in sensor_outputs:
        if not isinstance(out_val, _IRValue):
            raise TypeError(
                f"Output '{out_name}': must be an IR value; got "
                f"{type(out_val).__name__}")
        g_ctx.output(out_val, out_name)

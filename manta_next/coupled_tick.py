"""World-tick compilation: one shared CasADi function over every craft.

`World.compile()` always routes through `compile_coupled_tick` —
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

The per-craft physics pipeline mirrors the original
`Craft.compile_tick` (inertials → state inputs → TickContext → part
wrench aggregation → Newton-Euler → symplectic integration → outputs).
It diverges by: (a) prefixing all per-craft IO names with the craft
name, (b) sharing one `dt` + `t` input across all crafts, and (c)
splicing in coupling wrenches between the part-aggregation and N-E
steps.
"""

from __future__ import annotations

from typing import Any

import casadi as ca
import numpy as np

from . import ir
from .craft import TickContext, _aggregate_inertials, _wrench_to_craft
from .inertia import symbolic_inertia_rollup
from .kinematics import kinematic_pass
from .ir.frames import WorldFrame, CraftFrame
from .ir.manifold import SO3
from .parts.base import Part, PartUpdate
from .ir.wrench import Wrench


def compile_coupled_tick(crafts: list,
                          couplings: list,
                          *,
                          gravity_field=None,
                          fluid_field=None,
                          mag_field=None,
                          collision_field=None,
                          ) -> "ir.graph.CompiledGraph":
    """Compile a CasADi-MX tick over multiple coupled crafts.

    Args:
        crafts    — list of Craft instances. Must contain every craft
                    referenced by `couplings`.
        couplings — list of Coupling instances (subclasses with
                    compute_wrenches_sym(ctx_a, ctx_b) → (Wrench, Wrench)).
        gravity_field, fluid_field, mag_field, collision_field — same
                    semantics as Craft.compile_tick. Shared across all
                    crafts in this component. Any unset field defaults
                    to its empty form (zero gravity / zero density /
                    zero B / zero penetration).

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
    from .fields import (
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
        raise ValueError("compile_coupled_tick: needs at least one craft.")

    # Validate couplings reference only crafts in this component.
    craft_set = {id(c) for c in crafts}
    for cp in couplings:
        if id(cp.craft_a) not in craft_set or id(cp.craft_b) not in craft_set:
            raise ValueError(
                "compile_coupled_tick: coupling references a craft not in "
                "the given crafts list.")

    # Quick numpy snapshot per craft (mass-positivity guard only). The
    # actual COM / I_com used in Newton-Euler are symbolic and built
    # inside the ir.Graph block below — they pick up joint-angle
    # dependence when a Joint reorients a rotor.
    for craft in crafts:
        ai = _aggregate_inertials(craft._parts)
        if ai["m_total"] <= 0.0:
            raise ValueError(
                f"Craft '{craft.name}': total mass is "
                f"{ai['m_total']}; need m > 0.")

    name = "_".join(c.name for c in crafts) + "_coupled_tick"
    with ir.Graph(name=name) as g:
        dt = ir.Scalar.input("dt")
        t  = ir.Scalar.input("t")

        # Pass 0a: rigid-body state inputs for every craft, stashed on
        # the craft so disturbances (and the per-craft trace below) can
        # reference any craft's symbolic state. This must precede
        # disturbance plumbing because a disturbance might close over a
        # craft's position (e.g. CraftWindBubble).
        for craft in crafts:
            prefix = f"{craft.name}."
            craft._sym_state = {
                "position":         ir.Vec3[WorldFrame].input(prefix + "position"),
                "orientation":      ir.Quat[WorldFrame, CraftFrame].input(prefix + "orientation"),
                "velocity":         ir.Vec3[WorldFrame].input(prefix + "velocity"),
                "angular_velocity": ir.Vec3[CraftFrame].input(prefix + "angular_velocity"),
            }

        # Pass 0b: plumb State/Noise declarations on field disturbances.
        # These rebinds need to be in scope BEFORE any part's update()
        # queries a field.
        all_fields = [gravity_field, fluid_field, mag_field, collision_field]
        dist_saved_attrs, dist_state_outputs = _plumb_field_disturbances(
            all_fields, dt)

        # Pass 1: per-craft trace. The rigid-body state symbols are
        # already created (Pass 0a); the per-craft helper picks them up
        # from craft._sym_state.
        per_craft: dict[int, dict[str, Any]] = {}
        for craft in crafts:
            per_craft[id(craft)] = _trace_craft_pass1(
                g, craft, gravity_field, fluid_field, mag_field, dt, t,
                collision_field=collision_field)

        # Pass 2: coupling wrench injection.
        for cp in couplings:
            pc_a = per_craft[id(cp.craft_a)]
            pc_b = per_craft[id(cp.craft_b)]
            w_a, w_b = cp.compute_wrenches_sym(pc_a["ctx"], pc_b["ctx"])
            pc_a["net"] = pc_a["net"] + w_a
            pc_b["net"] = pc_b["net"] + w_b

        # Pass 3: restore part attrs (so each Craft instance stays
        # reusable across compiles) + finalize per-craft dynamics.
        for craft in crafts:
            pc = per_craft[id(craft)]
            _restore_part_attrs(craft, pc)
            _emit_per_craft_dynamics(g, craft, pc, dt)

        # Disturbance state outputs (deterministic passthrough for plain
        # State; bias_next = bias + sqrt(dt)·driver for RW Noise).
        for out_name, out_val in dist_state_outputs:
            g.output(out_val, out_name)

        # Restore disturbance attributes (so the disturbance instance
        # stays reusable across compiles).
        for dist, saved in dist_saved_attrs:
            for attr_name, attr_val in saved.items():
                object.__setattr__(dist, attr_name, attr_val)

    # Clear the per-craft symbolic-state stash; the compiled function
    # carries the references internally now.
    for craft in crafts:
        craft._sym_state = None

    return g.compile(defaults={"t": 0.0})


# ---------------------------------------------------------------------------
# Field-disturbance state plumbing
# ---------------------------------------------------------------------------

def _plumb_field_disturbances(fields, dt) -> tuple[list, list[tuple[str, Any]]]:
    """Walk every disturbance on each field, rebinding its declared
    State / Noise attributes to symbolic graph inputs. Returns:

      * `saved_attrs` — list of `(disturbance, {attr_name: prev_value})`
        for the restoration step.
      * `state_outputs` — list of `(output_name, output_value)` that the
        caller emits as graph outputs after the part-loop completes.
    """
    from .fields.base import Disturbance

    saved_attrs: list = []
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
                    f"compile_coupled_tick: duplicate disturbance name "
                    f"{dist.name!r}. Disturbance names must be unique "
                    f"within a world.")
            seen_names.add(dist.name)

            prefix  = f"{dist.name}."
            saved: dict[str, Any] = {}

            # User-declared State slots: input → identity passthrough.
            # (Disturbance state advances only via paired RW Noise.)
            for sname, sdecl in sdecls.items():
                if sdecl.manifold != "R1":
                    raise NotImplementedError(
                        f"{type(dist).__name__}('{dist.name}'): State "
                        f"manifold {sdecl.manifold!r} not yet supported "
                        f"on disturbance-declared state.")
                sym = ir.Scalar.input(prefix + sname)
                saved[sname] = getattr(dist, sname)
                object.__setattr__(dist, sname, sym)
                state_outputs.append((prefix + sname, sym))

            # Noise channels.
            for nname, ndecl in ndecls.items():
                frame = ndecl.frame or WorldFrame
                if ndecl.kind == "white":
                    if ndecl.shape == "scalar":
                        sym = ir.Scalar.input(prefix + nname)
                    else:
                        sym = ir.Vec3[frame].input(prefix + nname)
                    saved[nname] = getattr(dist, nname)
                    object.__setattr__(dist, nname, sym)
                    continue
                # kind == "rw".
                sigma = float(getattr(dist, f"{nname}_sigma"))
                if sigma <= 0.0:
                    # Inert: bind to zero, no slot, no driver.
                    if ndecl.shape == "scalar":
                        zero_sym = ir.Scalar.from_mx(ca.MX(0.0))
                    else:
                        zero_sym = ir.Vec3[frame].from_mx(
                            ca.MX.zeros(3, 1))
                    saved[nname] = getattr(dist, nname)
                    object.__setattr__(dist, nname, zero_sym)
                    continue
                # Active RW: bias state + driver input.
                driver_name = prefix + f"{nname}_driver"
                if ndecl.shape == "scalar":
                    bias_sym   = ir.Scalar.input(prefix + nname)
                    driver_sym = ir.Scalar.input(driver_name)
                    sqrt_dt    = ca.sqrt(dt._mx)
                    bias_next  = ir.Scalar.from_mx(
                        bias_sym._mx + sqrt_dt * driver_sym._mx)
                else:
                    bias_sym   = ir.Vec3[frame].input(prefix + nname)
                    driver_sym = ir.Vec3[frame].input(driver_name)
                    sqrt_dt    = ca.sqrt(dt._mx)
                    bias_next  = ir.Vec3[frame].from_mx(
                        bias_sym._mx + sqrt_dt * driver_sym._mx)
                saved[nname] = getattr(dist, nname)
                object.__setattr__(dist, nname, bias_sym)
                state_outputs.append((prefix + nname, bias_next))

            saved_attrs.append((dist, saved))

    return saved_attrs, state_outputs


# ---------------------------------------------------------------------------
# Per-craft helpers
# ---------------------------------------------------------------------------

def _trace_craft_pass1(g_ctx,
                       craft,
                       gravity_field,
                       fluid_field,
                       mag_field,
                       dt,
                       t,
                       collision_field=None) -> dict[str, Any]:
    """Set up one craft's state inputs, part rebinds, TickContext, run
    all parts' update(), and collect their wrench contributions + state
    outputs + sensor outputs. Returns a dict carrying everything the
    later passes need."""
    prefix = f"{craft.name}."

    # Rigid-body state symbols were created in `compile_coupled_tick`'s
    # Pass 0a and stashed on `craft._sym_state` (so disturbances anchored
    # to other crafts can reference them). Pick them up here.
    position    = craft._sym_state["position"]
    orientation = craft._sym_state["orientation"]
    velocity    = craft._sym_state["velocity"]
    ang_vel     = craft._sym_state["angular_velocity"]
    # Placeholder MX symbols for current-tick body acceleration / α
    # (substituted with the real Newton-Euler outputs after the wrench
    # sum is known — see Craft.compile_tick for the rationale).
    a_world_sym = ca.MX.sym(f"{craft.name}_a_anchor", 3, 1)
    alpha_sym    = ca.MX.sym(f"{craft.name}_alpha", 3, 1)
    a_world_placeholder = ir.Vec3[WorldFrame].from_mx(a_world_sym)
    alpha_placeholder    = ir.Vec3[CraftFrame].from_mx(alpha_sym)

    if collision_field is None:
        from .fields import CollisionField as _CF
        collision_field = _CF()

    # State + Input rebinds on each part — must happen BEFORE the
    # kinematic pass, so it sees the joint angle/rate symbols when
    # building per-part kinematic states.
    state_input_nodes: dict[Part, dict[str, Any]] = {}
    saved_state_attrs: dict[Part, dict[str, Any]] = {}
    saved_input_attrs: dict[Part, dict[str, Any]] = {}
    for part in craft._parts:
        decls = part.state_declarations()
        if decls:
            part_states: dict[str, Any] = {}
            saved: dict[str, Any] = {}
            for sname, sdecl in decls.items():
                if sdecl.manifold != "R1":
                    raise NotImplementedError(
                        f"{type(part).__name__}('{part.name}'): "
                        f"State manifold {sdecl.manifold!r} unsupported.")
                sym = ir.Scalar.input(prefix + f"{part.name}.{sname}")
                part_states[sname] = sym
                saved[sname] = getattr(part, sname)
                object.__setattr__(part, sname, sym)
            state_input_nodes[part] = part_states
            saved_state_attrs[part] = saved

        idecls = part.input_declarations()
        if idecls:
            saved_i: dict[str, Any] = {}
            for iname, idecl in idecls.items():
                sym = ir.Scalar.input(prefix + f"{part.name}.{iname}")
                saved_i[iname] = getattr(part, iname)
                object.__setattr__(part, iname, sym)
            saved_input_attrs[part] = saved_i

    # Per-part Noise plumbing — same shape as the single-craft path.
    # White channels: one graph input, bind directly. RW channels: a
    # state input (bias) + a driver input; after-tick state update
    # `bias_next = bias + sqrt(dt) · driver`.
    saved_noise_attrs: dict[Part, dict[str, Any]] = {}
    rw_bias_updates: list[tuple[str, Any]] = []   # (state_name, bias_next)
    for part in craft._parts:
        ndecls = part.noise_declarations()
        if not ndecls:
            continue
        saved_n: dict[str, Any] = {}
        for nname, ndecl in ndecls.items():
            input_name = prefix + f"{part.name}.{nname}"
            frame = ndecl.frame or CraftFrame
            if ndecl.kind == "white":
                if ndecl.shape == "scalar":
                    sym = ir.Scalar.input(input_name)
                else:
                    sym = ir.Vec3[frame].input(input_name)
                saved_n[nname] = getattr(part, nname)
                object.__setattr__(part, nname, sym)
                continue
            # kind == "rw"
            sigma = float(getattr(part, f"{nname}_sigma"))
            if sigma <= 0.0:
                if ndecl.shape == "scalar":
                    zero_sym = ir.Scalar.from_mx(ca.MX(0.0))
                else:
                    zero_sym = ir.Vec3[frame].from_mx(ca.MX.zeros(3, 1))
                saved_n[nname] = getattr(part, nname)
                object.__setattr__(part, nname, zero_sym)
                continue
            driver_name = f"{input_name}_driver"
            if ndecl.shape == "scalar":
                bias_sym   = ir.Scalar.input(input_name)
                driver_sym = ir.Scalar.input(driver_name)
                sqrt_dt    = ca.sqrt(dt._mx)
                bias_next_mx = bias_sym._mx + sqrt_dt * driver_sym._mx
                bias_next = ir.Scalar.from_mx(bias_next_mx)
            else:
                bias_sym   = ir.Vec3[frame].input(input_name)
                driver_sym = ir.Vec3[frame].input(driver_name)
                sqrt_dt    = ca.sqrt(dt._mx)
                bias_next_mx = bias_sym._mx + sqrt_dt * driver_sym._mx
                bias_next = ir.Vec3[frame].from_mx(bias_next_mx)
            saved_n[nname] = getattr(part, nname)
            object.__setattr__(part, nname, bias_sym)
            rw_bias_updates.append((input_name, bias_next))
        saved_noise_attrs[part] = saved_n

    # Symbolic kinematic + inertia passes over the part tree.
    kin_states = kinematic_pass(
        craft.root, position, orientation, velocity, ang_vel,
        gravity_field, t,
        body_acceleration_world=a_world_placeholder,
        body_angular_acceleration=alpha_placeholder)
    inertia = symbolic_inertia_rollup(craft.root)

    # Per-craft TickContext (root-body view) for couplings to read. Per-
    # part TickContexts are built per part below for `update()`.
    root_kin = kin_states[craft.root]
    ctx = TickContext(
        gravity=root_kin.gravity_in_craft,
        gravity_field=gravity_field,
        fluid_field=fluid_field,
        mag_field=mag_field,
        collision_field=collision_field,
        t=t,
        dt=dt,
        world=getattr(craft, "_world", None),
        position=position,
        orientation=orientation,
        velocity=velocity,
        angular_velocity=ang_vel,
        velocity_body=root_kin.velocity_body_in_craft,
        R_craft_from_input=root_kin.R_craft_from_input,
        acceleration_world=root_kin.acceleration_world,
        acceleration_body=root_kin.acceleration_body,
        angular_acceleration=root_kin.angular_acceleration,
    )

    # Aggregate wrenches + collect state/sensor outputs.
    net = Wrench.zero(CraftFrame)
    new_state_outputs: list[tuple[str, Any]] = []
    sensor_outputs:    list[tuple[str, Any]] = []
    for part in craft._parts:
        kin = kin_states[part]
        ctx_part = TickContext(
            gravity=kin.gravity_in_craft,
            gravity_field=gravity_field,
            fluid_field=fluid_field,
            mag_field=mag_field,
            collision_field=collision_field,
            t=t,
            dt=dt,
            position=kin.origin_in_world,
            orientation=orientation,
            velocity=kin.velocity_origin,
            angular_velocity=kin.angular_velocity_input,
            velocity_body=kin.velocity_body_in_craft,
            R_craft_from_input=kin.R_craft_from_input,
            acceleration_world=kin.acceleration_world,
            acceleration_body=kin.acceleration_body,
            angular_acceleration=kin.angular_acceleration,
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
        if w_part.frame is not CraftFrame:
            from .ir.frames import FrameError, _capture_user_source
            raise FrameError(
                f"{type(part).__name__}.update",
                expected="Wrench in CraftFrame",
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
        for sname in decls:
            val = new_state.get(sname, state_input_nodes[part][sname])
            new_state_outputs.append((prefix + f"{part.name}.{sname}", val))

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

        # Wrench at-part-offset → at-craft-origin via symbolic position.
        w_craft = _wrench_to_craft(w_part, kin.r_in_craft)
        net = net + w_craft

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
        "rw_bias_updates":   rw_bias_updates,
        "saved_state_attrs": saved_state_attrs,
        "saved_input_attrs": saved_input_attrs,
        "saved_noise_attrs": saved_noise_attrs,
        "a_world_sym":      a_world_sym,
        "alpha_sym":         alpha_sym,
    }


def _restore_part_attrs(craft, pc) -> None:
    for part, saved in pc["saved_state_attrs"].items():
        for sname, sval in saved.items():
            object.__setattr__(part, sname, sval)
    for part, saved in pc["saved_input_attrs"].items():
        for iname, ival in saved.items():
            object.__setattr__(part, iname, ival)
    for part, saved in pc["saved_noise_attrs"].items():
        for nname, nval in saved.items():
            object.__setattr__(part, nname, nval)


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
    if np.linalg.det(I_com_at_zero_np) > 1e-18:
        alpha_mx = ca.solve(I_com_mx, tau_eff._mx)
        alpha = ir.Vec3[CraftFrame].from_mx(alpha_mx)
    else:
        alpha = ir.Vec3[CraftFrame].constant((0.0, 0.0, 0.0))

    r_OC = -r_com
    offset_term = alpha.cross(r_OC) + ang_vel.cross(ang_vel.cross(r_OC))
    a_origin_world = a_com_world + orientation.apply(offset_term)

    # --- Validate wrench independence from placeholder dynamics ----------
    a_world_sym = pc["a_world_sym"]
    alpha_sym    = pc["alpha_sym"]
    for sym_name, sym_mx in (("acceleration_world", a_world_sym),
                              ("acceleration_body", a_world_sym),
                              ("angular_acceleration", alpha_sym)):
        if (ca.depends_on(net.force._mx, sym_mx)
                or ca.depends_on(net.torque._mx, sym_mx)):
            raise ValueError(
                f"Craft '{craft.name}': a part's wrench or a coupling "
                f"depends on ctx.{sym_name}. Wrenches must be a function "
                f"of state only.")

    # --- Substitute placeholders → real dynamics in emitted outputs -----
    placeholders = ca.vertcat(a_world_sym, alpha_sym)
    real_values  = ca.vertcat(a_origin_world._mx, alpha._mx)
    from .ir.types import _IRValue

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
    sensor_outputs    = [(n, _resolve(v))
                          for n, v in pc["sensor_outputs"]]

    new_velocity = velocity + a_origin_world * dt
    new_position = position + velocity * dt + a_origin_world * (0.5 * dt * dt)
    new_ang_vel  = ang_vel + alpha * dt
    current_so3  = SO3.from_quat(orientation)
    omega_dt_world = orientation.apply(ang_vel * dt)
    new_so3 = current_so3.boxplus(omega_dt_world)
    new_orientation = new_so3.quat.normalize()

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

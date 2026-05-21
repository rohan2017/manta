"""Multi-craft coupled tick compilation.

When World.compile discovers a connected component with more than one
craft (via registered Couplings), it routes the component through
`compile_coupled_tick`. The resulting CompiledGraph takes the
concatenated state of all crafts in the component (with `<craft_name>.`
prefixed slot names) and advances them by one tick. Coupling forces
are computed symbolically inside the graph and injected into each
craft's net wrench BEFORE Newton-Euler, so the per-craft dynamics see
the full inter-craft coupling.

This function shares the per-craft physics pipeline with
Craft.compile_tick (inertials → state inputs → TickContext → part
wrench aggregation → Newton-Euler → symplectic integration → outputs).
It diverges by: (a) prefixing all per-craft IO names with the craft
name, (b) sharing one `dt` input across all crafts, and (c) splicing
in coupling wrenches between the part-aggregation and N-E steps.
"""

from __future__ import annotations

from typing import Any

import casadi as ca
import numpy as np

from . import ir
from .craft import TickContext, _aggregate_inertials, _wrench_to_craft
from .ir.frames import AnchorFrame, CraftFrame
from .math.manifold import SO3
from .ir.types import Mat3, Quat, Scalar, Vec3
from .parts.base import Part, PartUpdate, State
from .math.wrench import Wrench


def compile_coupled_tick(crafts: list,
                          couplings: list,
                          *,
                          gravity_field=None,
                          fluid_field=None,
                          mag_field=None,
                          ) -> "ir.graph.CompiledGraph":
    """Compile a CasADi-MX tick over multiple coupled crafts.

    Args:
        crafts    — list of Craft instances. Must contain every craft
                    referenced by `couplings`.
        couplings — list of Coupling instances (subclasses with
                    compute_wrenches_sym(ctx_a, ctx_b) → (Wrench, Wrench)).
        gravity_field, fluid_field, mag_field — same semantics as
                    Craft.compile_tick. Shared across all crafts in this
                    component.

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
    # Coupled-tick callers don't currently expose collision_field —
    # default to empty. Wiring it through is a small follow-up.
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

    # Compute inertials per craft (numpy).
    inertials = {}
    for craft in crafts:
        ai = _aggregate_inertials(craft._parts)
        if ai["m_total"] <= 0.0:
            raise ValueError(
                f"Craft '{craft.name}': total mass is {ai['m_total']}; need m > 0.")
        ai["I_com_inv"] = (
            np.linalg.inv(ai["I_com"])
            if np.linalg.det(ai["I_com"]) > 1e-18 else None)
        inertials[id(craft)] = ai

    name = "_".join(c.name for c in crafts) + "_coupled_tick"
    with ir.Graph(name=name) as g:
        dt = ir.Scalar.input("dt")

        # Pass 1: per-craft inputs, TickContext, part wrench aggregation,
        # state/input rebinds. We keep all the symbolic handles in
        # `per_craft` so we can splice in coupling wrenches between
        # passes.
        per_craft: dict[int, dict[str, Any]] = {}
        for craft in crafts:
            per_craft[id(craft)] = _trace_craft_pass1(
                g, craft, gravity_field, fluid_field, mag_field, dt,
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
            _emit_per_craft_dynamics(
                g, craft, inertials[id(craft)], pc, dt)

    return g.compile()


# ---------------------------------------------------------------------------
# Per-craft helpers
# ---------------------------------------------------------------------------

def _trace_craft_pass1(g_ctx,
                       craft,
                       gravity_field,
                       fluid_field,
                       mag_field,
                       dt,
                       collision_field=None) -> dict[str, Any]:
    """Set up one craft's state inputs, part rebinds, TickContext, run
    all parts' update(), and collect their wrench contributions + state
    outputs + sensor outputs. Returns a dict carrying everything the
    later passes need."""
    prefix = f"{craft.name}."

    # State inputs (prefixed).
    position    = ir.Vec3[AnchorFrame].input(prefix + "position")
    orientation = ir.Quat[AnchorFrame, CraftFrame].input(prefix + "orientation")
    velocity    = ir.Vec3[AnchorFrame].input(prefix + "velocity")
    ang_vel     = ir.Vec3[CraftFrame].input(prefix + "angular_velocity")

    # Per-tick context.
    g_anchor = gravity_field.state_at_sym(position)
    g_craft  = orientation.conjugate().apply(g_anchor)
    v_body   = orientation.conjugate().apply(velocity)
    if collision_field is None:
        from .fields import CollisionField as _CF
        collision_field = _CF()
    ctx = TickContext(
        gravity=g_craft,
        gravity_field=gravity_field,
        fluid_field=fluid_field,
        mag_field=mag_field,
        collision_field=collision_field,
        dt=dt,
        position=position,
        orientation=orientation,
        velocity=velocity,
        angular_velocity=ang_vel,
        velocity_body=v_body,
    )

    # State + Input rebinds on each part.
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

    # Aggregate wrenches + collect state/sensor outputs.
    net = Wrench.zero(CraftFrame)
    new_state_outputs: list[tuple[str, Any]] = []
    sensor_outputs:    list[tuple[str, Any]] = []
    for part in craft._parts:
        result = part.update(ctx)
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

        # Wrench at-part-offset → at-craft-origin.
        w_craft = _wrench_to_craft(w_part, part.transform)
        net = net + w_craft

    return {
        "position":          position,
        "orientation":       orientation,
        "velocity":          velocity,
        "ang_vel":           ang_vel,
        "ctx":               ctx,
        "net":               net,
        "new_state_outputs": new_state_outputs,
        "sensor_outputs":    sensor_outputs,
        "saved_state_attrs": saved_state_attrs,
        "saved_input_attrs": saved_input_attrs,
    }


def _restore_part_attrs(craft, pc) -> None:
    for part, saved in pc["saved_state_attrs"].items():
        for sname, sval in saved.items():
            object.__setattr__(part, sname, sval)
    for part, saved in pc["saved_input_attrs"].items():
        for iname, ival in saved.items():
            object.__setattr__(part, iname, ival)


def _emit_per_craft_dynamics(g_ctx, craft, inertials, pc, dt) -> None:
    """Newton-Euler + symplectic integration + emit state/sensor outputs
    for one craft. Reads pc["net"] (which by this point includes any
    coupling-injected wrenches)."""
    prefix = f"{craft.name}."

    position    = pc["position"]
    orientation = pc["orientation"]
    velocity    = pc["velocity"]
    ang_vel     = pc["ang_vel"]
    net         = pc["net"]

    m_total     = inertials["m_total"]
    com_np      = inertials["com"]
    I_com_np    = inertials["I_com"]
    I_com_inv_np = inertials["I_com_inv"]

    F_craft    = net.force
    tau_origin = net.torque
    r_com = ir.Vec3[CraftFrame].constant(tuple(com_np))
    tau_com = tau_origin - r_com.cross(F_craft)

    f_anchor = orientation.apply(F_craft / m_total)
    a_com_anchor = f_anchor

    I_com   = ir.Mat3[CraftFrame, CraftFrame].constant(I_com_np)
    I_omega = I_com @ ang_vel
    tau_eff = tau_com - ang_vel.cross(I_omega)
    if I_com_inv_np is not None:
        I_com_inv = ir.Mat3[CraftFrame, CraftFrame].constant(I_com_inv_np)
        alpha = I_com_inv @ tau_eff
    else:
        alpha = ir.Vec3[CraftFrame].constant((0.0, 0.0, 0.0))

    r_OC = -r_com
    offset_term = alpha.cross(r_OC) + ang_vel.cross(ang_vel.cross(r_OC))
    a_origin_anchor = a_com_anchor + orientation.apply(offset_term)

    new_velocity = velocity + a_origin_anchor * dt
    new_position = position + velocity * dt + a_origin_anchor * (0.5 * dt * dt)
    new_ang_vel  = ang_vel + alpha * dt
    current_so3  = SO3.from_quat(orientation)
    omega_dt_anchor = orientation.apply(ang_vel * dt)
    new_so3 = current_so3.boxplus(omega_dt_anchor)
    new_orientation = new_so3.quat.normalize()

    g_ctx.output(new_position,    prefix + "position")
    g_ctx.output(new_orientation, prefix + "orientation")
    g_ctx.output(new_velocity,    prefix + "velocity")
    g_ctx.output(new_ang_vel,     prefix + "angular_velocity")
    for out_name, out_val in pc["new_state_outputs"]:
        g_ctx.output(out_val, out_name)
    from .ir.types import _IRValue
    for out_name, out_val in pc["sensor_outputs"]:
        if not isinstance(out_val, _IRValue):
            raise TypeError(
                f"Output '{out_name}': must be an IR value; got "
                f"{type(out_val).__name__}")
        g_ctx.output(out_val, out_name)

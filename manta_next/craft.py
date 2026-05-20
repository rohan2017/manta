"""Craft — full rigid-body dynamics with orientation and Newton-Euler.

M2 scope:
- 13-DOF rigid-body state: position (3) + orientation quaternion (4) +
  linear velocity (3) + angular velocity (3).
- Parts have static position offsets (`Part.transform`) in craft frame.
- Mass parts declare a diagonal MOI tensor about their own origin.
- Aggregation: total mass, COM in craft frame, MOI about craft origin
  (parallel-axis lifts from each part's position).
- Newton-Euler:
      a_com_scene = R_craft_to_scene · (F_net / m_total)
      I_com · α   = τ_com − ω × (I_com · ω)             (in craft frame)
  with τ_com = τ_origin − r_com × F_net.
  Origin acceleration is recovered via
      a_origin = a_com + R · [α × r_OC + ω × (ω × r_OC)]
  where r_OC = -r_com (origin minus COM in craft frame).
- Integration: position via symplectic-flavored Euler; orientation via
  SO3 boxplus on ω·dt; velocities via Euler.

M2 omissions: articulated joints (M3+), inputs/state on parts (M3+),
field-based gravity beyond a static world-frame vector (M3+), part
static orientations beyond identity (M3+).
"""

from __future__ import annotations

from typing import Any

import casadi as ca
import numpy as np

from . import ir
from .ir.frames import AnchorFrame, CraftFrame
from .ir.manifold import SO3
from .ir.types import Mat3, Quat, Scalar, Vec3
from .parts.base import Part, PartUpdate, State
from .parts.wrench import Wrench


# ---------------------------------------------------------------------------
# TickContext
# ---------------------------------------------------------------------------

class TickContext:
    """The per-tick context passed to each `Part.update(ctx)` call.

    Exposes the symbolic craft body state + environment to part code so
    a Part's `update()` can read everything it physically depends on:

      gravity          : Vec3[CraftFrame]            — gravity at the
                                                       craft origin,
                                                       rotated into craft
                                                       frame. Convenience
                                                       for Mass-like parts
                                                       that don't care
                                                       about field spatial
                                                       variation across
                                                       the body.
      gravity_field    : GravityField                — registered field
                                                       instance. Use for
                                                       position-sensitive
                                                       queries (e.g.
                                                       point-mass-gravity
                                                       sampled at a
                                                       buoyancy point).
                                                       Always non-None —
                                                       empty when
                                                       unregistered.
      fluid_field      : FluidField                  — registered fluid
                                                       field. Same idiom
                                                       as gravity_field.
      dt               : Scalar                      — integrator timestep.
      position         : Vec3[AnchorFrame]           — craft origin in
                                                       anchor frame.
      orientation      : Quat[AnchorFrame,CraftFrame]— craft attitude.
      velocity         : Vec3[AnchorFrame]           — anchor-frame linear
                                                       velocity.
      angular_velocity : Vec3[CraftFrame]            — body angular velocity
                                                       ω. What a rate gyro
                                                       reads.
      velocity_body    : Vec3[CraftFrame]            — body linear velocity
                                                       (R^T · v_anchor).
                                                       What a DVL reads.

    Body-frame inertial acceleration is NOT here — it depends on the
    aggregated wrench, which is the very thing parts are contributing.
    A post-Newton-Euler sensor phase would expose it cleanly; deferred.
    """

    __slots__ = ("gravity", "gravity_field", "fluid_field", "mag_field",
                 "dt", "position", "orientation", "velocity",
                 "angular_velocity", "velocity_body")

    def __init__(self,
                 *,
                 gravity: Vec3,
                 gravity_field,
                 fluid_field,
                 mag_field,
                 dt: Scalar,
                 position: Vec3,
                 orientation: Quat,
                 velocity: Vec3,
                 angular_velocity: Vec3,
                 velocity_body: Vec3) -> None:
        self.gravity = gravity
        self.gravity_field = gravity_field
        self.fluid_field = fluid_field
        self.mag_field = mag_field
        self.dt = dt
        self.position = position
        self.orientation = orientation
        self.velocity = velocity
        self.angular_velocity = angular_velocity
        self.velocity_body = velocity_body


# ---------------------------------------------------------------------------
# Inertial aggregation (pure Python, runs at compile time)
# ---------------------------------------------------------------------------

def _aggregate_inertials(parts: list[Part]) -> dict[str, Any]:
    """Compute total mass, COM offset, and MOI-about-craft-origin from
    a list of parts. Returns concrete Python/numpy values — these are
    constants in the traced graph, not symbolic nodes.

    Parts without `mass` are skipped (Wrench-only contributors like
    actuators in a future world). For M2 every contributing part is a
    Mass (or subclass), so `mass`, `moi`, and `transform` are the only
    attributes consulted.
    """
    m_total = 0.0
    com_sum = np.zeros(3)        # m_i · r_i, then divide
    I_about_origin = np.zeros((3, 3))

    for part in parts:
        m = float(getattr(part, "mass", 0.0))
        if m <= 0.0:
            continue
        r = np.array(part.transform, dtype=float)
        moi_diag = getattr(part, "moi", (0.0, 0.0, 0.0))
        I_own = np.diag([float(moi_diag[0]),
                          float(moi_diag[1]),
                          float(moi_diag[2])])

        m_total += m
        com_sum += m * r
        # Parallel-axis lift: I_about_origin += I_own + m·(|r|²·I − r·r^T).
        I_about_origin += I_own + m * (
            float(r @ r) * np.eye(3) - np.outer(r, r))

    if m_total <= 0.0:
        return {"m_total": 0.0, "com": np.zeros(3),
                "I_origin": np.zeros((3, 3)), "I_com": np.zeros((3, 3))}

    com = com_sum / m_total
    # I_com = I_origin − m_total · (|com|² · I − com·com^T)
    I_com = I_about_origin - m_total * (
        float(com @ com) * np.eye(3) - np.outer(com, com))

    return {
        "m_total":  m_total,
        "com":      com,            # in CraftFrame
        "I_origin": I_about_origin,
        "I_com":    I_com,
    }


# ---------------------------------------------------------------------------
# Wrench transformation (part frame → craft frame)
# ---------------------------------------------------------------------------

def _wrench_to_craft(wrench_part: Wrench, transform: tuple) -> Wrench:
    """Roll a Wrench[CraftFrame] (which is what parts emit in M2 — they're
    composed in CraftFrame even though they conceptually live in
    PartFrame) up through a static position offset.

    Assumes identity orientation between part and craft (M2 constraint).
    For a force F applied at offset r from craft origin:
        F_craft = F_part
        τ_craft = τ_part + r × F_part

    Inputs:
        wrench_part: a Wrench whose frame tag is already CraftFrame (the
                     parts emit forces in CraftFrame for M2; the static
                     part frame conventionally aligns with craft for
                     non-articulated parts).
        transform:   (x, y, z) position offset of the part from craft origin.

    Output: an equivalent Wrench at the craft origin.
    """
    if wrench_part.frame is not CraftFrame:
        from .ir.frames import FrameError, _capture_user_source
        raise FrameError(
            "_wrench_to_craft",
            expected="part Wrench in CraftFrame",
            got=f"frame={wrench_part.frame.__name__}",
            source=_capture_user_source(),
        )
    r = Vec3[CraftFrame].constant(tuple(transform))
    extra_torque = r.cross(wrench_part.force)
    return Wrench(
        force=wrench_part.force,
        torque=wrench_part.torque + extra_torque,
    )


# ---------------------------------------------------------------------------
# Craft
# ---------------------------------------------------------------------------

class Craft:
    """A collection of parts with shared rigid-body dynamics.

    M2 scope: full 6-DOF rigid body. State is
        position    : Vec3[AnchorFrame]
        orientation : Quat[AnchorFrame, CraftFrame]
        velocity    : Vec3[AnchorFrame]
        angular_velocity : Vec3[CraftFrame]
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self._parts: list[Part] = []

    def add(self, part: Part) -> Part:
        if not isinstance(part, Part):
            raise TypeError(f"Craft.add: expected Part, got {type(part).__name__}")
        self._parts.append(part)
        return part

    @property
    def parts(self) -> tuple[Part, ...]:
        return tuple(self._parts)

    @property
    def total_mass(self) -> float:
        return sum(float(getattr(p, "mass", 0.0)) for p in self._parts)

    def aggregate_inertials(self) -> dict[str, Any]:
        """Public-facing accessor: see `_aggregate_inertials`. Useful for
        external inspection and tests."""
        return _aggregate_inertials(self._parts)

    # ----- Tick compilation ------------------------------------------------

    def compile_tick(self,
                     *,
                     gravity_field: "Field | None" = None,
                     fluid_field:   "Field | None" = None,
                     mag_field:     "Field | None" = None,
                     gravity_anchor: tuple[float, float, float] | None = None,
                     ) -> "ir.graph.CompiledGraph":
        """Trace the 6-DOF rigid-body tick into a callable function.

        Args:
            gravity_field   — registered GravityField to query for the
                              craft's instantaneous gravity. Per tick the
                              anchor-frame gravity is `field.state_at_sym(
                              position)`, which becomes a position-
                              dependent symbolic expression when non-
                              uniform disturbances are present.
            gravity_anchor  — escape hatch for direct use without a
                              field: pass a (gx, gy, gz) tuple and the
                              compile creates an internal one-shot
                              GravityField with one UniformGravity
                              disturbance. Convenient for tests and
                              codegen extract.

        Exactly one of `gravity_field` / `gravity_anchor` must be
        provided (or neither, in which case gravity defaults to zero
        — useful for in-vacuum scenarios).

        State (both input and output):
            position         : Vec3[AnchorFrame]            (3,)
            orientation      : Quat[AnchorFrame, CraftFrame] (4,)  (w, x, y, z)
            velocity         : Vec3[AnchorFrame]            (3,)
            angular_velocity : Vec3[CraftFrame]             (3,)
        Other input: dt (scalar).
        """
        # Resolve gravity source.
        from .fields import (
            FluidField as _FluidField,
            GravityField as _GravityField,
            MagField as _MagField,
            UniformGravity as _UniformGravity,
        )
        if gravity_field is not None and gravity_anchor is not None:
            raise ValueError(
                "Craft.compile_tick: pass `gravity_field` OR "
                "`gravity_anchor`, not both.")
        if gravity_field is None and gravity_anchor is not None:
            gravity_field = _GravityField()
            gravity_field.add(_UniformGravity(gravity_anchor))
        if gravity_field is None:
            # No gravity specified → zero field (in-vacuum).
            gravity_field = _GravityField()
        if fluid_field is None:
            # No fluid specified → zero density / zero velocity. Parts
            # like PointBuoy will produce zero contribution.
            fluid_field = _FluidField()
        if mag_field is None:
            # No magnetic field specified → zero B everywhere.
            mag_field = _MagField()
        if not self._parts:
            raise ValueError(f"Craft '{self.name}': no parts added.")

        inertials = _aggregate_inertials(self._parts)
        m_total = inertials["m_total"]
        if m_total <= 0.0:
            raise ValueError(
                f"Craft '{self.name}': total mass is {m_total}; need m > 0.")
        com_np    = inertials["com"]
        I_com_np  = inertials["I_com"]
        I_com_inv_np = np.linalg.inv(I_com_np) if np.linalg.det(I_com_np) > 1e-18 else None

        with ir.Graph(name=f"{self.name}_tick") as g:
            # --- State inputs --------------------------------------------
            position    = ir.Vec3[AnchorFrame].input("position")
            orientation = ir.Quat[AnchorFrame, CraftFrame].input("orientation")
            velocity    = ir.Vec3[AnchorFrame].input("velocity")
            ang_vel     = ir.Vec3[CraftFrame].input("angular_velocity")
            dt          = ir.Scalar.input("dt")

            # --- Per-tick context (gravity rotated to CraftFrame) --------
            # Query the registered GravityField at the craft's anchor
            # position. For a uniform field this folds to a constant; for
            # a point-mass field it becomes a position-dependent MX.
            g_anchor = gravity_field.state_at_sym(position)
            g_craft  = orientation.conjugate().apply(g_anchor)
            v_body   = orientation.conjugate().apply(velocity)

            ctx = TickContext(
                gravity=g_craft,
                gravity_field=gravity_field,
                fluid_field=fluid_field,
                mag_field=mag_field,
                dt=dt,
                position=position,
                orientation=orientation,
                velocity=velocity,
                angular_velocity=ang_vel,
                velocity_body=v_body,
            )

            # --- Per-part state plumbing ---------------------------------
            # For each part with State declarations, create a graph input
            # named "<part_name>.<state_name>" and rebind the attribute on
            # the part to the symbolic node so `self.<state_name>` reads
            # the current value inside update(). Saved Python defaults
            # are restored after tracing so the part instance stays
            # reusable across compiles.
            state_input_nodes: dict[Part, dict[str, Any]] = {}
            saved_state_attrs: dict[Part, dict[str, Any]] = {}
            for part in self._parts:
                decls = part.state_declarations()
                if not decls:
                    continue
                part_states: dict[str, Any] = {}
                saved: dict[str, Any] = {}
                for sname, sdecl in decls.items():
                    input_name = f"{part.name}.{sname}"
                    if sdecl.manifold != "R1":
                        raise NotImplementedError(
                            f"{type(part).__name__}('{part.name}'): "
                            f"State manifold {sdecl.manifold!r} not supported "
                            f"in M3.")
                    sym = ir.Scalar.input(input_name)
                    part_states[sname] = sym
                    saved[sname] = getattr(part, sname)
                    object.__setattr__(part, sname, sym)
                state_input_nodes[part] = part_states
                saved_state_attrs[part] = saved

            # --- Per-part Input plumbing ---------------------------------
            # Same rebind trick as State, but inputs don't get an output
            # in the tick — they're read-only from the part's perspective,
            # and the user supplies values each step.
            saved_input_attrs: dict[Part, dict[str, Any]] = {}
            for part in self._parts:
                idecls = part.input_declarations()
                if not idecls:
                    continue
                saved: dict[str, Any] = {}
                for iname, idecl in idecls.items():
                    input_name = f"{part.name}.{iname}"
                    sym = ir.Scalar.input(input_name)
                    saved[iname] = getattr(part, iname)
                    object.__setattr__(part, iname, sym)
                saved_input_attrs[part] = saved

            # --- Aggregate wrenches + collect state updates --------------
            net = Wrench.zero(CraftFrame)
            new_state_outputs: list[tuple[str, Any]] = []
            sensor_outputs:    list[tuple[str, Any]] = []
            for part in self._parts:
                result = part.update(ctx)
                if isinstance(result, Wrench):
                    w_part = result
                    new_state = {}
                    outputs   = {}
                elif isinstance(result, PartUpdate):
                    w_part    = result.wrench
                    new_state = result.new_state
                    outputs   = result.outputs
                else:
                    raise TypeError(
                        f"{type(part).__name__}('{part.name}').update(): "
                        f"must return a Wrench or PartUpdate, got "
                        f"{type(result).__name__}")
                if w_part.frame is not CraftFrame:
                    from .ir.frames import FrameError, _capture_user_source
                    raise FrameError(
                        f"{type(part).__name__}.update",
                        expected="Wrench in CraftFrame",
                        got=f"frame={w_part.frame.__name__}",
                        source=_capture_user_source(),
                    )

                # Validate state writes against declarations + queue outputs.
                decls = part.state_declarations()
                unknown = set(new_state) - set(decls)
                if unknown:
                    raise KeyError(
                        f"{type(part).__name__}('{part.name}').update(): "
                        f"unknown state slot(s) in new_state: {sorted(unknown)}. "
                        f"Declared: {sorted(decls)}")
                for sname in decls:
                    val = new_state.get(sname, state_input_nodes[part][sname])
                    new_state_outputs.append((f"{part.name}.{sname}", val))

                # Validate + queue Output writes.
                out_decls = part.output_declarations()
                unknown_out = set(outputs) - set(out_decls)
                if unknown_out:
                    raise KeyError(
                        f"{type(part).__name__}('{part.name}').update(): "
                        f"unknown output slot(s): {sorted(unknown_out)}. "
                        f"Declared: {sorted(out_decls)}")
                # Each declared output must be written (we don't bake in a
                # default — a sensor that didn't compute its value is a bug).
                missing_out = set(out_decls) - set(outputs)
                if missing_out:
                    raise KeyError(
                        f"{type(part).__name__}('{part.name}').update(): "
                        f"output slot(s) declared but not written: "
                        f"{sorted(missing_out)}.")
                for oname, oval in outputs.items():
                    sensor_outputs.append((f"{part.name}.{oname}", oval))

                # Aggregate wrench (in CraftFrame, transformed for offset).
                w_craft = _wrench_to_craft(w_part, part.transform)
                net = net + w_craft

            # Restore the part attributes to their Python defaults so the
            # part instance is reusable across compiles.
            for part, saved in saved_state_attrs.items():
                for sname, sval in saved.items():
                    object.__setattr__(part, sname, sval)
            for part, saved in saved_input_attrs.items():
                for iname, ival in saved.items():
                    object.__setattr__(part, iname, ival)

            # --- Newton-Euler --------------------------------------------
            F_craft   = net.force         # Vec3[CraftFrame]
            tau_origin = net.torque       # Vec3[CraftFrame], about craft origin

            # τ_com = τ_origin − r_com × F   (in craft frame)
            r_com = ir.Vec3[CraftFrame].constant(tuple(com_np))
            tau_com = tau_origin - r_com.cross(F_craft)

            # Linear: a_com (anchor) = R · (F_craft / m_total)
            f_anchor = orientation.apply(F_craft / m_total)
            a_com_anchor = f_anchor       # already divided by m_total

            # Angular: I_com · α = τ_com − ω × (I_com · ω)  in craft frame
            I_com   = ir.Mat3[CraftFrame, CraftFrame].constant(I_com_np)
            I_omega = I_com @ ang_vel
            tau_eff = tau_com - ang_vel.cross(I_omega)
            if I_com_inv_np is not None:
                I_com_inv = ir.Mat3[CraftFrame, CraftFrame].constant(I_com_inv_np)
                alpha = I_com_inv @ tau_eff
            else:
                # Degenerate inertia (e.g., single point mass at origin):
                # treat α as zero. Rotational dynamics are undefined.
                alpha = ir.Vec3[CraftFrame].constant((0.0, 0.0, 0.0))

            # a_origin (anchor) = a_com + R · [α × r_OC + ω × (ω × r_OC)]
            r_OC = -r_com   # origin − COM, in CraftFrame
            offset_term = alpha.cross(r_OC) + ang_vel.cross(ang_vel.cross(r_OC))
            a_origin_anchor = a_com_anchor + orientation.apply(offset_term)

            # --- Symplectic-flavored Euler integration -------------------
            # Linear: position += v·dt + ½·a·dt²;  velocity += a·dt.
            new_velocity = velocity + a_origin_anchor * dt
            new_position = position + velocity * dt + a_origin_anchor * (0.5 * dt * dt)
            # Angular: ω_new = ω + α·dt;  q_new = q ⊞ (ω·dt).
            new_ang_vel  = ang_vel + alpha * dt
            current_so3  = SO3.from_quat(orientation)
            # boxplus expects a Vec3 in the rotation's from_frame; SO3.boxplus
            # in our convention uses left trivialization with from=Anchor.
            # We're updating by the body-frame ω_dt — convert via rotation
            # (R · ω_dt lives in Anchor frame, which is SO3.from_frame).
            omega_dt_anchor = orientation.apply(ang_vel * dt)
            new_so3 = current_so3.boxplus(omega_dt_anchor)
            # Renormalize quaternion after every step (Euler error in
            # boxplus is tiny per tick but accumulates over thousands).
            new_orientation = new_so3.quat.normalize()

            # --- Outputs --------------------------------------------------
            g.output(new_position,    "position")
            g.output(new_orientation, "orientation")
            g.output(new_velocity,    "velocity")
            g.output(new_ang_vel,     "angular_velocity")
            # Per-part state outputs (names like "motor.angle").
            for out_name, out_val in new_state_outputs:
                g.output(out_val, out_name)
            # Per-part sensor outputs (names like "imu.gyro").
            from .ir.types import _IRValue
            for out_name, out_val in sensor_outputs:
                if not isinstance(out_val, _IRValue):
                    raise TypeError(
                        f"Output '{out_name}': must be an IR value (Vec3, "
                        f"Scalar, Quat); got {type(out_val).__name__}")
                g.output(out_val, out_name)

        return g.compile()

    # ----- Helpers --------------------------------------------------------

    def initial_state(self, **overrides) -> dict:
        """Build the initial state dict for the compiled tick.

        Returns a dict with the rigid-body slots (position, orientation,
        velocity, angular_velocity) AND a `"<part_name>.<state_name>"`
        entry for every part that declares state. Defaults come from each
        State declaration's `init`; keyword overrides replace them by name.
        """
        state: dict[str, Any] = {
            "position":         np.asarray((0.0, 0.0, 0.0), dtype=float),
            "orientation":      np.asarray((1.0, 0.0, 0.0, 0.0), dtype=float),
            "velocity":         np.asarray((0.0, 0.0, 0.0), dtype=float),
            "angular_velocity": np.asarray((0.0, 0.0, 0.0), dtype=float),
        }
        for part in self._parts:
            for sname, sdecl in part.state_declarations().items():
                state[f"{part.name}.{sname}"] = float(sdecl.init)
            # Input slots: seed from the part's current attribute (which
            # is either the constructor-time override or the declaration
            # default). These pass through CompiledWorld.step's merge so
            # the user can update them per-tick or leave them alone.
            for iname in part.input_declarations():
                state[f"{part.name}.{iname}"] = float(getattr(part, iname))
        unknown = set(overrides) - set(state)
        if unknown:
            raise KeyError(
                f"Craft.initial_state: unknown slot(s) {sorted(unknown)}. "
                f"Available: {sorted(state)}")
        for k, v in overrides.items():
            current = state[k]
            if isinstance(current, np.ndarray):
                state[k] = np.asarray(v, dtype=float)
            else:
                state[k] = float(v)
        return state

    # ----- Introspection --------------------------------------------------

    def __repr__(self) -> str:
        parts = ", ".join(p.name for p in self._parts)
        return f"<Craft '{self.name}' parts=[{parts}]>"

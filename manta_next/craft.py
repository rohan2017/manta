"""Craft — full 6-DOF rigid-body dynamics with Newton-Euler integration.

Scope:
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
- Single-phase parts: each Part implements exactly one `update(ctx)`
  function. `ctx.acceleration_world` / `ctx.acceleration_body` /
  `ctx.angular_acceleration` reflect **current-tick** dynamics — the
  framework runs `update()` against MX placeholders, then substitutes
  the real Newton-Euler outputs into the emitted sensor expressions
  before compiling the graph. Wrenches must not depend on those
  placeholders (the substitution would create an unsolved fixpoint);
  the compile step validates this and raises otherwise.

Articulation: nested `Joint` chains compose symbolically through the
kinematic pass; `r_com`, `I_com`, and per-part offsets pick up joint-
angle dependence via `inertia.symbolic_inertia_rollup`. Native multi-
DOF joints (ball, universal) are still future work — for now stack
single-DOF Joints to build them.

Known omissions: non-identity static orientation between part and craft
frames; field disturbances tied to per-craft motion (only queried, not
contributed-to by parts).
"""

from __future__ import annotations

from typing import Any

import casadi as ca
import numpy as np

from . import ir
from .ir.frames import WorldFrame, CraftFrame
from .ir.manifold import SO3
from .ir.types import Mat3, Quat, Scalar, Vec3
from .parts.base import Part, PartUpdate, State
from .ir.wrench import Wrench


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
      position         : Vec3[WorldFrame]           — craft origin in
                                                       world frame.
      orientation      : Quat[WorldFrame,CraftFrame]— craft attitude.
      velocity         : Vec3[WorldFrame]           — world-frame linear
                                                       velocity.
      angular_velocity : Vec3[CraftFrame]            — body angular velocity
                                                       ω. What a rate gyro
                                                       reads.
      velocity_body    : Vec3[CraftFrame]            — body linear velocity
                                                       (R^T · v_anchor).
                                                       What a DVL reads.
      R_craft_from_input : Mat3[CraftFrame, CraftFrame]
                                                     — rotates a vector
                                                       in the part's
                                                       INPUT-frame coords
                                                       into body-frame
                                                       coords. Identity
                                                       for parts mounted
                                                       directly on the
                                                       craft root; for a
                                                       Joint nested
                                                       inside another
                                                       Joint, carries
                                                       the outer joint's
                                                       angle.

      acceleration_world  : Vec3[WorldFrame]      — world-frame
                                                       inertial accel
                                                       at the part's
                                                       mount point.
      acceleration_body    : Vec3[CraftFrame]        — same, in body-
                                                       frame coords.
                                                       Specific force
                                                       (accelerometer
                                                       reading) is
                                                       `acceleration_body
                                                       - gravity`.
      angular_acceleration : Vec3[CraftFrame]        — body α.

    Note on the acceleration fields: these reflect the **current**
    tick's Newton-Euler output (`α`, `a_origin`) with the lever-arm
    contribution lifted to this part's mount point using current-tick
    ω and `r_in_craft`. At compile time the framework hands `update()`
    MX placeholder symbols for a/α and substitutes the real
    expressions after Newton-Euler. Wrenches must not depend on these
    fields (the substitute pass doesn't solve fixpoints); the compile
    step validates and raises otherwise.
    """

    __slots__ = ("gravity", "gravity_field", "fluid_field", "mag_field",
                 "collision_field",
                 "t", "dt", "position", "orientation", "velocity",
                 "angular_velocity", "velocity_body",
                 "R_craft_from_input",
                 "acceleration_world", "acceleration_body",
                 "angular_acceleration")

    def __init__(self,
                 *,
                 gravity: Vec3,
                 gravity_field,
                 fluid_field,
                 mag_field,
                 collision_field,
                 t: Scalar,
                 dt: Scalar,
                 position: Vec3,
                 orientation: Quat,
                 velocity: Vec3,
                 angular_velocity: Vec3,
                 velocity_body: Vec3,
                 R_craft_from_input: Mat3,
                 acceleration_world: Vec3,
                 acceleration_body: Vec3,
                 angular_acceleration: Vec3) -> None:
        self.gravity = gravity
        self.gravity_field = gravity_field
        self.fluid_field = fluid_field
        self.mag_field = mag_field
        self.collision_field = collision_field
        self.t = t
        self.dt = dt
        self.position = position
        self.orientation = orientation
        self.velocity = velocity
        self.angular_velocity = angular_velocity
        self.velocity_body = velocity_body
        # Body-frame rotation of the part's INPUT frame. Identity for any
        # part mounted directly on the craft root; for a Mass / Joint
        # child of an outer Joint, this carries the outer joint's angle.
        # Articulated parts use this to express their input-frame axes
        # (e.g., a Joint's spin axis) in body-frame coords.
        self.R_craft_from_input = R_craft_from_input
        self.acceleration_world  = acceleration_world
        self.acceleration_body    = acceleration_body
        self.angular_acceleration = angular_acceleration


# ---------------------------------------------------------------------------
# Inertial aggregation (pure Python, runs at compile time)
# ---------------------------------------------------------------------------

def _aggregate_inertials(parts: list[Part]) -> dict[str, Any]:
    """Compute total mass, COM offset, and MOI-about-craft-origin from
    a list of parts. Returns concrete Python/numpy values — these are
    constants in the traced graph, not symbolic nodes.

    Walks `part.mass`, `part.moi`, `part.transform` on every part. Parts
    without `mass` (Thruster, sensors, drag surfaces, …) are skipped.
    `Joint` exposes `.mass` and `.moi` as @property aggregates over its
    rotor children, so the body sees the rotor's inertia distribution
    via the same code path as a plain `Mass`.
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

def _wrench_to_craft(wrench_part: Wrench, r_in_craft: Vec3) -> Wrench:
    """Lift a part-emitted wrench to a wrench acting at the craft origin.

    The part's wrench is assumed already expressed in body-frame
    (CraftFrame) coords — any part whose native quantities live in a
    rotated input frame (Joint axis, articulated thruster, …) is
    responsible for rotating them into body-frame inside its own
    `update()` using `ctx.R_craft_from_input`. The lift here is
    purely the parallel-axis-style force-to-torque coupling:

        F_at_origin = F_at_part
        τ_at_origin = τ_at_part + r × F_at_part

    `r_in_craft` is the part's body-frame position (from the
    symbolic kinematic pass), which for a flat craft simplifies to
    `part.transform` and for a part on a joint chain composes through
    the chain.
    """
    if wrench_part.frame is not CraftFrame:
        from .ir.frames import FrameError, _capture_user_source
        raise FrameError(
            "_wrench_to_craft",
            expected="part Wrench in CraftFrame",
            got=f"frame={wrench_part.frame.__name__}",
            source=_capture_user_source(),
        )
    extra_torque = r_in_craft.cross(wrench_part.force)
    return Wrench(
        force=wrench_part.force,
        torque=wrench_part.torque + extra_torque,
    )


# ---------------------------------------------------------------------------
# Craft
# ---------------------------------------------------------------------------

class Craft:
    """A collection of parts with shared rigid-body dynamics.

    Internally a craft is a tree of parts rooted at `Craft.root` (a
    `RootPart`). `craft.add(part)` is sugar for `craft.root.add(part)`;
    `craft.parts` returns a flat tuple of all parts in the tree (DFS
    order). Nested composition (e.g. `Joint` hosting another `Joint`
    for a pan-tilt gimbal) is supported via the standard composite
    `add()` chain on individual parts.

    State (13 DOF):
        position         : Vec3[WorldFrame]
        orientation      : Quat[WorldFrame, CraftFrame]
        velocity         : Vec3[WorldFrame]
        angular_velocity : Vec3[CraftFrame]
    plus one Scalar per `R1` State slot declared on any of the parts.
    """

    def __init__(self, name: str) -> None:
        from .parts.base import RootPart
        self.name = name
        self.root = RootPart(f"{name}_root")

    def add(self, part: Part) -> Part:
        """Attach a part to the craft's root. Equivalent to
        `craft.root.add(part)`."""
        return self.root.add(part)

    @property
    def parts(self) -> tuple[Part, ...]:
        """Flat tuple of every part in the tree, root first (DFS order).
        Excludes the root itself."""
        return tuple(p for p in self.root.walk() if p is not self.root)

    @property
    def _parts(self) -> list[Part]:
        """Backwards-compatible alias used by compile_tick / _aggregate
        helpers that haven't yet been refactored to walk the tree."""
        return list(self.parts)

    @property
    def total_mass(self) -> float:
        return sum(float(getattr(p, "mass", 0.0)) for p in self.parts)

    def aggregate_inertials(self) -> dict[str, Any]:
        """Public-facing accessor: see `_aggregate_inertials`. Useful for
        external inspection and tests."""
        return _aggregate_inertials(list(self.parts))

    # ----- Tick compilation ------------------------------------------------

    def compile_tick(self,
                     *,
                     gravity_field:   "Field | None" = None,
                     fluid_field:     "Field | None" = None,
                     mag_field:       "Field | None" = None,
                     collision_field: "Field | None" = None,
                     ) -> "ir.graph.CompiledGraph":
        """Trace the 6-DOF rigid-body tick into a callable function.

        Args:
            gravity_field   — registered GravityField to query for the
                              craft's instantaneous gravity. Per tick the
                              world-frame gravity is `field.state_at_sym(
                              position)`, which becomes a position-
                              dependent symbolic expression when non-
                              uniform disturbances are present. None ⇒
                              in-vacuum (zero gravity everywhere).

        State (both input and output):
            position         : Vec3[WorldFrame]            (3,)
            orientation      : Quat[WorldFrame, CraftFrame] (4,)  (w, x, y, z)
            velocity         : Vec3[WorldFrame]            (3,)
            angular_velocity : Vec3[CraftFrame]             (3,)
        Other input: dt (scalar).
        """
        # Default any unspecified field to an empty instance (zero
        # contribution everywhere). Empty fields fold to constants
        # symbolically — no runtime cost.
        from .fields import (
            CollisionField as _CollisionField,
            FluidField as _FluidField,
            GravityField as _GravityField,
            MagField as _MagField,
        )
        if gravity_field is None:
            gravity_field = _GravityField()
        if fluid_field is None:
            # No fluid specified → zero density / zero velocity. Parts
            # like PointBuoy will produce zero contribution.
            fluid_field = _FluidField()
        if mag_field is None:
            # No magnetic field specified → zero B everywhere.
            mag_field = _MagField()
        if collision_field is None:
            # No obstacles → zero penetration vector everywhere. Collider
            # parts contribute zero contact force.
            collision_field = _CollisionField()
        if not self._parts:
            raise ValueError(f"Craft '{self.name}': no parts added.")

        # Quick numpy snapshot for the m_total > 0 guard. The actual COM
        # and I_com used in Newton-Euler are computed symbolically below
        # (inside the ir.Graph block) so they pick up joint-angle
        # dependence when a Joint reorients a rotor.
        snapshot = _aggregate_inertials(self._parts)
        if snapshot["m_total"] <= 0.0:
            raise ValueError(
                f"Craft '{self.name}': total mass is "
                f"{snapshot['m_total']}; need m > 0.")

        with ir.Graph(name=f"{self.name}_tick") as g:
            # --- State inputs --------------------------------------------
            position    = ir.Vec3[WorldFrame].input("position")
            orientation = ir.Quat[WorldFrame, CraftFrame].input("orientation")
            velocity    = ir.Vec3[WorldFrame].input("velocity")
            ang_vel     = ir.Vec3[CraftFrame].input("angular_velocity")
            dt          = ir.Scalar.input("dt")
            t           = ir.Scalar.input("t")
            # Compile-time placeholder symbols for current-tick body
            # acceleration / angular acceleration. update() reads these
            # via TickContext; after Newton-Euler builds the real MX
            # expressions, ca.substitute replaces the placeholders. No
            # runtime state — purely a compile-time wiring trick.
            a_world_sym = ca.MX.sym(f"{self.name}_a_anchor", 3, 1)
            alpha_sym    = ca.MX.sym(f"{self.name}_alpha", 3, 1)
            a_world_placeholder = ir.Vec3[WorldFrame].from_mx(a_world_sym)
            alpha_placeholder    = ir.Vec3[CraftFrame].from_mx(alpha_sym)

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
                            f"State manifold {sdecl.manifold!r} not yet "
                            f"supported on part-declared state. Only 'R1' "
                            f"is wired through compile_tick today.")
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

            # --- Per-part Noise plumbing ---------------------------------
            # Each Noise declaration becomes a graph input vector. The
            # default initial-state value is zero (clean signal — what
            # the EKF wants). Sim callers draw a fresh sample per tick
            # via `Craft.sample_noise(rng)`.
            saved_noise_attrs: dict[Part, dict[str, Any]] = {}
            for part in self._parts:
                ndecls = part.noise_declarations()
                if not ndecls:
                    continue
                saved: dict[str, Any] = {}
                for nname, ndecl in ndecls.items():
                    input_name = f"{part.name}.{nname}"
                    frame = ndecl.frame or CraftFrame
                    if ndecl.shape == "scalar":
                        sym = ir.Scalar.input(input_name)
                    else:
                        sym = ir.Vec3[frame].input(input_name)
                    saved[nname] = getattr(part, nname)
                    object.__setattr__(part, nname, sym)
                saved_noise_attrs[part] = saved

            # --- Symbolic kinematic pass --------------------------------
            # Now that all Joint angles/rates have been rebound to MX
            # symbols above, walk the part tree to produce each part's
            # effective kinematic state (position, velocity, ω, gravity).
            # For a flat craft this just adds part-offset lever-arms to
            # the body state; for nested joints it composes through the
            # chain.
            from .kinematics import kinematic_pass
            kin_states = kinematic_pass(
                self.root, position, orientation, velocity, ang_vel,
                gravity_field, t,
                body_acceleration_world=a_world_placeholder,
                body_angular_acceleration=alpha_placeholder)

            # --- Symbolic inertia rollup --------------------------------
            # Build (m_total, com, I_com) as MX expressions in joint
            # angles. For a flat craft these simplify to the same
            # constants as the numpy snapshot above; for a craft with a
            # Joint, com and I_com vary symbolically with the joint
            # angle.
            from .inertia import symbolic_inertia_rollup
            inertia = symbolic_inertia_rollup(self.root)
            m_total          = inertia["m_total"]
            com_mx           = inertia["com_in_craft_mx"]
            I_com_mx         = inertia["I_com_in_craft_mx"]
            I_com_at_zero_np = inertia["I_com_at_zero"]

            # --- Aggregate wrenches + collect state updates --------------
            net = Wrench.zero(CraftFrame)
            new_state_outputs: list[tuple[str, Any]] = []
            sensor_outputs:    list[tuple[str, Any]] = []
            for part in self._parts:
                # Build a per-part TickContext from its kinematic state.
                # ctx.orientation stays the body's orientation so wrenches
                # the part emits remain in CraftFrame (the existing
                # aggregation expects that). ctx.position / velocity /
                # gravity / angular_velocity are the PART's effective
                # values — for a flat craft these equal `body + lever arm
                # from part.transform`, matching what every part used to
                # compute manually.
                kin = kin_states[part]
                ctx = TickContext(
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
                missing_out = set(out_decls) - set(outputs)
                if unknown_out:
                    raise KeyError(
                        f"{type(part).__name__}('{part.name}').update(): "
                        f"unknown output slot(s): {sorted(unknown_out)}. "
                        f"Declared: {sorted(out_decls)}")
                if missing_out:
                    raise KeyError(
                        f"{type(part).__name__}('{part.name}').update(): "
                        f"output slot(s) declared but not written: "
                        f"{sorted(missing_out)}.")
                for oname, oval in outputs.items():
                    sensor_outputs.append((f"{part.name}.{oname}", oval))

                # Aggregate wrench (in CraftFrame, lifted to body origin
                # via the part's symbolic body-frame position).
                w_craft = _wrench_to_craft(w_part, kin.r_in_craft)
                net = net + w_craft

            # --- Validate wrench independence from placeholder dynamics --
            # Wrenches must be a function of state only — if one
            # depends on the placeholder a/α, the substitution below
            # creates an unsolved fixpoint (would need added-mass-style
            # inertia augmentation). Raise so the part author rewrites
            # the dependency explicitly.
            for sym_name, sym_mx in (("acceleration_world", a_world_sym),
                                      ("acceleration_body", a_world_sym),
                                      ("angular_acceleration", alpha_sym)):
                if (ca.depends_on(net.force._mx, sym_mx)
                        or ca.depends_on(net.torque._mx, sym_mx)):
                    raise ValueError(
                        f"Craft '{self.name}': a part's wrench depends "
                        f"on ctx.{sym_name}. Wrenches must be a function "
                        f"of state only; reading current-tick dynamics "
                        f"in the wrench creates an implicit equation "
                        f"this compiler doesn't solve.")

            # --- Newton-Euler --------------------------------------------
            F_craft   = net.force         # Vec3[CraftFrame]
            tau_origin = net.torque       # Vec3[CraftFrame], about craft origin

            # τ_com = τ_origin − r_com × F   (in craft frame)
            r_com = ir.Vec3[CraftFrame].from_mx(com_mx)
            tau_com = tau_origin - r_com.cross(F_craft)

            # Linear: a_com (anchor) = R · (F_craft / m_total)
            f_world = orientation.apply(F_craft / m_total)
            a_com_world = f_world       # already divided by m_total

            # Angular: I_com · α = τ_com − ω × (I_com · ω)  in craft frame.
            # I_com is symbolic in joint angles; solve at runtime rather
            # than pre-inverting. If the at-rest snapshot is singular
            # (point mass at origin with zero MOI), fall back to α = 0
            # — rotational dynamics are physically undefined there.
            I_com   = ir.Mat3[CraftFrame, CraftFrame].from_mx(I_com_mx)
            I_omega = I_com @ ang_vel
            tau_eff = tau_com - ang_vel.cross(I_omega)
            if np.linalg.det(I_com_at_zero_np) > 1e-18:
                alpha_mx = ca.solve(I_com_mx, tau_eff._mx)
                alpha = ir.Vec3[CraftFrame].from_mx(alpha_mx)
            else:
                # Degenerate inertia (e.g., single point mass at origin):
                # treat α as zero. Rotational dynamics are undefined.
                alpha = ir.Vec3[CraftFrame].constant((0.0, 0.0, 0.0))

            # a_origin (anchor) = a_com + R · [α × r_OC + ω × (ω × r_OC)]
            r_OC = -r_com   # origin − COM, in CraftFrame
            offset_term = alpha.cross(r_OC) + ang_vel.cross(ang_vel.cross(r_OC))
            a_origin_world = a_com_world + orientation.apply(offset_term)

            # --- Substitute placeholders → real dynamics ----------------
            # Every output and per-part state-update expression that
            # referenced `ctx.acceleration_world` /
            # `ctx.acceleration_body` / `ctx.angular_acceleration` now
            # gets the actual Newton-Euler result wired in.
            placeholders = ca.vertcat(a_world_sym, alpha_sym)
            real_values  = ca.vertcat(a_origin_world._mx, alpha._mx)
            from .ir.types import _IRValue

            def _resolve(val):
                if not isinstance(val, _IRValue):
                    return val
                new_mx = ca.substitute(val._mx, placeholders, real_values)
                # Reconstruct the typed wrapper around the substituted MX,
                # preserving frame tags. Each IR type carries its frame
                # info in private attributes that match its constructor
                # kwargs — pull them out and re-build.
                cls = type(val)
                if isinstance(val, ir.Vec3):
                    return cls._from_mx(new_mx, frame=val._frame)
                if isinstance(val, (ir.Mat3, ir.Quat)):
                    return cls._from_mx(new_mx,
                                         from_frame=val._from_frame,
                                         to_frame=val._to_frame)
                # Scalar / fallback.
                return cls._from_mx(new_mx)

            new_state_outputs = [(n, _resolve(v)) for n, v in new_state_outputs]
            sensor_outputs    = [(n, _resolve(v)) for n, v in sensor_outputs]

            # Now restore part attrs.
            for part, saved in saved_state_attrs.items():
                for sname, sval in saved.items():
                    object.__setattr__(part, sname, sval)
            for part, saved in saved_input_attrs.items():
                for iname, ival in saved.items():
                    object.__setattr__(part, iname, ival)
            for part, saved in saved_noise_attrs.items():
                for nname, nval in saved.items():
                    object.__setattr__(part, nname, nval)

            # --- Symplectic-flavored Euler integration -------------------
            # Linear: position += v·dt + ½·a·dt²;  velocity += a·dt.
            new_velocity = velocity + a_origin_world * dt
            new_position = position + velocity * dt + a_origin_world * (0.5 * dt * dt)
            # Angular: ω_new = ω + α·dt;  q_new = q ⊞ (ω·dt).
            new_ang_vel  = ang_vel + alpha * dt
            current_so3  = SO3.from_quat(orientation)
            # boxplus expects a Vec3 in the rotation's from_frame; SO3.boxplus
            # in our convention uses left trivialization with from=Anchor.
            # We're updating by the body-frame ω_dt — convert via rotation
            # (R · ω_dt lives in World frame, which is SO3.from_frame).
            omega_dt_world = orientation.apply(ang_vel * dt)
            new_so3 = current_so3.boxplus(omega_dt_world)
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
            for out_name, out_val in sensor_outputs:
                if not isinstance(out_val, _IRValue):
                    raise TypeError(
                        f"Output '{out_name}': must be an IR value (Vec3, "
                        f"Scalar, Quat); got {type(out_val).__name__}")
                g.output(out_val, out_name)

        return g.compile(defaults={"t": 0.0})

    # ----- Helpers --------------------------------------------------------

    def sample_noise(self, rng) -> dict:
        """Draw one tick of white-Gaussian samples for every declared
        `Noise` slot on every part. Returns a dict of
        `"<part>.<noise>" → np.ndarray` ready to merge into the state
        dict before calling the compiled tick.

        Slots whose sigma is 0 return zero vectors without consuming
        RNG state (so a deterministic-seed sim stays reproducible
        regardless of which noise channels are active).
        """
        out: dict[str, Any] = {}
        for part in self._parts:
            for nname, ndecl in part.noise_declarations().items():
                key = f"{part.name}.{nname}"
                sigma = float(getattr(part, f"{nname}_sigma"))
                if ndecl.shape == "scalar":
                    out[key] = (rng.normal(0.0, sigma)
                                if sigma > 0.0 else 0.0)
                else:
                    out[key] = (rng.normal(0.0, sigma, 3)
                                if sigma > 0.0
                                else np.zeros(3, dtype=float))
        return out

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
            # Noise slots: seed at zero (clean signal). Sim callers
            # overwrite via `craft.sample_noise(rng)` each tick; the EKF
            # leaves them at zero and reads `part.noise_R(name)` for the
            # measurement-noise covariance instead.
            for nname, ndecl in part.noise_declarations().items():
                shape = 3 if ndecl.shape == "vec3" else 1
                state[f"{part.name}.{nname}"] = (
                    np.zeros(shape, dtype=float) if shape > 1 else 0.0)
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

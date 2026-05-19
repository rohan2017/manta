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
from .parts.base import Part
from .parts.wrench import Wrench


# ---------------------------------------------------------------------------
# TickContext
# ---------------------------------------------------------------------------

class TickContext:
    """The per-tick context passed to each `Part.update(ctx)` call.

    M2 fields:
      gravity   : Vec3[CraftFrame]   — world-frame gravity rotated into
                                       craft frame each tick (so a tilted
                                       craft sees gravity tilt with it).
      dt        : Scalar             — integrator timestep.
    """

    __slots__ = ("gravity", "dt")

    def __init__(self, *, gravity: Vec3, dt: Scalar) -> None:
        self.gravity = gravity
        self.dt = dt


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
                     gravity_anchor: tuple[float, float, float] = (0.0, 0.0, -9.81)
                     ) -> "ir.graph.CompiledGraph":
        """Trace the 6-DOF rigid-body tick into a callable function.

        State (both input and output):
            position         : Vec3[AnchorFrame]            (3,)
            orientation      : Quat[AnchorFrame, CraftFrame] (4,)  (w, x, y, z)
            velocity         : Vec3[AnchorFrame]            (3,)
            angular_velocity : Vec3[CraftFrame]             (3,)
        Other input: dt (scalar).
        """
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
            # World-frame gravity (= AnchorFrame for now, since the world
            # → anchor relationship is identity in M2).
            g_anchor = ir.Vec3[AnchorFrame].constant(tuple(gravity_anchor))
            g_craft  = orientation.conjugate().apply(g_anchor)

            ctx = TickContext(gravity=g_craft, dt=dt)

            # --- Aggregate wrenches --------------------------------------
            net = Wrench.zero(CraftFrame)
            for part in self._parts:
                w_part = part.update(ctx)
                if not isinstance(w_part, Wrench):
                    raise TypeError(
                        f"{type(part).__name__}('{part.name}').update(): "
                        f"must return a Wrench, got {type(w_part).__name__}")
                if w_part.frame is not CraftFrame:
                    from .ir.frames import FrameError, _capture_user_source
                    raise FrameError(
                        f"{type(part).__name__}.update",
                        expected="Wrench in CraftFrame",
                        got=f"frame={w_part.frame.__name__}",
                        source=_capture_user_source(),
                    )
                w_craft = _wrench_to_craft(w_part, part.transform)
                net = net + w_craft

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

        return g.compile()

    # ----- Helpers --------------------------------------------------------

    @staticmethod
    def initial_state(*,
                      position=(0.0, 0.0, 0.0),
                      orientation=(1.0, 0.0, 0.0, 0.0),
                      velocity=(0.0, 0.0, 0.0),
                      angular_velocity=(0.0, 0.0, 0.0)) -> dict:
        """Convenience builder for the state dict passed to the tick fn."""
        return {
            "position":         np.asarray(position, dtype=float),
            "orientation":      np.asarray(orientation, dtype=float),
            "velocity":         np.asarray(velocity, dtype=float),
            "angular_velocity": np.asarray(angular_velocity, dtype=float),
        }

    # ----- Introspection --------------------------------------------------

    def __repr__(self) -> str:
        parts = ", ".join(p.name for p in self._parts)
        return f"<Craft '{self.name}' parts=[{parts}]>"

"""Craft — container for parts, source of a compiled tick function.

A Craft holds a list of Parts and knows how to assemble them into a
single IR tick graph::

    state_t  ──┐
    inputs   ──┤───▶ [ Craft.tick ] ───▶ state_{t+1}
    dt       ──┘                         (telemetry, optional)

For M1, the tick does point-mass linear dynamics only:
  * State is position + linear velocity in AnchorFrame.
  * Parts contribute Wrench[CraftFrame] (force + torque); the M1 craft
    body frame is *aligned with* AnchorFrame (no orientation tracking
    yet), so the wrench is summed in AnchorFrame directly.
  * Newton: a = F_total / m_total. Symplectic-flavored Euler step.

M2 will add: orientation (Quat[Anchor,Craft]) state, full Newton-Euler
with MOI, frame transformations for off-axis force application, etc.
"""

from __future__ import annotations

from typing import Any, Iterable

from . import ir
from .ir.frames import AnchorFrame, CraftFrame
from .ir.types import Scalar, Vec3
from .parts.base import Part, TickContext
from .parts.wrench import Wrench


# ---------------------------------------------------------------------------
# TickContext — what `Part.update(ctx)` receives.
# ---------------------------------------------------------------------------

class TickContext:
    """The per-tick context passed to each `Part.update(ctx)` call.

    M1 fields:
      gravity   : Vec3[CraftFrame]   — the world-frame gravity vector,
                                       rotated into craft frame (= same as
                                       anchor in M1 since orientation is
                                       identity).
      dt        : Scalar             — the integrator timestep.

    M2 will add: per-craft kinematic queries (position/velocity/orientation
    of any part relative to any frame), field queries (gravity, fluid,
    magnetic), and named input access.
    """

    __slots__ = ("gravity", "dt")

    def __init__(self, *, gravity: Vec3, dt: Scalar) -> None:
        self.gravity = gravity
        self.dt = dt


# ---------------------------------------------------------------------------
# Craft
# ---------------------------------------------------------------------------

class Craft:
    """A collection of parts that share rigid-body dynamics.

    M1 scope: linear dynamics only. Position + velocity in AnchorFrame.
    No orientation, no MOI, no torques yet.

    Usage::

        c = Craft("falling")
        c.add(Mass("body", mass=1.0))
        tick = c.compile_tick(gravity_anchor=(0, 0, -9.81))

        state = {"position": [0,0,100], "velocity": [0,0,0]}
        for _ in range(100):
            state = tick(dt=0.01, **state)
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
        """Sum of declared `mass` parameter across all parts that have one.
        Parts without a `mass` attribute are skipped (zero contribution)."""
        return sum(getattr(p, "mass", 0.0) for p in self._parts)

    # ----- Tick compilation -----------------------------------------------

    def compile_tick(self,
                     *,
                     gravity_anchor: tuple[float, float, float] = (0.0, 0.0, -9.81)
                     ) -> "ir.graph.CompiledGraph":
        """Trace the per-tick scene update into a CasADi-callable function.

        Inputs: position (Vec3[Anchor]), velocity (Vec3[Anchor]), dt.
        Outputs: position, velocity (post-tick).

        Returns the CompiledGraph for the resulting `casadi.Function`. The
        caller invokes it with keyword args by input name.
        """
        if not self._parts:
            raise ValueError(
                f"Craft '{self.name}': no parts added; nothing to compile.")
        if self.total_mass <= 0.0:
            raise ValueError(
                f"Craft '{self.name}': total mass is {self.total_mass}; "
                f"need m > 0 for linear dynamics.")

        with ir.Graph(name=f"{self.name}_tick") as g:
            # --- Inputs: rigid-body state and dt --------------------------
            position = ir.Vec3[AnchorFrame].input("position")
            velocity = ir.Vec3[AnchorFrame].input("velocity")
            dt       = ir.Scalar.input("dt")

            # M1: anchor frame and craft frame are identical (no orientation
            # state). Gravity in anchor → gravity in craft, same numbers.
            g_anchor = ir.Vec3[AnchorFrame].constant(tuple(gravity_anchor))
            g_craft  = ir.Vec3[CraftFrame].constant(tuple(gravity_anchor))

            ctx = TickContext(gravity=g_craft, dt=dt)

            # --- Aggregate part contributions -----------------------------
            net = Wrench.zero(CraftFrame)
            for part in self._parts:
                w = part.update(ctx)
                if not isinstance(w, Wrench):
                    raise TypeError(
                        f"{type(part).__name__}('{part.name}').update(): "
                        f"must return a Wrench, got {type(w).__name__}")
                if w.frame is not CraftFrame:
                    from .ir.frames import FrameError, _capture_user_source
                    raise FrameError(
                        f"{type(part).__name__}.update",
                        expected=f"Wrench in CraftFrame",
                        got=f"frame={w.frame.__name__}",
                        source=_capture_user_source(),
                    )
                net = net + w

            # --- Linear dynamics (point mass) -----------------------------
            # Sum of mass parameters as the inertial mass (M1: constants).
            m_total = ir.Scalar.constant(self.total_mass)

            # F_craft = net.force (in CraftFrame; CraftFrame == AnchorFrame in M1).
            # Reinterpret to AnchorFrame for the integrator step.
            f_anchor = Vec3(net.force._mx, frame=AnchorFrame)

            accel    = f_anchor / m_total
            new_vel  = velocity + accel * dt
            new_pos  = position + velocity * dt + accel * 0.5 * dt * dt

            g.output(new_pos, "position")
            g.output(new_vel, "velocity")

        return g.compile()

    # ----- Introspection --------------------------------------------------

    def __repr__(self) -> str:
        parts = ", ".join(p.name for p in self._parts)
        return f"<Craft '{self.name}' parts=[{parts}]>"

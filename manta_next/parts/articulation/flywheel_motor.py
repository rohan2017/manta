"""FlywheelMotor — a simplified reaction wheel.

Implements a 1-DOF flywheel about a body-fixed axis. A commanded torque
spins up the wheel; an equal-and-opposite reaction torque (Newton's
third) is applied to the craft body. This is enough to demonstrate
state + wrench contribution without the full ArticulatedPart machinery
(joint-axis projection, parent-chain MOI etc.) — that lands when we add
the full Motor for a real top.

Parameters:
    I_axial    (float)            — flywheel's axial moment of inertia (kg·m²).
    axis       (3-tuple)          — joint axis in CraftFrame (defaults to +z).
    torque_cmd (float)            — commanded torque on the rotor.
                                    Constant for M3; M4 will let this be
                                    a per-tick `Input(...)` from the controller.

State:
    angle (Scalar)                — joint angle, rad.
    rate  (Scalar)                — joint angular rate, rad/s.

Limitations vs. legacy Motor:
    * No saturating torque clamp (no stall_torque parameter — defer).
    * Reaction torque on the body is treated as -torque_cmd along the
      axis, NOT the resolved axial torque accounting for the rotor's
      own dynamics (legacy Motor's resolve()). For M3 that's fine: with
      a stationary rotor and a commanded torque, the reaction is exactly
      -torque_cmd by Newton's 3rd. As soon as the rotor accelerates this
      stays valid (the rotor's I·α is what consumes the actuator torque;
      the parent feels equal-and-opposite by definition).
"""

from __future__ import annotations

from ...ir.frames import CraftFrame
from ...ir.types import Vec3
from ..base import Input, Part, Parameter, PartUpdate, State
from ..wrench import Wrench


class FlywheelMotor(Part):
    I_axial:    float                       = Parameter(0.01)
    axis:       "tuple[float, float, float]" = Parameter((0.0, 0.0, 1.0))
    torque_cmd: float                       = Input(default=0.0)

    angle = State(init=0.0, manifold="R1")
    rate  = State(init=0.0, manifold="R1")

    def update(self, ctx) -> PartUpdate:
        # Rotor dynamics about its own axis. (Self-consistent on the joint
        # alone — no cross-coupling with body angular velocity in M3; that
        # was patch territory in legacy manta and lands in M5.)
        accel = self.torque_cmd / self.I_axial

        # Symplectic Euler for the joint state — matches the body
        # integrator's behavior.
        new_rate  = self.rate + accel * ctx.dt
        new_angle = self.angle + self.rate * ctx.dt + 0.5 * accel * ctx.dt * ctx.dt

        # Reaction torque on the craft body in CraftFrame: -τ_cmd along axis.
        axis_vec = Vec3[CraftFrame].constant(tuple(self.axis))
        reaction = axis_vec * (-self.torque_cmd)
        zero_force = Vec3[CraftFrame].constant((0.0, 0.0, 0.0))

        return PartUpdate(
            wrench=Wrench(force=zero_force, torque=reaction),
            new_state={"angle": new_angle, "rate": new_rate},
        )

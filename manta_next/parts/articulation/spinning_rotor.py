"""SpinningRotor — a purely kinematic joint that advances an angle at
a constant rate. No coupling to body dynamics; useful for visualization
and as the simplest example of a part with internal state.
"""

from __future__ import annotations

from ...ir.frames import CraftFrame
from ..base import Part, Parameter, PartUpdate, State
from ...math.wrench import Wrench


class SpinningRotor(Part):
    """Kinematic spinning joint. The `angle` state advances at `spin_rate`
    every tick. Contributes zero wrench to the craft.

    Parameters:
        spin_rate (float)  — rad/s.

    State:
        angle (Scalar)     — current joint angle, integrated each tick.
    """

    spin_rate: float = Parameter(0.0)

    angle = State(init=0.0, manifold="R1")

    def update(self, ctx) -> PartUpdate:
        new_angle = self.angle + self.spin_rate * ctx.dt
        return PartUpdate(
            wrench=Wrench.zero(CraftFrame),
            new_state={"angle": new_angle},
        )

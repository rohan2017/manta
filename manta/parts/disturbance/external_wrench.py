"""World-frame load port for a numerically solved external interaction.

Mount at the interaction's frame, including below articulated joints. The
ordinary part wrench pass accounts for lever arms and joint reactions.
"""

import casadi as ca

from ...ir.frames import WorldFrame
from ...ir.types import Scalar, Vec3
from ...ir.wrench import Wrench
from .._declarations import Input
from ..base import Part


class ExternalWrench(Part):
    fx = Input(0.0)
    fy = Input(0.0)
    fz = Input(0.0)
    tx = Input(0.0)
    ty = Input(0.0)
    tz = Input(0.0)

    def update(self, ctx):
        force = Vec3[WorldFrame].from_mx(
            ca.vertcat(*(Scalar.coerce(x).mx for x in (self.fx, self.fy, self.fz)))
        )
        torque = Vec3[WorldFrame].from_mx(
            ca.vertcat(*(Scalar.coerce(x).mx for x in (self.tx, self.ty, self.tz)))
        )
        return Wrench(
            force=ctx.orientation.conjugate().apply(force),
            torque=ctx.orientation.conjugate().apply(torque),
        )

"""HeadingSensor — horizontal body-forward direction in the world frame.

The sensor emits a horizontal three-component unit vector instead of an angle. This gives
EKF/UKF residuals continuous behavior through the -pi/pi wrap while retaining
exactly the yaw information supplied by a moving-baseline GNSS solution.
"""

from __future__ import annotations

import casadi as ca

from ...ir.frames import PartFrame, WorldFrame
from ...ir.types import Vec3
from ...ir.wrench import Wrench
from .._declarations import Output, Parameter, PartUpdate, WhiteNoise
from ..base import Part, PartRole


class HeadingSensor(Part):
    """Outputs the normalized horizontal projection of sensor-frame +x.

    ``heading_vector`` is ``[cos(heading), sin(heading), 0]`` in world axes.
    The vehicle must not operate this measurement model at a vertical
    attitude, where horizontal heading is physically undefined.
    """

    role = PartRole.SENSOR

    rate: float = Parameter(None)
    heading_vector_noise = WhiteNoise("R3", frame=WorldFrame, sigma=0.0)
    heading_vector = Output()

    def update(self, ctx) -> PartUpdate:
        forward = ctx.orientation.apply(Vec3[PartFrame].constant((1.0, 0.0, 0.0)))
        norm = ca.sqrt(forward.x.mx * forward.x.mx + forward.y.mx * forward.y.mx)
        reading = (
            Vec3[WorldFrame].from_mx(
                ca.vertcat(forward.x.mx / norm, forward.y.mx / norm, 0.0)
            )
            + self.heading_vector_noise
        )
        zero = Vec3[PartFrame].constant((0.0, 0.0, 0.0))
        return PartUpdate(
            wrench=Wrench(force=zero, torque=zero),
            outputs={"heading_vector": reading},
            rates={"heading_vector": self.rate},
        )


__all__ = ["HeadingSensor"]

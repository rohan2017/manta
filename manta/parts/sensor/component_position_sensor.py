"""Position observation split into horizontal and vertical update groups.

GNSS receivers produce one three-dimensional fix, but a bad altitude must not
force an otherwise consistent horizontal fix out of an estimator. This part
keeps one physical mount and one three-axis noise source while exposing two
measurement ports so each statistically distinct component group receives its
own innovation gate.
"""

from __future__ import annotations

from ...ir.frames import PartFrame, WorldFrame
from ...ir.types import Vec3, VecN
from ...ir.wrench import Wrench
from .._declarations import Output, Parameter, PartUpdate, WhiteNoise
from ..base import Part, PartRole


class ComponentPositionSensor(Part):
    """World position observer with independent XY and Z measurement ports."""

    role = PartRole.SENSOR

    rate: float = Parameter(None)
    position_noise = WhiteNoise("R3", frame=WorldFrame, sigma=0.0)

    horizontal_position = Output()
    vertical_position = Output()

    def update(self, ctx) -> PartUpdate:
        zero = Vec3[PartFrame].constant((0.0, 0.0, 0.0))
        reading = ctx.position[WorldFrame] + self.position_noise
        return PartUpdate(
            wrench=Wrench(force=zero, torque=zero),
            outputs={
                "horizontal_position": VecN[2].from_mx(reading.mx[:2]),
                "vertical_position": reading.z,
            },
            rates={
                "horizontal_position": self.rate,
                "vertical_position": self.rate,
            },
        )


__all__ = ["ComponentPositionSensor"]

"""RF antenna mount marker with a world-frame position output."""
from ...ir.frames import PartFrame, WorldFrame
from ...ir.types import Vec3
from ...ir.wrench import Wrench
from .._declarations import Output, Parameter, PartUpdate
from ..base import Part


class Antenna(Part):
    frequency_hz: float = Parameter(2.4e9)
    tx_power_dbm: float = Parameter(20.0)
    gain_dbi: float = Parameter(0.0)
    position = Output()

    def update(self, ctx) -> PartUpdate:
        zero = Vec3[PartFrame].constant((0.0, 0.0, 0.0))
        return PartUpdate(wrench=Wrench(force=zero, torque=zero),
                          outputs={"position": ctx.position[WorldFrame]})


__all__ = ["Antenna"]

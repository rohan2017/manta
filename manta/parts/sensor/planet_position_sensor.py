"""Planet-fixed Cartesian position at a sensor mount point."""

from __future__ import annotations

from typing import Any, ClassVar

from ...fields import PlanetBindingField
from ...ir.frames import PartFrame, PlanetFrame, WorldFrame
from ...ir.types import Vec3
from ...ir.wrench import Wrench
from .._declarations import Output, Parameter, PartUpdate, WhiteNoise
from ..base import Part, PartRole


class PlanetPositionSensor(Part):
    """Ideal Cartesian position relative to the craft's bound planet.

    This is the estimator-facing counterpart of a GNSS position solution.  It
    outputs planet-centred, planet-fixed Cartesian metres; latitude, longitude,
    altitude, datums, and receiver behavior belong to the device adapter.
    """

    role = PartRole.SENSOR
    requires_fields: ClassVar[list[type]] = [PlanetBindingField]

    rate: Any = Parameter(None)
    position_noise = WhiteNoise("R3", frame=PlanetFrame, sigma=0.0)
    position = Output()

    def update(self, ctx) -> PartUpdate:
        binding = ctx.field(PlanetBindingField)
        p_planet, _ = binding.planet.world_to_planet_sym(
            ctx.position[WorldFrame], ctx.velocity[WorldFrame], ctx.t
        )
        zero = Vec3[PartFrame].constant((0.0, 0.0, 0.0))
        return PartUpdate(
            wrench=Wrench(force=zero, torque=zero),
            outputs={"position": p_planet + self.position_noise},
            rates={"position": self.rate},
        )


__all__ = ["PlanetPositionSensor"]

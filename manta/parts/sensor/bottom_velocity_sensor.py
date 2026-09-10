"""Lightweight bottom-relative velocity observation for navigation filters."""

from __future__ import annotations

from typing import Any, ClassVar

from ...fields import PlanetBindingField
from ...ir.frames import PartFrame, WorldFrame
from ...ir.types import Vec3
from ...ir.wrench import Wrench
from .._declarations import Output, Parameter, PartUpdate, WhiteNoise
from ..base import Part, PartRole


class BottomVelocitySensor(Part):
    """Planet-surface-relative mount velocity in the sensor's own frame.

    The observation function assumes a stationary bottom in the bound
    planet's Cartesian frame.  It models the exact mounted-point kinematics
    and planet rotation, but deliberately contains no beams, bathymetry,
    acoustic propagation, or lock decision.  A navigation adapter may use
    reported altitude/beam quality and estimated attitude to gate or widen the
    covariance of each delivered measurement.
    """

    role = PartRole.SENSOR
    requires_fields: ClassVar[list[type]] = [PlanetBindingField]

    rate: Any = Parameter(None)
    velocity_noise = WhiteNoise("R3", frame=PartFrame, sigma=0.0)
    velocity = Output()

    def update(self, ctx) -> PartUpdate:
        binding = ctx.field(PlanetBindingField)
        planet = binding.planet
        _, v_planet = planet.world_to_planet_sym(
            ctx.position[WorldFrame], ctx.velocity[WorldFrame], ctx.t
        )
        # Re-express the planet-relative vector in WorldFrame, then in this
        # sensor's PartFrame.  This rotates a relative vector only; the
        # co-rotation term was already removed by world_to_planet_sym.
        v_relative_world = Vec3[WorldFrame].from_mx(
            planet.R_world_from_planet_sym(ctx.t) @ v_planet._mx
        )
        v_sensor = ctx.orientation.conjugate().apply(v_relative_world)
        zero = Vec3[PartFrame].constant((0.0, 0.0, 0.0))
        return PartUpdate(
            wrench=Wrench(force=zero, torque=zero),
            outputs={"velocity": v_sensor + self.velocity_noise},
            rates={"velocity": self.rate},
        )


__all__ = ["BottomVelocitySensor"]

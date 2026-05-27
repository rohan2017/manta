"""PointBuoy — single-point buoyancy displacing a fixed volume.

A craft submerged in a fluid feels a buoyant force opposing gravity at
each sampled volume element. PointBuoy is the simplest such element: a
single sample with displacement volume V at the part's mount offset.

  F = -ρ(p) · V · g(p)

evaluated at the buoy's world-frame position. ρ comes from the
registered FluidField; g from the registered GravityField. With no
FluidField registered, ρ defaults to zero → no buoyancy contribution
(useful for in-vacuum sanity tests). With Part.transform set to a
non-zero offset, the framework's wrench-at-offset lift rolls the force
up into the parent's torque automatically.

A future Hull part will be a multi-sample version: a list of points
distributing the buoyancy across the submerged shape. With enough
samples + a smooth water/air boundary, surface-crossing torques (the
"righting moment" of a hull) emerge naturally.
"""

from __future__ import annotations

from ...fields import FluidField, GravityField
from ...ir.frames import PartFrame, WorldFrame
from ...ir.types import Vec3
from ..base import Parameter, Part, PartUpdate
from ...ir.wrench import Wrench


class PointBuoy(Part):
    """Single-point buoyancy displacing a fixed volume.

    Parameters:
        volume — m³ displaced by the buoyancy element. Default 1e-3.

    Force = -ρ(p_world) · V · g(p_world) at the part's mount point,
    rotated from anchor to craft frame, applied at the offset (so the
    framework lifts force-at-offset → body-frame torque for tilt
    response).
    """

    volume: float = Parameter(1e-3)     # m³

    def update(self, ctx) -> PartUpdate:
        # ctx.position is already the buoy's world-frame mount point (the
        # kinematic pass composed the transform + any joints). Field queries
        # there capture the correct local value for spatially varying fields;
        # for uniform fields it's the same as the craft origin.
        p_world = ctx.position[WorldFrame]
        fluid    = ctx.field(FluidField).state_at_sym(p_world, ctx.t)
        g_world  = ctx.field(GravityField).state_at_sym(p_world, ctx.t)

        # F = -ρ·V·g  (opposes gravity, scaled by displaced mass), rotated
        # into the buoy's own frame for the wrench return (the framework
        # rotates it back to body and lifts force-at-offset → torque).
        scale = fluid.density * self.volume
        f_world = g_world * (-1.0) * scale
        f_part  = ctx.orientation.conjugate().apply(f_world)

        zero_t = Vec3[PartFrame].constant((0.0, 0.0, 0.0))
        return PartUpdate(wrench=Wrench(force=f_part, torque=zero_t))

"""PointBuoy — single-point buoyancy displacing a fixed volume.

A craft submerged in a fluid feels a buoyant force opposing gravity at
each sampled volume element. PointBuoy is the simplest such element: a
single sample with displacement volume V at the part's mount offset.

  F = -ρ(p) · V · g(p)

evaluated at the buoy's anchor-frame position. ρ comes from the
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

from ...ir.frames import AnchorFrame, CraftFrame
from ...ir.types import Vec3
from ..base import Parameter, Part, PartUpdate
from ...math.wrench import Wrench


class PointBuoy(Part):
    """Single-point buoyancy displacing a fixed volume.

    Parameters:
        volume — m³ displaced by the buoyancy element. Default 1e-3.

    Force = -ρ(p_anchor) · V · g(p_anchor) at the part's mount point,
    rotated from anchor to craft frame, applied at the offset (so the
    framework lifts force-at-offset → body-frame torque for tilt
    response).
    """

    volume: float = Parameter(1e-3)     # m³

    def update(self, ctx) -> PartUpdate:
        # Anchor-frame position of the buoyancy center.
        offset_craft = Vec3[CraftFrame].constant(tuple(self.transform))
        p_anchor = ctx.position + ctx.orientation.apply(offset_craft)

        # Field queries at the buoy's actual position. For uniform fields
        # this is the same as querying at the craft origin; for spatially
        # varying fields it captures the correct local value.
        fluid    = ctx.fluid_field.state_at_sym(p_anchor)
        g_anchor = ctx.gravity_field.state_at_sym(p_anchor)

        # F_anchor = -ρ·V·g  (opposes gravity, scaled by displaced mass).
        scale = fluid.density * self.volume
        # We want F = -scale * g_anchor. Vec3 supports scalar*Vec3 and
        # negation via *(-1.0).
        f_anchor = g_anchor * (-1.0) * scale
        # Rotate into CraftFrame for the wrench return.
        f_craft  = ctx.orientation.conjugate().apply(f_anchor)

        zero_t = Vec3[CraftFrame].constant((0.0, 0.0, 0.0))
        return PartUpdate(wrench=Wrench(force=f_craft, torque=zero_t))

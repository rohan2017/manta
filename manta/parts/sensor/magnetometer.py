"""Magnetometer — body-frame magnetic flux reading.

Queries the registered MagField at the sensor's mount point in
WorldFrame, rotates the result into CraftFrame. That's exactly what a
3-axis fluxgate magnetometer reads when the craft is oriented in the
field.

With no MagField registered, ctx.mag_field defaults to an empty field
(zero everywhere); the magnetometer reads zero. With a Part.transform
offset, the sensor reads the field at that offset point — for spatially
uniform B this is identical to the craft origin; for a nearby dipole
it captures the local gradient.
"""

from __future__ import annotations

from ...ir.frames import CraftFrame
from ...ir.types import Vec3
from ..base import Output, Part, PartUpdate
from ...ir.wrench import Wrench


class Magnetometer(Part):
    """3-axis magnetometer.

    Outputs:
        B : Vec3[CraftFrame] — magnetic flux density at the sensor
                                position, rotated into craft frame. SI
                                units (Tesla).
    """

    B = Output(shape="vec3")

    def update(self, ctx) -> PartUpdate:
        offset_craft = Vec3[CraftFrame].constant(tuple(self.transform))
        p_world = ctx.position + ctx.orientation.apply(offset_craft)

        B_world = ctx.mag_field.state_at_sym(p_world, ctx.t)
        B_craft  = ctx.orientation.conjugate().apply(B_world)

        zero_v = Vec3[CraftFrame].constant((0.0, 0.0, 0.0))
        return PartUpdate(
            wrench=Wrench(force=zero_v, torque=zero_v),
            outputs={"B": B_craft},
        )

"""PositionSensor — anchor-frame position observer (GPS / motion capture).

Emits the craft's anchor-frame position as a per-tick Output. The
sensor's Output declaration *is* the measurement model for the EKF;
noise can be injected externally (sim) or modeled on the part itself
(see `IMU` for the pattern).
"""

from __future__ import annotations

from ...ir.frames import AnchorFrame, CraftFrame
from ...ir.types import Vec3
from ..base import Output, Part, PartUpdate
from ...math.wrench import Wrench


class PositionSensor(Part):
    """Outputs the craft origin's anchor-frame position each tick.

    Outputs:
        position : Vec3[AnchorFrame]   — craft position in anchor frame,
                                         exactly what a GPS or mocap rig
                                         reads at the origin.

    For a sensor mounted at a static offset from the craft origin, the
    Part.transform parameter shifts the apparent reading by `R·offset` —
    handled by emitting `ctx.position + R·offset`. Default offset is zero
    (sensor at craft origin).
    """

    position = Output(shape="vec3")

    def update(self, ctx) -> PartUpdate:
        # Anchor-frame position of the sensor's mount point.
        offset_craft = Vec3[CraftFrame].constant(tuple(self.transform))
        offset_anchor = ctx.orientation.apply(offset_craft)
        p_anchor = ctx.position + offset_anchor

        zero_v = Vec3[CraftFrame].constant((0.0, 0.0, 0.0))
        return PartUpdate(
            wrench=Wrench(force=zero_v, torque=zero_v),
            outputs={"position": p_anchor},
        )

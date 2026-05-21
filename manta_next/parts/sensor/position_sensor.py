"""PositionSensor — anchor-frame position observer (GPS / motion capture).

Emits the sensor's anchor-frame position as a per-tick Output. The
Output declaration *is* the measurement model for the EKF; noise can
be injected externally (sim) or modeled on the part itself (see `IMU`
for the pattern).

`ctx.position` is already the sensor's own anchor-frame position — the
framework's kinematic pass composed it from the body state and the
chain of `Part.transform` / joint rotations above the sensor. So the
part code is a one-liner.
"""

from __future__ import annotations

from ...ir.frames import CraftFrame
from ...ir.types import Vec3
from ..base import Output, Part, PartUpdate
from ...ir.wrench import Wrench


class PositionSensor(Part):
    """Outputs the sensor's anchor-frame position each tick.

    Outputs:
        position : Vec3[AnchorFrame]   — sensor mount-point position in
                                         anchor frame; exactly what a GPS
                                         or mocap marker reads.
    """

    position = Output(shape="vec3")

    def update(self, ctx) -> PartUpdate:
        zero_v = Vec3[CraftFrame].constant((0.0, 0.0, 0.0))
        return PartUpdate(
            wrench=Wrench(force=zero_v, torque=zero_v),
            outputs={"position": ctx.position},
        )

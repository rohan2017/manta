"""TetherEndpoint — marker for one end of a tether.

A `TetherEndpoint` Part contributes zero wrench during the per-part
update loop. Its only job is to mark a location on the craft where a
`Tether` coupling object connects. Part.transform encodes the
attachment point in the craft's body frame.

The Tether coupling reads this part's transform when computing the
spring-damper force and applies the wrench at the offset — through the
framework's force-at-offset → body-frame torque lift, this gives
correct rotational dynamics from off-axis tether attachments.
"""

from __future__ import annotations

from ...ir.frames import PartFrame
from ...ir.types import Vec3
from .._declarations import PartUpdate
from ..base import Part
from ...ir.wrench import Wrench


class TetherEndpoint(Part):
    """Marker Part for one end of a tether. No wrench contribution; the
    Tether coupling applies the actual force using this part's transform
    as the attachment offset."""

    def update(self, ctx) -> PartUpdate:
        zero_v = Vec3[PartFrame].constant((0.0, 0.0, 0.0))
        return PartUpdate(wrench=Wrench(force=zero_v, torque=zero_v))

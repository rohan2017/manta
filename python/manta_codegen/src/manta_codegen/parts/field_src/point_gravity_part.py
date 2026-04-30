"""PointGravityPart — applies inverse-square gravity to the craft."""

from __future__ import annotations

from ...core import PartDescriptor
from ...fields import PointGravityField


class PointGravityPart(PartDescriptor):
    """Applies F = m_total * g(p) to the craft each tick, where g(p) is the
    inverse-square gravity vector from a registered PointGravityField at the
    craft's scene-frame position.

    Required fields: PointGravityField.
    Telemetry: none.
    """

    cpp_class          = "manta::parts::PointGravityPart"
    cpp_class_template = "manta::parts::PointGravityPartT"
    cpp_header         = "manta/parts/field_src/point_gravity_part.hpp"
    requires_fields    = [PointGravityField]

    def __init__(self, name: str = "point_gravity", **kwargs) -> None:
        super().__init__(name=name, **kwargs)

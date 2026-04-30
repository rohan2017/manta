"""GravityPart — applies the registered uniform gravity to the craft."""

from __future__ import annotations

from ...core import PartDescriptor
from ...fields import GravityField


class GravityPart(PartDescriptor):
    """Applies F = m_total * g to the craft each tick, where g is read from
    the registered uniform GravityField. Add to the craft root after all
    mass-bearing parts so `compute_params()` reflects the total mass.

    Required fields: GravityField.
    Telemetry: none.
    """

    cpp_class          = "manta::parts::GravityPart"
    cpp_class_template = "manta::parts::GravityPartT"
    cpp_header         = "manta/parts/field_src/gravity_part.hpp"
    requires_fields    = [GravityField]

    def __init__(self, name: str = "gravity", **kwargs) -> None:
        super().__init__(name=name, **kwargs)

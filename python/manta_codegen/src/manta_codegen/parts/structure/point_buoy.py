"""PointBuoy — single-point buoyancy from a configured volume."""

from __future__ import annotations

from ..._format import cpp_float as _f
from ...core import PartDescriptor
from ...fields import GravityField


class PointBuoy(PartDescriptor):
    """Single-point buoyancy: F = -ρ(p) * V * g, applied at the part origin.
    Density comes from the registered FluidField. For a body where the buoyant
    distribution matters (boats crossing a surface, righting moments), prefer
    Hull (sampled-points version).

    Required fields: FluidField, GravityField.
    """

    cpp_class          = "manta::parts::PointBuoy"
    cpp_class_template = "manta::parts::PointBuoyT"
    cpp_header         = "manta/parts/structure/point_buoy.hpp"
    requires_fields    = [GravityField]

    def __init__(self, name: str, volume: float, **kwargs) -> None:
        super().__init__(name=name, **kwargs)
        self.volume = float(volume)

    def emit_constructor_args(self, scalar: str = "manta::Real") -> str:
        return f'"{self.name}", {scalar}({_f(self.volume)})'

"""PointGravityField — inverse-square gravity from a center point."""

from __future__ import annotations

from .._format import cpp_float as _f
from ..core import FieldDescriptor


class PointGravityField(FieldDescriptor):
    """Inverse-square gravity. g(p) = -mu * (p - center) / |p - center|^3.

    `mu` is GM (gravitational parameter, m^3/s^2). Earth: 3.986e14, Moon: 4.9e12.
    """

    cpp_class     = "manta::fields::PointGravityField"
    cpp_header    = "manta/fields/point_gravity_field.hpp"
    feature_macro = "MANTA_HAS_POINT_GRAVITY_FIELD"

    def __init__(self,
                 mu: float,
                 center: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> None:
        super().__init__()
        self.mu = float(mu)
        self.center = tuple(float(x) for x in center)

    def emit_construction(self, var_name: str) -> str:
        cx, cy, cz = self.center
        return (f"{self.cpp_class} {var_name}{{"
                f"{_f(self.mu)}, "
                f"manta::geom::Vec3<manta::SceneFrame>{{{_f(cx)}, {_f(cy)}, {_f(cz)}}}}};")

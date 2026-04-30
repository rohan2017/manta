"""GravityField — uniform gravity with a configurable g vector."""

from __future__ import annotations

from .._format import cpp_float as _f
from ..core import FieldDescriptor


class GravityField(FieldDescriptor):
    """Uniform gravitational field. Default: g = (0, 0, -9.81) m/s^2."""

    cpp_class     = "manta::fields::GravityField"
    cpp_header    = "manta/fields/gravity_field.hpp"
    feature_macro = "MANTA_HAS_GRAVITY_FIELD"

    def __init__(self,
                 g: tuple[float, float, float] = (0.0, 0.0, -9.81)) -> None:
        super().__init__()
        self.g = tuple(float(x) for x in g)

    def emit_construction(self, var_name: str) -> str:
        gx, gy, gz = self.g
        return (f"{self.cpp_class} {var_name}{{"
                f"manta::geom::Vec3<manta::SceneFrame>{{{_f(gx)}, {_f(gy)}, {_f(gz)}}}}};")

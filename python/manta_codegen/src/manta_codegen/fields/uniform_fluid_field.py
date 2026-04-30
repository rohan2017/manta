"""UniformFluidField — constant density and velocity everywhere."""

from __future__ import annotations

from .._format import cpp_float as _f
from ..core import FieldDescriptor


class UniformFluidField(FieldDescriptor):
    """Constant density and bulk velocity. Useful for tests and quick sanity
    checks (e.g. immerse a craft in 'infinite ocean')."""

    cpp_class     = "manta::fields::UniformFluidField"
    cpp_header    = "manta/fields/uniform_fluid_field.hpp"
    feature_macro = "MANTA_HAS_UNIFORM_FLUID_FIELD"
    # Concrete FluidField — also register under the abstract base slot so
    # parts that query field<FluidField>() find it.
    register_as = ["manta::fields::FluidField"]

    def __init__(self,
                 density: float,
                 velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> None:
        super().__init__()
        self.density = float(density)
        self.velocity = tuple(float(x) for x in velocity)

    def emit_construction(self, var_name: str) -> str:
        vx, vy, vz = self.velocity
        return (f"{self.cpp_class} {var_name}{{{_f(self.density)}, "
                f"manta::geom::Vec3<manta::SceneFrame>{{{_f(vx)}, {_f(vy)}, {_f(vz)}}}}};")

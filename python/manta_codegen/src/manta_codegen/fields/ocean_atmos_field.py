"""OceanAtmosField — flat-sea two-layer fluid (water below, air above)."""

from __future__ import annotations

from .._format import cpp_float as _f
from ..core import FieldDescriptor


class OceanAtmosField(FieldDescriptor):
    """Two-layer fluid with a horizontal sea surface at z = sea_level. Below:
    water_density (with optional current). Above: air_density (with optional
    wind). Provides `height_above_sea_level(pos)` for boundary-aware Parts.
    """

    cpp_class     = "manta::fields::OceanAtmosField"
    cpp_header    = "manta/fields/ocean_atmos_field.hpp"
    feature_macro = "MANTA_HAS_OCEAN_ATMOS_FIELD"
    # Hull queries field<FluidField>(); register the concrete instance under
    # the abstract base slot too, then runtime dynamic_cast detects the
    # OceanAtmosField inside the Hull update for the smoothstep-blend branch.
    register_as = ["manta::fields::FluidField"]

    def __init__(self,
                 sea_level: float = 0.0,
                 water_density: float = 1000.0,
                 air_density:   float = 1.225) -> None:
        super().__init__()
        self.sea_level = float(sea_level)
        self.water_density = float(water_density)
        self.air_density = float(air_density)

    def emit_construction(self, var_name: str) -> str:
        return (f"{self.cpp_class} {var_name}{{"
                f"{_f(self.sea_level)}, {_f(self.water_density)}, {_f(self.air_density)}}};")

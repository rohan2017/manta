"""Earth — concrete Planet for near-Earth simulations."""

from __future__ import annotations

from .._format import cpp_float as _f
from ..core import PlanetDescriptor


class Earth(PlanetDescriptor):
    """When added to a Craft, codegen emits
        world.add_planet<manta::planets::Earth>(sea_level, water_density,
                                                 air_density, rotation_rate);
    and anchors the first Scene to this planet via Scene::set_planet.

    Earth automatically registers a FluidField (water-below + air-above
    profile) and a SeaSurface, so Hull-style buoyancy parts on a craft
    in this scene work without any extra field configuration.

    rotation_rate defaults to 0 (non-rotating Earth) for sim simplicity;
    pass `rotation_rate=Earth.SIDEREAL` (≈7.29e-5 rad/s) to enable
    Coriolis + centrifugal in the rotating frame.
    """

    SIDEREAL: float = 7.2921159e-5  # rad/s

    cpp_class  = "manta::planets::Earth"
    cpp_header = "manta/planets/earth.hpp"

    def __init__(self,
                 sea_level: float = 0.0,
                 water_density: float = 1000.0,
                 air_density: float = 1.225,
                 rotation_rate: float = 0.0) -> None:
        self.sea_level     = float(sea_level)
        self.water_density = float(water_density)
        self.air_density   = float(air_density)
        self.rotation_rate = float(rotation_rate)

    def emit_constructor_args(self) -> str:
        return (f"{_f(self.sea_level)}, "
                f"{_f(self.water_density)}, "
                f"{_f(self.air_density)}, "
                f"{_f(self.rotation_rate)}")

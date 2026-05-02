"""Earth — concrete Planet for near-Earth simulations."""

from __future__ import annotations

from .._format import cpp_float as _f
from ..core import PlanetDescriptor


class Earth(PlanetDescriptor):
    """When added to a World, codegen emits

        world.add_planet<manta::planets::Earth>(sea_level, water_density,
                                                 air_density, rotation_rate,
                                                 gravity_mu, include_j2,
                                                 dipole_moment);

    and anchors the first Scene to this planet via Scene::set_planet.

    Earth always registers a FluidField with two persistent disturbances
    (water below sea level, air above). Optionally registers a GravityField
    with a point-mass + J2 disturbance (`gravity_mu > 0`) and/or a MagField
    with a dipole disturbance (`dipole_moment != 0`).

    rotation_rate defaults to 0 (non-rotating Earth) for sim simplicity;
    pass `rotation_rate=Earth.SIDEREAL` (≈7.29e-5 rad/s) to enable
    Coriolis + centrifugal in the rotating frame.
    """

    SIDEREAL: float = 7.2921159e-5            # rad/s
    MU:       float = 3.986004418e14          # m^3/s^2
    R_EQ:     float = 6.378137e6              # m
    J2:       float = 1.0826267e-3
    DIPOLE:   float = 7.94e22                 # A·m^2

    cpp_class  = "manta::planets::Earth"
    cpp_header = "manta/planets/earth.hpp"

    def __init__(self,
                 sea_level: float = 0.0,
                 water_density: float = 1000.0,
                 air_density: float = 1.225,
                 rotation_rate: float = 0.0,
                 gravity_mu: float = 0.0,
                 include_j2: bool = False,
                 dipole_moment: float = 0.0) -> None:
        self.sea_level     = float(sea_level)
        self.water_density = float(water_density)
        self.air_density   = float(air_density)
        self.rotation_rate = float(rotation_rate)
        self.gravity_mu    = float(gravity_mu)
        self.include_j2    = bool(include_j2)
        self.dipole_moment = float(dipole_moment)

    def emit_constructor_args(self) -> str:
        j2 = "true" if self.include_j2 else "false"
        return (f"{_f(self.sea_level)}, "
                f"{_f(self.water_density)}, "
                f"{_f(self.air_density)}, "
                f"{_f(self.rotation_rate)}, "
                f"{_f(self.gravity_mu)}, "
                f"{j2}, "
                f"{_f(self.dipole_moment)}")

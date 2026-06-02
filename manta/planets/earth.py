"""Earth — concrete Planet preset for near-Earth simulations.

`Earth` registers its standing field contributions on the World's
shared GravityField, FluidField, and MagField:

  * GravityField : point-mass at planet origin (μ = gravity_mu) plus
                   an optional J2 oblateness perturbation. Skipped
                   entirely when `gravity_mu == 0`.
  * FluidField   : an ocean co-rotating with the planet (density =
                   water_density, active below sea level) and an
                   atmosphere co-rotating with the planet (density =
                   air_density, active above sea level). Bound by
                   `planet_radius + sea_level` from the planet center.
  * MagField     : a magnetic dipole at planet origin, aligned along
                   the spin axis (pointing along `-rotation_axis`).
                   Skipped when `dipole_moment == 0`.

Default rotation rate is 0 (non-rotating Earth, for sim simplicity).
Pass `rotation_rate=Earth.SIDEREAL` to enable Coriolis + centrifugal
effects (and a co-rotating ocean).
"""

from __future__ import annotations

import casadi as ca

from ..fields import (
    DipoleMag, GravityField, FluidField, J2Gravity, MagField,
    PointMassGravity,
)
from .base import Planet
from .disturbances import PlanetFrameFluid


class Earth(Planet):
    """Standard Earth preset.

    Args:
        name           — identifier. Default "earth".
        position       — planet center in WorldFrame (m).
        rotation_rate  — angular rate, rad/s. Earth sidereal:
                         `Earth.SIDEREAL`. Default 0.
        sea_level      — elevation of the ocean's top above the
                         planet's equatorial radius, m. Default 0
                         (sea-level surface coincides with R_EQ).
        water_density  — ocean density, kg/m³. Default 1025 (seawater).
        air_density    — atmosphere density, kg/m³. Default 1.225 (ISA
                         sea-level).
        gravity_mu     — gravitational parameter μ (m³/s²). 0 disables
                         gravity. Default `Earth.MU`.
        include_j2     — register a J2 oblateness perturbation
                         alongside the point-mass term. Default False.
        dipole_moment  — magnetic dipole strength, A·m². 0 disables
                         magnetic. Default 0.
    """

    # CODATA / WGS84 / IGRF leading-term constants.
    SIDEREAL: float = 7.2921159e-5       # rad/s
    MU:       float = 3.986004418e14     # m^3/s^2
    R_EQ:     float = 6.378137e6         # m
    J2:       float = 1.0826267e-3
    DIPOLE:   float = 7.94e22            # A·m²

    # ISA-style atmospheric scale height (m). Density falls off as
    # exp(-altitude / SCALE_HEIGHT) above sea level.
    ATMOSPHERE_SCALE_HEIGHT: float = 8500.0

    def __init__(self,
                 name: str = "earth",
                 *,
                 position: tuple[float, float, float] = (0.0, 0.0, 0.0),
                 rotation_rate: float = 0.0,
                 sea_level: float = 0.0,
                 water_density: float = 1025.0,
                 air_density: float = 1.225,
                 atmosphere_scale_height: float | None = None,
                 gravity_mu: float = MU,
                 include_j2: bool = False,
                 dipole_moment: float = 0.0) -> None:
        super().__init__(name=name,
                         position=position,
                         rotation_axis=(0.0, 0.0, 1.0),
                         omega=rotation_rate)
        self.sea_level     = float(sea_level)
        self.water_density = float(water_density)
        self.air_density   = float(air_density)
        self.atmosphere_scale_height = float(
            atmosphere_scale_height
            if atmosphere_scale_height is not None
            else self.ATMOSPHERE_SCALE_HEIGHT)
        self.gravity_mu    = float(gravity_mu)
        self.include_j2    = bool(include_j2)
        self.dipole_moment = float(dipole_moment)

    # ------------------------------------------------------------------

    @property
    def planet_radius(self) -> float:
        """Radius from planet center to sea-level surface.
        Equal to R_EQ + sea_level."""
        return self.R_EQ + self.sea_level

    # ------------------------------------------------------------------

    def register_disturbances(self, world) -> None:
        # Gravity: point-mass + optional J2.
        if self.gravity_mu > 0.0:
            gf = world.get_or_create_field(GravityField)
            gf.add(PointMassGravity(
                position=tuple(self.center.tolist()),
                GM=self.gravity_mu))
            if self.include_j2:
                gf.add(J2Gravity(
                    position=tuple(self.center.tolist()),
                    GM=self.gravity_mu,
                    J2=self.J2,
                    eq_radius=self.R_EQ,
                    polar_axis=tuple(self.axis.tolist())))

        # Fluid: a single PlanetFrameFluid whose density function
        # branches on altitude (water below sea level, exponential
        # atmosphere above). The lambda uses `ca.if_else` — a hard
        # boundary is fine; CasADi's autodiff handles it cleanly
        # everywhere except the (measure-zero) interface itself.
        ff = world.get_or_create_field(FluidField)
        R_planet  = self.planet_radius
        rho_w     = self.water_density
        rho_air   = self.air_density
        scale_h   = self.atmosphere_scale_height

        def density_fn(p_planet):
            # Radial altitude above the sea-level sphere.
            r = ca.sqrt(ca.dot(p_planet, p_planet) + 1e-30)
            altitude = r - R_planet
            return ca.if_else(altitude < 0.0,
                              rho_w,
                              rho_air * ca.exp(-altitude / scale_h))

        ff.add(PlanetFrameFluid(planet=self,
                                density_fn=density_fn,
                                name=f"{self.name}_fluid"))

        # Magnetic dipole along the spin axis (-axis for a planet whose
        # spin axis points away from the geographic-north magnetic dip).
        if self.dipole_moment != 0.0:
            mf = world.get_or_create_field(MagField)
            moment_vec = (-self.dipole_moment * self.axis).tolist()
            mf.add(DipoleMag(
                position=tuple(self.center.tolist()),
                moment=tuple(moment_vec)))

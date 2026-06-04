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

from dataclasses import dataclass

import casadi as ca
import numpy as np

from ..fields import (
    CollisionField, DipoleMag, GravityField, FluidField, J2Gravity,
    MagField, PointMassGravity,
)
from .base import Planet
from .disturbances import PlanetFrameFluid


@dataclass(frozen=True)
class SeaWaves:
    """Planar deep-water sinusoid riding a planet's sea surface.

    The surface elevation (above the sea-level sphere) is

        η(p, t) = amplitude · cos(k·ξ − ω·t),   ξ = p_planet · direction

    with k = 2π/wavelength and ω = k·c. The phase speed c defaults to
    the deep-water dispersion relation c = √(g·λ / 2π). Underwater, the
    fluid carries the matching first-order orbital velocity — particles
    circle with radius `amplitude` at the surface, decaying as e^{k·z}
    with depth — so drag surfaces and foils feel the moving water, not
    just the moving boundary.

    `direction` is a planet-frame vector (normalized; its radial
    component at the point of interest should be ~0). The wave is a
    PLANAR field in planet coordinates — valid for a local patch of
    ocean, not a globe-wrapping solution.

    Args:
        amplitude   — m (crest height above mean sea level).
        wavelength  — m (crest-to-crest).
        direction   — planet-frame propagation direction. Default +x.
        speed       — phase speed override, m/s. None → deep-water
                      dispersion using the planet's surface gravity.
    """

    amplitude: float
    wavelength: float
    direction: tuple = (1.0, 0.0, 0.0)
    speed: float | None = None


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
        waves          — optional `SeaWaves`: a sinusoidal moving sea
                         surface (boundary elevation + underwater
                         orbital velocity). Default None (flat sea).
        surface_collision — register the sea-level sphere as a solid
                         `CollisionField` obstacle (a rough model of
                         the surface), so `Collider`-footed craft can
                         stand anywhere on the planet without a
                         per-site ground plane. Default True.
        surface_smoothing — m. Blend the water/air density switch over
                         this length (logistic in altitude) instead of
                         a hard `if_else`. Physically: a finite-size
                         volume element crosses the surface over its
                         own diameter; numerically it turns point-
                         sampled buoyancy from bang-bang into a smooth
                         ramp (a floating hull finds a stable draft, a
                         surface-piercing foil gets a smooth lift-vs-
                         height slope). Default 0 (hard boundary).
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
                 dipole_moment: float = 0.0,
                 waves: SeaWaves | None = None,
                 surface_collision: bool = True,
                 surface_smoothing: float = 0.0) -> None:
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
        self.waves         = waves
        self.surface_collision = bool(surface_collision)
        self.surface_smoothing = float(surface_smoothing)

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
        # branches on altitude (water below the — possibly waving —
        # sea surface, exponential atmosphere above), optionally
        # blended over `surface_smoothing` metres. With waves, the
        # underwater bulk velocity carries the first-order deep-water
        # orbital motion.
        ff = world.get_or_create_field(FluidField)
        R_planet  = self.planet_radius
        rho_w     = self.water_density
        rho_air   = self.air_density
        scale_h   = self.atmosphere_scale_height
        delta     = self.surface_smoothing
        waves     = self.waves

        if waves is not None:
            wave_dir = np.asarray(waves.direction, dtype=float)
            wave_dir = wave_dir / np.linalg.norm(wave_dir)
            k_wave   = 2.0 * np.pi / float(waves.wavelength)
            if waves.speed is not None:
                c_wave = float(waves.speed)
            else:
                g0 = (self.gravity_mu / R_planet**2
                      if self.gravity_mu > 0.0 else 9.80665)
                c_wave = float(np.sqrt(g0 * waves.wavelength
                                       / (2.0 * np.pi)))
            omega_wave = k_wave * c_wave
            dir_dm = ca.DM(wave_dir.reshape(3, 1))

        def _altitude(p_planet, t):
            """Signed height above the local (waving) sea surface, plus
            the mean-sea-level altitude (for depth-decay terms)."""
            r = ca.sqrt(ca.dot(p_planet, p_planet) + 1e-30)
            alt_mean = r - R_planet
            if waves is None:
                return alt_mean, alt_mean
            xi    = ca.dot(p_planet, dir_dm)
            eta   = waves.amplitude * ca.cos(k_wave * xi - omega_wave * t)
            return alt_mean - eta, alt_mean

        def _wet(altitude):
            """Water fraction: 1 below −δ, 0 above +δ, C¹ smoothstep
            between. Compact support matters — with a ~1000:1 density
            ratio, a logistic tail would leak water density metres into
            the air."""
            if delta <= 0.0:
                return ca.if_else(altitude < 0.0, 1.0, 0.0)
            s = ca.fmax(-1.0, ca.fmin(1.0, altitude / delta))
            return 0.5 - 0.75 * s + 0.25 * s**3

        def density_fn(p_planet, t):
            altitude, _ = _altitude(p_planet, t)
            # Air density continues its profile down to the surface;
            # clamp the exponent so it doesn't grow below.
            rho_air_at = rho_air * ca.exp(-ca.fmax(altitude, 0.0) / scale_h)
            if delta <= 0.0:
                return ca.if_else(altitude < 0.0, rho_w, rho_air_at)
            wet = _wet(altitude)
            return wet * rho_w + (1.0 - wet) * rho_air_at

        if waves is None:
            velocity_fn = None
        else:
            def velocity_fn(p_planet, t):
                # First-order deep-water orbital velocity: circles of
                # radius `amplitude·e^{k·z}`, in phase with the crest,
                # gated to the wet side of the (waving) surface.
                altitude, alt_mean = _altitude(p_planet, t)
                r = ca.sqrt(ca.dot(p_planet, p_planet) + 1e-30)
                up = p_planet / r
                xi = ca.dot(p_planet, dir_dm)
                phase = k_wave * xi - omega_wave * t
                decay = ca.exp(k_wave * ca.fmin(alt_mean, 0.0))
                speed = waves.amplitude * omega_wave * decay \
                    * _wet(altitude)
                return speed * (ca.cos(phase) * dir_dm
                                + ca.sin(phase) * up)

        ff.add(PlanetFrameFluid(planet=self,
                                density_fn=density_fn,
                                velocity_fn=velocity_fn,
                                name=f"{self.name}_fluid"))

        # Solid surface: the sea-level sphere as a collision obstacle —
        # locally indistinguishable from a ground plane (the outward
        # normal is the local radial), valid anywhere on the planet.
        if self.surface_collision:
            cf = world.get_or_create_field(CollisionField)
            cf.add_sphere(center=tuple(self.center.tolist()),
                          radius=R_planet)

        # Magnetic dipole along the spin axis (-axis for a planet whose
        # spin axis points away from the geographic-north magnetic dip).
        if self.dipole_moment != 0.0:
            mf = world.get_or_create_field(MagField)
            moment_vec = (-self.dipole_moment * self.axis).tolist()
            mf.add(DipoleMag(
                position=tuple(self.center.tolist()),
                moment=tuple(moment_vec)))

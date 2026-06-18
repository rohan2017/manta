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
    MagField, PointMassGravity, below_surface,
)
from ..smoothing import soft_norm
from .atmosphere import LAPSE_ISA, R_AIR, T0_ISA
from .base import Planet
from .disturbances import Atmosphere, Ocean


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

    def __post_init__(self) -> None:
        if self.amplitude < 0.0:
            raise ValueError(
                f"SeaWaves: amplitude must be >= 0; got {self.amplitude!r}")
        if self.wavelength <= 0.0:
            raise ValueError(
                f"SeaWaves: wavelength must be > 0; got {self.wavelength!r}")
        if self.speed is not None and self.speed <= 0.0:
            raise ValueError(
                f"SeaWaves: speed must be > 0; got {self.speed!r}")
        d = np.asarray(self.direction, dtype=float)
        if d.shape != (3,) or not np.all(np.isfinite(d)) \
                or np.linalg.norm(d) == 0.0:
            raise ValueError(
                f"SeaWaves: direction must be a finite nonzero 3-vector; "
                f"got {self.direction!r}")


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
        air_density    — atmosphere density at sea level, kg/m³. Default
                         1.225 (ISA). Sets the sea-level pressure via the
                         ideal-gas law `P0 = ρ0·R·T0`; aloft the air
                         follows the ISA troposphere (lapse + ideal gas),
                         so density is no longer a pure exponential.
        sea_level_temperature — ISA sea-level temperature T0, K. Default
                         288.15. Drops with altitude at `lapse_rate`.
        lapse_rate     — ISA troposphere temperature lapse, K/m. Default
                         6.5e-3.
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
        surface_smoothing — m. Blend the water/air switch over this
                         length (a C¹ Hermite step in altitude) instead
                         of a hard `if_else`. Physically: a finite-size
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

    def __init__(self,
                 name: str = "earth",
                 *,
                 position: tuple[float, float, float] = (0.0, 0.0, 0.0),
                 rotation_rate: float = 0.0,
                 rotation_axis: tuple[float, float, float] = (0.0, 0.0, 1.0),
                 sea_level: float = 0.0,
                 water_density: float = 1025.0,
                 air_density: float = 1.225,
                 sea_level_temperature: float = T0_ISA,
                 lapse_rate: float = LAPSE_ISA,
                 gravity_mu: float = MU,
                 include_j2: bool = False,
                 dipole_moment: float = 0.0,
                 waves: SeaWaves | None = None,
                 surface_collision: bool = True,
                 surface_smoothing: float = 0.0) -> None:
        # rotation_axis lets a LOCAL-tangent sim sit at a latitude: tilt the
        # spin axis off local-up so the inertial Earth rate the IMU senses has
        # a horizontal (north) component — the gyrocompass signal. Default +z
        # (sub at the pole / spin axis = local vertical).
        super().__init__(name=name,
                         position=position,
                         rotation_axis=rotation_axis,
                         omega=rotation_rate)
        self.sea_level     = float(sea_level)
        self.water_density = float(water_density)
        self.air_density   = float(air_density)
        self.sea_level_temperature = float(sea_level_temperature)
        self.lapse_rate    = float(lapse_rate)
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

        # Fluid: two baseline regimes on one FluidField — an `Atmosphere`
        # (global background, ISA troposphere) overlaid by an `Ocean`
        # (membership below the — possibly waving — sea surface, blended
        # over `surface_smoothing` metres). The FluidField layers them by
        # membership: `(1−wet)·air + wet·ocean`. The (ρ,P,T) physics live
        # in `manta.planets.atmosphere`, so other planets reuse this.
        ff = world.get_or_create_field(FluidField)
        R_planet = self.planet_radius
        rho_w    = self.water_density
        T0       = self.sea_level_temperature
        L        = self.lapse_rate
        # Sea-level pressure implied by the sea-level air density (P0 =
        # ρ0·R·T0); with the defaults this is the conventional 101325 Pa.
        P0       = self.air_density * R_AIR * T0
        delta    = self.surface_smoothing
        waves    = self.waves

        # Surface gravity for barometric + hydrostatic pressure. Needed
        # even when the gravity *force* is disabled (gravity_mu=0): fall
        # back to the class default μ rather than a magic g.
        mu = self.gravity_mu if self.gravity_mu > 0.0 else self.MU
        g0 = mu / R_planet ** 2

        if waves is not None:
            wave_dir = np.asarray(waves.direction, dtype=float)
            wave_dir = wave_dir / np.linalg.norm(wave_dir)
            k_wave   = 2.0 * np.pi / float(waves.wavelength)
            c_wave   = (float(waves.speed) if waves.speed is not None
                        else float(np.sqrt(g0 * waves.wavelength
                                           / (2.0 * np.pi))))
            omega_wave = k_wave * c_wave
            dir_dm = ca.DM(wave_dir.reshape(3, 1))

        def _signed_altitude(p_planet, t):
            """Signed height (m) of a PlanetFrame point above the local
            (waving) sea surface — negative when submerged."""
            alt_mean = soft_norm(p_planet) - R_planet
            if waves is None:
                return alt_mean
            xi  = ca.dot(p_planet, dir_dm)
            eta = waves.amplitude * ca.cos(k_wave * xi - omega_wave * t)
            return alt_mean - eta

        def _depth(p_planet, t):
            """Depth (m, positive downward) below the sea surface."""
            return ca.fmax(0.0, -_signed_altitude(p_planet, t))

        def _surface_height_world(point, t):
            """Signed altitude above the surface for a WORLD-frame query
            point — the ocean's membership runs on world coords, so do
            the world→planet transform here (the field weights by it)."""
            t = t._mx if hasattr(t, "_mx") else t
            r_world  = point._mx - self.position_world_sym()
            R_pw     = ca.transpose(self.R_world_from_planet_sym(t))
            return _signed_altitude(R_pw @ r_world, t)

        if waves is None:
            ocean_velocity_fn = None
        else:
            def ocean_velocity_fn(p_planet, t):
                # First-order deep-water orbital velocity: circles of
                # radius `amplitude·e^{k·z}` in phase with the crest. No
                # explicit wet-gate — the ocean membership in the field's
                # baseline blend confines it to the water.
                alt_mean = soft_norm(p_planet) - R_planet
                up = p_planet / soft_norm(p_planet)
                xi = ca.dot(p_planet, dir_dm)
                phase = k_wave * xi - omega_wave * t
                decay = ca.exp(k_wave * ca.fmin(alt_mean, 0.0))
                speed = waves.amplitude * omega_wave * decay
                return speed * (ca.cos(phase) * dir_dm
                                + ca.sin(phase) * up)

        # Background air everywhere (added first), ocean carving in below
        # the surface (added second → overrides where its membership is
        # high).
        ff.add(Atmosphere(self, _signed_altitude,
                          surface_gravity=g0,
                          sea_level_temperature=T0,
                          sea_level_pressure=P0,
                          lapse_rate=L,
                          gas_constant=R_AIR,
                          name=f"{self.name}_air"))
        ff.add(Ocean(self, _depth,
                     surface_gravity=g0,
                     water_density=rho_w,
                     surface_pressure=P0,
                     temperature=T0,
                     velocity_fn=ocean_velocity_fn,
                     membership=below_surface(_surface_height_world, delta),
                     name=f"{self.name}_ocean"))

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

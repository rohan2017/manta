"""Planet-frame fluid disturbances — generic regime + Ocean / Atmosphere.

A `PlanetFrameFluid` is a fluid disturbance whose properties are
functions of *PlanetFrame* coordinates. The framework handles the
world↔planet transform (using the planet's symbolic rotation at time t)
and adds the `ω × (point − planet.center)` co-rotation contribution to
the WorldFrame velocity automatically. The caller supplies per-point
density / pressure / temperature / rest-frame velocity as branch-OK
CasADi callables — use `ca.if_else`, `ca.fmax`, exponentials, anything
in the symbolic graph.

`Ocean` and `Atmosphere` are thin baseline regimes built on top: they
wire the **planet-independent** equations in `manta.planets.atmosphere`
(ISA lapse + ideal gas for air; incompressible hydrostatic for water)
into `PlanetFrameFluid`, parameterised by an `altitude_fn` / `depth_fn`
the planet supplies (its sea-surface geometry, possibly waving) plus its
constants. So a different planet reuses the machinery with its own
profile.
"""

from __future__ import annotations

from typing import Callable

import casadi as ca

from ..fields import Disturbance, FluidState
from ..ir.frames import WorldFrame
from ..ir.types import Vec3
from .atmosphere import (
    LAPSE_ISA, P0_ISA, R_AIR, T0_ISA,
    hydrostatic_pressure, ideal_gas_density, isa_pressure, isa_temperature,
)


_VEC3_W = Vec3[WorldFrame]


class PlanetFrameFluid(Disturbance):
    """A fluid disturbance authored in PlanetFrame coordinates.

    Args:
        planet         — owning Planet. Provides the world↔planet
                         transform and the rotation axis / rate.
        density_fn     — callable `(p_planet: ca.MX(3,1), t: ca.MX) →
                         ca.MX` scalar density at the queried point in
                         PlanetFrame coordinates at time t (time supports
                         e.g. surface waves).
        pressure_fn    — optional `(p_planet, t) → ca.MX` scalar pressure
                         (Pa). Defaults to zero (unmodelled).
        temperature_fn — optional `(p_planet, t) → ca.MX` scalar
                         temperature (K). Defaults to zero (unmodelled).
        velocity_fn    — optional `(p_planet, t) → ca.MX(3,1)` bulk fluid
                         velocity *in PlanetFrame*. Defaults to zero (a
                         fluid at rest in the planet frame). Rotated back
                         to WorldFrame and augmented with `ω × r`.
        membership     — optional `(point: Vec3[World], t) → ca.MX in
                         [0,1]` spatial support (e.g. an ocean half-space);
                         see `Disturbance.membership`. Default: global.
        combining      — combining role on the host FluidField. Defaults
                         to "baseline" (a regime medium).
    """

    field_value_shape = FluidState

    def __init__(self,
                 planet,
                 density_fn: Callable[[ca.MX, ca.MX], ca.MX],
                 *,
                 pressure_fn: Callable[[ca.MX, ca.MX], ca.MX] | None = None,
                 temperature_fn: Callable[[ca.MX, ca.MX], ca.MX] | None = None,
                 velocity_fn: Callable[[ca.MX, ca.MX], ca.MX] | None = None,
                 name: str | None = None,
                 membership=None,
                 combining: str = "baseline") -> None:
        super().__init__(name=name, combining=combining, membership=membership)
        self.planet         = planet
        self.density_fn     = density_fn
        self.pressure_fn    = pressure_fn
        self.temperature_fn = temperature_fn
        self.velocity_fn    = velocity_fn

    def _to_planet(self, point, t):
        """World query point → PlanetFrame coords + the world-frame
        offset from the planet centre (shared by contribute + membership)."""
        t = t._mx if hasattr(t, "_mx") else t
        r_world  = point._mx - self.planet.position_world_sym()
        R_pw     = ca.transpose(self.planet.R_world_from_planet_sym(t))
        return R_pw @ r_world, r_world, t

    def contribute_at_sym(self, point, t) -> FluidState:
        p_planet, r_world, t = self._to_planet(point, t)

        # Scalars (PlanetFrame → MX). Branches via ca.if_else are safe —
        # CasADi walks both and the Jacobian is well-defined except at
        # the measure-zero interface.
        rho  = self.density_fn(p_planet, t)
        pres = (self.pressure_fn(p_planet, t)
                if self.pressure_fn is not None else ca.MX(0.0))
        temp = (self.temperature_fn(p_planet, t)
                if self.temperature_fn is not None else ca.MX(0.0))

        # Velocity in PlanetFrame (defaults to zero) → WorldFrame.
        if self.velocity_fn is not None:
            v_planet = self.velocity_fn(p_planet, t)
        else:
            v_planet = ca.MX.zeros(3, 1)
        R_wp    = self.planet.R_world_from_planet_sym(t)
        v_world = R_wp @ v_planet
        # Co-rotation: a stationary-in-planet-frame fluid carries
        # tangential WorldFrame velocity `ω × r_world`.
        omega_w = self.planet.omega_world_sym()
        v_world = v_world + ca.cross(omega_w, r_world)

        return FluidState(
            density     = rho,
            pressure    = pres,
            temperature = temp,
            velocity    = _VEC3_W.from_mx(v_world),
        )

    def __repr__(self) -> str:
        return f"<{type(self).__name__} planet={self.planet.name!r}>"


class Atmosphere(PlanetFrameFluid):
    """A gaseous baseline regime — ISA troposphere (lapse + ideal gas).

    The `(T, P, ρ)` triple is mutually consistent by construction:
    `T = T0 − lapse·alt`, `P = P0·(T/T0)^(g/(R·lapse))`, `ρ = P/(R·T)`.
    Defaults to a global membership (air everywhere); pass a `membership`
    to bound it.

    Args:
        planet        — owning Planet.
        altitude_fn   — `(p_planet, t) → ca.MX` signed height above the
                        surface (m), the planet's sea-surface geometry.
        surface_gravity — g at the surface (m/s²), for the barometric
                        exponent.
        sea_level_temperature / sea_level_pressure / lapse_rate /
        gas_constant  — ISA reference constants (Earth defaults).
        velocity_fn   — optional PlanetFrame bulk velocity.
        membership / name — see `PlanetFrameFluid`.
    """

    def __init__(self,
                 planet,
                 altitude_fn: Callable[[ca.MX, ca.MX], ca.MX],
                 *,
                 surface_gravity: float,
                 sea_level_temperature: float = T0_ISA,
                 sea_level_pressure: float = P0_ISA,
                 lapse_rate: float = LAPSE_ISA,
                 gas_constant: float = R_AIR,
                 velocity_fn: Callable[[ca.MX, ca.MX], ca.MX] | None = None,
                 membership=None,
                 name: str | None = None) -> None:
        T0, P0, L, R, g = (sea_level_temperature, sea_level_pressure,
                           lapse_rate, gas_constant, surface_gravity)

        def temperature_fn(p_planet, t):
            return isa_temperature(altitude_fn(p_planet, t), T0, L)

        def pressure_fn(p_planet, t):
            return isa_pressure(altitude_fn(p_planet, t), P0, T0, L, g, R)

        def density_fn(p_planet, t):
            return ideal_gas_density(pressure_fn(p_planet, t),
                                     temperature_fn(p_planet, t), R)

        super().__init__(planet, density_fn,
                         pressure_fn=pressure_fn,
                         temperature_fn=temperature_fn,
                         velocity_fn=velocity_fn,
                         membership=membership, name=name)


class Ocean(PlanetFrameFluid):
    """A liquid baseline regime — incompressible water, hydrostatic P.

    `ρ = water_density` (const), `P = P_surface + ρ·g·depth`, `T` const.
    Pass a `membership` (typically `below_surface(...)`) to confine it to
    the submerged half-space.

    Args:
        planet        — owning Planet.
        depth_fn      — `(p_planet, t) → ca.MX` depth below the surface
                        (m, positive downward).
        surface_gravity — g at the surface (m/s²), for hydrostatic P.
        water_density — kg/m³ (default 1025, seawater).
        surface_pressure — Pa at the sea surface (default ISA P0).
        temperature   — K (constant, default ISA T0).
        velocity_fn   — optional PlanetFrame bulk velocity (wave orbital).
        membership / name — see `PlanetFrameFluid`.
    """

    def __init__(self,
                 planet,
                 depth_fn: Callable[[ca.MX, ca.MX], ca.MX],
                 *,
                 surface_gravity: float,
                 water_density: float = 1025.0,
                 surface_pressure: float = P0_ISA,
                 temperature: float = T0_ISA,
                 velocity_fn: Callable[[ca.MX, ca.MX], ca.MX] | None = None,
                 membership=None,
                 name: str | None = None) -> None:
        rho_w, P_surf, g, T = (water_density, surface_pressure,
                               surface_gravity, temperature)

        def density_fn(p_planet, t):
            return ca.MX(rho_w)

        def pressure_fn(p_planet, t):
            return hydrostatic_pressure(P_surf, rho_w, g, depth_fn(p_planet, t))

        def temperature_fn(p_planet, t):
            return ca.MX(T)

        super().__init__(planet, density_fn,
                         pressure_fn=pressure_fn,
                         temperature_fn=temperature_fn,
                         velocity_fn=velocity_fn,
                         membership=membership, name=name)

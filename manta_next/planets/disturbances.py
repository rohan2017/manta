"""Planet-frame disturbance helpers.

A `PlanetFrameFluid` is a fluid disturbance whose density and rest-
frame velocity are functions of *PlanetFrame* coordinates. The
framework handles the world↔planet transform (using the planet's
symbolic rotation at time t) and adds the `ω × (point − planet.center)`
contribution to the WorldFrame velocity automatically. The user
supplies the per-point density and rest-frame velocity as branch-OK
CasADi callables.

This replaces the previous `PlanetCoRotatingFluid` (which baked in a
specific sphere shape + soft-smoothed cutoff). Callers are now free
to use `ca.if_else`, `ca.fmax`, exponential profiles — anything
expressible in the symbolic graph.
"""

from __future__ import annotations

from typing import Callable

import casadi as ca

from ..fields import Disturbance, FluidState
from ..ir.frames import WorldFrame
from ..ir.types import Vec3


_VEC3_W = Vec3[WorldFrame]


class PlanetFrameFluid(Disturbance):
    """A fluid disturbance authored in PlanetFrame coordinates.

    Args:
        planet         — owning Planet. Provides the world↔planet
                         transform and the rotation axis / rate.
        density_fn     — callable `(p_planet: ca.MX(3,1)) → ca.MX`
                         scalar density at the queried point,
                         expressed in PlanetFrame coordinates.
        velocity_fn    — optional callable
                         `(p_planet: ca.MX(3,1)) → ca.MX(3,1)`
                         returning the bulk fluid velocity *in
                         PlanetFrame coordinates* at the query
                         point. Defaults to zero (a fluid at rest
                         in the planet's frame). Whatever this
                         returns is rotated back to WorldFrame and
                         augmented with the planet's `ω × r`
                         co-rotation contribution.
        combining      — combining flag for the host FluidField; see
                         `Disturbance` docs.
    """

    field_value_shape = FluidState

    def __init__(self,
                 planet,
                 density_fn: Callable[[ca.MX], ca.MX],
                 velocity_fn: Callable[[ca.MX], ca.MX] | None = None,
                 *,
                 name: str | None = None,
                 combining: str = "additive") -> None:
        super().__init__(name=name, combining=combining)
        self.planet      = planet
        self.density_fn  = density_fn
        self.velocity_fn = velocity_fn

    def contribute_at_sym(self, point, t) -> FluidState:
        # World→Planet transform of the query point.
        r_world  = point._mx - self.planet.position_world_sym()
        R_pw     = ca.transpose(self.planet.R_world_from_planet_sym(t))
        p_planet = R_pw @ r_world

        # Density (PlanetFrame → scalar). Branches via ca.if_else are
        # safe here — CasADi walks both branches and the EKF's Jacobian
        # is well-defined except at the (measure-zero) interface.
        rho = self.density_fn(p_planet)

        # Velocity in PlanetFrame (defaults to zero) → WorldFrame.
        if self.velocity_fn is not None:
            v_planet = self.velocity_fn(p_planet)
        else:
            v_planet = ca.MX.zeros(3, 1)
        R_wp     = self.planet.R_world_from_planet_sym(t)
        v_world  = R_wp @ v_planet
        # Co-rotation contribution: a stationary-in-planet-frame fluid
        # carries tangential WorldFrame velocity `ω × r_world`.
        omega_w  = self.planet.omega_world_sym()
        v_world  = v_world + ca.cross(omega_w, r_world)

        return FluidState(
            density  = rho,
            velocity = _VEC3_W.from_mx(v_world),
        )

    def __repr__(self) -> str:
        return f"<PlanetFrameFluid planet={self.planet.name!r}>"

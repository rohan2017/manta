"""Local z-up, deep-water wave field for worlds without a planet.

The linear wave law matches Earth's SeaWaves: a moving wet boundary,
depth-decaying orbital flow and wave pressure. This is not a slamming or
wave-radiation model. Components are (amplitude, wavelength, direction_xy, phase).
"""

import math
from collections.abc import Iterable
from dataclasses import replace
from typing import TypedDict, Unpack

import casadi as ca

from ..ir.frames import WorldFrame
from ..ir.types import Vec3
from ..smoothing import smooth_max0
from .fluid import FlatOcean, below_surface


WaveComponent = tuple[float, float, tuple[float, float], float]


class _OceanOptions(TypedDict, total=False):
    """Forwarded FlatOcean inputs; that constructor owns their defaults."""

    density: float
    surface_z: float
    surface_pressure: float
    gravity: float
    temperature: float
    viscosity: float
    velocity: tuple[float, float, float]
    surface_blend: float
    name: str | None
    combining: str | None


class WaveOcean(FlatOcean):
    def __init__(
        self,
        *,
        components: Iterable[WaveComponent] = (),
        **kwargs: Unpack[_OceanOptions],
    ) -> None:
        super().__init__(**kwargs)
        self.components = tuple(components)
        for amplitude, wavelength, direction, phase in self.components:
            if (
                len(direction) != 2
                or not all(
                    math.isfinite(v) for v in (amplitude, wavelength, *direction, phase)
                )
                or amplitude < 0
                or wavelength <= 0
                or math.hypot(*direction) == 0
            ):
                raise ValueError(
                    "wave components require finite amplitude >= 0, wavelength > 0 and nonzero direction"
                )
        self._membership = below_surface(
            lambda point, t: point._mx[2] - self.elevation_sym(point, t),
            self.surface_blend,
        )

    def _terms(self, point, t):
        t = t._mx if hasattr(t, "_mx") else t
        for amplitude, wavelength, (dx, dy), phase in self.components:
            norm = math.hypot(dx, dy)
            dx, dy = dx / norm, dy / norm
            k = 2 * math.pi / wavelength
            omega = math.sqrt(self.gravity * k)
            theta = k * (point._mx[0] * dx + point._mx[1] * dy) - omega * t + phase
            yield amplitude, k, omega, dx, dy, theta

    def elevation_sym(self, point, t):
        return self.surface_z + sum(
            a * ca.cos(theta) for a, _, _, _, _, theta in self._terms(point, t)
        )

    def contribute_at_sym(self, point, t):
        base = super().contribute_at_sym(point, t)
        height = point._mx[2] - self.surface_z
        flow = ca.MX(self.velocity)
        pressure_head = -height
        for a, k, omega, dx, dy, theta in self._terms(point, t):
            decay = ca.exp(k * ca.fmin(height, 0.0))
            flow += (
                a
                * omega
                * decay
                * ca.vertcat(dx * ca.cos(theta), dy * ca.cos(theta), ca.sin(theta))
            )
            pressure_head += a * decay * ca.cos(theta)
        return replace(
            base,
            pressure=self.surface_pressure
            + self.density
            * self.gravity
            * smooth_max0(pressure_head, self.surface_blend**2),
            velocity=Vec3(flow, WorldFrame),
        )

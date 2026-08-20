"""Axis-resolved white wrench noise for anisotropic rigid bodies.

Long, slender vehicles can have moments of inertia differing by orders of
magnitude.  One isotropic torque sigma therefore cannot express comparable
angular-acceleration uncertainty on every axis.  ``WrenchProcessNoise`` keeps
the physical wrench boundary while giving each force and torque axis its own
independent white-noise channel; EKF/UKF auto-Q assembly remains unchanged.
"""

from __future__ import annotations

import math
from typing import Iterable

import casadi as ca

from ...ir.frames import PartFrame
from ...ir.types import Vec3
from .._declarations import WhiteNoise
from ..base import Part
from ...ir.wrench import Wrench


def _sigma3(value: Iterable[float], *, name: str) -> tuple[float, float, float]:
    result = tuple(float(item) for item in value)
    if len(result) != 3:
        raise ValueError(f"{name} must contain exactly three axis sigmas")
    if not all(math.isfinite(item) and item >= 0.0 for item in result):
        raise ValueError(f"{name} axis sigmas must be finite and non-negative")
    return result


class WrenchProcessNoise(Part):
    """Independent body-axis force and torque model uncertainty.

    ``force_noise_sigma`` is an xyz tuple in N and
    ``torque_noise_sigma`` is an xyz tuple in N·m.  Values are per-tick white
    standard deviations, matching :class:`ProcessNoise` and Manta's noise
    driver/EKF covariance assembly semantics.
    """

    force_x_noise = WhiteNoise("R1", sigma=0.0)
    force_y_noise = WhiteNoise("R1", sigma=0.0)
    force_z_noise = WhiteNoise("R1", sigma=0.0)
    torque_x_noise = WhiteNoise("R1", sigma=0.0)
    torque_y_noise = WhiteNoise("R1", sigma=0.0)
    torque_z_noise = WhiteNoise("R1", sigma=0.0)

    def __init__(
        self,
        name: str,
        *,
        force_noise_sigma: Iterable[float] = (0.0, 0.0, 0.0),
        torque_noise_sigma: Iterable[float] = (0.0, 0.0, 0.0),
        **overrides,
    ) -> None:
        force = _sigma3(force_noise_sigma, name="force_noise_sigma")
        torque = _sigma3(torque_noise_sigma, name="torque_noise_sigma")
        super().__init__(
            name,
            force_x_noise_sigma=force[0],
            force_y_noise_sigma=force[1],
            force_z_noise_sigma=force[2],
            torque_x_noise_sigma=torque[0],
            torque_y_noise_sigma=torque[1],
            torque_z_noise_sigma=torque[2],
            **overrides,
        )

    def update(self, ctx):
        force = Vec3[PartFrame].from_mx(
            ca.vertcat(
                self.force_x_noise.mx,
                self.force_y_noise.mx,
                self.force_z_noise.mx,
            )
        )
        torque = Vec3[PartFrame].from_mx(
            ca.vertcat(
                self.torque_x_noise.mx,
                self.torque_y_noise.mx,
                self.torque_z_noise.mx,
            )
        )
        return Wrench(force=force, torque=torque)


__all__ = ["WrenchProcessNoise"]

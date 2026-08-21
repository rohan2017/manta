"""Axis-resolved or fully correlated white wrench model uncertainty.

Long, slender vehicles can have moments of inertia differing by orders of
magnitude, and model residuals can couple force and torque axes.  The part
keeps six independent scalar drivers (the framework's ordinary noise
currency), then optionally applies a fixed 6x6 covariance factor before the
wrench enters dynamics. EKF/UKF auto-Q assembly therefore remains unchanged
while preserving cross-axis and force/torque covariance.
"""

from __future__ import annotations

import math
from typing import Iterable

import casadi as ca
import numpy as np

from ...ir.frames import PartFrame
from ...ir.types import Vec3
from ...ir.wrench import Wrench
from .._declarations import Parameter, WhiteNoise
from ..base import Part


def _sigma3(value: Iterable[float], *, name: str) -> tuple[float, float, float]:
    result = tuple(float(item) for item in value)
    if len(result) != 3:
        raise ValueError(f"{name} must contain exactly three axis sigmas")
    if not all(math.isfinite(item) and item >= 0.0 for item in result):
        raise ValueError(f"{name} axis sigmas must be finite and non-negative")
    return result


class WrenchProcessNoise(Part):
    """Body-axis force and torque model uncertainty.

    ``force_noise_sigma`` is an xyz tuple in N and
    ``torque_noise_sigma`` is an xyz tuple in N·m. Alternatively,
    ``wrench_noise_covariance`` is a symmetric positive-semidefinite 6x6
    covariance in ``[Fx,Fy,Fz,Tx,Ty,Tz]`` coordinates. Values are per-tick,
    matching :class:`ProcessNoise` and Manta's noise-driver/EKF semantics.
    """

    force_x_noise = WhiteNoise("R1", sigma=0.0)
    force_y_noise = WhiteNoise("R1", sigma=0.0)
    force_z_noise = WhiteNoise("R1", sigma=0.0)
    torque_x_noise = WhiteNoise("R1", sigma=0.0)
    torque_y_noise = WhiteNoise("R1", sigma=0.0)
    torque_z_noise = WhiteNoise("R1", sigma=0.0)
    wrench_transform: tuple = Parameter((np.eye(6),))

    def __init__(
        self,
        name: str,
        *,
        force_noise_sigma: Iterable[float] = (0.0, 0.0, 0.0),
        torque_noise_sigma: Iterable[float] = (0.0, 0.0, 0.0),
        wrench_noise_covariance=None,
        **overrides,
    ) -> None:
        force = _sigma3(force_noise_sigma, name="force_noise_sigma")
        torque = _sigma3(torque_noise_sigma, name="torque_noise_sigma")
        channel_sigmas = (*force, *torque)
        transform = np.eye(6)
        if wrench_noise_covariance is not None:
            if any(value > 0.0 for value in channel_sigmas):
                raise ValueError(
                    "wrench_noise_covariance is mutually exclusive with "
                    "force_noise_sigma/torque_noise_sigma"
                )
            covariance = np.asarray(wrench_noise_covariance, dtype=float)
            if covariance.shape != (6, 6) or not np.isfinite(covariance).all():
                raise ValueError(
                    "wrench_noise_covariance must be a finite 6x6 matrix"
                )
            scale = max(1.0, float(np.max(np.abs(covariance))))
            if not np.allclose(covariance, covariance.T, rtol=0.0,
                               atol=1e-12 * scale):
                raise ValueError("wrench_noise_covariance must be symmetric")
            covariance = 0.5 * (covariance + covariance.T)
            eigenvalues, eigenvectors = np.linalg.eigh(covariance)
            if float(np.min(eigenvalues)) < -1e-10 * scale:
                raise ValueError(
                    "wrench_noise_covariance must be positive semidefinite"
                )
            eigenvalues = np.maximum(eigenvalues, 0.0)
            transform = eigenvectors @ np.diag(np.sqrt(eigenvalues))
            channel_sigmas = tuple(
                1.0 if np.any(np.abs(transform[:, index]) > 0.0) else 0.0
                for index in range(6)
            )
        super().__init__(
            name,
            force_x_noise_sigma=channel_sigmas[0],
            force_y_noise_sigma=channel_sigmas[1],
            force_z_noise_sigma=channel_sigmas[2],
            torque_x_noise_sigma=channel_sigmas[3],
            torque_y_noise_sigma=channel_sigmas[4],
            torque_z_noise_sigma=channel_sigmas[5],
            wrench_transform=(transform,),
            **overrides,
        )

    def update(self, ctx):
        independent = ca.vertcat(
            self.force_x_noise.mx,
            self.force_y_noise.mx,
            self.force_z_noise.mx,
            self.torque_x_noise.mx,
            self.torque_y_noise.mx,
            self.torque_z_noise.mx,
        )
        shaped = ca.MX(self.wrench_transform[0]) @ independent
        force = Vec3[PartFrame].from_mx(shaped[:3])
        torque = Vec3[PartFrame].from_mx(shaped[3:])
        return Wrench(force=force, torque=torque)


__all__ = ["WrenchProcessNoise"]

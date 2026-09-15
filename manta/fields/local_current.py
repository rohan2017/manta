"""Uniform local fluid velocity retained in estimation and MPC state."""

from __future__ import annotations

import math

import casadi as ca

from ..ir.frames import WorldFrame
from ..ir.types import Vec3
from ..parts._declarations import Parameter, RandomWalkNoise
from .base import Disturbance
from .fluid import FluidState


class _CurrentWalk(RandomWalkNoise):
    def is_active(self, owner, name):
        # A zero diffusion controller still needs the measured environmental
        # state. Ordinary zero-sigma bias declarations deliberately disappear.
        return True

    def initial_state_entries(self, name, owner):
        return {name: tuple(owner.initial_velocity), f"{name}_driver": (0.0, 0.0, 0.0)}


class LocalCurrent(Disturbance):
    """Locally uniform world-frame velocity, with random-walk diffusion.

    This is an additive velocity contribution only. For a *total* local
    current estimate, the caller must zero the baseline fluid velocity.
    Sigma is m/s/sqrt(s); zero retains a constant state for deterministic MPC.
    No spatial map or ocean-wide uniformity is implied by this local model.
    """

    field_value_shape = FluidState
    velocity = _CurrentWalk("R3", frame=WorldFrame)
    initial_velocity = Parameter((0.0, 0.0, 0.0))

    def __init__(
        self,
        name: str = "local_current",
        *,
        sigma: float = 0.0,
        initial_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> None:
        if not math.isfinite(sigma) or sigma < 0:
            raise ValueError("current diffusion must be finite and nonnegative")
        if len(initial_velocity) != 3 or not all(
            math.isfinite(v) for v in initial_velocity
        ):
            raise ValueError("initial current must contain three finite values")
        super().__init__(
            name=name, velocity_sigma=sigma, initial_velocity=tuple(initial_velocity)
        )

    def contribute_at_sym(self, point, t):
        return FluidState(
            density=ca.MX(0),
            pressure=ca.MX(0),
            temperature=ca.MX(0),
            viscosity=ca.MX(0),
            velocity=Vec3[WorldFrame].coerce(self.velocity),
        )

"""Manta control algorithms over the shared world model.

These consume the same declarative model as `Sim(world)` / `EKF(world)`.
Module-shaped algorithms lower through `Target*`; solver-backed MPC owns its
specialized runtime directly.

  * `LQR(world, ...)` — infinite-horizon discrete LQR about an operating
    point; a baked state-feedback control law.
  * `MPC(world, ...)` — sparse tangent-space RTI with a native OSQP solve.
    It is intentionally a direct runtime: solver-backed control does not
    pretend to lower through every generic Module backend.
  * `PID(kp, ki, kd, ...)` — a scalar PID controller; a freestanding
    `recurrence` block (no world needed), lowered by the same backends.
"""

from .lqr import LQR, LQRSolution
from .pid import PID
from .rti import (
    CraftHorizonReference,
    MPC,
    MPCReference,
    MPCResult,
    MPCTimings,
)

__all__ = [
    "CraftHorizonReference",
    "LQR",
    "LQRSolution",
    "MPC",
    "MPCReference",
    "MPCResult",
    "MPCTimings",
    "PID",
]

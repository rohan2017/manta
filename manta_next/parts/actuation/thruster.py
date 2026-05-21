"""Thruster — polynomial-in-throttle force + torque (legacy Thruster1..4).

Tensor-style model: per-order force and torque vectors in CraftFrame.

    F(t) = Σ_k F_k · throttle^k        (k = 0, 1, …, N;  N ≤ 4)
    τ(t) = Σ_k τ_k · throttle^k

The user supplies `forces = [F_0, F_1, …, F_N]` and (optionally) the
matching `torques`. `throttle` is a scalar `Input`. The polynomial order
is capped at 4 — beyond that you're modeling something CasADi's
single-symbol throttle can't represent cleanly anyway.

Common patterns:

  * EDF / rocket motor (linear thrust along axis, no torque):
      Thruster.linear("edf", max_thrust=10.0, axis=(0,0,1))
    → F_0=(0,0,0), F_1=(0,0,10)

  * Quadcopter prop (linear thrust + linear yaw reaction torque):
      Thruster.linear("prop_cw",
                      max_thrust=10.0, axis=(0,0,1),
                      torque_coefficient=+0.1, torque_axis=(0,0,1))
    → F_1=(0,0,10), τ_1=(0,0,+1.0)

  * Higher-order propeller (RPM² thrust):
      Thruster("blade", forces=[(0,0,0), (0,0,0), (0,0,K_T)])
    → F = K_T · throttle²

The force is applied at the mount offset (Part.transform); the framework
lifts that to a body-origin wrench, so off-axis thrusters produce
correct body torques automatically.
"""

from __future__ import annotations

import casadi as ca

from ...ir.frames import CraftFrame
from ...ir.types import Vec3
from ..base import Input, Parameter, Part
from ...math.wrench import Wrench


_MAX_ORDER = 4


def _normalize_polynomial(seq, name: str) -> tuple:
    """Validate + normalize a forces/torques argument to a tuple of
    3-tuples-of-floats. Each entry is one polynomial coefficient
    (F_k or τ_k)."""
    if seq is None:
        return ((0.0, 0.0, 0.0),)
    out: list[tuple[float, float, float]] = []
    for k, vec in enumerate(seq):
        if len(vec) != 3:
            raise ValueError(
                f"Thruster {name}: coefficient {k} must be length-3, "
                f"got {vec!r}")
        out.append(tuple(float(x) for x in vec))
    if len(out) - 1 > _MAX_ORDER:
        raise ValueError(
            f"Thruster {name}: polynomial order {len(out)-1} exceeds "
            f"the cap of {_MAX_ORDER}.")
    if not out:
        out.append((0.0, 0.0, 0.0))
    return tuple(out)


class Thruster(Part):
    """Polynomial-in-throttle thruster (legacy Thruster1..4 equivalent).

    Parameters:
        forces   — tuple/list of force coefficient 3-tuples in CraftFrame,
                   ordered [F_0, F_1, …, F_N]. Order N capped at 4.
        torques  — same shape for torque coefficients (applied at the
                   mount point — the offset lift adds extra torque via
                   Part.transform).

    Input:
        throttle — scalar control input. Units depend on how you scaled
                   the F_k / τ_k coefficients. For Thruster.linear(...)
                   the convention is throttle ∈ [0, 1] giving up to
                   max_thrust Newtons of force.
    """

    forces:   tuple = Parameter(((0.0, 0.0, 0.0),))
    torques:  tuple = Parameter(((0.0, 0.0, 0.0),))
    throttle: float = Input(default=0.0)

    @classmethod
    def linear(cls,
               name: str,
               *,
               max_thrust: float,
               axis: tuple[float, float, float] = (0.0, 0.0, 1.0),
               torque_coefficient: float = 0.0,
               torque_axis: tuple[float, float, float] | None = None,
               **kwargs) -> "Thruster":
        """1st-order linear-throttle thruster.

        Args:
            max_thrust         — N at throttle=1.0.
            axis               — body-frame unit thrust direction.
            torque_coefficient — N·m at throttle=1.0 about `torque_axis`.
                                  Use the same axis as thrust for prop
                                  yaw-reaction; reverse sign for the CCW
                                  rotors of a quadcopter.
            torque_axis        — body-frame torque direction. Defaults to
                                  `axis` (yaw reaction torque from a prop
                                  spinning about its thrust axis).
        """
        F_1 = tuple(max_thrust * a for a in axis)
        t_ax = torque_axis if torque_axis is not None else axis
        τ_1 = tuple(torque_coefficient * a for a in t_ax)
        return cls(name,
                   forces=((0.0, 0.0, 0.0), F_1),
                   torques=((0.0, 0.0, 0.0), τ_1),
                   **kwargs)

    def __init__(self, name: str, **overrides) -> None:
        # Validate + normalize before storing on the instance.
        if "forces" in overrides:
            overrides["forces"] = _normalize_polynomial(overrides["forces"],
                                                         name)
        if "torques" in overrides:
            overrides["torques"] = _normalize_polynomial(overrides["torques"],
                                                          name)
        super().__init__(name, **overrides)

    def update(self, ctx):
        # Pre-compute throttle powers 1, t, t², t³, t⁴ up to whatever
        # max-order shows up in forces / torques.
        max_order = max(len(self.forces), len(self.torques)) - 1
        # We always have throttle^0 = 1 (handled inline below).
        powers = [None] * (max_order + 1)
        if max_order >= 1:
            powers[1] = self.throttle      # symbolic Scalar
        for k in range(2, max_order + 1):
            powers[k] = powers[k - 1] * self.throttle

        F_total = Vec3[CraftFrame].constant((0.0, 0.0, 0.0))
        for k, F_k in enumerate(self.forces):
            F_vec_k = Vec3[CraftFrame].constant(F_k)
            if k == 0:
                F_total = F_total + F_vec_k
            else:
                F_total = F_total + F_vec_k * powers[k]

        τ_total = Vec3[CraftFrame].constant((0.0, 0.0, 0.0))
        for k, τ_k in enumerate(self.torques):
            τ_vec_k = Vec3[CraftFrame].constant(τ_k)
            if k == 0:
                τ_total = τ_total + τ_vec_k
            else:
                τ_total = τ_total + τ_vec_k * powers[k]

        return Wrench(force=F_total, torque=τ_total)

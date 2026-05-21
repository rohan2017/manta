"""Thruster — polynomial-in-throttle force + torque (legacy Thruster1..4).

Tensor-style model: per-order force and torque vectors in CraftFrame.

    F(t) = Σ_k F_k · throttle^k        (k = 0, 1, …, N;  N ≤ 4)
    τ(t) = Σ_k τ_k · throttle^k

Two equivalent construction styles:

  * **Linear (single-vector shortcut):** when you pass `force=(x,y,z)`
    and (optionally) `torque=(x,y,z)`, the thruster is 1st-order:
    F = throttle·(x,y,z) and τ = throttle·(x,y,z).

      Thruster("edf", force=(0, 0, 10))
      → F_1 = (0, 0, 10);  F = throttle · (0, 0, 10)

      Thruster("prop_cw", force=(0, 0, 10), torque=(0, 0, +0.5))
      → F_1 = (0, 0, 10), τ_1 = (0, 0, +0.5)

  * **Polynomial (full tensor form):** pass `forces=[F_0, F_1, …, F_N]`
    and optionally `torques=[…]`. Cap at 4th order.

      Thruster("blade", forces=[(0,0,0), (0,0,0), (0,0,K_T)])
      → F = K_T · throttle²

Mixing `force=...` and `forces=...` in the same constructor is an error.
The force is applied at the mount offset (Part.transform); the framework
lifts that to a body-origin wrench, so off-axis thrusters produce
correct body torques automatically.
"""

from __future__ import annotations

from ...ir.frames import CraftFrame
from ...ir.types import Vec3
from ..base import Input, Parameter, Part
from ...math.wrench import Wrench


_MAX_ORDER = 4


def _as_polynomial(*,
                    force=None,
                    forces=None,
                    name: str,
                    kind: str) -> tuple:
    """Resolve the `force`/`forces` (or `torque`/`torques`) inputs into
    the canonical polynomial form: a tuple of 3-tuples-of-floats giving
    [F_0, F_1, …, F_N].

    Validation:
      * `force` and `forces` are mutually exclusive.
      * `force=(x,y,z)` produces [(0,0,0), (x,y,z)] (a 1st-order linear).
      * `forces=[F_0, F_1, …]` is taken verbatim. Order capped at 4.
      * Missing → [(0,0,0)] (constant-zero, the no-op default).
    """
    if force is not None and forces is not None:
        raise ValueError(
            f"Thruster {name!r}: pass `{kind}=(x,y,z)` for the 1st-order "
            f"linear case OR `{kind}s=[F_0, F_1, …]` for the polynomial "
            f"case, not both.")
    if force is not None:
        if len(force) != 3:
            raise ValueError(
                f"Thruster {name!r}: `{kind}` must be length-3, got "
                f"{force!r}")
        return ((0.0, 0.0, 0.0),
                tuple(float(x) for x in force))
    if forces is not None:
        out: list[tuple[float, float, float]] = []
        for k, vec in enumerate(forces):
            if len(vec) != 3:
                raise ValueError(
                    f"Thruster {name!r}: `{kind}s[{k}]` must be length-3, "
                    f"got {vec!r}")
            out.append(tuple(float(x) for x in vec))
        if len(out) - 1 > _MAX_ORDER:
            raise ValueError(
                f"Thruster {name!r}: polynomial order {len(out)-1} "
                f"exceeds the cap of {_MAX_ORDER}.")
        if not out:
            out.append((0.0, 0.0, 0.0))
        return tuple(out)
    return ((0.0, 0.0, 0.0),)


class Thruster(Part):
    """Polynomial-in-throttle thruster.

    Constructor accepts either of:
      * `force=(x,y,z)` and/or `torque=(x,y,z)` for the 1st-order linear case
        (F = throttle · force_vec, τ = throttle · torque_vec).
      * `forces=[F_0, F_1, …]` and/or `torques=[τ_0, τ_1, …]` for the full
        polynomial case (order capped at 4).

    Input:
        throttle — scalar control input. Units depend on the scaling of
                   the F_k / τ_k coefficients. For the linear `force=v`
                   case, throttle is the standard 0–1 ratio if v is the
                   max-thrust vector; in general it's whatever the user
                   chose.
    """

    forces:   tuple = Parameter(((0.0, 0.0, 0.0),))
    torques:  tuple = Parameter(((0.0, 0.0, 0.0),))
    throttle: float = Input(default=0.0)

    def __init__(self,
                 name: str,
                 *,
                 force: tuple | None = None,
                 forces: list | tuple | None = None,
                 torque: tuple | None = None,
                 torques: list | tuple | None = None,
                 **overrides) -> None:
        overrides["forces"] = _as_polynomial(
            force=force, forces=forces, name=name, kind="force")
        overrides["torques"] = _as_polynomial(
            force=torque, forces=torques, name=name, kind="torque")
        super().__init__(name, **overrides)

    def update(self, ctx):
        # Pre-compute throttle powers 1, t, t², t³, t⁴ up to whatever
        # max-order appears in forces / torques.
        max_order = max(len(self.forces), len(self.torques)) - 1
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

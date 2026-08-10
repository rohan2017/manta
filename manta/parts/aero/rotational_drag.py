"""RotationalDrag — polynomial damping torque in the body's own spin.

    τ(ω) = ρ · Σ_{k=1..N} B_k · ω^(k)        (N ≤ 4)

where ω^(k) is the sign-preserving element-wise k-th power of the
part-frame angular velocity (k=1 linear vane damping, k=2 the
direction-preserving quadratic form). B_k are 3×3 tensors, per unit
fluid density, matching `DragSurface`'s conventions exactly — this IS
DragSurface's missing rotational twin.

Why a separate part: `DragSurface`'s `moment=` tensors act on the
TRANSLATIONAL relative wind — correct for flow-induced moments (a
canted vane, Magnus-like couplings), and exactly wrong for spin
damping: a flow-moment tensor abused as spin damping is a roll torque
proportional to FORWARD SPEED, which silently rolls a cruising hull
over (measured: a "spin-damped" test hull spun up at 5.7 rad/s² from
rest under thrust alone, and an optimizer promptly learned to steer
with the phantom torque). Rotational damping responds to ω, so it
needs ω — this part reads it.

The physically-derived alternative remains: offset `DragSurface`
panels see the ω×r lever-arm wind and damp rotation for real. Use
panels when the geometry is known; use this part for the lumped
coefficient a reduced model carries — the rotational rows of the
Fossen template's 6×6 damping are exactly `RotationalDrag` tensors.
"""

from __future__ import annotations

import casadi as ca
import numpy as np

from ...fields import FluidField
from ...ir.frames import PartFrame, WorldFrame
from ...ir.types import Vec3
from .._declarations import Parameter, PartUpdate
from ..base import Part
from ...ir.wrench import Wrench


def _as_polynomial(torque, tensors, name):
    if (torque is None) == (tensors is None):
        raise ValueError(
            f"RotationalDrag {name!r}: give exactly one of torque= "
            f"(a 3-vector of linear per-axis coefficients) or "
            f"torque_tensors= (a list of 3x3 tensors, one per order).")
    if torque is not None:
        return [np.diag([float(c) for c in torque])]
    out = [np.asarray(t, dtype=float).reshape(3, 3) for t in tensors]
    if not 1 <= len(out) <= 4:
        raise ValueError(
            f"RotationalDrag {name!r}: 1..4 tensor orders, "
            f"got {len(out)}.")
    return out


class RotationalDrag(Part):
    """Damping torque polynomial in the part-frame angular velocity.

    Parameters:
        torque         — (kx, ky, kz): the one-order shortcut,
                         τ_i = ρ·k_i·ω_i (linear damping; k_i < 0).
        torque_tensors — full per-order 3×3 tensor list instead.

    Tensors are per unit fluid density, like DragSurface's.
    """

    requires_fields = [FluidField]

    torque_tensors: list = Parameter(None)

    def __init__(self, name: str, *, torque: tuple | None = None,
                 torque_tensors: list | tuple | None = None,
                 **overrides) -> None:
        overrides["torque_tensors"] = _as_polynomial(
            torque, torque_tensors, name)
        super().__init__(name, **overrides)

    def update(self, ctx) -> PartUpdate:
        p_world = ctx.position[WorldFrame]
        fluid = ctx.field(FluidField).value_at_sym(p_world, ctx.t)
        rho = fluid.density

        R_part_from_world = ctx.orientation.conjugate()
        omega = R_part_from_world.apply(
            ctx.angular_velocity[WorldFrame])._mx

        tensors = self.torque_tensors
        abs_w = ca.fabs(omega) + 1e-30      # sign-preserving powers
        w_pow = omega
        tau = ca.MX.zeros(3, 1)
        for k, B_k in enumerate(tensors):
            if k > 0:
                w_pow = w_pow * abs_w
            if not np.all(B_k == 0.0):
                tau = tau + rho * (ca.MX(B_k) @ w_pow)

        return PartUpdate(wrench=Wrench(
            force=Vec3[PartFrame].from_mx(ca.MX.zeros(3, 1)),
            torque=Vec3[PartFrame].from_mx(tau)))

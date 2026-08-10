"""FossenDamping — the full 6×6 damping polynomial of a reduced model.

    wrench = ρ · Σ_{k=1..N} D_k · ν^(k),      ν = [v_rel; ω]   (N ≤ 4)

where ν stacks the fluid-relative velocity and the angular velocity in
the part's own frame, ν^(k) is the sign-preserving element-wise k-th
power (the house convention, matching `DragSurface` and
`RotationalDrag`), and each D_k is a 6×6 tensor per unit fluid
density. Rows 1–3 are force, rows 4–6 torque, so the quadrants are:

    ┌ F(v)   F(ω) ┐      top-left      DragSurface's force block
    └ τ(v)   τ(ω) ┘      bottom-right  RotationalDrag's block
                         off-diagonal  the cross-couplings neither
                                       lumped part carried: Y_r (sway
                                       force from yaw rate), N_v (yaw
                                       moment from sway), Z_q, M_w, …

This is the Fossen reduced template's D, as one part: what
`manta.fit` fits when shiver's reducer distils a config-derived world
into the controller's model. The diagonal blocks make `DragSurface`-
plus-`RotationalDrag` a special case (pinned by test); the
off-diagonal blocks are the slender-body couplings that single-point
geometric parts cannot express (same physics family as the Munk
moment, which lives in `AddedMass`).

Sign convention — read this: tensors are applied ADDITIVELY, so a
dissipative D has NEGATIVE diagonal entries, exactly like
`DragSurface`/`RotationalDrag` coefficients. This is the transpose of
the textbook Fossen form (ν̇ term −D(ν)ν with D positive-definite):
negate a textbook D before handing it over.

Lumped, craft-level, about the COM: mount it at the COM (the default)
and give it no offset — like `AddedMass`, its geometry is the craft,
not a place on the hull.

Promotable: each order's tensor is a Parameter on the R36 manifold
(column-major flattened 6×6), so `Sim(world,
parameters=["<craft>.<part>.D1", ...])` promotes it to a live input —
`manta.fit` fits D without recompiling the graph per iterate, and a
deployed runtime can be handed a refit D through `set_parameters`.
This part is why the manifold grammar generalized past R3.
"""

from __future__ import annotations

import casadi as ca
import numpy as np

from ...fields import FluidField
from ...ir.frames import PartFrame, WorldFrame
from ...ir.types import Vec3, _IRValue
from .._declarations import Parameter, PartUpdate
from ..base import Part
from ...ir.wrench import Wrench

_MAX_ORDER = 4
_ZERO36 = (0.0,) * 36


def _flat36(T, name, k):
    T = np.asarray(T, dtype=float)
    if T.shape != (6, 6):
        raise ValueError(
            f"FossenDamping {name!r}: tensors[{k}] must be shape "
            f"(6, 6), got {T.shape}")
    # column-major, so ca.reshape(flat, 6, 6) reconstructs it exactly
    return tuple(T.flatten(order="F"))


def _as_orders(damping, tensors, name):
    if (damping is None) == (tensors is None):
        raise ValueError(
            f"FossenDamping {name!r}: give exactly one of damping= "
            f"(a 6-vector of linear per-axis coefficients — the "
            f"diagonal shortcut) or tensors= (a list of 6x6 tensors, "
            f"one per polynomial order).")
    if damping is not None:
        if len(damping) != 6:
            raise ValueError(
                f"FossenDamping {name!r}: damping= must be length-6 "
                f"(vx, vy, vz, wx, wy, wz), got {damping!r}")
        tensors = [np.diag([float(c) for c in damping])]
    if not 1 <= len(tensors) <= _MAX_ORDER:
        raise ValueError(
            f"FossenDamping {name!r}: 1..{_MAX_ORDER} tensor orders, "
            f"got {len(tensors)}.")
    flats = [_flat36(T, name, k) for k, T in enumerate(tensors)]
    return flats + [_ZERO36] * (_MAX_ORDER - len(flats))


class FossenDamping(Part):
    """Full 6×6 damping wrench, polynomial in ν = [v_rel; ω].

    Parameters:
        damping — (kvx, kvy, kvz, kwx, kwy, kwz): the diagonal
                  one-order shortcut (dissipative entries are < 0).
        tensors — full per-order 6×6 tensor list instead.

    Tensors are per unit fluid density and applied additively — see
    the module docstring's sign-convention note before pasting a
    textbook D in here.
    """

    requires_fields = [FluidField]

    #: One promotable R36 per polynomial order, column-major flattened.
    #: Promote with `Sim(world, parameters=["<craft>.<part>.D1"])`.
    D1: tuple = Parameter(_ZERO36, manifold="R36")
    D2: tuple = Parameter(_ZERO36, manifold="R36")
    D3: tuple = Parameter(_ZERO36, manifold="R36")
    D4: tuple = Parameter(_ZERO36, manifold="R36")

    def __init__(self, name: str, *, damping: tuple | None = None,
                 tensors: list | tuple | None = None,
                 **overrides) -> None:
        flats = _as_orders(damping, tensors, name)
        for k, flat in enumerate(flats):
            overrides[f"D{k + 1}"] = flat
        super().__init__(name, **overrides)

    def update(self, ctx) -> PartUpdate:
        # ν in the part's own frame — the AddedMass idiom exactly:
        # fluid-relative point velocity (mount at the COM ⇒ ν_com)
        # plus the body spin, both rotated from world.
        p_world = ctx.position[WorldFrame]
        fluid = ctx.field(FluidField).value_at_sym(p_world, ctx.t)
        rho = fluid.density
        v_rel_world = ctx.velocity[WorldFrame] - fluid.velocity

        R_part_from_world = ctx.orientation.conjugate()
        v = R_part_from_world.apply(v_rel_world)._mx
        w = R_part_from_world.apply(
            ctx.angular_velocity[WorldFrame])._mx
        nu = ca.vertcat(v, w)

        abs_nu = ca.fabs(nu) + 1e-30        # sign-preserving powers
        nu_pow = nu
        wrench6 = ca.MX.zeros(6, 1)
        for k in range(_MAX_ORDER):
            if k > 0:
                nu_pow = nu_pow * abs_nu
            D_k = getattr(self, f"D{k + 1}")
            if isinstance(D_k, _IRValue):
                # promoted for system ID: a live flat-R36 input
                D_mx = ca.reshape(D_k._mx, 6, 6)
            else:
                D_np = np.asarray(D_k, dtype=float).reshape(6, 6,
                                                            order="F")
                if np.all(D_np == 0.0):
                    continue
                D_mx = ca.MX(D_np)
            wrench6 = wrench6 + rho * (D_mx @ nu_pow)

        return PartUpdate(wrench=Wrench(
            force=Vec3[PartFrame].from_mx(wrench6[0:3]),
            torque=Vec3[PartFrame].from_mx(wrench6[3:6])))

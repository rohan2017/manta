"""DragSurface — polynomial drag/lift in a fluid, per the legacy
Surface1..4 tensor form:

    F(v_rel) = ρ · Σ_{k=1..N} A_k · v_rel^(k)        (N ≤ 4)
    τ(v_rel) = ρ · Σ_{k=1..N} B_k · v_rel^(k)

Both polynomials are in the LOCAL FLOW at the mount point — the
translational relative wind, which includes the ω×r lever-arm
contribution when the surface is mounted off-centre, and nothing
rotational when it is not. The `moment=` tensors are therefore
FLOW-INDUCED moments (a canted vane's rolling moment with forward
speed, a panel's Cm-like pitching moment) — NOT damping in the body's
own spin. A centred surface is rotationally blind: a single-point
sample cannot feel the craft rotating about it. For spin damping use
`RotationalDrag` (the lumped coefficient), or mount several offset
DragSurfaces and let the ω×r wind produce it physically.

where v_rel^(k) is the **sign-preserving** element-wise k-th power of
the relative-wind vector in CraftFrame:

    v_rel^(k)_i  =  v_rel_i · |v_rel_i|^(k-1)

(so k=1 is just v, k=2 is v·|v| componentwise — the correct
direction-preserving quadratic drag, etc.). A_k, B_k are 3×3 matrices.
ρ is queried from the registered FluidField at the part's mount point
each tick, so the user-provided A_k / B_k tensors are "per unit fluid
density." A world with no FluidField is a configuration error
(`requires_fields`), rejected when the first transform is built — a
vacuum test world simply omits the DragSurface.

Constructor styles match Thruster's:

  * **One-vector shortcut** for a simple 1st-order isotropic linear
    drag along a particular body axis (rare in practice — most surfaces
    are at least quadratic):

        DragSurface("strip", force=(-0.5, 0, 0))
        → A_1 = diag(-0.5, 0, 0) · v_rel_x (drag scales linearly with v_x).

  * **Per-order tensor list** for the full polynomial:

        DragSurface("hull", force_tensors=[
            np.zeros((3,3)),                         # A_1 = 0 (no linear drag)
            -0.5 * area * Cd * np.eye(3),            # A_2 (quadratic per axis)
        ])

For convenience, `DragSurface.isotropic_quadratic(area, drag_coefficient)`
returns the classic single-Cd hull form (A_2 = -½·A·Cd·I).

Mount offset (Part.mount_offset) lifts force to body-frame torque via the
standard force-at-offset path. Off-axis drag surfaces produce body
torques automatically.

What's still deferred:
  * Lift (airfoils with angle of attack) — use Aerofoil for that.
  * Spanwise variation — DragSurface is a single panel.

Turbulence is an optional per-tick white perturbation of the local fluid
velocity.  It enters before the drag polynomial, so this surface's geometry
and anisotropy turn the disturbed flow into force instead of an unrelated
body wrench being added afterwards.  Its default sigma is zero; engage it
with ``flow_noise_sigma=...`` and a ``NoiseDriver``.
"""

from __future__ import annotations

from typing import ClassVar

import casadi as ca
import numpy as np

from ...fields import FluidField
from ...ir.frames import PartFrame, WorldFrame
from ...ir.types import Vec3
from ...ir.wrench import Wrench
from .._declarations import Parameter, PartUpdate, WhiteNoise
from ..base import Part
from ._flow import signed_powers

_MAX_ORDER = 4


def _as_tensor_polynomial(*,
                          force=None,
                          tensors=None,
                          name: str,
                          kind: str) -> tuple:
    """Resolve a `force`/`force_tensors` (or `moment`/`moment_tensors`)
    input into the canonical form: a tuple of 3×3 numpy arrays giving
    [A_1, A_2, …, A_N]. Note the polynomial starts at k=1 (no F_0 — a
    surface in still fluid produces no spontaneous force).

    Validation:
      * `force` and `tensors` are mutually exclusive.
      * `force=(x,y,z)` produces A_1 = diag(x, y, z).
      * `tensors=[A_1, A_2, …]` is taken verbatim, validated as 3×3.
      * Missing → [0_3x3] (no force).
    """
    if force is not None and tensors is not None:
        raise ValueError(
            f"DragSurface {name!r}: pass `{kind}=(x,y,z)` for the "
            f"diagonal-1st-order shortcut OR `{kind}_tensors=[A_1, A_2, …]` "
            f"for the polynomial form, not both.")
    if force is not None:
        if len(force) != 3:
            raise ValueError(
                f"DragSurface {name!r}: `{kind}` must be length-3, got "
                f"{force!r}")
        A_1 = np.diag([float(x) for x in force])
        return (A_1,)
    if tensors is not None:
        out: list[np.ndarray] = []
        for k, T in enumerate(tensors):
            T_arr = np.asarray(T, dtype=float)
            if T_arr.shape != (3, 3):
                raise ValueError(
                    f"DragSurface {name!r}: `{kind}_tensors[{k}]` must be "
                    f"shape (3, 3), got {T_arr.shape}")
            out.append(T_arr)
        if len(out) > _MAX_ORDER:
            raise ValueError(
                f"DragSurface {name!r}: polynomial order {len(out)} "
                f"exceeds the cap of {_MAX_ORDER}.")
        if not out:
            out.append(np.zeros((3, 3)))
        return tuple(out)
    return (np.zeros((3, 3)),)


class DragSurface(Part):
    """Polynomial drag/lift surface.

    Parameters:
        force_tensors   — list of 3×3 matrices [A_1, A_2, …, A_N], CraftFrame.
                          Default is a single zero matrix (no drag).
        moment_tensors  — same shape: the surface's FLOW-INDUCED moment
                          about the mount point (τ = ρ·Σ B_k·v_rel^(k);
                          not spin damping — see the module docstring).

    Convenience args (mutually exclusive with the *_tensors form):
        force=(x,y,z)   — sets A_1 = diag(x, y, z) (per-axis linear drag).
        moment=(x,y,z)  — sets B_1 = diag(x, y, z) (per-axis, linear in flow).
    """

    # Parameter descriptors expose this resolved edge spelling at runtime.
    flow_noise_sigma: float

    # `force_tensors` / `moment_tensors` carry a zero-3×3 default so the
    # declaration machinery sees a well-typed slot at class scope; the
    # actual value is always set by __init__ from `force`/`force_tensors`
    # (resp. moment variants), so this default never reaches update().
    requires_fields: ClassVar[list[type]] = [FluidField]

    force_tensors:  tuple = Parameter((np.zeros((3, 3)),))
    moment_tensors: tuple = Parameter((np.zeros((3, 3)),))
    flow_noise = WhiteNoise("R3", frame=PartFrame, sigma=0.0)

    @classmethod
    def isotropic_quadratic(cls,
                            name: str,
                            *,
                            area: float,
                            drag_coefficient: float,
                            **kwargs) -> DragSurface:
        """Single-Cd quadratic hull/sphere drag:
            F = -½·ρ·A·Cd · v_rel^(2)  (element-wise square per body axis)
        Identical to the v1 isotropic model, just expressed in tensor form
        so the user can mix it with other polynomial orders."""
        A_1 = np.zeros((3, 3))
        A_2 = -0.5 * area * drag_coefficient * np.eye(3)
        return cls(name, force_tensors=[A_1, A_2], **kwargs)

    @classmethod
    def directional_quadratic(cls,
                              name: str,
                              *,
                              areas: tuple,
                              drag_coefficient: float,
                              **kwargs) -> DragSurface:
        """Anisotropic quadratic drag — a per-body-axis reference area:
            F_i = -½·ρ·areas_i·Cd · v_i·|v_i|   (diagonal A_2)
        Use it for a slender body: a cylindrical fuselage along, say, body
        +z is ``areas=(side, side, frontal)`` with ``frontal ≪ side`` —
        low drag nose-on, high drag broadside (and an off-axis flow gets a
        restoring body torque through the standard force-at-offset lift)."""
        ax, ay, az = (float(a) for a in areas)
        A_1 = np.zeros((3, 3))
        A_2 = -0.5 * drag_coefficient * np.diag([ax, ay, az])
        return cls(name, force_tensors=[A_1, A_2], **kwargs)

    def __init__(self,
                 name: str,
                 *,
                 force: tuple | None = None,
                 force_tensors: list | tuple | None = None,
                 moment: tuple | None = None,
                 moment_tensors: list | tuple | None = None,
                 **overrides) -> None:
        overrides["force_tensors"] = _as_tensor_polynomial(
            force=force, tensors=force_tensors, name=name, kind="force")
        for legacy in ("torque", "torque_tensors"):
            if legacy in overrides:
                raise TypeError(
                    f"DragSurface {name!r}: `{legacy}=` was renamed "
                    f"`{legacy.replace('torque', 'moment')}=` — these "
                    f"tensors are moments induced by the incident FLOW "
                    f"(τ = ρ·Σ B_k·v_rel^k), NOT damping in the body's "
                    f"own spin. For spin damping use RotationalDrag.")
        overrides["moment_tensors"] = _as_tensor_polynomial(
            force=moment, tensors=moment_tensors, name=name, kind="moment")
        super().__init__(name, **overrides)

    def update(self, ctx) -> PartUpdate:
        # --- v_rel at the mount point, in the surface's own frame -----------
        # ctx.position / ctx.velocity are already the surface point's
        # world-frame pose + velocity (kinematic pass composed the transform
        # and the rotational lever arm, incl. any joint motion above it).
        p_world          = ctx.position[WorldFrame]
        v_surface_anchor = ctx.velocity[WorldFrame]

        fluid = ctx.field(FluidField).value_at_sym(p_world, ctx.t)
        rho   = fluid.density
        v_rel_world = v_surface_anchor - fluid.velocity

        # The polynomial tensors are in the surface's own frame; rotate v_rel
        # there. ctx.orientation is the part's own world attitude, so its
        # conjugate maps world → that frame. The framework rotates the
        # returned wrench back to body.
        # Each surface owns an independent channel: the white,
        # small-coherence-length approximation. Correlated or advected
        # turbulence belongs in FluidField instead.
        v_rel_craft = (ctx.orientation.conjugate().apply(v_rel_world)
                       - self.flow_noise)
        v_rel_mx    = v_rel_craft._mx

        # --- Σ_k ρ · A_k · (v_rel componentwise to k) ------------------------
        F_mx = ca.MX.zeros(3, 1)
        τ_mx = ca.MX.zeros(3, 1)

        # Sign-preserving element-wise powers of v_rel (the shared house
        # convention — see `_flow.signed_powers`): v_powers[k] = v·|v|^k.
        max_order = max(len(self.force_tensors), len(self.moment_tensors))
        v_powers = signed_powers(v_rel_mx, max_order)

        for k, A_k in enumerate(self.force_tensors):
            if np.all(A_k == 0.0):
                continue
            A_mx = ca.MX(A_k)
            F_mx = F_mx + rho * (A_mx @ v_powers[k])

        for k, B_k in enumerate(self.moment_tensors):
            if np.all(B_k == 0.0):
                continue
            B_mx = ca.MX(B_k)
            τ_mx = τ_mx + rho * (B_mx @ v_powers[k])

        F_part = Vec3[PartFrame].from_mx(F_mx)
        τ_part = Vec3[PartFrame].from_mx(τ_mx)
        return PartUpdate(wrench=Wrench(force=F_part, torque=τ_part))

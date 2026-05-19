"""GravityField + concrete gravity disturbances.

Single field class for "the gravitational acceleration field" the World
exposes. Per-source variation (uniform background, planet inverse-square
pull, body-of-mass attraction) is expressed by adding different
Disturbance subclasses to the same GravityField instance.

Shipped disturbances:
  * UniformGravity(g_vec)              — constant Vec3[AnchorFrame].
  * PointMassGravity(position, GM)     — Newtonian inverse-square pull.
"""

from __future__ import annotations

import casadi as ca

from ..ir.frames import AnchorFrame
from ..ir.types import Vec3
from .base import Disturbance, Field


_VEC3_ANCHOR = Vec3[AnchorFrame]


class GravityField(Field):
    """Gravitational acceleration `g(point)` in the AnchorFrame.

    `state_at_sym(point)` returns Vec3[AnchorFrame] giving the
    acceleration a free-falling test mass would experience at `point`.
    """

    value_shape = _VEC3_ANCHOR

    def _zero_value(self):
        return _VEC3_ANCHOR.constant((0.0, 0.0, 0.0))


class UniformGravity(Disturbance):
    """Position-independent gravity vector. The standard default for
    sims that don't care about altitude variation.

    Args:
        g_vec — (x, y, z) gravity acceleration in AnchorFrame, m/s².
                Conventionally (0, 0, -9.81) for Earth-near-surface
                with z pointing up.
    """

    field_value_shape = _VEC3_ANCHOR

    def __init__(self, g_vec: tuple[float, float, float]) -> None:
        self.g_vec = tuple(float(x) for x in g_vec)
        if len(self.g_vec) != 3:
            raise ValueError(
                f"UniformGravity: g_vec must be length-3, got {g_vec!r}")

    def contribute_at_sym(self, point):
        # Constant regardless of `point`; the symbolic dependency on
        # point is None — CasADi will fold this into a constant during
        # codegen.
        return _VEC3_ANCHOR.constant(self.g_vec)

    def __repr__(self) -> str:
        return f"<UniformGravity g_vec={self.g_vec}>"


class PointMassGravity(Disturbance):
    """Newtonian gravity from a point mass at a fixed anchor position.

    g(p) = -GM · (p - r_src) / |p - r_src|³

    Args:
        position — (x, y, z) source position in AnchorFrame, meters.
        GM       — gravitational parameter (G·M), m³/s². For Earth GM
                   ≈ 3.986e14; for the Moon ≈ 4.903e12.
        eps      — softening length to avoid singularity at r→0 (m).
                   Defaults to 1.0 — far below any realistic orbital
                   scale, well above numerical noise.
    """

    field_value_shape = _VEC3_ANCHOR

    def __init__(self,
                 position: tuple[float, float, float],
                 GM: float,
                 eps: float = 1.0) -> None:
        self.position = tuple(float(x) for x in position)
        self.GM = float(GM)
        self.eps = float(eps)

    def contribute_at_sym(self, point):
        r_src = _VEC3_ANCHOR.constant(self.position)
        # r = point − r_src   (Vec3 supports operator-)
        r = point - r_src
        # |r|² with softening floor → avoid /0 at the source.
        r_mx  = r._mx                      # underlying MX
        r_sq  = ca.dot(r_mx, r_mx) + self.eps**2
        r_mag = ca.sqrt(r_sq)
        # g = -GM · r / |r|³
        g_mx = (-self.GM / (r_sq * r_mag)) * r_mx
        return _VEC3_ANCHOR.from_mx(g_mx)

    def __repr__(self) -> str:
        return (f"<PointMassGravity position={self.position} GM={self.GM:.3e}>")

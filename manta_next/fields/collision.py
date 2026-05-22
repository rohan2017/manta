"""CollisionField + obstacle disturbances.

`CollisionField.state_at_sym(point)` returns a Vec3[WorldFrame]
giving the **outward penetration vector** at the query point:

  * `(0, 0, 0)` when the point is NOT inside any registered obstacle.
  * Otherwise, a vector along the obstacle's outward normal whose
    magnitude equals the penetration depth.

The natural use is a `Collider` Part: it queries the field at its
mount point and applies a spring (+ optional damper) force scaled by
the penetration vector. The Collider lives in parts/structure/.

Why a vector return (not a scalar penetration)? When multiple obstacles
overlap (e.g. corner of a room: floor + two walls), they each contribute
their own outward direction. Adding them gives a sensible composite
outward direction. The Field base's superposition pattern carries this
automatically — no special-case logic needed.

The smooth-max formula used by HalfSpace keeps the Jacobian regular:

    depth(signed_distance) = (−signed + sqrt(signed² + ε)) / 2

This is exactly `max(0, −signed)` away from signed_distance=0 (with
tiny rounding) and smoothly transitions across the boundary. Critical
for the EKF predict step to stay sane near contact.
"""

from __future__ import annotations

import casadi as ca

from ..ir.frames import WorldFrame
from ..ir.types import Vec3
from .base import Disturbance, Field


_VEC3_ANCHOR = Vec3[WorldFrame]

# Smoothing parameter for the penetration-depth `max(0, x)` regularizer.
# Small enough that depth ≈ |x| for |x| > 1mm; large enough to keep
# Jacobians well-conditioned near the boundary.
_SMOOTH_EPS_SQ = 1.0e-12


class CollisionField(Field):
    """Outward-penetration vector field for contact detection.

    Per the Field-base pattern, every registered Disturbance is an
    obstacle shape that contributes its own penetration vector when the
    query point is inside it. Multi-obstacle overlap composes additively.
    """

    value_shape = _VEC3_ANCHOR

    def _zero_value(self):
        return _VEC3_ANCHOR.constant((0.0, 0.0, 0.0))

    def add_half_space(self,
                       origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
                       normal: tuple[float, float, float] = (0.0, 0.0, 1.0)
                       ) -> "CollisionField":
        """Attach a half-space obstacle (infinite ground plane / wall).
        Returns self."""
        return self.add(HalfSpace(origin=origin, normal=normal))


class HalfSpace(Disturbance):
    """Infinite half-space below a plane.

    The plane is defined by an `origin` point on it and an outward
    `normal`. Points where `(p − origin) · normal < 0` are inside the
    obstacle (below the plane); the outward direction is `+normal`.

    Args:
        origin — point on the plane (world frame), m.
        normal — outward unit normal (world frame). For a ground plane
                 at z=0 with air above and solid below: origin=(0,0,0),
                 normal=(0,0,1).
    """

    field_value_shape = _VEC3_ANCHOR

    def __init__(self,
                 origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
                 normal: tuple[float, float, float] = (0.0, 0.0, 1.0),
                 ) -> None:
        self.origin = tuple(float(x) for x in origin)
        self.normal = tuple(float(x) for x in normal)
        if len(self.origin) != 3 or len(self.normal) != 3:
            raise ValueError(
                f"HalfSpace: origin and normal must be length-3; got "
                f"origin={origin!r}, normal={normal!r}")

    def contribute_at_sym(self, point):
        origin_v = _VEC3_ANCHOR.constant(self.origin)
        normal_v = _VEC3_ANCHOR.constant(self.normal)
        # Signed perpendicular distance from plane (positive = outside).
        diff_mx   = (point - origin_v)._mx
        normal_mx = normal_v._mx
        signed_d  = ca.dot(diff_mx, normal_mx)
        # Penetration depth = max(0, -signed_d), smoothed.
        neg = -signed_d
        depth = 0.5 * (neg + ca.sqrt(neg * neg + _SMOOTH_EPS_SQ))
        # Outward vector = depth · normal.
        out_mx = normal_mx * depth
        return _VEC3_ANCHOR.from_mx(out_mx)

    def __repr__(self) -> str:
        return f"<HalfSpace origin={self.origin} normal={self.normal}>"

"""CollisionField + obstacle disturbances.

`CollisionField.value_at_sym(point)` returns a Vec3[WorldFrame]
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
from ..smoothing import smooth_max0, soft_norm
from .base import Disturbance, Field


_VEC3_ANCHOR = Vec3[WorldFrame]

# Smoothing parameter for the penetration-depth `max(0, x)` regularizer:
# the kink is rounded over ±sqrt(1e-12) = ±1 µm of signed distance, so
# depth ≈ max(0, x) to better than 0.5 µm for |x| > 1 µm — invisible at
# the millimetre scales contact springs act on — while the contact
# Jacobian ramps smoothly instead of stepping at the boundary.
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

    def add_sphere(self,
                   center: tuple[float, float, float],
                   radius: float) -> "CollisionField":
        """Attach a solid-sphere obstacle (e.g. a planet surface).
        Returns self."""
        return self.add(Sphere(center=center, radius=radius))


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
                 *, name: str | None = None) -> None:
        super().__init__(name=name)
        self.origin = tuple(float(x) for x in origin)
        self.normal = tuple(float(x) for x in normal)
        if len(self.origin) != 3 or len(self.normal) != 3:
            raise ValueError(
                f"HalfSpace: origin and normal must be length-3; got "
                f"origin={origin!r}, normal={normal!r}")

    def contribute_at_sym(self, point, t):
        origin_v = _VEC3_ANCHOR.constant(self.origin)
        normal_v = _VEC3_ANCHOR.constant(self.normal)
        # Signed perpendicular distance from plane (positive = outside).
        diff_mx   = (point - origin_v)._mx
        normal_mx = normal_v._mx
        signed_d  = ca.dot(diff_mx, normal_mx)
        # Penetration depth = max(0, -signed_d), smoothed.
        depth = smooth_max0(-signed_d, _SMOOTH_EPS_SQ)
        # Outward vector = depth · normal.
        out_mx = normal_mx * depth
        return _VEC3_ANCHOR.from_mx(out_mx)

    def __repr__(self) -> str:
        return f"<HalfSpace origin={self.origin} normal={self.normal}>"


class Sphere(Disturbance):
    """Solid sphere obstacle — e.g. a whole planet's surface.

    Points with `|p − center| < radius` are inside; the outward
    direction is the local radial, so a craft standing anywhere on the
    sphere gets an up-is-outward contact normal — no per-site ground
    plane needed.

    Args:
        center — sphere centre (world frame), m.
        radius — sphere radius, m.
    """

    field_value_shape = _VEC3_ANCHOR

    def __init__(self,
                 center: tuple[float, float, float],
                 radius: float,
                 *, name: str | None = None) -> None:
        super().__init__(name=name)
        self.center = tuple(float(x) for x in center)
        self.radius = float(radius)
        if len(self.center) != 3:
            raise ValueError(
                f"Sphere: center must be length-3; got {center!r}")
        if self.radius <= 0.0:
            raise ValueError(f"Sphere: radius must be > 0; got {radius!r}")

    def contribute_at_sym(self, point, t):
        center_v = _VEC3_ANCHOR.constant(self.center)
        diff_mx  = (point - center_v)._mx
        r = soft_norm(diff_mx)
        # Signed distance from the surface (positive = outside).
        signed_d = r - self.radius
        depth = smooth_max0(-signed_d, _SMOOTH_EPS_SQ)
        out_mx = (diff_mx / r) * depth
        return _VEC3_ANCHOR.from_mx(out_mx)

    def __repr__(self) -> str:
        return f"<Sphere center={self.center} radius={self.radius}>"

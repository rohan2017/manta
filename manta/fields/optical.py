"""OpticalField + semantic ellipsoids — what a camera sees.

Unlike GravityField / MagField / FluidField, an OpticalField is NOT a
superposition: a camera needs each object SEPARATELY to draw a separate
bounding box, so summing contributions into one "value" is meaningless.
The field therefore just CARRIES its disturbances (semantic ellipsoids)
and exposes them for enumeration — `OpticalField.ellipsoids` — and the
camera part projects each one to an image-frame box.

A semantic ellipsoid is a coarse 3-D extent + a class label: the
quadric ``(x − c)ᵀ M (x − c) ≤ 1`` with ``M = R · diag(1/a², 1/b²,
1/c²) · Rᵀ`` (semi-axes a,b,c along the body axes R). It is the loosest
useful object model — exactly enough to project to a 2-D bounding box,
which is the standard detector output a perception stack consumes.

Two flavours:
  * `SemanticEllipsoid`  — fixed in the world (scenery / a landmark).
  * `BodySemanticEllipsoid` — rides a craft (emitted by an
    `OpticalSource` part), so a camera on another vehicle sees it move.
    Its world pose is read from the active trace, like CraftWindBubble.
"""

from __future__ import annotations

import casadi as ca

from ..ir._rotation import quat_to_rotmat
from ..ir.frames import WorldFrame
from ..ir.types import Vec3
from .base import Disturbance, Field, anchored_pose


_VEC3_W = Vec3[WorldFrame]


class SemanticQuadric:
    """Type tag for OpticalField's value shape (the `Field.add`
    compatibility check keys on it). Optical disturbances are quadrics,
    not a summable vector/scalar value."""


def _diag_shape_mx(semi_axes) -> ca.MX:
    """Λ = diag(1/a², 1/b², 1/c²) — the ellipsoid's body-frame quadric."""
    a, b, c = (float(s) for s in semi_axes)
    if min(a, b, c) <= 0.0:
        raise ValueError(
            f"semantic ellipsoid: semi-axes must be > 0, got {semi_axes!r}")
    return ca.diag(ca.DM([1.0 / a**2, 1.0 / b**2, 1.0 / c**2]))


class _EllipsoidBase(Disturbance):
    """Shared surface for the two ellipsoid flavours: a world-frame
    center, a world-frame shape matrix M (3×3 SPD), a class `label`, and
    a `source_craft` (None for scenery) so a camera can skip its own
    body. Subclasses provide `center_world` / `shape_world_mx`."""

    field_value_shape = SemanticQuadric
    source_craft = None
    label: int = 0

    def contribute_at_sym(self, point, t):
        # Optical disturbances are never summed (the camera enumerates
        # them); the base Field.value_at_sym is overridden to forbid it.
        raise NotImplementedError(
            "OpticalField disturbances are not summed — a camera enumerates "
            "OpticalField.ellipsoids and projects each quadric.")

    # ---- per-flavour ----------------------------------------------------
    def center_world(self) -> Vec3:
        raise NotImplementedError

    def shape_world_mx(self) -> ca.MX:
        raise NotImplementedError


class SemanticEllipsoid(_EllipsoidBase):
    """A fixed semantic ellipsoid in the world (a landmark / scenery).

    Args:
        center     — (x, y, z) world position of the ellipsoid center, m.
        semi_axes  — (a, b, c) half-extents along the ellipsoid's own
                     axes, m.
        orientation— wxyz quaternion (world ← ellipsoid). Default upright.
        label      — integer class id the camera reports with the box.
    """

    def __init__(self, center, semi_axes, *,
                 orientation=(1.0, 0.0, 0.0, 0.0),
                 label: int = 0, name: str | None = None) -> None:
        super().__init__(name=name)
        self.center = tuple(float(x) for x in center)
        self.semi_axes = tuple(float(x) for x in semi_axes)
        self._Lambda = _diag_shape_mx(self.semi_axes)
        q = ca.DM([float(x) for x in orientation])
        self._R = quat_to_rotmat(q / ca.norm_2(q))       # world ← body
        self.label = int(label)

    def center_world(self) -> Vec3:
        return _VEC3_W.constant(self.center)

    def shape_world_mx(self) -> ca.MX:
        return self._R @ self._Lambda @ self._R.T


class BodySemanticEllipsoid(_EllipsoidBase):
    """A semantic ellipsoid that rides a craft (emitted by an
    `OpticalSource`). Center and orientation track the carrying craft's
    pose, read from the active trace; the semi-axes are the body extent."""

    def __init__(self, craft, offset_body, semi_axes, *,
                 label: int = 0, name: str | None = None) -> None:
        super().__init__(name=name)
        self.source_craft = craft
        self.offset_body = tuple(float(x) for x in offset_body)
        self.semi_axes = tuple(float(x) for x in semi_axes)
        self._Lambda = _diag_shape_mx(self.semi_axes)
        self.label = int(label)

    def center_world(self) -> Vec3:
        center, _R, _q = anchored_pose(self.source_craft, self.offset_body)
        return center

    def shape_world_mx(self) -> ca.MX:
        _center, R, _q = anchored_pose(self.source_craft, self.offset_body)
        Rm = R._mx                                   # Mat3[World, Craft] → MX
        return Rm @ self._Lambda @ Rm.T


class OpticalField(Field):
    """A scene of semantic ellipsoids. Carries (does not sum) its
    disturbances; the camera part enumerates `ellipsoids` and projects
    each quadric to an image-frame bounding box."""

    value_shape = SemanticQuadric

    @property
    def ellipsoids(self) -> tuple[_EllipsoidBase, ...]:
        """Every registered ellipsoid (scenery + body-anchored)."""
        return tuple(self._disturbances)

    def add_ellipsoid(self, center, semi_axes, *,
                      orientation=(1.0, 0.0, 0.0, 0.0),
                      label: int = 0) -> "OpticalField":
        """Attach a fixed scenery ellipsoid. Returns self for chaining."""
        return self.add(SemanticEllipsoid(
            center, semi_axes, orientation=orientation, label=label))

    def _zero_value(self):
        raise NotImplementedError(
            "OpticalField has no summed value — enumerate `ellipsoids`.")

    def value_at_sym(self, point, t):
        raise NotImplementedError(
            "OpticalField is not a superposition field; a camera enumerates "
            "OpticalField.ellipsoids and projects each quadric to a box.")

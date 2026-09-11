"""Numeric sphere contacts with axisymmetric solids.

The solid is the revolution of a closed convex polygon in (radius, z).
This represents cylinders, annular collars and tapered sockets without mesh
faceting. Contact geometry is evaluated outside the smooth vehicle kernel;
apply the returned equal/opposite loads through mounted ExternalWrench parts.
No capture, command, vehicle, or transport policy lives here.

Use this boundary for numeric inter-body contact and equal/opposite external
wrenches. ``manta.fields.CollisionField`` instead represents symbolic spatial
obstacles inside a differentiable plant. Their evaluation and coupling contracts
are distinct; applications choose according to the model they need.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np


def _vector(value: Any, size: int, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must have {size} finite components")
    return result


@dataclass(frozen=True)
class Kinematics:
    """World pose and twist of a contact frame; quaternion is scalar-first."""

    position: tuple[float, float, float]
    orientation: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)
    angular_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        for name, size in (
            ("position", 3),
            ("orientation", 4),
            ("velocity", 3),
            ("angular_velocity", 3),
        ):
            value = _vector(getattr(self, name), size, name)
            if name == "orientation":
                norm = np.linalg.norm(value)
                if norm < 1e-12:
                    raise ValueError("orientation must be nonzero")
                value = value / norm
            object.__setattr__(self, name, tuple(float(x) for x in value))

    @property
    def rotation(self) -> np.ndarray:
        w, x, y, z = self.orientation
        return np.array(
            (
                (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
                (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
                (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
            )
        )

    def point(self, local: Any) -> np.ndarray:
        return np.asarray(self.position) + self.rotation @ np.asarray(local)

    def point_velocity(self, world_point: Any) -> np.ndarray:
        return np.asarray(self.velocity) + np.cross(
            self.angular_velocity, np.asarray(world_point) - self.position
        )


@dataclass(frozen=True)
class RevolvedSolid:
    """Convex meridian polygon revolved about local Z (may have a bore).

    Positive signed distance is outside material. Polygon vertices may have
    either winding; the solid may be nonconvex in 3D, e.g. a hollow socket.
    """

    profile: tuple[tuple[float, float], ...]

    def __post_init__(self) -> None:
        p = np.asarray(self.profile, dtype=float)
        if (
            p.ndim != 2
            or p.shape[1] != 2
            or len(p) < 3
            or not np.isfinite(p).all()
            or np.any(p[:, 0] < 0)
        ):
            raise ValueError("profile requires finite nonnegative radii")
        edge = np.roll(p, -1, axis=0) - p
        if np.any(np.linalg.norm(edge, axis=1) < 1e-12):
            raise ValueError("profile has a zero-length edge")
        cross = edge[:, 0] * np.roll(edge[:, 1], -1) - edge[:, 1] * np.roll(
            edge[:, 0], -1
        )
        if not (np.all(cross >= -1e-12) or np.all(cross <= 1e-12)) or np.all(
            abs(cross) < 1e-12
        ):
            raise ValueError("meridian profile must be convex with nonzero area")
        object.__setattr__(
            self, "profile", tuple(tuple(float(x) for x in row) for row in p)
        )

    @classmethod
    def cylinder(cls, radius: float, length: float) -> RevolvedSolid:
        if not all(math.isfinite(x) and x > 0 for x in (radius, length)):
            raise ValueError("cylinder dimensions must be positive and finite")
        return cls(
            (
                (0.0, -length / 2),
                (radius, -length / 2),
                (radius, length / 2),
                (0.0, length / 2),
            )
        )

    def distance_normal(self, point: Any) -> tuple[float, np.ndarray]:
        """Exact signed distance and an outward unit normal, in local axes."""
        xyz = _vector(point, 3, "point")
        r = math.hypot(xyz[0], xyz[1])
        q = np.array((r, xyz[2]))
        p = np.asarray(self.profile)
        e = np.roll(p, -1, axis=0) - p
        d = q - p
        crosses = e[:, 0] * d[:, 1] - e[:, 1] * d[:, 0]
        inside = bool(np.all(crosses >= -1e-14) or np.all(crosses <= 1e-14))
        t = np.clip(np.sum(d * e, axis=1) / np.sum(e * e, axis=1), 0.0, 1.0)
        closest = p + t[:, None] * e
        delta = q - closest
        lengths = np.linalg.norm(delta, axis=1)
        # The axis is an interior seam, not a surface of the revolved solid.
        lengths[(p[:, 0] == 0) & (np.roll(p[:, 0], -1) == 0)] = np.inf
        i = int(np.argmin(lengths))
        distance = float(lengths[i])
        if distance > 1e-12:
            normal2 = delta[i] / distance * (-1 if inside else 1)
        else:
            area = np.sum(
                p[:, 0] * np.roll(p[:, 1], -1) - p[:, 1] * np.roll(p[:, 0], -1)
            )
            normal2 = np.array((e[i, 1], -e[i, 0])) * (1 if area > 0 else -1)
            normal2 /= np.linalg.norm(normal2)
        radial = xyz[:2] / r if r > 1e-12 else np.array((1.0, 0.0))
        normal = np.array((normal2[0] * radial[0], normal2[0] * radial[1], normal2[1]))
        return (-distance if inside else distance), normal


@dataclass(frozen=True)
class ContactMaterial:
    stiffness: float = 30_000.0
    damping: float = 250.0
    friction: float = 0.15
    slip_damping: float = 100.0

    def __post_init__(self) -> None:
        if (
            any(
                not math.isfinite(x) or x < 0
                for x in (
                    self.stiffness,
                    self.damping,
                    self.friction,
                    self.slip_damping,
                )
            )
            or self.stiffness == 0
        ):
            raise ValueError(
                "contact coefficients must be finite and nonnegative; stiffness positive"
            )


@dataclass(frozen=True)
class Contact:
    point_world: tuple[float, float, float]
    force_on_sphere_world: tuple[float, float, float]
    penetration: float

    def wrenches(
        self, sphere_origin: Any, solid_origin: Any
    ) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
        """Force and moment about each frame origin; common application point."""
        f = np.asarray(self.force_on_sphere_world)
        p = np.asarray(self.point_world)
        return (
            (f, np.cross(p - sphere_origin, f)),
            (-f, np.cross(p - solid_origin, -f)),
        )


def sphere_contact(
    sphere: Kinematics,
    radius: float,
    solid: RevolvedSolid,
    frame: Kinematics,
    material: ContactMaterial,
) -> Contact | None:
    if not math.isfinite(radius) or radius <= 0:
        raise ValueError("sphere radius must be positive and finite")
    center = np.asarray(sphere.position)
    rotation = frame.rotation
    local_center = rotation.T @ (center - frame.position)
    distance, normal = solid.distance_normal(local_center)
    penetration = radius - distance
    if penetration <= 0:
        return None
    # The nearest feature on the axis may be a ring. Average its normals so
    # a centered sphere receives no arbitrary lateral force. This lumped
    # resultant omits the distributed ring's spin friction.
    if np.linalg.norm(local_center[:2]) < 1e-12:
        normal[:2] = 0.0
    normal = rotation @ normal
    normal_scale = float(np.linalg.norm(normal))
    point = center - normal * (radius - penetration / 2)
    if normal_scale < 1e-12:
        return Contact(tuple(point), (0.0, 0.0, 0.0), penetration)
    normal = normal / normal_scale
    relative = sphere.point_velocity(point) - frame.point_velocity(point)
    speed = float(relative @ normal)
    force_n = normal_scale * max(
        0.0, material.stiffness * penetration - material.damping * speed * normal_scale
    )
    tangent = relative - speed * normal
    slip = float(np.linalg.norm(tangent))
    force = force_n * normal
    if slip > 1e-12:
        force -= (
            min(material.slip_damping * slip, material.friction * force_n)
            * tangent
            / slip
        )
    return Contact(tuple(point), tuple(force), penetration)


def cylinder_contacts(
    a: Kinematics,
    radius_a: float,
    length_a: float,
    b: Kinematics,
    radius_b: float,
    length_b: float,
    material: ContactMaterial,
    *,
    spacing: float = 0.025,
) -> tuple[Contact, ...]:
    """Cylinder contact quadrature: inscribed axial spheres against a solid.

    Both cylinders use local Z axes. The target is exact; the source uses
    overlapping inscribed spheres. Away from end caps, radial undercoverage is
    at most r-sqrt(r*r-spacing*spacing/4). End-rim contacts are rounded inward
    by the probe radius; callers needing sharp rim collisions need another
    discretization. This approximation never adds material outside a cylinder.
    The material stiffness is shared across the probes, avoiding stiffness
    growth when the quadrature resolution increases.
    """
    if any(
        not math.isfinite(v) or v <= 0
        for v in (radius_a, length_a, radius_b, length_b, spacing)
    ):
        raise ValueError("cylinder contact dimensions must be positive and finite")
    if spacing > radius_a:
        raise ValueError("cylinder probe spacing must not exceed its radius")
    bound_a = math.hypot(length_a / 2, radius_a)
    bound_b = math.hypot(length_b / 2, radius_b)
    if np.linalg.norm(np.array(a.position) - b.position) >= bound_a + bound_b:
        return ()
    radius = min(radius_a, length_a / 2)
    half = length_a / 2 - radius
    count = max(1, math.ceil(2 * half / spacing) + 1)
    if count > 4096:
        raise ValueError("cylinder contact quadrature exceeds 4096 probes")
    target = RevolvedSolid.cylinder(radius_b, length_b)
    shared = ContactMaterial(
        material.stiffness / count,
        material.damping / count,
        material.friction,
        material.slip_damping / count,
    )
    hits = []
    for z in np.linspace(-half, half, count):
        point = a.point((0.0, 0.0, z))
        probe = Kinematics(
            tuple(point),
            a.orientation,
            tuple(a.point_velocity(point)),
            a.angular_velocity,
        )
        hit = sphere_contact(probe, radius, target, b, shared)
        if hit is not None:
            hits.append(hit)
    return tuple(hits)

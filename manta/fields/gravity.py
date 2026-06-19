"""GravityField + concrete gravity disturbances.

Single field class for "the gravitational acceleration field" the World
exposes. Per-source variation (uniform background, planet inverse-square
pull, body-of-mass attraction) is expressed by adding different
Disturbance subclasses to the same GravityField instance.

Shipped disturbances:
  * UniformGravity(g_vec)              — constant Vec3[WorldFrame].
  * PointMassGravity(position, GM)     — Newtonian inverse-square pull.
"""

from __future__ import annotations

import casadi as ca

from ..ir.frames import WorldFrame
from ..ir.types import Vec3
from .base import Disturbance, SuperposedField


_VEC3_W = Vec3[WorldFrame]


class GravityField(SuperposedField):
    """Gravitational acceleration `g(point)` in the WorldFrame.

    `value_at_sym(point)` returns Vec3[WorldFrame] giving the
    acceleration a free-falling test mass would experience at `point`.

    The `add_uniform` builder returns self for chaining:
    `GravityField().add_uniform((0,0,-9.81))`. As a
    convenience, the constructor accepts `g=(gx,gy,gz)` for the common
    single-uniform case — equivalent to one `add_uniform` call.
    """

    value_shape = _VEC3_W

    def __init__(self, g: tuple[float, float, float] | None = None) -> None:
        super().__init__()
        if g is not None:
            self.add_uniform(g)

    def _zero_value(self):
        return _VEC3_W.constant((0.0, 0.0, 0.0))

    def add_uniform(self, g_vec: tuple[float, float, float]) -> "GravityField":
        """Attach a position-independent gravity vector. Returns self.

        Other disturbances (point-mass, J2, …) attach via the generic
        `field.add(PointMassGravity(...))`."""
        return self.add(UniformGravity(g_vec))


class UniformGravity(Disturbance):
    """Position-independent gravity vector. The standard default for
    sims that don't care about altitude variation.

    Args:
        g_vec — (x, y, z) gravity acceleration in WorldFrame, m/s².
                Conventionally (0, 0, -9.81) for Earth-near-surface
                with z pointing up.
    """

    field_value_shape = _VEC3_W

    def __init__(self, g_vec: tuple[float, float, float], *, name: str | None = None) -> None:
        super().__init__(name=name)
        self.g_vec = tuple(float(x) for x in g_vec)
        if len(self.g_vec) != 3:
            raise ValueError(
                f"UniformGravity: g_vec must be length-3, got {g_vec!r}")

    def contribute_at_sym(self, point, t):
        # Constant regardless of `point`; the symbolic dependency on
        # point is None — CasADi will fold this into a constant during
        # codegen.
        return _VEC3_W.constant(self.g_vec)

    def __repr__(self) -> str:
        return f"<UniformGravity g_vec={self.g_vec}>"


def _point_mass_accel(point, center, GM: float, eps: float):
    """g(p) = −GM·(p−c)/|p−c|³, eps-softened, for an already-resolved
    world-frame source center. Shared by the fixed and body-anchored
    point-mass disturbances."""
    r_mx = (point - center)._mx
    r_sq = ca.dot(r_mx, r_mx) + eps**2
    r_mag = ca.sqrt(r_sq)
    return _VEC3_W.from_mx((-GM / (r_sq * r_mag)) * r_mx)


class PointMassGravity(Disturbance):
    """Newtonian gravity from a point mass at a fixed anchor position.

    g(p) = -GM · (p - r_src) / |p - r_src|³

    Args:
        position — (x, y, z) source position in WorldFrame, meters.
        GM       — gravitational parameter (G·M), m³/s². For Earth GM
                   ≈ 3.986e14; for the Moon ≈ 4.903e12.
        eps      — softening length to avoid singularity at r→0 (m).
                   Defaults to 1.0 — far below any realistic orbital
                   scale, well above numerical noise.
    """

    field_value_shape = _VEC3_W

    def __init__(self,
                 position: tuple[float, float, float],
                 GM: float,
                 eps: float = 1.0, *, name: str | None = None) -> None:
        super().__init__(name=name)
        self.position = tuple(float(x) for x in position)
        self.GM = float(GM)
        self.eps = float(eps)
        if self.GM <= 0.0:
            raise ValueError(
                f"PointMassGravity: GM must be > 0, got {GM!r}")
        # eps == 0 is allowed: exact Newtonian, singular only at the source.
        if self.eps < 0.0:
            raise ValueError(
                f"PointMassGravity: eps must be >= 0, got {eps!r}")

    def contribute_at_sym(self, point, t):
        return _point_mass_accel(
            point, _VEC3_W.constant(self.position), self.GM, self.eps)

    def __repr__(self) -> str:
        return (f"<PointMassGravity position={self.position} GM={self.GM:.3e}>")


class BodyPointMassGravity(Disturbance):
    """Inverse-square gravity from a point mass that RIDES a craft — the
    field a `GravitySource` part emits to simulate a massive body (planet,
    asteroid, station) moving through the sim. Same law as
    `PointMassGravity`, but the source position tracks the carrying
    craft's pose (read from the active trace), not a fixed world point.

    Args:
        craft       — the craft carrying the source.
        offset_body — the source position in the craft's body frame, m.
        GM          — gravitational parameter (G·M), m³/s².
        eps         — softening length, m. Default 1.0.
    """

    field_value_shape = _VEC3_W

    def __init__(self, craft, offset_body, GM: float,
                 eps: float = 1.0, *, name: str | None = None) -> None:
        super().__init__(name=name)
        self.craft = craft
        self.offset_body = tuple(float(x) for x in offset_body)
        self.GM = float(GM)
        self.eps = float(eps)
        if self.GM <= 0.0:
            raise ValueError(f"BodyPointMassGravity: GM must be > 0, got {GM!r}")
        if self.eps < 0.0:
            raise ValueError(f"BodyPointMassGravity: eps must be >= 0, got {eps!r}")

    def contribute_at_sym(self, point, t):
        from .base import anchored_pose
        center, _q = anchored_pose(self.craft, self.offset_body)
        return _point_mass_accel(point, center, self.GM, self.eps)

    def __repr__(self) -> str:
        return (f"<BodyPointMassGravity craft={self.craft.name!r} "
                f"GM={self.GM:.3e}>")


class J2Gravity(Disturbance):
    """J2 oblateness perturbation around a point mass.

    For an oblate spheroid with equatorial bulge, the gravitational
    potential acquires a J2 term beyond the point-mass; the
    corresponding acceleration is

        g_J2(r) = -(3/2) · J2 · GM · R_eq² / r⁵ · [
            (1 - 5(ẑ·r̂)²) · r + 2(ẑ·r̂)·|r|·ẑ
        ]

    where ẑ is the polar (spin) axis. This is the perturbation only —
    add it alongside a `PointMassGravity` to model the full field.

    Args:
        position    — (x, y, z) planet center in WorldFrame, m.
        GM          — gravitational parameter (G·M), m³/s².
        J2          — dimensionless J2 coefficient. Earth: 1.0826e-3.
        eq_radius   — equatorial radius, m. Earth: 6.378e6.
        polar_axis  — unit polar/spin axis in WorldFrame. Default
                      (0, 0, 1).
        eps         — softening length, m.
    """

    field_value_shape = _VEC3_W

    def __init__(self,
                 position: tuple[float, float, float],
                 GM: float,
                 J2: float,
                 eq_radius: float,
                 polar_axis: tuple[float, float, float] = (0.0, 0.0, 1.0),
                 eps: float = 1.0, *, name: str | None = None) -> None:
        super().__init__(name=name)
        self.position = tuple(float(x) for x in position)
        self.GM = float(GM)
        self.J2 = float(J2)
        self.eq_radius = float(eq_radius)
        axis = ca.DM([float(polar_axis[0]),
                      float(polar_axis[1]),
                      float(polar_axis[2])])
        n = float(ca.norm_2(axis))
        if n == 0.0:
            raise ValueError("J2Gravity: polar_axis must be nonzero.")
        self._axis = axis / n
        self.eps = float(eps)
        if self.GM <= 0.0:
            raise ValueError(f"J2Gravity: GM must be > 0, got {GM!r}")
        if self.eq_radius <= 0.0:
            raise ValueError(
                f"J2Gravity: eq_radius must be > 0, got {eq_radius!r}")
        if self.eps < 0.0:
            raise ValueError(f"J2Gravity: eps must be >= 0, got {eps!r}")

    def contribute_at_sym(self, point, t):
        r_src = _VEC3_W.constant(self.position)
        r     = point - r_src
        r_mx  = r._mx
        r_sq  = ca.dot(r_mx, r_mx) + self.eps**2
        r_mag = ca.sqrt(r_sq)
        z_hat = self._axis
        z_dot_r = ca.dot(z_hat, r_mx)
        coef  = -1.5 * self.J2 * self.GM * (self.eq_radius ** 2) / (r_sq ** 2 * r_mag)
        # cos²(latitude-complement) = (ẑ · r̂)² = (ẑ · r)² / |r|²
        cos2 = (z_dot_r * z_dot_r) / r_sq
        bracket = (1.0 - 5.0 * cos2) * r_mx + 2.0 * z_dot_r * z_hat
        g_mx = coef * bracket
        return _VEC3_W.from_mx(g_mx)

    def __repr__(self) -> str:
        return (f"<J2Gravity position={self.position} GM={self.GM:.3e} "
                f"J2={self.J2:.4e} R_eq={self.eq_radius:.3e}>")

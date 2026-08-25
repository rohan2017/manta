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
import numpy as np

from ..ir.frames import WorldFrame
from ..ir.types import Vec3
from ..smoothing import smooth_max0, soft_norm
from .base import Disturbance, SuperposedField

_VEC3_W = Vec3[WorldFrame]

# Smoothing parameter for the penetration-depth `max(0, x)` regularizer:
# the kink is rounded over ±sqrt(1e-12) = ±1 µm of signed distance, so
# depth ≈ max(0, x) to better than 0.5 µm for |x| > 1 µm — invisible at
# the millimetre scales contact springs act on — while the contact
# Jacobian ramps smoothly instead of stepping at the boundary.
_SMOOTH_EPS_SQ = 1.0e-12


class CollisionField(SuperposedField):
    """Outward-penetration vector field for contact detection.

    Per the Field-base pattern, every registered Disturbance is an
    obstacle shape that contributes its own penetration vector when the
    query point is inside it. Multi-obstacle overlap composes additively.
    """

    value_shape = _VEC3_W

    def _zero_value(self):
        return _VEC3_W.constant((0.0, 0.0, 0.0))

    def add_half_space(self,
                       origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
                       normal: tuple[float, float, float] = (0.0, 0.0, 1.0)
                       ) -> CollisionField:
        """Attach a half-space obstacle (infinite ground plane / wall).
        Returns self."""
        return self.add(HalfSpace(origin=origin, normal=normal))

    def add_sphere(self,
                   center: tuple[float, float, float],
                   radius: float) -> CollisionField:
        """Attach a solid-sphere obstacle (e.g. a planet surface).
        Returns self."""
        return self.add(Sphere(center=center, radius=radius))

    def add_ellipsoid(self,
                      center: tuple[float, float, float],
                      equatorial_radius: float,
                      flattening: float,
                      polar_axis: tuple[float, float, float] = (0.0, 0.0, 1.0),
                      *, height: float = 0.0) -> Ellipsoid:
        """Attach a solid oblate-spheroid obstacle (a WGS-84 Earth).
        Returns the `Ellipsoid` (not self) so the caller can reuse its
        geodetic height for other surface-relative queries."""
        el = Ellipsoid(center=center, equatorial_radius=equatorial_radius,
                       flattening=flattening, polar_axis=polar_axis,
                       height=height)
        self.add(el)
        return el

    def add_heightfield(self, heights, *,
                        x0: float = 0.0, y0: float = 0.0,
                        dx: float = 1.0, dy: float = 1.0
                        ) -> Heightfield:
        """Attach gridded solid terrain `z = h(x, y)` (bathymetry, a
        ground DEM). Returns the `Heightfield` (not self) so the caller
        can keep it for `height_at` queries."""
        hf = Heightfield(heights, x0=x0, y0=y0, dx=dx, dy=dy)
        self.add(hf)
        return hf


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

    field_value_shape = _VEC3_W

    def __init__(self,
                 origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
                 normal: tuple[float, float, float] = (0.0, 0.0, 1.0),
                 *, name: str | None = None) -> None:
        from ..parts._declarations import unit_axis
        super().__init__(name=name)
        self.origin = tuple(float(x) for x in origin)
        if len(self.origin) != 3:
            raise ValueError(
                f"HalfSpace: origin must be length-3; got {origin!r}")
        # The penetration math assumes a unit normal (a non-unit one
        # would scale the response by |normal|²) — normalize at
        # construction, same convention as part axes.
        self.normal = unit_axis(normal, who="HalfSpace", what="normal")

    def contribute_at_sym(self, point, t):
        origin_v = _VEC3_W.constant(self.origin)
        normal_v = _VEC3_W.constant(self.normal)
        # Signed perpendicular distance from plane (positive = outside).
        diff_mx   = (point - origin_v)._mx
        normal_mx = normal_v._mx
        signed_d  = ca.dot(diff_mx, normal_mx)
        # Penetration depth = max(0, -signed_d), smoothed.
        depth = smooth_max0(-signed_d, _SMOOTH_EPS_SQ)
        # Outward vector = depth · normal.
        out_mx = normal_mx * depth
        return _VEC3_W.from_mx(out_mx)

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

    field_value_shape = _VEC3_W

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
        center_v = _VEC3_W.constant(self.center)
        diff_mx  = (point - center_v)._mx
        r = soft_norm(diff_mx)
        # Signed distance from the surface (positive = outside).
        signed_d = r - self.radius
        depth = smooth_max0(-signed_d, _SMOOTH_EPS_SQ)
        out_mx = (diff_mx / r) * depth
        return _VEC3_W.from_mx(out_mx)

    def __repr__(self) -> str:
        return f"<Sphere center={self.center} radius={self.radius}>"


class Ellipsoid(Disturbance):
    """Solid oblate spheroid — a planet with an equatorial bulge, e.g.
    the WGS-84 Earth (`flattening` = 1/298.257…).

    Points at negative geodetic height are inside; the outward direction
    is the geodetic normal (the direction gravity + centrifugal force
    hangs a plumb line along on a planet in hydrostatic balance), so a
    craft standing anywhere on the surface gets the same "up" the
    planet's `local_tangent_basis` and a GNSS receiver use. The spheroid
    is symmetric about `polar_axis`, so it is the same shape whether the
    planet spins beneath it or not — a world-fixed obstacle serves a
    rotating planet.

    `signed_height_sym` is the reusable core: the geodetic height of a
    point (Bowring's method, CasADi-symbolic and smooth) together with the
    geodetic up direction. `Earth` uses it for the sea surface (fluid
    membership, hydrostatic column, wave orbital direction) so the water
    line and the solid surface are one geometry. It mirrors the numpy
    `manta.planets.base.geodetic_from_cylindrical`; the two are
    cross-checked in the tests.

    Args:
        center            — spheroid centre (world frame), m.
        equatorial_radius — semi-major axis `a`, m.
        flattening        — `(a − b)/a`; 0 is a sphere.
        polar_axis        — symmetry (spin) axis, world frame.
        height            — m; the solid surface sits this far above the
                            reference spheroid along the geodetic normal
                            (a mean-sea-level offset). Default 0.
    """

    field_value_shape = _VEC3_W

    def __init__(self,
                 center: tuple[float, float, float],
                 equatorial_radius: float,
                 flattening: float,
                 polar_axis: tuple[float, float, float] = (0.0, 0.0, 1.0),
                 *, height: float = 0.0, name: str | None = None) -> None:
        from ..parts._declarations import unit_axis
        super().__init__(name=name)
        self.center = tuple(float(x) for x in center)
        if len(self.center) != 3:
            raise ValueError(
                f"Ellipsoid: center must be length-3; got {center!r}")
        self.equatorial_radius = float(equatorial_radius)
        if self.equatorial_radius <= 0.0:
            raise ValueError(
                f"Ellipsoid: equatorial_radius must be > 0; got "
                f"{equatorial_radius!r}")
        self.flattening = float(flattening)
        if not (0.0 <= self.flattening < 1.0):
            raise ValueError(
                f"Ellipsoid: flattening must be in [0, 1); got {flattening!r}")
        self.polar_axis = unit_axis(polar_axis, who="Ellipsoid",
                                    what="polar_axis")
        self.height = float(height)
        self._axis_dm = ca.DM(list(self.polar_axis))

    def signed_height_sym(self, r_mx: ca.MX) -> tuple[ca.MX, ca.MX]:
        """`(height, up)` for an offset `r_mx` (3×1 MX) from the centre —
        geodetic height above the surface (negative inside; the `height`
        offset already subtracted) and the geodetic up unit vector, both
        in the frame `r_mx` is expressed in. That frame must share the
        `polar_axis` coordinates (true for the world frame and for any
        frame rotated about the axis, e.g. a planet's body frame)."""
        a = self.equatorial_radius
        f = self.flattening
        e2 = f * (2.0 - f)
        b = a * (1.0 - f)
        ep2 = e2 / (1.0 - e2)
        axis = self._axis_dm
        z = ca.dot(r_mx, axis)
        rho_vec = r_mx - z * axis
        rho = soft_norm(rho_vec)
        beta = ca.atan2(a * z, b * rho)
        lat = ca.atan2(z + ep2 * b * ca.sin(beta) ** 3,
                       rho - e2 * a * ca.cos(beta) ** 3)
        beta = ca.atan2((1.0 - f) * ca.sin(lat), ca.cos(lat))
        lat = ca.atan2(z + ep2 * b * ca.sin(beta) ** 3,
                       rho - e2 * a * ca.cos(beta) ** 3)
        sin_lat, cos_lat = ca.sin(lat), ca.cos(lat)
        h = (rho * cos_lat + z * sin_lat
             - a * ca.sqrt(1.0 - e2 * sin_lat * sin_lat)) - self.height
        # e_rho = rho_vec / rho vanishes smoothly on the axis, where
        # cos(lat) → 0 anyway, so `up` stays a unit vector there.
        up = cos_lat * (rho_vec / rho) + sin_lat * axis
        return h, up

    def contribute_at_sym(self, point, t):
        center_v = _VEC3_W.constant(self.center)
        h, up = self.signed_height_sym((point - center_v)._mx)
        depth = smooth_max0(-h, _SMOOTH_EPS_SQ)
        return _VEC3_W.from_mx(up * depth)

    def __repr__(self) -> str:
        return (f"<Ellipsoid center={self.center} "
                f"a={self.equatorial_radius} f={self.flattening:.6g} "
                f"axis={self.polar_axis} height={self.height}>")


class Heightfield(Disturbance):
    """Solid terrain below a gridded height surface `z = h(x, y)` —
    bathymetry for a UUV, a DEM for a ground vehicle. World frame, z-up,
    solid everywhere below the surface.

    The grid is interpolated with a cubic B-spline (`ca.interpolant`),
    so the surface — and, crucially, its GRADIENT, which supplies the
    contact normal — is smooth across cell boundaries (a bilinear
    surface has normal jumps at every cell edge, which a contact spring
    turns into force chatter). Penetration is the vertical excess
    projected onto the local surface normal (exact for a plane, the
    standard near-surface approximation for gentle terrain):

        e        = h(x, y) − z                       (vertical excess)
        n_unnorm = (−∂h/∂x, −∂h/∂y, 1)
        depth    = smooth_max0(e / |n_unnorm|)       (⊥ distance)
        value    = depth · n_unnorm / |n_unnorm|     (outward vector)

    Two contracts to respect:

      * **Coverage** — the spline extrapolates polynomially outside the
        grid, which diverges fast; the grid must cover the operating
        area with margin. Size it generously; memory is cheap.
      * **Backend reach** — interpolant nodes are MX-only (no SX
        expansion), so a Heightfield world is for the numpy/C++
        targets: `TargetJax` refuses it loudly and `Fit` warns and
        takes the interpreted path. This is the accepted trade for
        real terrain in the tick.

    The same grid should back any *acoustic* raycasting numerically
    (shiver's sim) — one dataset, two views; keep the returned
    `Heightfield` and share its `heights` array.

    Args:
        heights — 2D array-like, shape (nx, ny): `heights[i, j]` is the
                  surface height at `(x0 + i·dx, y0 + j·dy)`. Cubic
                  B-spline needs at least 4 samples per axis.
        x0, y0  — world coordinates of grid corner `heights[0, 0]`, m.
        dx, dy  — grid spacing, m (> 0).
    """

    field_value_shape = _VEC3_W

    def __init__(self, heights, *,
                 x0: float = 0.0, y0: float = 0.0,
                 dx: float = 1.0, dy: float = 1.0,
                 name: str | None = None) -> None:
        super().__init__(name=name)
        H = np.asarray(heights, dtype=float)
        if H.ndim != 2 or H.shape[0] < 4 or H.shape[1] < 4:
            raise ValueError(
                f"Heightfield: heights must be a 2D grid with at least "
                f"4 samples per axis (cubic B-spline support); got shape "
                f"{H.shape}.")
        if not np.all(np.isfinite(H)):
            raise ValueError("Heightfield: heights contain non-finite "
                             "values.")
        if dx <= 0.0 or dy <= 0.0:
            raise ValueError(
                f"Heightfield: grid spacing must be > 0; got dx={dx!r}, "
                f"dy={dy!r}.")
        self.heights = H
        self.x0, self.y0 = float(x0), float(y0)
        self.dx, self.dy = float(dx), float(dy)
        xg = self.x0 + self.dx * np.arange(H.shape[0])
        yg = self.y0 + self.dy * np.arange(H.shape[1])
        # ca.interpolant data layout: first grid axis varies fastest —
        # heights[i, j] ↔ d[i + nx·j] is Fortran ravel order.
        self._h = ca.interpolant(
            "heightfield_h", "bspline",
            [xg.tolist(), yg.tolist()],
            H.ravel(order="F").tolist())
        xy = ca.MX.sym("xy", 2)
        self._grad = ca.Function("heightfield_grad", [xy],
                                 [ca.jacobian(self._h(xy), xy)])

    # ---- surface queries (symbolic + numeric) -------------------------

    def height_at_sym(self, x: ca.MX, y: ca.MX) -> ca.MX:
        """Surface height h(x, y) as an MX expression — for in-tick
        consumers (an altitude output, a terrain-following law)."""
        return self._h(ca.vertcat(x, y))

    def height_at(self, x: float, y: float) -> float:
        """Numeric h(x, y) — the spline the contact actually feels
        (which is NOT bit-identical to the raw grid between samples)."""
        return float(self._h(ca.DM([float(x), float(y)])))

    def altitude_of_sym(self, point) -> ca.MX:
        """Signed height of a world point above the surface (MX):
        `z − h(x, y)` — negative below terrain."""
        p = point._mx if hasattr(point, "_mx") else point
        return p[2] - self._h(ca.vertcat(p[0], p[1]))

    # ---- collision contribution ---------------------------------------

    def contribute_at_sym(self, point, t):
        p = point._mx
        xy = ca.vertcat(p[0], p[1])
        e = self._h(xy) - p[2]                       # vertical excess
        g = self._grad(xy)                           # (1×2) [∂h/∂x ∂h/∂y]
        n_unnorm = ca.vertcat(-g[0, 0], -g[0, 1], ca.MX(1.0))
        n_mag = soft_norm(n_unnorm)
        depth = smooth_max0(e / n_mag, _SMOOTH_EPS_SQ)
        out_mx = (n_unnorm / n_mag) * depth
        return _VEC3_W.from_mx(out_mx)

    def __repr__(self) -> str:
        nx, ny = self.heights.shape
        return (f"<Heightfield {nx}x{ny} origin=({self.x0}, {self.y0}) "
                f"spacing=({self.dx}, {self.dy})>")

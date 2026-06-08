"""MagField + Uniform/Dipole magnetic disturbances.

Magnetic flux density B at a point, in Tesla, expressed in WorldFrame.
Per the Field-redesign pattern, there's one MagField class; per-source
variation lives in Disturbance subclasses.

Shipped disturbances:
  * UniformMag(B_vec)            — constant background (Earth-field
                                   first-order model).
  * DipoleMag(position, moment) — point-dipole field B(p) =
                                   μ₀/(4π) · (3(m̂·r̂)r̂ − m̂) / r³
                                   for a magnet of moment m at position
                                   `position` (world frame, A·m²).
                                   Useful for nearby permanent magnets
                                   and small-body dipole approximations
                                   of larger sources.
"""

from __future__ import annotations

import casadi as ca

from ..ir.frames import WorldFrame
from ..ir.types import Vec3
from .base import Disturbance, Field


_VEC3_ANCHOR = Vec3[WorldFrame]

# Vacuum permeability over 4π. In SI units. μ₀/(4π) = 1e-7 H/m.
_MU0_OVER_4PI = 1.0e-7


class MagField(Field):
    """Magnetic flux density field, Vec3[WorldFrame] in Tesla.

    Builder methods (`add_uniform`, `add_dipole`) return self for
    chaining. The constructor accepts `B=(bx,by,bz)` for the common
    single-uniform case — equivalent to one `add_uniform`.
    """

    value_shape = _VEC3_ANCHOR

    def __init__(self, B: tuple[float, float, float] | None = None) -> None:
        super().__init__()
        if B is not None:
            self.add_uniform(B)

    def _zero_value(self):
        return _VEC3_ANCHOR.constant((0.0, 0.0, 0.0))

    def add_uniform(self, B_vec: tuple[float, float, float]) -> "MagField":
        """Attach a position-independent magnetic field. Returns self."""
        return self.add(UniformMag(B_vec))

    def add_dipole(self,
                   moment: tuple[float, float, float],
                   position: tuple[float, float, float] = (0.0, 0.0, 0.0),
                   eps: float = 1e-3) -> "MagField":
        """Attach a point-dipole source. Returns self."""
        return self.add(DipoleMag(position=position, moment=moment, eps=eps))


class UniformMag(Disturbance):
    """Position-independent magnetic field.

    For a quick first-order Earth-field model, common values (in Tesla):
      * Equator   ≈ 30 µT       — (3e-5, 0, 0) horizontal
      * Mid-lat   ≈ 50 µT       — (2e-5, 0, -4.5e-5) inclined
      * Pole      ≈ 60 µT       — (0, 0, -6e-5)
    """

    field_value_shape = _VEC3_ANCHOR

    def __init__(self, B_vec: tuple[float, float, float], *, name: str | None = None) -> None:
        super().__init__(name=name)
        self.B_vec = tuple(float(x) for x in B_vec)
        if len(self.B_vec) != 3:
            raise ValueError(
                f"UniformMag: B_vec must be length-3, got {B_vec!r}")

    def contribute_at_sym(self, point, t):
        return _VEC3_ANCHOR.constant(self.B_vec)

    def __repr__(self) -> str:
        return f"<UniformMag B_vec={self.B_vec}>"


class DipoleMag(Disturbance):
    """Point magnetic dipole.

      B(p) = μ₀/(4π) · [3(m·r̂)·r̂ − m] / r³

    where r = p − r_src, r̂ = r/|r|.

    Args:
        position — (x, y, z) dipole position in WorldFrame, m.
        moment   — (mx, my, mz) magnetic dipole moment in WorldFrame,
                   A·m². For Earth this is ~8e22 (purely fictitious
                   point-dipole approximation).
        eps      — softening length (m) to avoid singularity at the
                   dipole position. Default 1e-3.
    """

    field_value_shape = _VEC3_ANCHOR

    def __init__(self,
                 position: tuple[float, float, float],
                 moment: tuple[float, float, float],
                 eps: float = 1e-3, *, name: str | None = None) -> None:
        super().__init__(name=name)
        self.position = tuple(float(x) for x in position)
        self.moment   = tuple(float(x) for x in moment)
        self.eps      = float(eps)

    def contribute_at_sym(self, point, t):
        r_src = _VEC3_ANCHOR.constant(self.position)
        m     = _VEC3_ANCHOR.constant(self.moment)
        r     = point - r_src
        r_mx  = r._mx
        m_mx  = m._mx
        r_sq  = ca.dot(r_mx, r_mx) + self.eps**2
        r_mag = ca.sqrt(r_sq)
        r_cubed = r_sq * r_mag                    # |r|³
        m_dot_r = ca.dot(m_mx, r_mx)
        # B = μ₀/(4π) · [3·(m·r)·r / r⁵ − m / r³]
        # Note: 3(m·r̂)r̂ / r³ = 3(m·r)·r / r⁵
        B_mx = _MU0_OVER_4PI * (
            (3.0 * m_dot_r / (r_cubed * r_sq)) * r_mx
            - m_mx / r_cubed
        )
        return _VEC3_ANCHOR.from_mx(B_mx)

    def __repr__(self) -> str:
        return (f"<DipoleMag position={self.position} moment={self.moment}>")


class BodyDipoleMag(Disturbance):
    """A magnetic dipole that RIDES a craft — the field a `MagneticSource`
    part emits to model the magnetic signature of motors, magnets, or
    magnetized structure on a moving vehicle. Same law as `DipoleMag`, but
    BOTH the dipole position and its moment vector are body-fixed: the
    position tracks the craft and the moment rotates with it (read from
    the active trace), so a magnetometer on another craft sees the field
    swing as the source turns.

    Args:
        craft       — the craft carrying the source.
        offset_body — dipole position in the craft body frame, m.
        moment_body — dipole moment in the craft body frame, A·m².
        eps         — softening length, m. Default 1e-3.
    """

    field_value_shape = _VEC3_ANCHOR

    def __init__(self, craft, offset_body,
                 moment_body: tuple[float, float, float],
                 eps: float = 1e-3, *, name: str | None = None) -> None:
        super().__init__(name=name)
        self.craft = craft
        self.offset_body = tuple(float(x) for x in offset_body)
        self.moment_body = tuple(float(x) for x in moment_body)
        if len(self.moment_body) != 3:
            raise ValueError(
                f"BodyDipoleMag: moment_body must be length-3, "
                f"got {moment_body!r}")
        self.eps = float(eps)

    def contribute_at_sym(self, point, t):
        from .base import anchored_pose
        from ..ir.frames import CraftFrame
        center, _R, quat = anchored_pose(self.craft, self.offset_body)
        # Moment is body-fixed → rotate into world with the craft attitude.
        m_world = quat.apply(Vec3[CraftFrame].constant(self.moment_body))
        r = point - center
        r_mx, m_mx = r._mx, m_world._mx
        r_sq = ca.dot(r_mx, r_mx) + self.eps**2
        r_mag = ca.sqrt(r_sq)
        r_cubed = r_sq * r_mag
        m_dot_r = ca.dot(m_mx, r_mx)
        B_mx = _MU0_OVER_4PI * (
            (3.0 * m_dot_r / (r_cubed * r_sq)) * r_mx - m_mx / r_cubed)
        return _VEC3_ANCHOR.from_mx(B_mx)

    def __repr__(self) -> str:
        return (f"<BodyDipoleMag craft={self.craft.name!r} "
                f"moment={self.moment_body}>")

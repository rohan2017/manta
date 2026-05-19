"""Manifold helpers — SO3, R3, RigidBody.

These describe the structure of a *state*: how it composes, what its
tangent space looks like, and how `boxplus`/`boxminus` lift between the
ambient (4D for SO3, 3D for R3) and the tangent (3D for SO3, 3D for R3).

State declarations on parts will reference these manifold types (M1
work); the EKF will use boxplus/boxminus + autodiff to derive
tangent-space Jacobians. M0 lays the foundation: the types exist, their
ops emit CasADi expressions, and the well-known SO(3)-on-quaternion
math is correct.

No "manifold differentiation magic" — we just provide boxplus / boxminus
as plain CasADi expressions. CasADi's autodiff differentiates through
them naively; the *manifold-correct* Jacobian (tangent → tangent) is
recovered by wrapping the function as
    delta_out = boxminus(f(boxplus(x_ref, delta_in)), x_ref_post)
and asking CasADi for d(delta_out)/d(delta_in). That structural pattern
is what M3's EKF integration will use.
"""

from __future__ import annotations

import casadi as ca

from .frames import Frame, _validate_frame
from .types import Quat, Scalar, Vec3, _as_mx


# ---------------------------------------------------------------------------
# SO(3)
# ---------------------------------------------------------------------------

class SO3:
    """A rotation in SO(3), stored as a unit quaternion.

    Ambient dim: 4 (quaternion). Tangent dim: 3 (axis-angle vector).
    Used as a *manifold type* (decoration on state declarations); under
    the hood it just wraps a Quat. The tangent vectors live in the
    SAME frame as the rotation's `from_frame` by convention — that's
    where the EKF's error state lives in the standard ESKF formulation.
    """

    def __init__(self, q: Quat):
        if not isinstance(q, Quat):
            raise TypeError(
                f"SO3: expected a Quat, got {type(q).__name__}")
        self._q = q

    @property
    def quat(self) -> Quat:
        return self._q

    @property
    def from_frame(self):
        return self._q.from_frame

    @property
    def to_frame(self):
        return self._q.to_frame

    def __repr__(self) -> str:
        return f"<SO3 {self._q!r}>"

    # ---- Constructors ----------------------------------------------------

    @classmethod
    def identity(cls, *, from_frame, to_frame) -> "SO3":
        return cls(Quat.identity(from_frame=from_frame, to_frame=to_frame))

    @classmethod
    def from_axis_angle(cls, axis: Vec3, angle: Scalar) -> "SO3":
        """Build a rotation about `axis` (in PartFrame or whichever frame
        the axis lives in) by `angle` (radians). The resulting Quat's
        from/to frames are both the axis's frame (it's an endomorphism)."""
        _validate_frame("SO3.from_axis_angle axis", axis._frame)
        # axis must be unit; we normalize for safety.
        half = _as_mx(angle) * 0.5
        c = ca.cos(half)
        s = ca.sin(half)
        n = ca.norm_2(axis._mx)
        ax = axis._mx / n
        w = c
        v = s * ax
        q_mx = ca.vertcat(w, v[0], v[1], v[2])
        return cls(Quat(q_mx, from_frame=axis._frame, to_frame=axis._frame))

    @classmethod
    def from_quat(cls, q: Quat) -> "SO3":
        return cls(q)

    # ---- Manifold ops ---------------------------------------------------

    def boxplus(self, delta: Vec3) -> "SO3":
        """SO(3) boxplus: q ⊞ δ = q · exp(δ), where exp is the SO(3)
        exponential mapping the tangent (axis-angle) vector to a unit
        quaternion increment. δ lives in the same frame as the rotation's
        from_frame (the "left" frame in our convention)."""
        if not isinstance(delta, Vec3):
            raise TypeError(
                f"SO3.boxplus: delta must be a Vec3, got {type(delta).__name__}")
        # Note: the tangent frame is by convention the rotation's
        # from_frame in our left-trivialization convention.
        if delta._frame is not self.from_frame:
            from .frames import FrameError, _capture_user_source
            raise FrameError(
                "SO3.boxplus",
                expected=f"delta frame matches SO3.from_frame "
                         f"({self.from_frame.__name__})",
                got=f"delta_frame={delta._frame.__name__}",
                source=_capture_user_source(),
            )
        dq = _so3_exp(delta._mx)
        # q_new = dq * q   (left trivialization: rotation appended on the
        # left in the from_frame). Mathematically:
        #   delta_q has from=from, to=from; self._q has from=from, to=to.
        #   Result has from=from, to=to.
        dq_quat = Quat(dq, from_frame=self.from_frame, to_frame=self.from_frame)
        return SO3(dq_quat * self._q)

    def boxminus(self, other: "SO3") -> Vec3:
        """SO(3) boxminus: δ = log(self · other⁻¹), returned in the
        from_frame's tangent space."""
        if not isinstance(other, SO3):
            raise TypeError(
                f"SO3.boxminus: other must be an SO3, got {type(other).__name__}")
        if (other.from_frame is not self.from_frame
                or other.to_frame is not self.to_frame):
            from .frames import FrameError, _capture_user_source
            raise FrameError(
                "SO3.boxminus",
                expected=f"matching SO3 frames "
                         f"({self.from_frame.__name__}, {self.to_frame.__name__})",
                got=f"({other.from_frame.__name__}, {other.to_frame.__name__})",
                source=_capture_user_source(),
            )
        dq = (self._q * other._q.conjugate())._mx
        return Vec3(_so3_log(dq), frame=self.from_frame)


def _so3_exp(omega_mx) -> ca.MX:
    """Map a 3-vector to a unit quaternion via the SO(3) exponential.
    Numerically stable near zero via Taylor expansion of (sin θ/2)/θ.
    """
    half_theta_sq = ca.dot(omega_mx, omega_mx) * 0.25
    # cos(θ/2) and sinc(θ/2) via Taylor for stability near 0.
    #   cos(θ/2)   ≈ 1 - θ²/8                (= 1 - half_theta_sq/2)
    #   sin(θ/2)/θ ≈ 0.5 - θ²/48
    # Below threshold, use Taylor; above, use the closed form. We use
    # CasADi `if_else` so the symbolic graph can still differentiate.
    threshold = 1e-12   # ‖ω‖² threshold to swap formulas
    theta = ca.sqrt(half_theta_sq * 4)
    half_theta = theta * 0.5
    c = ca.cos(half_theta)
    s_over_theta = ca.if_else(
        half_theta_sq < threshold,
        0.5 - half_theta_sq / 12.0,    # Taylor near zero
        ca.sin(half_theta) / theta,     # exact otherwise
    )
    w = c
    v = omega_mx * s_over_theta
    return ca.vertcat(w, v[0], v[1], v[2])


def _so3_log(q_mx) -> ca.MX:
    """Map a unit quaternion to its 3-vector tangent. Stable near identity
    via the Taylor expansion of (atan2(|v|, w)) / |v|."""
    w = q_mx[0]
    v = ca.vertcat(q_mx[1], q_mx[2], q_mx[3])
    v_norm_sq = ca.dot(v, v)
    v_norm = ca.sqrt(v_norm_sq)
    # Always go through positive-w hemisphere to avoid the cover.
    sign = ca.sign(w + 1e-30)  # avoid sign(0)=0
    w_pos = w * sign
    v_pos = v * sign
    # 2 * atan2(|v|, w) / |v|, with Taylor near |v|=0.
    threshold = 1e-12
    coeff = ca.if_else(
        v_norm_sq < threshold,
        2.0 / (w_pos + 1e-30),   # Taylor: 2 * atan2(|v|, w) / |v| ≈ 2/w
        2.0 * ca.atan2(v_norm, w_pos) / (v_norm + 1e-30),
    )
    return v_pos * coeff


# ---------------------------------------------------------------------------
# R3 — Euclidean 3-space
# ---------------------------------------------------------------------------

class R3:
    """Trivial manifold: just R^3 with the natural addition as boxplus.

    Mostly a marker type — Vec3 already does the work. R3 makes the
    state-declaration story uniform across SO(3), R(n), and composites.
    """

    def __init__(self, v: Vec3):
        if not isinstance(v, Vec3):
            raise TypeError(f"R3: expected a Vec3, got {type(v).__name__}")
        self._v = v

    @property
    def value(self) -> Vec3:
        return self._v

    @property
    def frame(self):
        return self._v.frame

    def boxplus(self, delta: Vec3) -> "R3":
        return R3(self._v + delta)

    def boxminus(self, other: "R3") -> Vec3:
        return self._v - other._v


# ---------------------------------------------------------------------------
# Rigid body — R3 × SO3 × R3 × R3 (position, orientation, vel_linear, vel_angular)
# ---------------------------------------------------------------------------

class RigidBody:
    """The standard 13-D ambient / 12-D tangent rigid-body state.

    Ambient: position (3) + orientation (4) + vel_linear (3) + vel_angular (3) = 13.
    Tangent: δp (3) + δθ (3) + δv (3) + δω (3) = 12.

    Composite manifold: boxplus / boxminus dispatch component-wise.
    """

    def __init__(self,
                 position: Vec3,
                 orientation: SO3,
                 vel_linear: Vec3,
                 vel_angular: Vec3):
        self.position    = position
        self.orientation = orientation
        self.vel_linear  = vel_linear
        self.vel_angular = vel_angular

    def boxplus(self,
                d_position: Vec3,
                d_theta:    Vec3,
                d_vlinear:  Vec3,
                d_vangular: Vec3) -> "RigidBody":
        return RigidBody(
            position    = self.position + d_position,
            orientation = self.orientation.boxplus(d_theta),
            vel_linear  = self.vel_linear + d_vlinear,
            vel_angular = self.vel_angular + d_vangular,
        )

    def boxminus(self, other: "RigidBody") -> tuple[Vec3, Vec3, Vec3, Vec3]:
        return (
            self.position - other.position,
            self.orientation.boxminus(other.orientation),
            self.vel_linear - other.vel_linear,
            self.vel_angular - other.vel_angular,
        )

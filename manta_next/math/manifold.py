"""Manifold helpers — SO3, R3, RigidBody.

These describe the structure of a *state*: how it composes, what its
tangent space looks like, and how `boxplus`/`boxminus` lift between the
ambient (4D for SO3, 3D for R3) and the tangent (3D for SO3, 3D for R3).

The ESKF in `manta_next.estimation.ekf` uses boxplus/boxminus + CasADi
autodiff to derive tangent-space Jacobians via the standard pattern

    delta_out = boxminus(f(boxplus(x_ref, delta_in)), x_ref_post)

and asking CasADi for ∂δ_out / ∂δ_in at δ_in=0. No "manifold
differentiation magic" — boxplus/boxminus are plain CasADi expressions
and the framework just composes them.

`_so3_exp` and `_so3_log` are deliberately branch-free (eps-regularized
in place of an `if_else`) because CasADi's autodiff walks both branches
of a conditional, and the non-Taylor branch had `d(sin(0)/0)/dω = NaN`
that poisoned every Jacobian extracted from a tick using boxplus.
"""

from __future__ import annotations

import casadi as ca

from ..ir.frames import Frame, _validate_frame
from ..ir.types import Quat, Scalar, Vec3, _as_mx


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
            from ..ir.frames import FrameError, _capture_user_source
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
            from ..ir.frames import FrameError, _capture_user_source
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

    Branch-free, smooth everywhere. The trick: add a tiny epsilon under
    the sqrt so the denominator never hits zero. For ω=0 the formula
    `sin(eps/2) / eps` evaluates to ~0.5 (since sin(x) ≈ x for tiny x);
    for nonzero ω the eps is dominated by |ω|² and the result is exact
    to roundoff.

    Avoids `ca.if_else` because CasADi's autodiff evaluates BOTH
    branches; the non-Taylor branch had `d(sin(0)/0)/dω` = NaN, which
    poisoned every EKF Jacobian we tried to extract from a tick using
    boxplus.
    """
    eps_sq = 1.0e-30
    theta = ca.sqrt(ca.dot(omega_mx, omega_mx) + eps_sq)
    half_theta = theta * 0.5
    w = ca.cos(half_theta)
    s_over_theta = ca.sin(half_theta) / theta     # → 0.5 at ω=0 (smooth)
    v = omega_mx * s_over_theta
    return ca.vertcat(w, v[0], v[1], v[2])


def _so3_log(q_mx) -> ca.MX:
    """Map a unit quaternion to its 3-vector tangent.

    Branch-free via the same eps-regularization as _so3_exp. The
    composite `2·atan2(|v|, w) / |v|` is smooth at the identity
    quaternion (q = (1, 0, 0, 0)): both numerator and denominator scale
    as |v| there, so the ratio approaches the well-defined limit 2.
    """
    eps_sq = 1.0e-30
    w = q_mx[0]
    v = ca.vertcat(q_mx[1], q_mx[2], q_mx[3])
    v_norm = ca.sqrt(ca.dot(v, v) + eps_sq)
    # Always use the positive-w hemisphere to avoid the double cover.
    # sign(0) returns 0 in CasADi; offset slightly so sign(w=0) = +1.
    sign = ca.sign(w + 1.0e-30)
    w_pos = w * sign
    v_pos = v * sign
    coeff = 2.0 * ca.atan2(v_norm, w_pos) / v_norm
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
        if not isinstance(delta, Vec3):
            raise TypeError(
                f"R3.boxplus: delta must be a Vec3, got {type(delta).__name__}")
        if delta._frame is not self._v.frame:
            from ..ir.frames import FrameError, _capture_user_source
            raise FrameError(
                "R3.boxplus",
                expected=f"delta frame matches R3.frame "
                         f"({self._v.frame.__name__})",
                got=f"delta_frame={delta._frame.__name__}",
                source=_capture_user_source(),
            )
        return R3(self._v + delta)

    def boxminus(self, other: "R3") -> Vec3:
        if not isinstance(other, R3):
            raise TypeError(
                f"R3.boxminus: other must be an R3, got {type(other).__name__}")
        if other.frame is not self._v.frame:
            from ..ir.frames import FrameError, _capture_user_source
            raise FrameError(
                "R3.boxminus",
                expected=f"matching R3.frame ({self._v.frame.__name__})",
                got=f"other_frame={other.frame.__name__}",
                source=_capture_user_source(),
            )
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
        # Frame checks: each delta must match the frame of the
        # corresponding ambient slot. SO3.boxplus does its own check on
        # d_theta; the three Vec3 additions below need explicit checks
        # because Vec3.__add__ relies on its operand frames matching but
        # the wrong-frame failure mode is a TypeError in CasADi land,
        # not a FrameError. So we validate here for a clearer message.
        for label, ref, delta in (
            ("d_position",  self.position,    d_position),
            ("d_vlinear",   self.vel_linear,  d_vlinear),
            ("d_vangular",  self.vel_angular, d_vangular),
        ):
            if not isinstance(delta, Vec3):
                raise TypeError(
                    f"RigidBody.boxplus: {label} must be a Vec3, got "
                    f"{type(delta).__name__}")
            if delta._frame is not ref._frame:
                from ..ir.frames import FrameError, _capture_user_source
                raise FrameError(
                    f"RigidBody.boxplus[{label}]",
                    expected=f"frame {ref._frame.__name__}",
                    got=f"{delta._frame.__name__}",
                    source=_capture_user_source(),
                )
        return RigidBody(
            position    = self.position + d_position,
            orientation = self.orientation.boxplus(d_theta),
            vel_linear  = self.vel_linear + d_vlinear,
            vel_angular = self.vel_angular + d_vangular,
        )

    def boxminus(self, other: "RigidBody") -> tuple[Vec3, Vec3, Vec3, Vec3]:
        if not isinstance(other, RigidBody):
            raise TypeError(
                f"RigidBody.boxminus: other must be a RigidBody, got "
                f"{type(other).__name__}")
        # SO3.boxminus does its own frame check; same per-slot pattern
        # as boxplus for the three Vec3 differences.
        for label, ref, oth in (
            ("position",    self.position,    other.position),
            ("vel_linear",  self.vel_linear,  other.vel_linear),
            ("vel_angular", self.vel_angular, other.vel_angular),
        ):
            if oth._frame is not ref._frame:
                from ..ir.frames import FrameError, _capture_user_source
                raise FrameError(
                    f"RigidBody.boxminus[{label}]",
                    expected=f"frame {ref._frame.__name__}",
                    got=f"{oth._frame.__name__}",
                    source=_capture_user_source(),
                )
        return (
            self.position - other.position,
            self.orientation.boxminus(other.orientation),
            self.vel_linear - other.vel_linear,
            self.vel_angular - other.vel_angular,
        )

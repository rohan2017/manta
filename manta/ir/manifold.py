"""Manifold metadata + operations — Scalar, R3, SO(3).

A `Manifold` instance is the single source of truth for one state-vector
component: its structural metadata (`kind`, dims, `storage_shape`) AND the
boxplus/boxminus operations at every layer the framework needs them.

Three op flavors, ONE underlying math definition per manifold:

  * value-typed  — `boxplus(x, delta)` / `boxminus(a, b)` on frame-tagged
    `Scalar` / `Vec3` / `Quat` values. Frame-checked. This is what a
    Part's `update()` uses to integrate its own state slot
    manifold-correctly, and what the world tick uses for rigid-body
    orientation.
  * symbolic     — `boxplus_sym` / `boxminus_sym` on raw CasADi MX. Used
    by the ESKF Jacobian tracer and codegen. Frame-blind: the IR is
    already built, the frame tags have done their job.
  * numeric      — `boxplus_num` on flat numpy arrays. Used by the ESKF
    update to apply a Kalman correction onto the manifold.

The value and numeric paths delegate to the same kernel the symbolic
path emits (`_so3_exp` / `_so3_log` for SO(3)), so there is exactly one
definition of each manifold's math — no parallel implementation to drift.

`StateSlot.manifold` holds one of these instances. Backends map
`manifold.kind` to a concrete type via their own registry
(e.g. `codegen/cpp/types.py`); backend code never lives on this class, so
the IR stays backend-agnostic.

`_so3_exp` and `_so3_log` are deliberately branch-free (eps-regularized
in place of an `if_else`) because CasADi's autodiff walks both branches
of a conditional, and the non-Taylor branch had `d(sin(0)/0)/dω = NaN`
that poisoned every Jacobian extracted from a tick using boxplus.

Adding a new manifold kind:
  1. Subclass `Manifold` with a new `kind` string and structural dims.
  2. Implement the math kernel once; wire the three op flavors to it.
  3. Add an entry in each backend's type registry keyed on `kind`
     (e.g. `codegen/cpp/types.py::_REGISTRY`).
  4. Done — StateSpec, world_tick, the ESKF, and the wrappers all pick
     it up via the registry / metadata reads.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar

import casadi as ca
import numpy as np

from .frames import FrameError, _capture_user_source
from .types import Quat, Scalar, Vec3


# ---------------------------------------------------------------------------
# SO(3) math kernel — shared by every op flavor
# ---------------------------------------------------------------------------

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


def _quat_mul_mx(a_mx, b_mx) -> ca.MX:
    """Hamilton product of two (w, x, y, z) quaternion MX columns."""
    w1, x1, y1, z1 = a_mx[0], a_mx[1], a_mx[2], a_mx[3]
    w2, x2, y2, z2 = b_mx[0], b_mx[1], b_mx[2], b_mx[3]
    return ca.vertcat(
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    )


# ---------------------------------------------------------------------------
# ABC
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Manifold(ABC):
    """Structural descriptor + operations for a state-vector component.

    Subclasses set `kind` (the backend-registry key) and the dims as
    class-level constants. Instance attributes carry per-occurrence
    context (e.g. a frame tag).
    """

    kind:          ClassVar[str]
    ambient_dim:   ClassVar[int]
    tangent_dim:   ClassVar[int]
    storage_shape: ClassVar[tuple[int, ...]]

    # ---- Value-typed (frame-checked, for IR construction) ------------

    @abstractmethod
    def boxplus(self, x, delta):
        """Frame-checked boxplus on tagged values: ambient ⊞ tangent →
        ambient. `x` is a `Scalar`/`Vec3`/`Quat`, `delta` the matching
        tangent value. Returns a value of the same type as `x`."""

    @abstractmethod
    def boxminus(self, a, b):
        """Frame-checked boxminus on tagged values: ambient ⊟ ambient →
        tangent value."""

    # ---- Symbolic (CasADi MX, used by ESKF Jacobian extraction) ------

    @abstractmethod
    def boxplus_sym(self, x_mx, delta_mx):
        """Symbolic boxplus: ambient + tangent → ambient. CasADi MX of
        shape (ambient_dim, 1) and (tangent_dim, 1)."""

    @abstractmethod
    def boxminus_sym(self, a_mx, b_mx):
        """Symbolic boxminus: ambient − ambient → tangent."""

    # ---- Numeric (numpy, used by ESKF update onto manifold) ----------

    @abstractmethod
    def boxplus_num(self, x: np.ndarray, delta: np.ndarray) -> np.ndarray:
        """Numeric boxplus on flat arrays."""

    # ---- Initial value -----------------------------------------------

    @abstractmethod
    def default_value(self):
        """Python-side default value for fresh state (identity element
        for groups, zero for vectors). Returned as a numpy array of
        shape `storage_shape` or a scalar."""


# ---------------------------------------------------------------------------
# Concrete manifolds
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ScalarManifold(Manifold):
    """R^1 — scalar real. Backend kind: ``"scalar"``."""

    kind:          ClassVar[str]               = "scalar"
    ambient_dim:   ClassVar[int]               = 1
    tangent_dim:   ClassVar[int]               = 1
    storage_shape: ClassVar[tuple[int, ...]]   = (1,)

    def boxplus(self, x: Scalar, delta: Scalar) -> Scalar:
        if not isinstance(x, Scalar) or not isinstance(delta, Scalar):
            raise TypeError(
                "ScalarManifold.boxplus: x and delta must be Scalar, got "
                f"{type(x).__name__} / {type(delta).__name__}")
        return x + delta

    def boxminus(self, a: Scalar, b: Scalar) -> Scalar:
        if not isinstance(a, Scalar) or not isinstance(b, Scalar):
            raise TypeError(
                "ScalarManifold.boxminus: a and b must be Scalar, got "
                f"{type(a).__name__} / {type(b).__name__}")
        return a - b

    def boxplus_sym(self, x_mx, delta_mx):
        return x_mx + delta_mx

    def boxminus_sym(self, a_mx, b_mx):
        return a_mx - b_mx

    def boxplus_num(self, x, delta):
        return np.asarray(x, dtype=float) + np.asarray(delta, dtype=float)

    def default_value(self):
        return 0.0


@dataclass(frozen=True)
class R3Manifold(Manifold):
    """R^3 — 3-vector. Backend kind: ``"vec"`` (size carried in
    `storage_shape`, not in the kind string, so a future R6 reuses
    the same backend dispatch)."""

    kind:          ClassVar[str]               = "vec"
    ambient_dim:   ClassVar[int]               = 3
    tangent_dim:   ClassVar[int]               = 3
    storage_shape: ClassVar[tuple[int, ...]]   = (3,)

    frame: Any = None   # Frame class; codegen does not consume this.

    def boxplus(self, x: Vec3, delta: Vec3) -> Vec3:
        if not isinstance(x, Vec3) or not isinstance(delta, Vec3):
            raise TypeError(
                "R3Manifold.boxplus: x and delta must be Vec3, got "
                f"{type(x).__name__} / {type(delta).__name__}")
        if delta._frame is not x._frame:
            raise FrameError(
                "R3Manifold.boxplus",
                expected=f"delta frame matches x.frame ({x._frame.__name__})",
                got=f"delta_frame={delta._frame.__name__}",
                source=_capture_user_source(),
            )
        return x + delta

    def boxminus(self, a: Vec3, b: Vec3) -> Vec3:
        if not isinstance(a, Vec3) or not isinstance(b, Vec3):
            raise TypeError(
                "R3Manifold.boxminus: a and b must be Vec3, got "
                f"{type(a).__name__} / {type(b).__name__}")
        if a._frame is not b._frame:
            raise FrameError(
                "R3Manifold.boxminus",
                expected=f"matching frames ({a._frame.__name__})",
                got=f"b_frame={b._frame.__name__}",
                source=_capture_user_source(),
            )
        return a - b

    def boxplus_sym(self, x_mx, delta_mx):
        return x_mx + delta_mx

    def boxminus_sym(self, a_mx, b_mx):
        return a_mx - b_mx

    def boxplus_num(self, x, delta):
        return np.asarray(x, dtype=float) + np.asarray(delta, dtype=float)

    def default_value(self):
        return np.zeros(3, dtype=float)


@dataclass(frozen=True)
class SO3Manifold(Manifold):
    """SO(3) — rotations stored as a unit quaternion (w, x, y, z).

    Ambient 4 (quat) / tangent 3 (axis-angle). Backend kind: ``"quat"``.
    Left-trivialization convention: the tangent vector lives in the
    rotation's `from_frame`, and

        q ⊞ δ = exp(δ) ⊗ q

    `from_frame` / `to_frame` parametrize the underlying Quat — a
    `Quat[from, to]` rotates a vector in `to` coords into `from` coords.
    Both must be provided when used as a user-declared State manifold
    (no Frame default — pick the application's convention; the
    framework's rigid-body orientation uses Quat[WorldFrame,
    CraftFrame], so an attitude estimator typically matches that).
    Codegen consumes only `kind` / `storage_shape`; the value-typed ops
    derive their frames from the `Quat` argument, so a frameless
    `SO3Manifold()` is a valid operator for already-frame-tagged values.
    """

    kind:          ClassVar[str]               = "quat"
    ambient_dim:   ClassVar[int]               = 4
    tangent_dim:   ClassVar[int]               = 3
    storage_shape: ClassVar[tuple[int, ...]]   = (4,)

    from_frame: Any = None
    to_frame:   Any = None

    def boxplus(self, q: Quat, delta: Vec3) -> Quat:
        """q ⊞ δ with δ in q's from_frame (left trivialization). Returns
        a Quat carrying q's frames."""
        if not isinstance(q, Quat):
            raise TypeError(
                f"SO3Manifold.boxplus: q must be a Quat, got {type(q).__name__}")
        if not isinstance(delta, Vec3):
            raise TypeError(
                "SO3Manifold.boxplus: delta must be a Vec3, got "
                f"{type(delta).__name__}")
        if delta._frame is not q._from_frame:
            raise FrameError(
                "SO3Manifold.boxplus",
                expected=f"delta frame matches q.from_frame "
                         f"({q._from_frame.__name__})",
                got=f"delta_frame={delta._frame.__name__}",
                source=_capture_user_source(),
            )
        new_mx = self.boxplus_sym(q._mx, delta._mx)
        return Quat(new_mx, from_frame=q._from_frame, to_frame=q._to_frame)

    def boxminus(self, a: Quat, b: Quat) -> Vec3:
        """δ = log(a ⊗ b⁻¹), returned in a's from_frame tangent space."""
        if not isinstance(a, Quat) or not isinstance(b, Quat):
            raise TypeError(
                "SO3Manifold.boxminus: a and b must be Quat, got "
                f"{type(a).__name__} / {type(b).__name__}")
        if (a._from_frame is not b._from_frame
                or a._to_frame is not b._to_frame):
            raise FrameError(
                "SO3Manifold.boxminus",
                expected=f"matching Quat frames "
                         f"({a._from_frame.__name__}, {a._to_frame.__name__})",
                got=f"({b._from_frame.__name__}, {b._to_frame.__name__})",
                source=_capture_user_source(),
            )
        return Vec3(self.boxminus_sym(a._mx, b._mx), frame=a._from_frame)

    def boxplus_sym(self, x_mx, delta_mx):
        return _quat_mul_mx(_so3_exp(delta_mx), x_mx)

    def boxminus_sym(self, a_mx, b_mx):
        b_conj = ca.vertcat(b_mx[0], -b_mx[1], -b_mx[2], -b_mx[3])
        return _so3_log(_quat_mul_mx(a_mx, b_conj))

    def boxplus_num(self, x, delta):
        omega = np.asarray(delta, dtype=float)
        theta = float(np.linalg.norm(omega) + 1e-30)
        half  = 0.5 * theta
        w     = np.cos(half)
        v     = omega * (np.sin(half) / theta)
        dq    = np.array([w, v[0], v[1], v[2]])
        w1, x1, y1, z1 = dq
        w2, x2, y2, z2 = np.asarray(x, dtype=float)
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
        ])

    def default_value(self):
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)


# ---------------------------------------------------------------------------
# String shortcut → instance
# ---------------------------------------------------------------------------

def manifold_from_shortcut(shortcut, *, frame=None) -> Manifold:
    """Map a user-facing shortcut to a Manifold instance.

    Accepts:
      * a `Manifold` instance — passed through.
      * `"R1"` → `ScalarManifold()`.
      * `"R3"` → `R3Manifold(frame=frame)`.
      * `"SO3"` → `SO3Manifold()`.
    """
    if isinstance(shortcut, Manifold):
        return shortcut
    if shortcut == "R1":
        return ScalarManifold()
    if shortcut == "R3":
        return R3Manifold(frame=frame)
    if shortcut == "SO3":
        return SO3Manifold()
    raise ValueError(
        f"manifold_from_shortcut: unknown manifold shortcut {shortcut!r}. "
        f"Pass 'R1', 'R3', 'SO3', or a Manifold instance.")

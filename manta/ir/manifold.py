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

All three flavors delegate to the shared kernels in `ir._rotation`
(`so3_exp` / `so3_log` / `quat_mul` / `quat_conj` for SO(3), plus their
numpy twins), so there is exactly one definition of each manifold's math —
no parallel implementation to drift.

`StateSlot.manifold` holds one of these instances. Backends map
`manifold.kind` to a concrete type via their own registry
(e.g. `codegen/cpp/types.py`); backend code never lives on this class, so
the IR stays backend-agnostic.

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

from ._rotation import quat_conj, quat_mul, quat_mul_np, so3_exp, so3_exp_np, so3_log
from .frames import FrameError, _capture_user_source
from .types import Quat, Scalar, Vec3


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

    # ---- IR construction (typed input / zero / add) ------------------
    # These build the typed IR value for this manifold occurrence, so the
    # world tick and the Noise plumbing never branch on `kind` — the
    # manifold subclass IS the dispatch. A vector manifold declared
    # without a frame takes `default_frame` (the world tick passes
    # CraftFrame for Part state, WorldFrame for disturbance state).

    @abstractmethod
    def ir_input(self, name: str, *, default_frame=None):
        """Typed IR *input* symbol — `Scalar`/`Vec3[frame]`/`Quat[from,to]`."""

    @abstractmethod
    def ir_zero(self, *, default_frame=None):
        """Typed IR zero / identity value for this manifold."""

    @abstractmethod
    def ir_add(self, value, delta_mx, *, default_frame=None):
        """Type-preserving `value ⊕ delta_mx` (raw MX), used by random-walk
        state updates: `bias_next = bias + √dt · driver`."""


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

    def ir_input(self, name, *, default_frame=None):
        return Scalar.input(name)

    def ir_zero(self, *, default_frame=None):
        return Scalar.from_mx(ca.MX(0.0))

    def ir_add(self, value, delta_mx, *, default_frame=None):
        return Scalar.from_mx(value._mx + delta_mx)


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

    def _resolved_frame(self, default_frame):
        frame = self.frame or default_frame
        if frame is None:
            raise ValueError(
                "R3Manifold: no frame available — set frame= on the manifold "
                "or pass default_frame to ir_input/ir_zero/ir_add.")
        return frame

    def ir_input(self, name, *, default_frame=None):
        return Vec3[self._resolved_frame(default_frame)].input(name)

    def ir_zero(self, *, default_frame=None):
        return Vec3[self._resolved_frame(default_frame)].from_mx(
            ca.MX.zeros(3, 1))

    def ir_add(self, value, delta_mx, *, default_frame=None):
        return Vec3[self._resolved_frame(default_frame)].from_mx(
            value._mx + delta_mx)


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
        return quat_mul(so3_exp(delta_mx), x_mx)

    def boxminus_sym(self, a_mx, b_mx):
        return so3_log(quat_mul(a_mx, quat_conj(b_mx)))

    def boxplus_num(self, x, delta):
        return quat_mul_np(so3_exp_np(delta), x)

    def default_value(self):
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)

    def ir_input(self, name, *, default_frame=None):
        return Quat[self.from_frame, self.to_frame].input(name)

    def ir_zero(self, *, default_frame=None):
        return Quat[self.from_frame, self.to_frame].from_mx(
            ca.MX([1.0, 0.0, 0.0, 0.0]))

    def ir_add(self, value, delta_mx, *, default_frame=None):
        raise NotImplementedError(
            "SO3Manifold.ir_add: random-walk on SO(3) is undefined "
            "(quaternions don't add Euclidean-ly). Use a vector manifold for "
            "an RW bias, or boxplus for attitude integration.")


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

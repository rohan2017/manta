"""Manifold metadata — structural descriptors for state-vector slots.

`StateSlot.manifold` holds an instance of one of these classes. The
ESKF, the world-tick compiler, and codegen backends consume the
*metadata* (`kind`, `ambient_dim`, `tangent_dim`, `storage_shape`) to
lay out memory and pick wrap/unwrap code. Backends key on `kind` via
their own registry — Eigen / nalgebra / numpy mappings do NOT live on
this class; the IR stays backend-agnostic.

Distinct from `manta.ir.manifold` (`R3`, `SO3`, `RigidBody`) which are
*value* wrappers carrying a CasADi expression in the manifold's
ambient space. Those are what `Part.update()` receives when reading a
State slot; these describe the *type* of that slot. A future cleanup
may unify the two.

Adding a new manifold kind:
  1. Subclass `Manifold` here with a new `kind` string and structural
     dims.
  2. Implement `boxplus_sym`/`boxminus_sym` (CasADi) and `boxplus`
     (numpy) — pure math, no backend code.
  3. Add an entry in each backend's type registry
     (e.g. `codegen/cpp/types.py::_REGISTRY`) keyed on the same
     `kind` string.
  4. Done — every consumer (StateSpec, world_tick, EKF, wrappers)
     picks it up via the registry / metadata reads.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar

import casadi as ca
import numpy as np


# ---------------------------------------------------------------------------
# ABC
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Manifold(ABC):
    """Structural descriptor for a state-vector component.

    Subclasses set `kind` (the backend-registry key) and the dims as
    class-level constants. Instance attributes carry per-occurrence
    context (e.g., frame tag for a vector manifold).
    """

    kind:          ClassVar[str]
    ambient_dim:   ClassVar[int]
    tangent_dim:   ClassVar[int]
    storage_shape: ClassVar[tuple[int, ...]]

    # ---- Symbolic math (CasADi, used by ESKF Jacobian extraction) ----

    @abstractmethod
    def boxplus_sym(self, x_mx, delta_mx):
        """Symbolic boxplus: ambient + tangent → ambient. Inputs and
        output are CasADi MX of shape (ambient_dim, 1) and
        (tangent_dim, 1)."""

    @abstractmethod
    def boxminus_sym(self, a_mx, b_mx):
        """Symbolic boxminus: ambient − ambient → tangent."""

    # ---- Numeric math (numpy, used by ESKF update onto manifold) -----

    @abstractmethod
    def boxplus(self, x: np.ndarray, delta: np.ndarray) -> np.ndarray:
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

    def boxplus_sym(self, x_mx, delta_mx):
        return x_mx + delta_mx

    def boxminus_sym(self, a_mx, b_mx):
        return a_mx - b_mx

    def boxplus(self, x, delta):
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

    def boxplus_sym(self, x_mx, delta_mx):
        return x_mx + delta_mx

    def boxminus_sym(self, a_mx, b_mx):
        return a_mx - b_mx

    def boxplus(self, x, delta):
        return np.asarray(x, dtype=float) + np.asarray(delta, dtype=float)

    def default_value(self):
        return np.zeros(3, dtype=float)


@dataclass(frozen=True)
class SO3Manifold(Manifold):
    """SO(3) — rotations stored as unit quaternion (w, x, y, z).

    Ambient 4 (quat) / tangent 3 (axis-angle). Backend kind:
    ``"quat"``. Boxplus uses the same left-trivialization convention
    as `manta.ir.manifold.SO3.boxplus`:
        q_new = exp(δ) ⊗ q

    `from_frame` / `to_frame` parametrize the underlying Quat — a
    `Quat[from, to]` rotates a vector in `to` coords into `from` coords.
    Both must be provided when used as a user-declared State manifold
    (no Frame default — pick the application's convention; the
    framework's rigid-body orientation uses Quat[WorldFrame,
    CraftFrame], so an attitude estimator typically matches that).
    Codegen consumes only `kind` / `storage_shape`; frames are for
    frame-checked IR construction at compile time.
    """

    kind:          ClassVar[str]               = "quat"
    ambient_dim:   ClassVar[int]               = 4
    tangent_dim:   ClassVar[int]               = 3
    storage_shape: ClassVar[tuple[int, ...]]   = (4,)

    from_frame: Any = None
    to_frame:   Any = None

    def boxplus_sym(self, x_mx, delta_mx):
        # Imported lazily to avoid circular import with manta.ir.manifold.
        from .manifold import _so3_exp
        dq = _so3_exp(delta_mx)
        w1, x1, y1, z1 = dq[0], dq[1], dq[2], dq[3]
        w2 = x_mx[0]; x2 = x_mx[1]
        y2 = x_mx[2]; z2 = x_mx[3]
        return ca.vertcat(
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
        )

    def boxminus_sym(self, a_mx, b_mx):
        from .manifold import _so3_log
        aw = a_mx[0]; ax = a_mx[1]
        ay = a_mx[2]; az = a_mx[3]
        bw =  b_mx[0]; bx = -b_mx[1]
        by = -b_mx[2]; bz = -b_mx[3]   # conjugate
        dq = ca.vertcat(
            aw*bw - ax*bx - ay*by - az*bz,
            aw*bx + ax*bw + ay*bz - az*by,
            aw*by - ax*bz + ay*bw + az*bx,
            aw*bz + ax*by - ay*bx + az*bw,
        )
        return _so3_log(dq)

    def boxplus(self, x, delta):
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
# String shortcut → instance (back-compat for `State(manifold='R1')`)
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

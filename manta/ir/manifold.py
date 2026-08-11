"""Manifold metadata + operations — Scalar (R^1), R^3, R^n, SO(3).

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

import re

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar

import casadi as ca
import numpy as np

from ._rotation import quat_conj, quat_mul, quat_mul_np, so3_exp, so3_exp_np, so3_log
from .frames import FrameError, _capture_user_source
from .types import Quat, Scalar, Vec3, VecN


# ---------------------------------------------------------------------------
# ABC
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Manifold(ABC):
    """Structural descriptor + operations for a state-vector component.

    Subclasses set `kind` (the backend-registry key) as a class-level
    constant. The three dims are *instance-level* abstract properties:
    fixed-dimension manifolds (Scalar/R3/SO3) satisfy them with plain
    class attributes, while `RnManifold` derives them from its `dim`
    instance field — no `ClassVar`-vs-property override friction.
    Instance attributes carry per-occurrence context (e.g. a frame tag).
    """

    kind: ClassVar[str]

    @property
    @abstractmethod
    def ambient_dim(self) -> int:
        """Length of this component's packed ambient block."""

    @property
    @abstractmethod
    def tangent_dim(self) -> int:
        """Length of this component's tangent (error/perturbation) block."""

    @property
    @abstractmethod
    def storage_shape(self) -> tuple[int, ...]:
        """Shape of the stored ambient value (backends size their concrete
        types from this, not from `kind`)."""

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
# Shared guards / Euclidean base
# ---------------------------------------------------------------------------

def _require(x, T, who: str) -> None:
    """Loud isinstance guard shared by the value-typed manifold ops — one
    error shape instead of a hand-rolled copy per manifold per op."""
    if not isinstance(x, T):
        raise TypeError(
            f"{who}: must be {T.__name__}, got {type(x).__name__}")


@dataclass(frozen=True)
class _EuclideanManifold(Manifold):
    """Base for the flat (R^1 / R^3 / R^n) manifolds: ⊞ is `+` and ⊟ is
    `−` in every flavor, so the symbolic and numeric ops — which were
    byte-identical across the three subclasses — live here exactly once.
    Subclasses keep only the value-typed (type/frame-checked) pair, the
    dims, and the IR constructors."""

    def boxplus_sym(self, x_mx, delta_mx):
        return x_mx + delta_mx

    def boxminus_sym(self, a_mx, b_mx):
        return a_mx - b_mx

    def boxplus_num(self, x, delta):
        return np.asarray(x, dtype=float) + np.asarray(delta, dtype=float)


# ---------------------------------------------------------------------------
# Concrete manifolds
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ScalarManifold(_EuclideanManifold):
    """R^1 — scalar real. Backend kind: ``"scalar"``."""

    kind:          ClassVar[str]               = "scalar"
    ambient_dim:   ClassVar[int]               = 1
    tangent_dim:   ClassVar[int]               = 1
    storage_shape: ClassVar[tuple[int, ...]]   = (1,)

    def boxplus(self, x: Scalar, delta: Scalar) -> Scalar:
        _require(x,     Scalar, "ScalarManifold.boxplus: x")
        _require(delta, Scalar, "ScalarManifold.boxplus: delta")
        return x + delta

    def boxminus(self, a: Scalar, b: Scalar) -> Scalar:
        _require(a, Scalar, "ScalarManifold.boxminus: a")
        _require(b, Scalar, "ScalarManifold.boxminus: b")
        return a - b

    def default_value(self):
        return 0.0

    def ir_input(self, name, *, default_frame=None):
        return Scalar.input(name)

    def ir_zero(self, *, default_frame=None):
        return Scalar.from_mx(ca.MX(0.0))

    def ir_add(self, value, delta_mx, *, default_frame=None):
        return Scalar.from_mx(value._mx + delta_mx)


@dataclass(frozen=True)
class R3Manifold(_EuclideanManifold):
    """R^3 — spatial 3-vector, frame-checked. Backend kind: ``"vec"``
    (fixed size 3, carried in `storage_shape`). Other Euclidean sizes
    do NOT reuse this kind: they are `RnManifold` (kind ``"vecn"``,
    frame-free), which backends dispatch separately."""

    kind:          ClassVar[str]               = "vec"
    ambient_dim:   ClassVar[int]               = 3
    tangent_dim:   ClassVar[int]               = 3
    storage_shape: ClassVar[tuple[int, ...]]   = (3,)

    frame: Any = None   # Frame class; codegen does not consume this.

    # The frame checks live on Vec3's own operators — delegating keeps ONE
    # definition of "same frame" while the re-raise keeps the error's op
    # label pointing at the manifold call the user actually made.

    def boxplus(self, x: Vec3, delta: Vec3) -> Vec3:
        _require(x,     Vec3, "R3Manifold.boxplus: x")
        _require(delta, Vec3, "R3Manifold.boxplus: delta")
        try:
            return x + delta
        except FrameError as e:
            raise FrameError("R3Manifold.boxplus", expected=e.expected,
                             got=e.got, source=e.source) from None

    def boxminus(self, a: Vec3, b: Vec3) -> Vec3:
        _require(a, Vec3, "R3Manifold.boxminus: a")
        _require(b, Vec3, "R3Manifold.boxminus: b")
        try:
            return a - b
        except FrameError as e:
            raise FrameError("R3Manifold.boxminus", expected=e.expected,
                             got=e.got, source=e.source) from None

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
class RnManifold(_EuclideanManifold):
    """R^n — parameterized-dimension Euclidean. Backend kind: ``"vecn"``.

    ONE class for every n: the dimension is INSTANCE DATA, not a new
    type. `dim` is required — n = 1 and n = 3 have specialized classes
    (`ScalarManifold`, `R3Manifold`), so there is no honest default and
    a silent `dim=1` would just shadow ScalarManifold. Backends size
    their concrete type from `storage_shape`; the shortcut vocabulary is
    a grammar (``"R7"``, ``"R36"``, any ``"R<n>"``), not a table.

    Frame-free by design: an R^n quantity is a coefficient block (a
    fitted 6x6 damping tensor travelling as flat R36), not a spatial
    vector. Spatial 3-vectors keep `R3Manifold` and its frame
    checking — which is why n = 1 and n = 3 shortcuts still resolve to
    the specialized classes."""

    dim: int

    kind: ClassVar[str] = "vecn"

    def __post_init__(self):
        if int(self.dim) < 1:
            raise ValueError(f"RnManifold: dim must be >= 1, got "
                             f"{self.dim}")

    @property
    def ambient_dim(self) -> int:
        return int(self.dim)

    @property
    def tangent_dim(self) -> int:
        return int(self.dim)

    @property
    def storage_shape(self) -> tuple[int, ...]:
        return (int(self.dim),)

    def boxplus(self, x: VecN, delta: VecN) -> VecN:
        _require(x,     VecN, "RnManifold.boxplus: x")
        _require(delta, VecN, "RnManifold.boxplus: delta")
        return VecN(x._mx + delta._mx, self.dim)

    def boxminus(self, a: VecN, b: VecN) -> VecN:
        _require(a, VecN, "RnManifold.boxminus: a")
        _require(b, VecN, "RnManifold.boxminus: b")
        return VecN(a._mx - b._mx, self.dim)

    def default_value(self):
        return np.zeros(int(self.dim), dtype=float)

    def ir_input(self, name, *, default_frame=None):
        return VecN.input(name, self.dim)

    def ir_zero(self, *, default_frame=None):
        return VecN.constant(np.zeros(int(self.dim)), self.dim)

    def ir_add(self, value, delta_mx, *, default_frame=None):
        return VecN(value._mx + delta_mx, self.dim)


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
        _require(q,     Quat, "SO3Manifold.boxplus: q")
        _require(delta, Vec3, "SO3Manifold.boxplus: delta")
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
        _require(a, Quat, "SO3Manifold.boxminus: a")
        _require(b, Quat, "SO3Manifold.boxminus: b")
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
        # Renormalize defensively: the tangent step may be large, and float
        # error drifts ‖q‖ off 1 over repeated numeric updates.
        q = quat_mul_np(so3_exp_np(delta), x)
        return q / np.linalg.norm(q)

    def default_value(self):
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)

    def ir_input(self, name, *, default_frame=None):
        return Quat[self.from_frame, self.to_frame].input(name)

    def ir_zero(self, *, default_frame=None):
        # `Quat.identity` is the single spelling of the identity rotation —
        # no restated (1, 0, 0, 0) literal to drift.
        return Quat.identity(from_frame=self.from_frame,
                             to_frame=self.to_frame)

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
      * `"R<n>"` for any n ≥ 1 (no leading zeros) — the Euclidean
        grammar. `"R1"` and `"R3"` resolve to their specialized classes
        (`ScalarManifold`, frame-checked `R3Manifold`); every other n
        gives the parameterized, frame-free `RnManifold(n)` (`"R36"` =
        a flattened 6x6 tensor, and so on).
      * `"SO3"` → `SO3Manifold()`.

    `frame=` is consumed only by `"R3"` (the one frame-tagged shortcut).
    Passing it with `"SO3"` (whose frames are the dual from/to pair on
    `SO3Manifold`) or a frame-free `"R<n>"` raises instead of silently
    dropping it — a dropped frame would resurface later as an unchecked
    frame bug, far from the declaration."""
    if isinstance(shortcut, Manifold):
        return shortcut

    def _reject_frame(why: str) -> None:
        if frame is not None:
            raise ValueError(
                f"manifold_from_shortcut: frame= is not consumed by "
                f"{shortcut!r} — {why}")

    if shortcut == "SO3":
        _reject_frame("SO(3) frames are the from/to pair on "
                      "SO3Manifold(from_frame=..., to_frame=...).")
        return SO3Manifold()
    # No leading zeros: "R03" would silently alias "R3" via int().
    m = re.fullmatch(r"R([1-9][0-9]*)", shortcut) \
        if isinstance(shortcut, str) else None
    if m:
        n = int(m.group(1))
        if n == 1:
            return ScalarManifold()
        if n == 3:
            return R3Manifold(frame=frame)
        _reject_frame(f"R{n} is frame-free by design (RnManifold); only "
                      f"'R3' carries a frame tag.")
        return RnManifold(n)
    raise ValueError(
        f"manifold_from_shortcut: unknown manifold shortcut {shortcut!r}. "
        f"Pass 'R<n>' (e.g. 'R1', 'R3', 'R36'), 'SO3', or a Manifold "
        f"instance.")

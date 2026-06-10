"""IR value types — Scalar, Vec3[F], Mat3[A,B], Quat[A,B].

Each type wraps a `casadi.MX` node and carries one or two frame tags. The
operators dispatch to CasADi while preserving and checking frame tags.

Type parameterization uses `__class_getitem__` so the syntax reads cleanly:

    Vec3[WorldFrame]
    Mat3[CraftFrame, PartFrame]
    Quat[WorldFrame, CraftFrame]

`Cls[args]` returns a small typed-constructor object (not the bare class).
Its `.input(name)` and `.constant(value)` factories produce instances of the
underlying class with the frame tags pre-filled.

All operations are pure (no in-place mutation). Each op returns a new IR
value wrapping a freshly built CasADi expression.
"""

from __future__ import annotations

from typing import Any, ClassVar

import casadi as ca

from ._rotation import quat_conj, quat_mul, quat_to_rotmat
from .frames import (
    FrameError,
    _capture_user_source,
    _format_frame,
    _is_frame,
    _validate_frame,
)
from .graph import _current_graph


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _as_mx(x) -> ca.MX:
    """Promote a Python int/float to a CasADi MX scalar, pass MX through."""
    if isinstance(x, ca.MX):
        return x
    if isinstance(x, (int, float)):
        return ca.MX(float(x))
    if hasattr(x, "_mx"):
        return x._mx
    raise TypeError(f"_as_mx: cannot promote {type(x).__name__}")


# ---------------------------------------------------------------------------
# IRValue base
# ---------------------------------------------------------------------------

class _IRValue:
    """Marker base class for all IR-typed values. Subclasses store a CasADi
    MX in `_mx` plus frame metadata."""

    _mx: ca.MX

    @property
    def shape(self) -> tuple[int, int]:
        return tuple(self._mx.shape)

    def __repr__(self) -> str:
        return f"<{type(self).__name__} shape={self.shape}>"


# ---------------------------------------------------------------------------
# Typed-constructor objects
# ---------------------------------------------------------------------------

class _ParameterizedConstructor:
    """Returned by `Cls[args]`. Holds the bound type parameters and exposes
    factories that instantiate the underlying class with them filled in."""

    def __init__(self, cls, **kwargs):
        self._cls = cls
        self._kwargs = kwargs

    def input(self, name: str) -> Any:
        """Create a named symbolic input registered with the current graph."""
        g = _current_graph()
        shape = self._cls._mx_shape
        mx = ca.MX.sym(name, *shape)
        value = self._cls._from_mx(mx, **self._kwargs)
        g._register_input(name, value)
        return value

    def constant(self, value) -> Any:
        return self._cls._make_constant(value, **self._kwargs)

    def coerce(self, value) -> Any:
        """Accept `value` as-is when it is already an IR value of this type
        (frame-checked), else build a constant from the Python value.

        The idiom for consuming a promotable `Parameter` inside `update()`:
        normally the attribute is a plain Python value (→ baked constant);
        when promoted for system ID, the trace binds it to a typed graph
        input and `coerce` passes it through.
        """
        if isinstance(value, self._cls):
            want = self._kwargs.get("frame")
            got = getattr(value, "_frame", None)
            if want is not None and got is not want:
                raise FrameError(
                    f"{self._cls.__name__}.coerce",
                    expected=f"value in {want.__name__}",
                    got=f"frame={got.__name__ if got else None}",
                    source=_capture_user_source(),
                )
            return value
        if isinstance(value, _IRValue):
            raise TypeError(
                f"{self!r}.coerce: expected {self._cls.__name__} or a plain "
                f"Python value, got {type(value).__name__}")
        return self._cls._make_constant(value, **self._kwargs)

    def from_mx(self, mx) -> Any:
        return self._cls._from_mx(mx, **self._kwargs)

    def __repr__(self) -> str:
        params = ", ".join(
            f"{k}={_format_frame(v) if _is_frame(v) else v!r}"
            for k, v in self._kwargs.items())
        return f"{self._cls.__name__}[{params}]"


# ---------------------------------------------------------------------------
# Scalar
# ---------------------------------------------------------------------------

class Scalar(_IRValue):
    """A scalar IR value. Has no frame tag — scalars are frame-agnostic."""

    _mx_shape: ClassVar[tuple[int, int]] = (1, 1)

    def __init__(self, mx):
        # Accept any 1×1 (or zero-D) MX. CasADi sometimes returns shape
        # (1, 1) for scalars; we normalize.
        mx = _as_mx(mx)
        if mx.shape not in ((1, 1), (1,)):
            raise ValueError(
                f"Scalar: expected shape (1,1), got {tuple(mx.shape)}")
        self._mx = mx

    @classmethod
    def _from_mx(cls, mx):
        return cls(mx)

    @classmethod
    def _make_constant(cls, value):
        return cls(ca.MX(float(value)))

    @classmethod
    def input(cls, name: str) -> "Scalar":
        g = _current_graph()
        mx = ca.MX.sym(name, 1, 1)
        v = cls(mx)
        g._register_input(name, v)
        return v

    @classmethod
    def constant(cls, value) -> "Scalar":
        return cls._make_constant(value)

    @classmethod
    def coerce(cls, value) -> "Scalar":
        """`value` as-is when already a Scalar, else a constant — the
        promotable-Parameter idiom (see _ParameterizedConstructor.coerce)."""
        if isinstance(value, cls):
            return value
        if isinstance(value, _IRValue):
            raise TypeError(
                f"Scalar.coerce: expected Scalar or a plain number, got "
                f"{type(value).__name__}")
        return cls._make_constant(value)

    # --- Arithmetic -------------------------------------------------------

    def __add__(self, other):
        return _scalar_op(self, other, lambda a, b: a + b)
    __radd__ = __add__

    def __sub__(self, other):
        return _scalar_op(self, other, lambda a, b: a - b)
    def __rsub__(self, other):
        return _scalar_op(other, self, lambda a, b: a - b)

    def __mul__(self, other):
        # Scalar * Vec3 / Mat3 / Quat: dispatch to that type's reverse op.
        if isinstance(other, (Vec3, Mat3, Quat)):
            return other.__rmul__(self)
        return _scalar_op(self, other, lambda a, b: a * b)
    __rmul__ = __mul__

    def __truediv__(self, other):
        return _scalar_op(self, other, lambda a, b: a / b)
    def __rtruediv__(self, other):
        return _scalar_op(other, self, lambda a, b: a / b)

    def __neg__(self):
        return Scalar(-self._mx)

    def __pow__(self, exponent):
        if isinstance(exponent, Scalar):
            return Scalar(self._mx ** exponent._mx)
        return Scalar(self._mx ** float(exponent))


def _scalar_op(a, b, fn):
    return Scalar(fn(_as_mx(a), _as_mx(b)))


# ---------------------------------------------------------------------------
# Vec3
# ---------------------------------------------------------------------------

class Vec3(_IRValue):
    """A 3-vector tagged with a single frame."""

    _mx_shape: ClassVar[tuple[int, int]] = (3, 1)

    def __init__(self, mx, frame):
        _validate_frame("Vec3", frame)
        mx = _as_mx(mx)
        if mx.shape != (3, 1):
            raise ValueError(
                f"Vec3: expected shape (3,1), got {tuple(mx.shape)}")
        self._mx = mx
        self._frame = frame

    # ---- Constructors via __class_getitem__ -----------------------------

    def __class_getitem__(cls, frame):
        _validate_frame("Vec3[...]", frame)
        return _ParameterizedConstructor(cls, frame=frame)

    @classmethod
    def _from_mx(cls, mx, *, frame):
        return cls(mx, frame=frame)

    @classmethod
    def _make_constant(cls, value, *, frame):
        if len(value) != 3:
            raise ValueError(
                f"Vec3[{frame.__name__}].constant: expected length-3 sequence, "
                f"got len={len(value)}")
        mx = ca.MX(ca.DM([float(value[0]), float(value[1]), float(value[2])]))
        return cls(mx, frame=frame)

    # ---- Properties -----------------------------------------------------

    @property
    def frame(self):
        return self._frame

    def __repr__(self) -> str:
        return f"<Vec3[{self._frame.__name__}] shape={self.shape}>"

    # ---- Component access ------------------------------------------------

    @property
    def x(self) -> Scalar:
        return Scalar(self._mx[0])
    @property
    def y(self) -> Scalar:
        return Scalar(self._mx[1])
    @property
    def z(self) -> Scalar:
        return Scalar(self._mx[2])

    # ---- Arithmetic ------------------------------------------------------

    def __add__(self, other: "Vec3") -> "Vec3":
        _check_vec3_same_frame(self, other, op="Vec3 + Vec3")
        return Vec3(self._mx + other._mx, frame=self._frame)

    def __sub__(self, other: "Vec3") -> "Vec3":
        _check_vec3_same_frame(self, other, op="Vec3 - Vec3")
        return Vec3(self._mx - other._mx, frame=self._frame)

    def __neg__(self) -> "Vec3":
        return Vec3(-self._mx, frame=self._frame)

    def __mul__(self, scalar) -> "Vec3":
        return Vec3(self._mx * _as_mx(scalar), frame=self._frame)
    __rmul__ = __mul__

    def __truediv__(self, scalar) -> "Vec3":
        return Vec3(self._mx / _as_mx(scalar), frame=self._frame)

    # ---- Vector ops ------------------------------------------------------

    def dot(self, other: "Vec3") -> Scalar:
        _check_vec3_same_frame(self, other, op="Vec3.dot")
        return Scalar(ca.dot(self._mx, other._mx))

    def cross(self, other: "Vec3") -> "Vec3":
        _check_vec3_same_frame(self, other, op="Vec3.cross")
        return Vec3(ca.cross(self._mx, other._mx), frame=self._frame)

    def norm(self) -> Scalar:
        # CasADi: norm_2 returns a 1x1.
        return Scalar(ca.norm_2(self._mx))

    def normalize(self) -> "Vec3":
        # NB: division by zero produces NaN here. Callers can wrap in a
        # `where(norm > 0, ...)` if they need a safe variant.
        return Vec3(self._mx / ca.norm_2(self._mx), frame=self._frame)


def _check_vec3_same_frame(a: "Vec3", b: "Vec3", *, op: str) -> None:
    if not isinstance(b, Vec3):
        raise TypeError(f"{op}: rhs must be a Vec3, got {type(b).__name__}")
    if a._frame is not b._frame:
        raise FrameError(
            op,
            expected=f"both operands in {a._frame.__name__}",
            got=f"{a._frame.__name__} vs {b._frame.__name__}",
            source=_capture_user_source(),
        )


# ---------------------------------------------------------------------------
# Mat3
# ---------------------------------------------------------------------------

class Mat3(_IRValue):
    """A 3×3 matrix mapping `Vec3[B] → Vec3[A]` (i.e., `from_frame=A`, `to_frame=B`).

    Composition: `Mat3[A,B] @ Mat3[B,C] → Mat3[A,C]`.
    Application:  `Mat3[A,B] @ Vec3[B]  → Vec3[A]`.
    """

    _mx_shape: ClassVar[tuple[int, int]] = (3, 3)

    def __init__(self, mx, from_frame, to_frame):
        _validate_frame("Mat3 from_frame", from_frame)
        _validate_frame("Mat3 to_frame", to_frame)
        mx = _as_mx(mx)
        if mx.shape != (3, 3):
            raise ValueError(
                f"Mat3: expected shape (3,3), got {tuple(mx.shape)}")
        self._mx = mx
        self._from_frame = from_frame
        self._to_frame = to_frame

    def __class_getitem__(cls, frames):
        if not (isinstance(frames, tuple) and len(frames) == 2):
            raise TypeError(
                "Mat3[...]: requires two frame args, e.g. Mat3[A, B]")
        from_frame, to_frame = frames
        _validate_frame("Mat3[...]", from_frame)
        _validate_frame("Mat3[...]", to_frame)
        return _ParameterizedConstructor(
            cls, from_frame=from_frame, to_frame=to_frame)

    @classmethod
    def _from_mx(cls, mx, *, from_frame, to_frame):
        return cls(mx, from_frame=from_frame, to_frame=to_frame)

    @classmethod
    def _make_constant(cls, value, *, from_frame, to_frame):
        import numpy as np
        arr = np.asarray(value, dtype=float)
        if arr.shape != (3, 3):
            raise ValueError(
                f"Mat3.constant: expected 3×3 array-like, got shape {arr.shape}")
        return cls(ca.MX(ca.DM(arr)), from_frame=from_frame, to_frame=to_frame)

    # ---- Properties -----------------------------------------------------

    @property
    def from_frame(self):
        return self._from_frame
    @property
    def to_frame(self):
        return self._to_frame

    def __repr__(self) -> str:
        return (f"<Mat3[{self._from_frame.__name__}, "
                f"{self._to_frame.__name__}] shape={self.shape}>")

    # ---- Ops ------------------------------------------------------------

    def __matmul__(self, other):
        if isinstance(other, Mat3):
            # Mat3[A,B] @ Mat3[B,C] → Mat3[A,C]
            if self._to_frame is not other._from_frame:
                raise FrameError(
                    "Mat3 @ Mat3",
                    expected=f"middle frames match "
                             f"({self._to_frame.__name__} == "
                             f"{other._from_frame.__name__})",
                    got=f"{self._to_frame.__name__} vs {other._from_frame.__name__}",
                    source=_capture_user_source(),
                )
            return Mat3(self._mx @ other._mx,
                        from_frame=self._from_frame,
                        to_frame=other._to_frame)
        if isinstance(other, Vec3):
            # Mat3[A,B] @ Vec3[B] → Vec3[A]
            if self._to_frame is not other._frame:
                raise FrameError(
                    "Mat3 @ Vec3",
                    expected=f"to_frame matches vec frame "
                             f"({self._to_frame.__name__})",
                    got=f"to_frame={self._to_frame.__name__}, "
                        f"vec_frame={other._frame.__name__}",
                    source=_capture_user_source(),
                )
            return Vec3(self._mx @ other._mx, frame=self._from_frame)
        raise TypeError(
            f"Mat3 @ {type(other).__name__}: unsupported rhs.")

    def __mul__(self, scalar):
        return Mat3(self._mx * _as_mx(scalar),
                    from_frame=self._from_frame, to_frame=self._to_frame)
    __rmul__ = __mul__

    def transpose(self):
        """Mat3[A,B] → Mat3[B,A]."""
        return Mat3(self._mx.T,
                    from_frame=self._to_frame, to_frame=self._from_frame)

    def inv(self):
        """Mat3[A,A] → Mat3[A,A]. Frames must match (only same-frame
        matrices are square endomorphisms with sensible inverse semantics)."""
        if self._from_frame is not self._to_frame:
            raise FrameError(
                "Mat3.inv",
                expected="from_frame == to_frame (endomorphism)",
                got=f"{self._from_frame.__name__} vs {self._to_frame.__name__}",
                source=_capture_user_source(),
            )
        return Mat3(ca.inv(self._mx),
                    from_frame=self._from_frame, to_frame=self._to_frame)


# ---------------------------------------------------------------------------
# Quat — unit quaternion as a rotation Frame_To → Frame_From
# ---------------------------------------------------------------------------

class Quat(_IRValue):
    """Unit quaternion representing a rotation from one frame to another.

    Convention: a `Quat[From, To]`,
    when applied to a vector in `To`, returns the vector expressed in `From`
    components::

        q : Quat[From, To]
        v_in_to : Vec3[To]
        v_in_from = q.apply(v_in_to)    # → Vec3[From]

    Composition `Quat[A, B] * Quat[B, C] → Quat[A, C]`.

    Storage: (w, x, y, z) as a 4×1 MX. Normalization is the caller's
    responsibility (use `.normalize()` after numerical updates).
    """

    _mx_shape: ClassVar[tuple[int, int]] = (4, 1)

    def __init__(self, mx, from_frame, to_frame):
        _validate_frame("Quat from_frame", from_frame)
        _validate_frame("Quat to_frame", to_frame)
        mx = _as_mx(mx)
        if mx.shape != (4, 1):
            raise ValueError(
                f"Quat: expected shape (4,1), got {tuple(mx.shape)}")
        self._mx = mx
        self._from_frame = from_frame
        self._to_frame = to_frame

    def __class_getitem__(cls, frames):
        if not (isinstance(frames, tuple) and len(frames) == 2):
            raise TypeError(
                "Quat[...]: requires two frame args, e.g. Quat[From, To]")
        from_frame, to_frame = frames
        _validate_frame("Quat[...]", from_frame)
        _validate_frame("Quat[...]", to_frame)
        return _ParameterizedConstructor(
            cls, from_frame=from_frame, to_frame=to_frame)

    @classmethod
    def _from_mx(cls, mx, *, from_frame, to_frame):
        return cls(mx, from_frame=from_frame, to_frame=to_frame)

    @classmethod
    def _make_constant(cls, value, *, from_frame, to_frame):
        if len(value) != 4:
            raise ValueError(
                f"Quat.constant: expected (w,x,y,z) length-4, got len={len(value)}")
        mx = ca.MX(ca.DM([float(v) for v in value]))
        return cls(mx, from_frame=from_frame, to_frame=to_frame)

    @classmethod
    def identity(cls, *, from_frame, to_frame):
        return cls._make_constant((1.0, 0.0, 0.0, 0.0),
                                   from_frame=from_frame, to_frame=to_frame)

    # ---- Properties -----------------------------------------------------

    @property
    def from_frame(self):
        return self._from_frame
    @property
    def to_frame(self):
        return self._to_frame

    @property
    def w(self) -> Scalar: return Scalar(self._mx[0])
    @property
    def vx(self) -> Scalar: return Scalar(self._mx[1])
    @property
    def vy(self) -> Scalar: return Scalar(self._mx[2])
    @property
    def vz(self) -> Scalar: return Scalar(self._mx[3])

    def __repr__(self) -> str:
        return (f"<Quat[{self._from_frame.__name__}, "
                f"{self._to_frame.__name__}] shape={self.shape}>")

    # ---- Ops ------------------------------------------------------------

    def __mul__(self, other: "Quat") -> "Quat":
        """Compose: Quat[A,B] * Quat[B,C] → Quat[A,C]."""
        if not isinstance(other, Quat):
            raise TypeError(
                f"Quat * {type(other).__name__}: rhs must be a Quat.")
        if self._to_frame is not other._from_frame:
            raise FrameError(
                "Quat * Quat",
                expected=f"middle frames match "
                         f"({self._to_frame.__name__} == "
                         f"{other._from_frame.__name__})",
                got=f"{self._to_frame.__name__} vs {other._from_frame.__name__}",
                source=_capture_user_source(),
            )
        return Quat(quat_mul(self._mx, other._mx),
                    from_frame=self._from_frame, to_frame=other._to_frame)

    def conjugate(self) -> "Quat":
        """Quat[A,B] → Quat[B,A]. For unit quaternions this is the inverse."""
        return Quat(
            quat_conj(self._mx),
            from_frame=self._to_frame, to_frame=self._from_frame,
        )

    def apply(self, vec: Vec3) -> Vec3:
        """q.apply(v_in_To) → v_in_From. Standard quaternion-vector product."""
        if not isinstance(vec, Vec3):
            raise TypeError(
                f"Quat.apply: arg must be a Vec3, got {type(vec).__name__}")
        if vec._frame is not self._to_frame:
            raise FrameError(
                "Quat.apply",
                expected=f"vec frame matches Quat to_frame "
                         f"({self._to_frame.__name__})",
                got=f"vec_frame={vec._frame.__name__}, "
                    f"to_frame={self._to_frame.__name__}",
                source=_capture_user_source(),
            )
        # Build the rotation matrix and apply. CasADi's CSE collapses the
        # repeated subterms; clearer than the direct sandwich formula.
        R = self._rotmat_mx()
        return Vec3(R @ vec._mx, frame=self._from_frame)

    def to_rotmat(self) -> Mat3:
        """Return the rotation matrix Mat3[From, To]."""
        return Mat3(self._rotmat_mx(),
                    from_frame=self._from_frame, to_frame=self._to_frame)

    def normalize(self) -> "Quat":
        n = ca.norm_2(self._mx)
        return Quat(self._mx / n,
                    from_frame=self._from_frame, to_frame=self._to_frame)

    # ---- Internal: rotation matrix from quaternion (column-major) -------

    def _rotmat_mx(self) -> ca.MX:
        # Standard quaternion → R formula. Assumes unit quaternion; the
        # caller is responsible for keeping ‖q‖ ≈ 1.
        return quat_to_rotmat(self._mx)

"""Named-op dispatch — the 30-op vocabulary.

Most ops are thin wrappers around methods on the IR value classes. This
module exists so backends and simplifiers have a single named entry point
per op (instead of relying on Python operator dunders).

A user generally doesn't import from here — operator overloads on
Scalar/Vec3/Mat3/Quat cover ergonomic authoring. Backends will walk a
CasADi expression graph directly, so they bypass this layer entirely.
"""

from __future__ import annotations

import casadi as ca

from .types import Mat3, Quat, Scalar, Vec3, _as_mx


# ---- Scalar arithmetic ----------------------------------------------------

def add(a, b):          return Scalar(_as_mx(a) + _as_mx(b))
def sub(a, b):          return Scalar(_as_mx(a) - _as_mx(b))
def mul(a, b):          return Scalar(_as_mx(a) * _as_mx(b))
def div(a, b):          return Scalar(_as_mx(a) / _as_mx(b))
def neg(a):             return Scalar(-_as_mx(a))


# ---- Math -----------------------------------------------------------------

def sqrt(a):            return Scalar(ca.sqrt(_as_mx(a)))
def exp(a):             return Scalar(ca.exp(_as_mx(a)))
def log(a):             return Scalar(ca.log(_as_mx(a)))
def sin(a):             return Scalar(ca.sin(_as_mx(a)))
def cos(a):             return Scalar(ca.cos(_as_mx(a)))
def atan2(y, x):        return Scalar(ca.atan2(_as_mx(y), _as_mx(x)))
def abs(a):             return Scalar(ca.fabs(_as_mx(a)))
def maximum(a, b):      return Scalar(ca.fmax(_as_mx(a), _as_mx(b)))
def minimum(a, b):      return Scalar(ca.fmin(_as_mx(a), _as_mx(b)))


def clamp(x, lo, hi):
    """Saturate x to [lo, hi]. Branchless: piecewise via min/max."""
    return Scalar(ca.fmin(ca.fmax(_as_mx(x), _as_mx(lo)), _as_mx(hi)))


# ---- Conditional select (branchless `if`) --------------------------------

def where(cond, if_true, if_false):
    """Element-wise branchless select. `cond` must be a Scalar (interpreted
    as cond > 0 = true). `if_true` and `if_false` must be the same IR type."""
    if type(if_true) is not type(if_false):
        raise TypeError(
            f"where: branches must have the same type, got "
            f"{type(if_true).__name__} vs {type(if_false).__name__}")
    mx_cond = _as_mx(cond)
    mx_t    = _as_mx(if_true)
    mx_f    = _as_mx(if_false)
    chosen  = ca.if_else(mx_cond > 0, mx_t, mx_f)

    # Preserve frame metadata from the (matching) branches.
    if isinstance(if_true, Scalar):
        return Scalar(chosen)
    if isinstance(if_true, Vec3):
        return Vec3(chosen, frame=if_true._frame)
    if isinstance(if_true, Mat3):
        return Mat3(chosen,
                    from_frame=if_true._from_frame,
                    to_frame=if_true._to_frame)
    if isinstance(if_true, Quat):
        return Quat(chosen,
                    from_frame=if_true._from_frame,
                    to_frame=if_true._to_frame)
    raise TypeError(f"where: unsupported IR type {type(if_true).__name__}")


# ---- Vec3 ops --------------------------------------------------------------

def vec3_add(a: Vec3, b: Vec3) -> Vec3:        return a + b
def vec3_sub(a: Vec3, b: Vec3) -> Vec3:        return a - b
def vec3_scale(v: Vec3, s) -> Vec3:            return v * s
def vec3_dot(a: Vec3, b: Vec3) -> Scalar:      return a.dot(b)
def vec3_cross(a: Vec3, b: Vec3) -> Vec3:      return a.cross(b)
def vec3_norm(v: Vec3) -> Scalar:              return v.norm()
def vec3_normalize(v: Vec3) -> Vec3:           return v.normalize()


# ---- Mat3 ops --------------------------------------------------------------

def mat3_matmul(a: Mat3, b: Mat3) -> Mat3:     return a @ b
def mat3_apply(m: Mat3, v: Vec3) -> Vec3:      return m @ v
def mat3_transpose(m: Mat3) -> Mat3:           return m.transpose()
def mat3_inv(m: Mat3) -> Mat3:                 return m.inv()


# ---- Quat ops --------------------------------------------------------------

def q_mul(a: Quat, b: Quat) -> Quat:           return a * b
def q_conj(q: Quat) -> Quat:                   return q.conjugate()
def q_apply(q: Quat, v: Vec3) -> Vec3:         return q.apply(v)
def q_to_rotmat(q: Quat) -> Mat3:              return q.to_rotmat()
def q_normalize(q: Quat) -> Quat:              return q.normalize()


# ---- Manifold ops ---------------------------------------------------------

def so3_exp(omega: Vec3):
    """SO(3) exponential: tangent → Quat (endomorphism on omega's frame)."""
    from .manifold import _so3_exp
    return Quat(_so3_exp(omega._mx),
                from_frame=omega._frame, to_frame=omega._frame)


def so3_log(q: Quat) -> Vec3:
    """SO(3) logarithm: Quat → tangent vector in from_frame."""
    if q._from_frame is not q._to_frame:
        from .frames import FrameError, _capture_user_source
        raise FrameError(
            "so3_log",
            expected="endomorphism (from_frame == to_frame)",
            got=f"{q._from_frame.__name__} → {q._to_frame.__name__}",
            source=_capture_user_source(),
        )
    from .manifold import _so3_log
    return Vec3(_so3_log(q._mx), frame=q._from_frame)


def boxplus(state, *deltas):
    """Manifold boxplus. Dispatches on the state type's `boxplus` method."""
    return state.boxplus(*deltas)


def boxminus(a, b):
    """Manifold boxminus. Dispatches on `a.boxminus(b)`."""
    return a.boxminus(b)


# ---- Op-name registry ------------------------------------------------------
#
# A flat enumeration of the 30 supported ops, for backends / debug tooling
# to introspect. Not used at trace time — operator overloads call CasADi
# directly — but useful for documentation and future graph walks.

OP_NAMES: tuple[str, ...] = (
    # arithmetic (5)
    "add", "sub", "mul", "div", "neg",
    # math (10)
    "sqrt", "exp", "log", "sin", "cos", "atan2",
    "abs", "maximum", "minimum", "clamp",
    # control (1)
    "where",
    # vec3 (7)
    "vec3_add", "vec3_sub", "vec3_scale",
    "vec3_dot", "vec3_cross", "vec3_norm", "vec3_normalize",
    # mat3 (4)
    "mat3_matmul", "mat3_apply", "mat3_transpose", "mat3_inv",
    # quat (5)
    "q_mul", "q_conj", "q_apply", "q_to_rotmat", "q_normalize",
    # manifold (4)
    "so3_exp", "so3_log", "boxplus", "boxminus",
)
# Sanity: 5 + 10 + 1 + 7 + 4 + 5 + 4 = 36 distinct entry points exposed —
# more than the 30-op gut estimate, but several map to the same underlying
# CasADi primitive (add/sub/mul/div for scalars vs vectors). The number
# that actually need backend implementations is closer to 30.

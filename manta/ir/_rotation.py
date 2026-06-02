"""Rotation kernels — the single source of truth for SO(3) / quaternion math.

Pure CasADi-MX and numpy functions, with **no dependency on the rest of
`manta`** (not even `ir.types` or `ir.frames`). That is deliberate: every
other rotation-bearing module — `ir.manifold`, `ir.types`, `kinematics`,
`inertia` — imports its math from here, so this module must sit *below* all
of them in the import graph (it also breaks the otherwise-circular
`manifold ↔ types` edge).

Two families:

  * MX kernels (`so3_exp`, `so3_log`, `quat_mul`, `quat_conj`,
    `quat_from_axis_angle`, `rotate_vec_by_quat`, `R_from_axis_angle`,
    `quat_to_rotmat`) — operate on raw `ca.MX` columns. Frame-blind: the
    callers that need frame-checking (the value-typed `Quat`/`Vec3` ops)
    wrap these and supply the tags.
  * numpy twins (`so3_exp_np`, `quat_mul_np`) — for the numeric ESKF
    update path, which applies a Kalman correction on flat arrays.

`so3_exp` / `so3_log` are branch-free (eps-regularized rather than
`if_else`) because CasADi's autodiff walks *both* branches of a
conditional, and the non-Taylor branch carried `d(sin(0)/0)/dω = NaN`
that poisoned every Jacobian extracted from a boxplus-using tick.
"""

from __future__ import annotations

import casadi as ca
import numpy as np

_EPS_SQ = 1.0e-30


# ---------------------------------------------------------------------------
# SO(3) exp / log
# ---------------------------------------------------------------------------

def so3_exp(omega_mx) -> ca.MX:
    """Map a 3-vector to a unit quaternion via the SO(3) exponential.

    Branch-free, smooth everywhere. A tiny epsilon under the sqrt keeps the
    denominator off zero. For ω=0 the factor `sin(eps/2)/eps` → ~0.5 (since
    sin(x) ≈ x for tiny x); for nonzero ω the eps is dominated by |ω|² and
    the result is exact to roundoff.
    """
    theta = ca.sqrt(ca.dot(omega_mx, omega_mx) + _EPS_SQ)
    half_theta = theta * 0.5
    w = ca.cos(half_theta)
    s_over_theta = ca.sin(half_theta) / theta     # → 0.5 at ω=0 (smooth)
    v = omega_mx * s_over_theta
    return ca.vertcat(w, v[0], v[1], v[2])


def so3_log(q_mx) -> ca.MX:
    """Map a unit quaternion to its 3-vector tangent.

    Branch-free via the same eps-regularization as `so3_exp`. The composite
    `2·atan2(|v|, w) / |v|` is smooth at the identity quaternion
    (q = (1, 0, 0, 0)): numerator and denominator both scale as |v| there,
    so the ratio approaches the well-defined limit 2.
    """
    w = q_mx[0]
    v = ca.vertcat(q_mx[1], q_mx[2], q_mx[3])
    v_norm = ca.sqrt(ca.dot(v, v) + _EPS_SQ)
    # Always use the positive-w hemisphere to avoid the double cover.
    # sign(0) returns 0 in CasADi; offset slightly so sign(w=0) = +1.
    sign = ca.sign(w + _EPS_SQ)
    w_pos = w * sign
    v_pos = v * sign
    coeff = 2.0 * ca.atan2(v_norm, w_pos) / v_norm
    return v_pos * coeff


# ---------------------------------------------------------------------------
# Quaternion algebra
# ---------------------------------------------------------------------------

def quat_mul(a_mx, b_mx) -> ca.MX:
    """Hamilton product of two (w, x, y, z) quaternion MX columns."""
    w1, x1, y1, z1 = a_mx[0], a_mx[1], a_mx[2], a_mx[3]
    w2, x2, y2, z2 = b_mx[0], b_mx[1], b_mx[2], b_mx[3]
    return ca.vertcat(
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    )


def quat_conj(q_mx) -> ca.MX:
    """Conjugate of a (w, x, y, z) quaternion MX column (= inverse for unit
    quaternions)."""
    return ca.vertcat(q_mx[0], -q_mx[1], -q_mx[2], -q_mx[3])


def quat_from_axis_angle(axis_mx, angle_mx) -> ca.MX:
    """(w, x, y, z) MX quaternion from a length-3 axis MX and a scalar angle
    MX. The axis is normalized symbolically with an eps softener; a zero
    axis gives identity (within float rounding)."""
    n = ca.sqrt(ca.dot(axis_mx, axis_mx) + _EPS_SQ)
    axis_unit = axis_mx / n
    half = 0.5 * angle_mx
    w = ca.cos(half)
    s = ca.sin(half)
    return ca.vertcat(w, s * axis_unit[0], s * axis_unit[1], s * axis_unit[2])


# ---------------------------------------------------------------------------
# Quaternion → rotation matrix, and vector rotation
# ---------------------------------------------------------------------------

def quat_to_rotmat(q_mx) -> ca.MX:
    """3×3 rotation matrix from a (w, x, y, z) quaternion MX. Assumes a unit
    quaternion (the caller keeps ‖q‖ ≈ 1)."""
    w, x, y, z = q_mx[0], q_mx[1], q_mx[2], q_mx[3]
    r00 = 1 - 2 * (y * y + z * z)
    r01 = 2 * (x * y - w * z)
    r02 = 2 * (x * z + w * y)
    r10 = 2 * (x * y + w * z)
    r11 = 1 - 2 * (x * x + z * z)
    r12 = 2 * (y * z - w * x)
    r20 = 2 * (x * z - w * y)
    r21 = 2 * (y * z + w * x)
    r22 = 1 - 2 * (x * x + y * y)
    return ca.vertcat(
        ca.horzcat(r00, r01, r02),
        ca.horzcat(r10, r11, r12),
        ca.horzcat(r20, r21, r22),
    )


def rotate_vec_by_quat(q_mx, v_mx) -> ca.MX:
    """Rotate a 3-vector `v_mx` by the quaternion `q_mx = (w, x, y, z)` via
    the rotation matrix R(q) (CasADi CSE collapses the shared subterms)."""
    return quat_to_rotmat(q_mx) @ v_mx


def R_from_axis_angle(axis_mx, angle_mx) -> ca.MX:
    """Rodrigues' rotation matrix for an MX axis + angle. Eps-softens the
    axis norm so a degenerate (zero) axis doesn't crash. Accepts a Python
    list or a 3-element / 3×1 MX axis."""
    if isinstance(axis_mx, list):
        axis_mx = ca.MX(axis_mx)
    if axis_mx.shape == (3,):
        axis_mx = ca.reshape(axis_mx, 3, 1)
    n = ca.sqrt(ca.mtimes(axis_mx.T, axis_mx) + _EPS_SQ)
    u = axis_mx / n
    c = ca.cos(angle_mx)
    s = ca.sin(angle_mx)
    ux, uy, uz = u[0], u[1], u[2]
    zero = ca.MX(0.0)
    K = ca.vertcat(
        ca.horzcat(zero, -uz, uy),
        ca.horzcat(uz, zero, -ux),
        ca.horzcat(-uy, ux, zero),
    )
    return ca.MX.eye(3) + s * K + (1.0 - c) * ca.mtimes(K, K)


# ---------------------------------------------------------------------------
# numpy twins (numeric ESKF update path)
# ---------------------------------------------------------------------------

def so3_exp_np(omega: np.ndarray) -> np.ndarray:
    """numpy `so3_exp`: 3-vector → unit quaternion (w, x, y, z)."""
    omega = np.asarray(omega, dtype=float)
    theta = float(np.linalg.norm(omega) + 1e-30)
    half = 0.5 * theta
    w = np.cos(half)
    v = omega * (np.sin(half) / theta)
    return np.array([w, v[0], v[1], v[2]])


def quat_mul_np(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """numpy Hamilton product of two (w, x, y, z) quaternions."""
    w1, x1, y1, z1 = np.asarray(a, dtype=float)
    w2, x2, y2, z2 = np.asarray(b, dtype=float)
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])

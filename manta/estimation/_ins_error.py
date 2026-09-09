"""Gravity-referenced finite error coordinates for nonlinear strapdown INS.

The zero-error differential retains Manta's physical tangent layout. Finite
errors instead keep rotation about the reference vertical explicit, express
navigation vectors through the rotated frame, and retain the curvature between
attitude and sensor biases. The covariance reset is differentiated from this
same retraction, including every off-diagonal block.

For relative orientation D = Z(psi) S(tilt), theta = tilt + psi * up:

  v_true = D (v + delta_v - theta x v)
  bg_true = bg + R_sensor.T S.T Jr(psi * up) R_sensor delta_bg
  ba_true = ba + delta_ba + R_sensor.T (g - S.T g - tilt x g)

Here g is the *reference specific-force vector*, opposite effective gravity.
The last expression makes the stationary gravity observation affine in tilt
and accelerometer-bias error. The reference is a coordinate choice; it does
not replace gravity or Earth rotation in the process model.

A physical Gaussian prior must be mapped into these coordinates. Covariance
and error scoring refer to boxminus(truth, estimate), which is not generally
antisymmetric. These are local coordinates: the swing/twist decomposition is
singular at a 180-degree relative swing and has the usual angular branch cut.
"""

from functools import cached_property

import casadi as ca
import numpy as np

from ..ir._rotation import quat_conj, quat_mul, quat_to_rotmat, so3_exp, so3_log
from ..ir.state_spec import StateSpec
from ._kalman import _so3_reset_jacobian


def _ambient(slot):
    return slice(slot.ambient_offset, slot.ambient_offset + slot.ambient_dim)


def _tangent(slot):
    return slice(slot.tangent_offset, slot.tangent_offset + slot.tangent_dim)


def _left_inverse(theta):
    t2 = ca.dot(theta, theta)
    t = ca.sqrt(t2 + 1e-30)
    coefficient = ca.if_else(
        t2 < 1e-8,
        1 / 12 + t2 / 720 + t2 * t2 / 30240,
        (1 - 0.5 * t * ca.cos(0.5 * t) / ca.sin(0.5 * t)) / (t2 + 1e-30),
    )
    k = ca.skew(theta)
    return ca.MX.eye(3) - 0.5 * k + coefficient * k @ k


class INSStateSpec(StateSpec):
    error_model = "gravity_referenced_swing_twist_v1"

    def __init__(
        self, product, *, craft, imu, rotation_body_from_imu, reference_specific_force
    ):
        super().__init__(list(product.slots))
        self.product_spec = product
        self.orientation = self.slot(f"{craft}.orientation")
        self.navigation_vectors = tuple(
            self.slot(f"{craft}.{name}") for name in ("position", "velocity")
        )
        self.biases = tuple(
            self.slot(f"{imu}.{name}")
            for name in ("gyro_bias", "accel_bias")
            if f"{imu}.{name}" in self
        )
        self.mount = ca.DM(rotation_body_from_imu)
        reference = np.asarray(reference_specific_force, dtype=float).reshape(3)
        magnitude = np.linalg.norm(reference)
        if not np.all(np.isfinite(reference)) or magnitude == 0:
            raise ValueError(
                "nonlinear INS requires a finite, nonzero gravity reference"
            )
        self.reference_specific_force = tuple(float(v) for v in reference)
        self.gravity = ca.DM(reference)
        self.up = ca.DM(reference / magnitude)

    def _plus(self, x, d):
        out = self.product_spec.boxplus_sym(x, d)
        theta = d[_tangent(self.orientation)]
        twist_vector = self.up * ca.dot(self.up, theta)
        tilt = theta - twist_vector
        swing = so3_exp(tilt)
        Dq = quat_mul(so3_exp(twist_vector), swing)
        D = quat_to_rotmat(Dq)
        out[_ambient(self.orientation)] = quat_mul(Dq, x[_ambient(self.orientation)])
        for slot in self.navigation_vectors:
            value = x[_ambient(slot)]
            out[_ambient(slot)] = D @ (
                value + d[_tangent(slot)] - ca.cross(theta, value)
            )
        sensor = quat_to_rotmat(x[_ambient(self.orientation)]) @ self.mount
        S = quat_to_rotmat(swing)
        B = sensor.T @ S.T @ _so3_reset_jacobian(-twist_vector, symbolic=True) @ sensor
        offset = sensor.T @ (
            self.gravity - S.T @ self.gravity - ca.cross(tilt, self.gravity)
        )
        for slot in self.biases:
            increment = (
                offset + d[_tangent(slot)]
                if slot.name.endswith("accel_bias")
                else B @ d[_tangent(slot)]
            )
            out[_ambient(slot)] = x[_ambient(slot)] + increment
        # A local Gaussian must not wrap sigma points across the chart branch
        # and silently turn a broad heading distribution into a narrow one.
        # This guard is emitted into every backend along with the retraction.
        valid = ca.logic_and(
            ca.dot(twist_vector, twist_vector) < np.pi**2, ca.dot(tilt, tilt) < np.pi**2
        )
        return ca.if_else(valid, out, ca.MX.nan(self.ambient_dim, 1))

    def _minus(self, a, b):
        out = self.product_spec.boxminus_sym(a, b)
        q = quat_mul(
            a[_ambient(self.orientation)], quat_conj(b[_ambient(self.orientation)])
        )
        q = ca.if_else(q[0] < 0, -q, q)
        psi = 2 * ca.atan2(ca.dot(q[1:4], self.up), q[0])
        twist_vector = psi * self.up
        swing = quat_mul(so3_exp(-twist_vector), q)
        tilt = so3_log(swing)
        tilt -= self.up * ca.dot(self.up, tilt)
        theta = tilt + twist_vector
        out[_tangent(self.orientation)] = theta
        D = quat_to_rotmat(q)
        for slot in self.navigation_vectors:
            value = b[_ambient(slot)]
            out[_tangent(slot)] = (
                D.T @ a[_ambient(slot)] - value + ca.cross(theta, value)
            )
        sensor = quat_to_rotmat(b[_ambient(self.orientation)]) @ self.mount
        S = quat_to_rotmat(swing)
        inverse_B = sensor.T @ _left_inverse(-twist_vector) @ S @ sensor
        offset = sensor.T @ (
            self.gravity - S.T @ self.gravity - ca.cross(tilt, self.gravity)
        )
        for slot in self.biases:
            diff = a[_ambient(slot)] - b[_ambient(slot)]
            out[_tangent(slot)] = (
                diff - offset if slot.name.endswith("accel_bias") else inverse_B @ diff
            )
        return out

    @cached_property
    def plus(self):
        x = ca.MX.sym("x", self.ambient_dim)
        d = ca.MX.sym("d", self.tangent_dim)
        return ca.Function("ins_error_plus", [x, d], [self._plus(x, d)])

    @cached_property
    def minus(self):
        a = ca.MX.sym("a", self.ambient_dim)
        b = ca.MX.sym("b", self.ambient_dim)
        return ca.Function("ins_error_minus", [a, b], [self._minus(a, b)])

    def pack_projected(self, source):
        return self.product_spec.pack_projected(source)

    def boxplus_sym(self, x, d):
        return self.plus(x, d)

    def boxminus_sym(self, a, b):
        return self.minus(a, b)

    def boxplus_num(self, x, d):
        return np.asarray(self.plus(x, d)).ravel()

    @cached_property
    def exact_reset(self):
        x = ca.MX.sym("x", self.ambient_dim)
        d = ca.MX.sym("d", self.tangent_dim)
        e = ca.MX.sym("e", self.tangent_dim)
        error = self.minus(self.plus(x, d + e), self.plus(x, d))
        G = ca.substitute(ca.jacobian(error, e), e, ca.MX.zeros(self.tangent_dim))
        return ca.Function("ins_error_reset", [x, d], [G])

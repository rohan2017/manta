"""Coupled navigation/bias retraction and covariance transport for INS.

The ambient state and differential tangent units/frames are unchanged. Finite
errors are coupled: navigation vectors rotate with attitude and sensor-frame
bias increments integrate through that rotation. The first derivative of
boxplus at zero is the product StateSpec derivative, so the existing local
strapdown F, L, and measurement H remain valid.

Covariance is transported in the navigation/bias coordinates, not reset as
an independent quaternion plus additive vectors. See the INS error-model
explanation for the coordinate map and the finite-error/prior contract.
"""

from functools import cached_property, lru_cache

import casadi as ca
import numpy as np

from manta.ir._rotation import quat_to_rotmat, so3_exp
from manta.ir.state_spec import StateSpec
from manta.estimation._kalman import _skew, _so3_reset_jacobian


def _left_inverse(theta):
    """SO(3) left-Jacobian inverse, regular at zero and at pi."""
    t2 = ca.dot(theta, theta)
    t = ca.sqrt(t2 + 1e-30)
    coefficient = ca.if_else(
        t2 < 1e-8,
        1 / 12 + t2 / 720 + t2 * t2 / 30240,
        (1 - 0.5 * t * ca.cos(0.5 * t) / ca.sin(0.5 * t)) / (t2 + 1e-30),
    )
    k = _skew(theta, symbolic=True)
    return ca.MX.eye(3) - 0.5 * k + coefficient * (k @ k)


def _ambient(slot):
    return slice(slot.ambient_offset, slot.ambient_offset + slot.ambient_dim)


def _tangent(slot):
    return slice(slot.tangent_offset, slot.tangent_offset + slot.tangent_dim)


@lru_cache(maxsize=1)
def _right_cross():
    theta=ca.MX.sym('theta',3);velocity=ca.MX.sym('velocity',3)
    right=_so3_reset_jacobian(-theta,symbolic=True)
    cross=ca.reshape(ca.jtimes(ca.reshape(right,9,1),theta,velocity),3,3)
    return ca.Function('right_cross',[theta,velocity],[cross])


class INSStateSpec(StateSpec):
    """INS finite-error chart with the same differential layout as StateSpec.

    Covariance is local to the estimate. Score errors as boxminus(truth,
    estimate); finite boxminus is not antisymmetric for this chart. Its
    Euclidean-slot entries are coupled error coordinates, not independent
    ambient differences. Other craft/part states retain their own manifolds.
    """

    error_model = "semidirect_exact_differential_reset"

    def __init__(self, product, *, craft, imu, rotation_body_from_imu):
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
        self.rotation_body_from_imu = ca.DM(rotation_body_from_imu)

    def _sensor_rotation(self, x):
        return quat_to_rotmat(x[_ambient(self.orientation)]) @ self.rotation_body_from_imu

    def boxplus_sym(self, x_ambient, delta_tangent):
        out = self.product_spec.boxplus_sym(x_ambient, delta_tangent)
        theta = delta_tangent[_tangent(self.orientation)]
        rotation = quat_to_rotmat(so3_exp(theta))
        left = _so3_reset_jacobian(theta, symbolic=True)
        for slot in self.navigation_vectors:
            value = x_ambient[_ambient(slot)]
            delta = delta_tangent[_tangent(slot)]
            out[_ambient(slot)] = rotation @ value + left @ (
                delta - ca.cross(theta, value)
            )
        sensor = self._sensor_rotation(x_ambient)
        right = _so3_reset_jacobian(-theta, symbolic=True)
        for slot in self.biases:
            out[_ambient(slot)] = x_ambient[_ambient(slot)] + (
                sensor.T @ right @ sensor @ delta_tangent[_tangent(slot)]
            )
        if len(self.biases)==2:
            gyro,accel=self.biases
            velocity=self.navigation_vectors[1]
            cross=_right_cross()(theta,delta_tangent[_tangent(velocity)])
            out[_ambient(accel)]+=sensor.T@cross@sensor@delta_tangent[_tangent(gyro)]
        return out

    def boxminus_sym(self, x_a, x_b):
        out = self.product_spec.boxminus_sym(x_a, x_b)
        theta = out[_tangent(self.orientation)]
        rotation = quat_to_rotmat(so3_exp(theta))
        left_inverse = _left_inverse(theta)
        for slot in self.navigation_vectors:
            value = x_b[_ambient(slot)]
            out[_tangent(slot)] = left_inverse @ (
                x_a[_ambient(slot)] - rotation @ value
            ) + ca.cross(theta, value)
        sensor = self._sensor_rotation(x_b)
        right_inverse = _left_inverse(-theta)
        for slot in self.biases:
            out[_tangent(slot)] = sensor.T @ right_inverse @ sensor @ (
                x_a[_ambient(slot)] - x_b[_ambient(slot)]
            )
        if len(self.biases)==2:
            gyro,accel=self.biases
            velocity=self.navigation_vectors[1]
            cross=_right_cross()(theta,out[_tangent(velocity)])
            out[_tangent(accel)]-=sensor.T@right_inverse@cross@sensor@out[_tangent(gyro)]
        return out

    @cached_property
    def exact_reset(self):
        x = ca.MX.sym('x', self.ambient_dim)
        d = ca.MX.sym('d', self.tangent_dim)
        e = ca.MX.sym('e', self.tangent_dim)
        new = self.boxplus_sym(x, d)
        residual = self.boxminus_sym(self.boxplus_sym(x, d + e), new)
        jac = ca.substitute(ca.jacobian(residual, e), e, ca.MX.zeros(self.tangent_dim))
        return ca.Function('geometric_exact_reset', [x,d], [jac])

    @cached_property
    def _plus_function(self):
        x = ca.MX.sym("x", self.ambient_dim)
        delta = ca.MX.sym("delta", self.tangent_dim)
        return ca.Function("ins_boxplus", [x, delta], [self.boxplus_sym(x, delta)])

    def boxplus_num(self, x_ambient, delta_tangent):
        return np.asarray(self._plus_function(x_ambient, delta_tangent)).ravel()

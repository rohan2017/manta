"""Cartesian kinematics of a fixed, uniformly rotating navigation frame.

No geodesy lives here: the deployment boundary supplies the anchor and the
planet-axis projection. GravityField values are interpreted explicitly as
normal/effective gravity or gravitation. A moving local-level frame is not
this frame: it would require transport rate and different equations.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import casadi as ca

from ..ir._rotation import quat_mul, so3_exp

# Cubic rotation quadrature remainder <= exp(a)*a**4/24.
# a <= 1e-3 bounds the relative remainder below 4.2e-14.
MAX_PACKET_FRAME_ROTATION_RAD = 1e-3


@dataclass(frozen=True, kw_only=True)
class NavigationFrame:
    """Fixed Cartesian tangent frame rigidly attached to a rotating planet.

    ``angular_velocity`` is frame relative to inertial space, expressed in
    navigation axes (rad/s). ``origin_from_rotation_center`` is the vector
    from the planet rotation center to the anchor, in those same axes (m).
    ``frame_id`` and ``epoch`` identify the externally resolved Cartesian
    anchor/axes; they are retained in the hashed estimator metadata.

    With ``gravity_convention='effective'``, the supplied GravityField already
    includes centrifugal acceleration. With ``'gravitation'`` it does not,
    and INS subtracts omega x (omega x r) exactly once. Neither convention
    adds a vehicle transport rate. Constant effective gravity is an explicit
    local-patch approximation belonging to the caller's field declaration.
    """

    frame_id: str
    epoch: str
    angular_velocity: tuple[float, float, float]
    origin_from_rotation_center: tuple[float, float, float]
    gravity_convention: str

    def __post_init__(self):
        for name in ("frame_id", "epoch"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"NavigationFrame {name} must be a nonempty string")
        for name in ("angular_velocity", "origin_from_rotation_center"):
            value = tuple(float(v) for v in getattr(self, name))
            if len(value) != 3 or not all(math.isfinite(v) for v in value):
                raise ValueError(f"NavigationFrame {name} requires three finite values")
            object.__setattr__(self, name, value)
        if self.gravity_convention not in {"effective", "gravitation"}:
            raise ValueError("gravity_convention must be effective or gravitation")

    def metadata(self):
        return {
            "frame_definition": "fixed_planet_attached_cartesian",
            "frame_id": self.frame_id,
            "epoch": self.epoch,
            "angular_velocity": self.angular_velocity,
            "origin_from_rotation_center": self.origin_from_rotation_center,
            "gravity_convention": self.gravity_convention,
        }

    def effective_gravity(self, gravity, position):
        if self.gravity_convention == "effective":
            return gravity
        w = ca.DM(self.angular_velocity)
        r = position + ca.DM(self.origin_from_rotation_center)
        return gravity - ca.cross(w, ca.cross(w, r))

    def relative_rate(self, inertial_body_rate, rotation_nav_from_body):
        return inertial_body_rate - rotation_nav_from_body.T @ ca.DM(
            self.angular_velocity
        )

    def attitude(self, orientation, inertial_delta, dt):
        return quat_mul(
            so3_exp(-ca.DM(self.angular_velocity) * dt),
            quat_mul(orientation, inertial_delta),
        )

    def translation(
        self,
        position,
        velocity,
        gravity,
        delta_v,
        delta_p,
        dt,
        velocity_moments=None,
        position_moments=None,
    ):
        """Consume inertial force deltas expressed in start navigation axes.

        The packet's deterministic quadrature moments reconstruct the rotation
        of each left-held integration interval. Inverting those quadratures
        is exact for a constant navigation-frame force up to the fourth-order
        rotation remainder. For varying force it is a short-packet average;
        its error must be checked by packet-duration refinement. Covariance
        and bias sensitivities pass through these same linear maps by AD.

        Coriolis uses the implicit midpoint rule (norm preserving for an
        unforced velocity); raw and packet paths share this correction.
        """
        w = ca.DM(self.angular_velocity)
        W = ca.skew(w)
        if velocity_moments is not None:

            def undo_quadrature(moments, scale, value):
                # W**3 = -|w|**2 W: the cubic matrix is I+aW+bW².
                a = moments[1] / scale - ca.dot(w, w) * moments[3] / (6 * scale)
                b = moments[2] / (2 * scale)
                return _solve_rotation_polynomial(w, a, b, value)

            delta_v = undo_quadrature(velocity_moments, dt, delta_v)
            delta_p = undo_quadrature(position_moments, 0.5 * dt * dt, delta_p)
        v_next = _solve_rotation_polynomial(
            w, dt, 0, (ca.MX.eye(3) - W * dt) @ velocity + delta_v + gravity * dt
        )
        p_next = (
            position
            + velocity * dt
            + delta_p
            + 0.5 * gravity * dt * dt
            - 0.5 * ca.cross(w, velocity + v_next) * dt * dt
        )
        if velocity_moments is not None:
            valid = ca.dot(w, w) * dt * dt <= MAX_PACKET_FRAME_ROTATION_RAD**2
            for moments, scale in (
                (velocity_moments, dt),
                (position_moments, 0.5 * dt * dt),
            ):
                valid = ca.logic_and(valid, ca.fabs(moments[0] - scale) <= 1e-9 * scale)
                for j in range(1, 4):
                    valid = ca.logic_and(
                        valid,
                        ca.logic_and(
                            moments[j] >= 0, moments[j] <= scale * dt**j * (1 + 1e-9)
                        ),
                    )
            poison = ca.if_else(valid, 0, ca.MX.nan(1, 1))
            p_next += poison
            v_next += poison
        return p_next, v_next


def _solve_rotation_polynomial(w, a, b, value):
    """Solve (I+a[w]x+b[w]x²)y=value using scalar operations.

    Rodrigues' minimal polynomial avoids a pivoting linear-solver node, so
    this kernel supports the common C++, numpy, and SX/JAX lowering paths.
    """
    z = ca.dot(w, w)
    denominator = (1 - b * z) ** 2 + a * a * z
    return (
        value
        - (a / denominator) * ca.cross(w, value)
        + ((a * a - b + b * b * z) / denominator) * ca.cross(w, ca.cross(w, value))
    )

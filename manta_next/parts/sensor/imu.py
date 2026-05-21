"""IMU — gyro + accelerometer (body-frame specific force).

Outputs:
  * `gyro`  (Vec3[CraftFrame]) — body angular velocity ω (what a rate
                                  gyro reads).
  * `accel` (Vec3[CraftFrame]) — body-frame specific force at the IMU's
                                  mount point: f = a_imu_body − g_body.
                                  This is exactly what a real
                                  accelerometer measures: zero in free
                                  fall, +1g upward when stationary on
                                  the ground, plus any non-gravitational
                                  acceleration.

`gyro` is computed in `update()` (depends only on input state). `accel`
is computed in `post_update()` so it can read the just-computed
Newton-Euler outputs (body acceleration + angular acceleration) needed
for the lever-arm correction when the IMU is mounted at an offset:

    a_imu_anchor = a_origin_anchor + R · (α × r + ω × (ω × r))
    a_imu_body   = R^T · a_imu_anchor
    accel        = a_imu_body − g_body

For an IMU at the body origin (r=0), the lever-arm terms vanish and
this reduces to the standard `a_origin_body − g_body`.

Noise model (v1, additive white Gaussian, sampled externally):
  * `gyro_noise_sigma`  — rad/s, 1-σ on each axis.
  * `accel_noise_sigma` — m/s², 1-σ on each axis.

Noise is not injected into the symbolic graph — the IMU emits the
clean signal as the Output, and the user (or a sim wrapper) calls
`imu.sample_noise(rng)` per tick to get the noise vector to add. The
sigmas double as the EKF's R-matrix scaling: `imu.gyro_R` /
`imu.accel_R` return σ²·I3.
"""

from __future__ import annotations

import casadi as ca
import numpy as np

from ...ir.frames import CraftFrame
from ...ir.types import Vec3
from ..base import Output, Parameter, Part, PartUpdate
from ...math.wrench import Wrench


class IMU(Part):
    """Inertial-measurement unit.

    Parameters:
        gyro_noise_sigma  — additive white Gaussian noise standard
                            deviation on the gyro channel (rad/s).
                            Default 0 (clean sim).
        accel_noise_sigma — same for the accelerometer (m/s²).
                            Default 0.
    """

    gyro_noise_sigma:  float = Parameter(0.0)
    accel_noise_sigma: float = Parameter(0.0)

    gyro  = Output(shape="vec3")
    accel = Output(shape="vec3")

    # ----- Symbolic outputs ------------------------------------------------

    def update(self, ctx) -> PartUpdate:
        # `gyro` only needs the input state. `accel` waits for post_update
        # since it depends on the just-computed body acceleration.
        zero_v = Vec3[CraftFrame].constant((0.0, 0.0, 0.0))
        return PartUpdate(
            wrench=Wrench(force=zero_v, torque=zero_v),
            outputs={"gyro": ctx.angular_velocity},
        )

    def post_update(self, ctx_post) -> dict:
        # Lever-arm-corrected body-frame acceleration at the IMU mount.
        offset_craft = Vec3[CraftFrame].constant(tuple(self.transform))
        # Anchor-frame inertial acceleration at the IMU:
        #   a_imu_anchor = a_origin_anchor + R · (α × r + ω × (ω × r))
        omega = ctx_post.angular_velocity
        alpha = ctx_post.angular_acceleration
        lever = alpha.cross(offset_craft) + omega.cross(omega.cross(offset_craft))
        lever_anchor = ctx_post.orientation.apply(lever)
        a_imu_anchor = ctx_post.acceleration_anchor + lever_anchor
        # Body frame.
        a_imu_body = ctx_post.orientation.conjugate().apply(a_imu_anchor)
        # Specific force: inertial accel minus gravity (both body frame).
        specific_force = a_imu_body - ctx_post.gravity
        return {"accel": specific_force}

    # ----- Noise helpers (used by sim wrappers + EKF wiring) --------------

    def sample_noise(self, rng) -> dict:
        """Draw one tick of white Gaussian noise from `rng` (a
        numpy.random.Generator). Returns
            {"gyro": np.ndarray(3,), "accel": np.ndarray(3,)}
        which the caller adds to the clean Output readings.

        Returns zero vectors when both sigmas are 0 (no work, no
        spurious RNG draws so seeds stay reproducible)."""
        gs = float(self.gyro_noise_sigma)
        as_ = float(self.accel_noise_sigma)
        gyro = (rng.normal(0.0, gs, 3) if gs > 0.0 else np.zeros(3))
        acc  = (rng.normal(0.0, as_, 3) if as_ > 0.0 else np.zeros(3))
        return {"gyro": gyro, "accel": acc}

    @property
    def gyro_R(self) -> np.ndarray:
        """Measurement noise covariance for the gyro channel, σ²·I3.
        Plug into `EKF.update(..., R=imu.gyro_R)`."""
        return (float(self.gyro_noise_sigma) ** 2) * np.eye(3)

    @property
    def accel_R(self) -> np.ndarray:
        """Measurement noise covariance for the accelerometer channel."""
        return (float(self.accel_noise_sigma) ** 2) * np.eye(3)

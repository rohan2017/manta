"""IMU — gyro + accelerometer (body-frame specific force).

Outputs:
  * `gyro`  (Vec3[CraftFrame]) — body angular velocity ω + bias + white.
  * `accel` (Vec3[CraftFrame]) — body-frame specific force
                                  f = a_body − g_body, + bias + white.
                                  Zero in free fall; +1g upward when
                                  stationary on the ground.

Noise channels (Kalibr 4-parameter model — set σ to engage each):

  * `gyro_noise`       — vec3, white, per-tick rad/s.
  * `accel_noise`      — vec3, white, per-tick m/s².
  * `gyro_bias`        — vec3, RW, rad/s²/√Hz drift density.
                         The bias becomes an estimated state on the EKF.
  * `accel_bias`       — vec3, RW, m/s³/√Hz drift density.

Sigmas default to 0 (clean sim with no bias drift). Override on
construction: `IMU("imu", gyro_noise_sigma=0.005, gyro_bias_sigma=1e-4)`.

The framework substitutes the current-tick `a/α` into
`ctx.acceleration_body` at compile time, so the accel output is a
one-line subtraction — no fixed-point iteration, no auxiliary state,
no lag.
"""

from __future__ import annotations

from ...ir.frames import CraftFrame
from ...ir.types import Vec3
from ..base import Noise, Output, Part, PartUpdate
from ...ir.wrench import Wrench


class IMU(Part):
    """Inertial-measurement unit with Kalibr-style 4-parameter noise.

    Channels (override sigmas via construction):
        gyro_noise  — vec3 white, per-tick rad/s.
        accel_noise — vec3 white, per-tick m/s².
        gyro_bias   — vec3 RW,    rad/s²/√Hz drift density.
        accel_bias  — vec3 RW,    m/s³/√Hz  drift density.

    The two RW channels add bias state slots that the EKF can estimate;
    skip them by leaving sigma at 0.
    """

    gyro_noise  = Noise(shape="vec3", kind="white", frame=CraftFrame, sigma=0.0)
    accel_noise = Noise(shape="vec3", kind="white", frame=CraftFrame, sigma=0.0)
    gyro_bias   = Noise(shape="vec3", kind="rw",    frame=CraftFrame, sigma=0.0)
    accel_bias  = Noise(shape="vec3", kind="rw",    frame=CraftFrame, sigma=0.0)

    gyro  = Output(shape="vec3")
    accel = Output(shape="vec3")

    def update(self, ctx) -> PartUpdate:
        zero_v = Vec3[CraftFrame].constant((0.0, 0.0, 0.0))
        return PartUpdate(
            wrench=Wrench(force=zero_v, torque=zero_v),
            outputs={
                "gyro":  (ctx.angular_velocity
                          + self.gyro_bias
                          + self.gyro_noise),
                "accel": (ctx.acceleration_body - ctx.gravity
                          + self.accel_bias
                          + self.accel_noise),
            },
        )

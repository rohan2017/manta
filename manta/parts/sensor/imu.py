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

from ...fields import GravityField
from ...ir.frames import PartFrame, WorldFrame
from ...ir.types import Vec3
from ..base import Output, Part, PartUpdate, RandomWalkNoise, WhiteNoise
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

    gyro_noise  = WhiteNoise(     shape="vec3", frame=PartFrame, sigma=0.0)
    accel_noise = WhiteNoise(     shape="vec3", frame=PartFrame, sigma=0.0)
    gyro_bias   = RandomWalkNoise(shape="vec3", frame=PartFrame, sigma=0.0)
    accel_bias  = RandomWalkNoise(shape="vec3", frame=PartFrame, sigma=0.0)

    gyro  = Output(shape="vec3")
    accel = Output(shape="vec3")

    def update(self, ctx) -> PartUpdate:
        zero_v = Vec3[PartFrame].constant((0.0, 0.0, 0.0))
        # A real IMU measures inertial ω and specific force in its own case
        # frame. Read the inertial (WorldFrame) quantities and rotate into
        # the sensor's own frame via ctx.orientation — for a root-mounted IMU
        # that frame is the body frame; on a rotor it spins with the joint.
        # Gravity at the mount point folds to zero with no GravityField.
        # Bias/noise live in the sensor frame.
        g_world = ctx.field(GravityField).state_at_sym(
            ctx.position[WorldFrame], ctx.t)
        R_part_from_world = ctx.orientation.conjugate()
        omega_sensor = R_part_from_world.apply(ctx.angular_velocity[WorldFrame])
        accel_sensor = R_part_from_world.apply(
            ctx.acceleration[WorldFrame] - g_world)
        return PartUpdate(
            wrench=Wrench(force=zero_v, torque=zero_v),
            outputs={
                "gyro":  (omega_sensor
                          + self.gyro_bias
                          + self.gyro_noise),
                "accel": (accel_sensor
                          + self.accel_bias
                          + self.accel_noise),
            },
        )

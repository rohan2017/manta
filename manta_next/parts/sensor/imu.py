"""IMU — gyro + accelerometer (body-frame specific force).

Outputs:
  * `gyro`  (Vec3[CraftFrame]) — body angular velocity ω + per-tick
                                  Gaussian noise.
  * `accel` (Vec3[CraftFrame]) — body-frame specific force
                                  f = a_body − g_body, plus per-tick
                                  Gaussian noise. Zero in free fall;
                                  +1g upward when stationary on the
                                  ground.

Noise is declared via the `Noise(shape="vec3", sigma=...)` sentinel and
read inside `update()` as if it were a vector. The framework wires
each noise channel as a per-tick graph input — sim callers draw fresh
samples via `craft.sample_noise(rng)`; the EKF leaves the noise inputs
at zero and reads `imu.noise_R("gyro_noise")` / `imu.noise_R("accel_noise")`
for the measurement-noise covariance instead.

The framework substitutes the current-tick `a/α` into `ctx.acceleration_body`
at compile time, so the accel output is a one-line subtraction — no
fixed-point iteration, no auxiliary state, no lag.
"""

from __future__ import annotations

from ...ir.frames import CraftFrame
from ...ir.types import Vec3
from ..base import Noise, Output, Part, PartUpdate
from ...ir.wrench import Wrench


class IMU(Part):
    """Inertial-measurement unit.

    Noise:
        gyro_noise  — vec3 in body frame, σ rad/s.
        accel_noise — vec3 in body frame, σ m/s².

    Sigmas default to 0 (clean sim). Set via construction:
        `IMU("imu", gyro_noise=Noise(sigma=0.001))` — no, that won't
    work because `Noise(...)` is a class-attribute sentinel. Use a
    subclass or pass sigma at construction once the noise declaration
    supports overrides (TODO if needed).
    """

    gyro_noise  = Noise(shape="vec3", frame=CraftFrame, sigma=0.0)
    accel_noise = Noise(shape="vec3", frame=CraftFrame, sigma=0.0)

    gyro  = Output(shape="vec3")
    accel = Output(shape="vec3")

    def update(self, ctx) -> PartUpdate:
        zero_v = Vec3[CraftFrame].constant((0.0, 0.0, 0.0))
        return PartUpdate(
            wrench=Wrench(force=zero_v, torque=zero_v),
            outputs={
                "gyro":  ctx.angular_velocity + self.gyro_noise,
                "accel": ctx.acceleration_body - ctx.gravity + self.accel_noise,
            },
        )

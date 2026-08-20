"""Dynamics-predicted specific force as an ordinary pseudo-sensor.

``ModelForce`` is the measurement side of Manta's strapdown INS. It asks the
compiled world model for the specific force at its mount and adds one explicit
white model-error channel. The part has no estimator knowledge: EKF, UKF, INS,
NoiseFit, observability, and NEES all see the same ordinary ``Output``/``Noise``
contract.

Mount this part at the IMU proof-mass frame. The normal kinematic pass then
includes the complete rigid lever-arm acceleration, while actuator, fluid,
buoyancy, and estimable disturbance states reach the output through the
compiled wrench model.
"""

from __future__ import annotations

from ...fields import gravity_at
from ...ir.frames import PartFrame, WorldFrame
from ...ir.types import Vec3
from ...ir.wrench import Wrench
from .._declarations import Output, Parameter, PartUpdate, WhiteNoise
from ..base import Part, PartRole
from .imu import IMU


class ModelForce(Part):
    """Model-predicted sensor-frame specific force.

    ``model_error_sigma`` is the per-sample 1σ acceleration-model error in
    m/s². It must be nonzero when ``specific_force`` is selected by a filter,
    exactly like every other measurement-noise channel.
    """

    role = PartRole.SENSOR

    accelerometer = Parameter(None, numeric=False)
    rate: float | None = Parameter(None)
    model_error = WhiteNoise("R3", frame=PartFrame, sigma=0.0)
    specific_force = Output()

    def __init__(self, name: str, *, imu: IMU, **overrides) -> None:
        if not isinstance(imu, IMU):
            raise TypeError("ModelForce: imu must be an IMU part")
        # The pseudo-reading is sourced from this accelerometer, so its
        # default cadence must agree.  A caller may still override ``rate``
        # explicitly for a downsampled observer channel.
        overrides.setdefault("rate", imu.rate)
        super().__init__(name, accelerometer=imu, **overrides)

    def update(self, ctx) -> PartUpdate:
        zero = Vec3[PartFrame].constant((0.0, 0.0, 0.0))
        g_world = gravity_at(ctx, ctx.position[WorldFrame])
        predicted = ctx.orientation.conjugate().apply(
            ctx.acceleration[WorldFrame] - g_world)
        # The observation supplied to this pseudo-sensor is the selected
        # accelerometer's raw sample. Include that IMU's estimable bias in h
        # so innovation r = z - h has dr/db_a = -I. White IMU noise belongs
        # to INS propagation Q; ModelForce.model_error alone owns this R.
        expected_sample = predicted + self.accelerometer.accel_bias
        return PartUpdate(
            wrench=Wrench(force=zero, torque=zero),
            outputs={"specific_force": expected_sample + self.model_error},
            rates={"specific_force": self.rate},
        )


__all__ = ["ModelForce"]

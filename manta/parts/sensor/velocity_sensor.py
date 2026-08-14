"""VelocitySensor — body-frame linear-velocity sensor.

The simple, idealized velocity observer: it reads the craft's inertial
velocity (relative to the world / ground) expressed in its own case
frame, with an optional white-noise channel. This is the model an AUV's
Doppler velocity log (DVL) or an aircraft's ground-relative radar/lidar
velocity reduces to when you ignore the sensor's own dynamics.

A future `DVL` part will model the nuanced Doppler-velocity-log behavior
on top of this — bottom-lock vs water-track (velocity relative to the
*fluid* via a FluidField query rather than the ground), beam geometry,
and dropout. Likewise the simple `PositionSensor` is the seed of a
future `GPS` part. `VelocitySensor` stays the clean baseline.

Pattern matches IMU.gyro — both are TickContext-fed sensor reads that
require no fields and no part-frame physics.
"""

from __future__ import annotations

from ...ir.frames import PartFrame, WorldFrame
from ...ir.types import Vec3
from .._declarations import Output, Parameter, PartUpdate, WhiteNoise
from ..base import Part, PartRole
from ...ir.wrench import Wrench


class VelocitySensor(Part):
    """Body-frame linear-velocity sensor.

    Outputs:
        velocity : Vec3[PartFrame] — the craft's inertial (ground-
                                      relative) velocity in the sensor's
                                      own case frame (R^T·v_anchor). For
                                      a root-mounted sensor that frame
                                      coincides with CraftFrame; on a
                                      rotor it spins with the joint.

    Noise channel (set σ to engage):
        velocity_noise — vec3 white, per-tick m/s. Becomes the EKF's
                         measurement R, exactly as PositionSensor's
                         `position_noise`. Defaults to 0 (an ideal read).
    """

    role = PartRole.SENSOR

    #: Measurement rate, Hz. `None` ⇒ every tick (family-uniform knob).
    rate: float = Parameter(None)

    velocity_noise = WhiteNoise("R3", frame=PartFrame, sigma=0.0)

    velocity = Output()

    def update(self, ctx) -> PartUpdate:
        zero_v = Vec3[PartFrame].constant((0.0, 0.0, 0.0))
        # Reads the platform's inertial velocity (rel. to the ground) in
        # its own case frame: rotate the world-frame velocity through
        # ctx.orientation. For a root-mounted sensor that frame is the
        # body frame; on a rotor it spins with the joint.
        v_sensor = ctx.orientation.conjugate().apply(ctx.velocity[WorldFrame])
        return PartUpdate(
            wrench=Wrench(force=zero_v, torque=zero_v),
            outputs={"velocity": v_sensor + self.velocity_noise},
            rates={"velocity": self.rate},
        )

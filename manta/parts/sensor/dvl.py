"""DVL — Doppler velocity log.

An onboard sensor that emits the craft's body-frame linear velocity.
Underwater AUVs use these against the seafloor; airborne platforms
sometimes use radar/lidar variants for ground-relative velocity. The
sensor model here is the ideal case: noise-free read of v_body.

Pattern matches IMU.gyro — both are TickContext-fed sensor reads that
require no fields and no part-frame physics.
"""

from __future__ import annotations

from ...ir.frames import PartFrame, WorldFrame
from ...ir.types import Vec3
from .._declarations import Output, PartUpdate, WhiteNoise
from ..base import Part
from ...ir.wrench import Wrench


class DVL(Part):
    """Body-frame linear-velocity sensor.

    Outputs:
        velocity : Vec3[CraftFrame] — body-frame velocity (R^T·v_anchor).
                                      What a DVL reads when locked to a
                                      reference (seafloor / ground).

    Noise channel (set σ to engage):
        velocity_noise — vec3 white, per-tick m/s. Becomes the EKF's
                         measurement R, exactly as PositionSensor's
                         `position_noise`. Defaults to 0 (an ideal read).
    """

    velocity_noise = WhiteNoise("R3", frame=PartFrame, sigma=0.0)

    velocity = Output()

    def update(self, ctx) -> PartUpdate:
        zero_v = Vec3[PartFrame].constant((0.0, 0.0, 0.0))
        # A DVL reads the platform's inertial velocity (rel. to the ground)
        # in its own case frame: rotate the world-frame velocity through
        # ctx.orientation. For a root-mounted DVL that frame is the body
        # frame; on a rotor it spins with the joint.
        v_sensor = ctx.orientation.conjugate().apply(ctx.velocity[WorldFrame])
        return PartUpdate(
            wrench=Wrench(force=zero_v, torque=zero_v),
            outputs={"velocity": v_sensor + self.velocity_noise},
        )

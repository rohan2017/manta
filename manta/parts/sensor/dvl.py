"""DVL — Doppler velocity log.

An onboard sensor that emits the craft's body-frame linear velocity.
Underwater AUVs use these against the seafloor; airborne platforms
sometimes use radar/lidar variants for ground-relative velocity. The
sensor model here is the ideal case: noise-free read of v_body.

Pattern matches IMU.gyro — both are TickContext-fed sensor reads that
require no fields and no part-frame physics.
"""

from __future__ import annotations

from ...ir.frames import CraftFrame
from ...ir.types import Vec3
from ..base import Output, Part, PartUpdate
from ...ir.wrench import Wrench


class DVL(Part):
    """Body-frame linear-velocity sensor.

    Outputs:
        velocity : Vec3[CraftFrame] — body-frame velocity (R^T·v_anchor).
                                      What a DVL reads when locked to a
                                      reference (seafloor / ground).
    """

    velocity = Output(shape="vec3")

    def update(self, ctx) -> PartUpdate:
        zero_v = Vec3[CraftFrame].constant((0.0, 0.0, 0.0))
        # ctx.velocity_body is the mount-point velocity in the sensor's own
        # frame (the framework rotated it) — for a root-mounted DVL that's
        # the body frame; on a rotor it spins with the joint.
        return PartUpdate(
            wrench=Wrench(force=zero_v, torque=zero_v),
            outputs={"velocity": ctx.velocity_body},
        )

"""Magnetometer — magnetic flux reading in the sensor's own frame.

Queries the registered MagField at the sensor's mount point in
WorldFrame, rotates the result into the sensor's frame. That's exactly
what a 3-axis fluxgate magnetometer reads when the craft is oriented in
the field.

With no MagField registered, `ctx.field(MagField)` returns an empty
field (zero everywhere); the magnetometer reads zero. `ctx.position` is
already the sensor's mount-point in WorldFrame (the kinematic pass
composes it through the part's `transform` and any joints above it), so
the field is sampled there directly — for spatially uniform B this is
identical to the craft origin; for a nearby dipole it captures the
local gradient.
"""

from __future__ import annotations

from ...fields import MagField
from ...ir.frames import PartFrame, WorldFrame
from ...ir.types import Vec3
from .._declarations import Output, PartUpdate, WhiteNoise
from ..base import Part
from ...ir.wrench import Wrench


class Magnetometer(Part):
    """3-axis magnetometer.

    Outputs:
        B : Vec3[CraftFrame] — magnetic flux density at the sensor
                                position, in the sensor's own frame. SI
                                units (Tesla). For a sensor mounted
                                directly on the craft root that frame is
                                CraftFrame; on a joint rotor it spins
                                with the rotor.

    Noise channel (set σ to engage):
        B_noise — vec3 white, per-tick Tesla. Becomes the EKF's
                  measurement R for a heading/attitude fix, exactly as
                  PositionSensor's `position_noise` does. Defaults to 0
                  (a clean reading).
    """

    B_noise = WhiteNoise("R3", frame=PartFrame, sigma=0.0)

    B = Output()

    def update(self, ctx) -> PartUpdate:
        # ctx.position is already the sensor's world-frame mount point
        # (chain-composed through transform + any joints), so sample the
        # field there directly. ctx.orientation is the sensor's own world
        # attitude (the framework composed any joint rotation in), so its
        # conjugate maps world → sensor frame directly — for a root-mounted
        # magnetometer that's CraftFrame; on a rotor it spins with the joint.
        B_world  = ctx.field(MagField).value_at_sym(
            ctx.position[WorldFrame], ctx.t)
        B_sensor = ctx.orientation.conjugate().apply(B_world)

        zero_v = Vec3[PartFrame].constant((0.0, 0.0, 0.0))
        return PartUpdate(
            wrench=Wrench(force=zero_v, torque=zero_v),
            outputs={"B": B_sensor + self.B_noise},
        )

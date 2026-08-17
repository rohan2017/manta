"""RF antenna mount marker with world-frame kinematic outputs.

``Antenna`` deliberately contains no propagation or link policy.  It exposes
the mount point and antenna-frame motion that a downstream RF model needs:

* ``position`` is the antenna phase-center position in ``WorldFrame``;
* ``orientation`` is the world-from-antenna quaternion (w, x, y, z); and
* ``angular_velocity`` is the antenna frame's absolute angular velocity,
  expressed in ``WorldFrame`` (rad/s).

The framework's kinematic pass has already composed craft pose, static mount
orientation, and every articulation above the antenna.  Keeping this part as a
direct observer avoids creating a second transform implementation in RF code.
"""

from ...ir.frames import PartFrame, WorldFrame
from ...ir.types import Vec3
from ...ir.wrench import Wrench
from .._declarations import Output, Parameter, PartUpdate
from ..base import Part, PartRole


class Antenna(Part):
    """A kinematic marker for an RF antenna phase center and local frame.

    Outputs:
        position : Vec3[WorldFrame]
            Antenna phase-center position in world coordinates.
        orientation : Quat[WorldFrame, PartFrame]
            World-from-antenna unit quaternion in ``(w, x, y, z)`` order.
        angular_velocity : Vec3[WorldFrame]
            Absolute angular velocity of the antenna frame, expressed in
            world coordinates, in rad/s.

    These are ideal kinematic outputs.  Radiation patterns, propagation,
    tracking, link quality, and packet behavior belong to downstream users of
    the model, not to this marker.
    """

    role = PartRole.SENSOR
    frequency_hz: float = Parameter(2.4e9)
    tx_power_dbm: float = Parameter(20.0)
    gain_dbi: float = Parameter(0.0)
    position = Output()
    orientation = Output()
    angular_velocity = Output()

    def update(self, ctx) -> PartUpdate:
        zero = Vec3[PartFrame].constant((0.0, 0.0, 0.0))
        return PartUpdate(
            wrench=Wrench(force=zero, torque=zero),
            outputs={
                "position": ctx.position[WorldFrame],
                "orientation": ctx.orientation,
                "angular_velocity": ctx.angular_velocity[WorldFrame],
            },
        )


__all__ = ["Antenna"]

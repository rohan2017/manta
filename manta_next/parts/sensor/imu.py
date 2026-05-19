"""IMU — gyro-only sensor (M8 scope).

A real IMU outputs (specific force, angular velocity). M8 ships the
angular-velocity channel only: the accelerometer channel needs the
craft's body-frame inertial acceleration, which depends on the
aggregated wrench — a circular dependency inside `update()`. A
post-Newton-Euler "sensor phase" (run after the body Euler step) will
expose accel cleanly; that lands in a future milestone.

The IMU's output is a Vec3[CraftFrame]. For an IMU mounted at the body
origin (the default in M8), gyro = body ω directly. Mounting at an
offset with a static transform inherits Part.transform; the gyro
reading is still ω about the body axes — the rotation rate is the
same at every point of a rigid body.
"""

from __future__ import annotations

from ...ir.frames import CraftFrame
from ...ir.types import Vec3
from ..base import Output, Part, PartUpdate
from ..wrench import Wrench


class IMU(Part):
    """Inertial-measurement unit. Emits a gyro reading per tick.

    Outputs:
        gyro : Vec3[CraftFrame]   — body angular velocity. What an
                                    onboard rate gyro reads.

    Parameters (none yet):
        Noise modeling, bias states, rate gates, and accel-channel
        outputs all land in follow-up work. The point of the M8 IMU
        is to exercise the Output declaration plumbing and the
        TickContext-as-body-state-window pattern.
    """

    gyro = Output(shape="vec3")

    def update(self, ctx) -> PartUpdate:
        # Sensor parts don't contribute to dynamics — zero wrench.
        zero_v = Vec3[CraftFrame].constant((0.0, 0.0, 0.0))
        return PartUpdate(
            wrench=Wrench(force=zero_v, torque=zero_v),
            outputs={"gyro": ctx.angular_velocity},
        )

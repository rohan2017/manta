"""PositionSensor — world-frame position observer (GPS / motion capture).

Emits the sensor's world-frame position as a per-tick Output. The
Output declaration *is* the measurement model for the EKF; noise can
be injected externally (sim) or modeled on the part itself (see `IMU`
for the pattern).

`ctx.position` is already the sensor's own world-frame position — the
framework's kinematic pass composed it from the body state and the
chain of `Part.mount_offset` / joint rotations above the sensor. So the
part code is a one-liner.
"""

from __future__ import annotations

from ...ir.frames import PartFrame, WorldFrame
from ...ir.types import Vec3
from ...ir.wrench import Wrench
from .._declarations import Output, Parameter, PartUpdate, WhiteNoise
from ..base import Part, PartRole


class PositionSensor(Part):
    """Outputs the sensor's world-frame position each tick.

    Outputs:
        position : Vec3[WorldFrame]   — sensor mount-point position in
                                         world frame; exactly what a GPS
                                         or mocap marker reads.

    Noise channel (set σ to engage — leave at 0 for a noiseless oracle):
        position_noise : world-frame white noise on the reading. Engage
                         it (`PositionSensor("gps", position_noise_sigma=0.5)`)
                         to give the EKF an auto-built R for this sensor,
                         e.g. when driving the filter through `step()`.

    Rate (Hz):
        rate : measurement rate. `None` (default) ⇒ a fresh fix every
               tick. Set it (`PositionSensor("gps", rate=1.0)`) to model a
               slow sensor: the Sim publishes a new reading once per
               1/rate window and holds it in between, and the EKF folds
               each fix in exactly once. Pure metadata — the tick stays a
               smooth function (the estimator sees the continuous model).
    """

    role = PartRole.SENSOR

    rate: float = Parameter(None)

    position_noise = WhiteNoise("R3", frame=WorldFrame, sigma=0.0)

    position = Output()

    def update(self, ctx) -> PartUpdate:
        zero_v = Vec3[PartFrame].constant((0.0, 0.0, 0.0))
        reading = ctx.position[WorldFrame] + self.position_noise
        return PartUpdate(
            wrench=Wrench(force=zero_v, torque=zero_v),
            outputs={"position": reading},
            rates={"position": self.rate},
        )

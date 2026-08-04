"""Barometer — absolute fluid-pressure sensor.

Queries the world `FluidField` at the sensor's mount point and emits the
local pressure (Pa) as a per-tick Output — what a real barometer reads,
and the basis for a barometric altimeter (air) or a depth gauge (water),
since pressure encodes altitude/depth through the regime's hydrostatic /
ISA profile.

`ctx.position` is already the sensor's own world-frame position (the
kinematic pass composed it from the body state and the chain of
`Part.transform` / joint rotations above the sensor), so the part code is
a one-line field query — the same pattern as `PointBuoy`. With no
FluidField registered the reading is zero.

This is the pressure sibling of `VelocitySensor`/`PositionSensor`; a
`Thermometer` (reading `FluidState.temperature`) and a `FlowSensor`
(reading `velocity` relative to the craft) follow the identical shape.
"""

from __future__ import annotations

from ...fields import FluidField
from ...ir.frames import PartFrame, WorldFrame
from ...ir.types import Scalar, Vec3
from .._declarations import Output, Parameter, PartUpdate, WhiteNoise
from ..base import Part
from ...ir.wrench import Wrench


class Barometer(Part):
    """Outputs the local fluid pressure at the sensor each tick.

    Outputs:
        pressure : scalar — absolute pressure (Pa) of the fluid at the
                            sensor's world-frame position, read from the
                            FluidField.

    Noise channel (set σ to engage — leave at 0 for a noiseless oracle):
        pressure_noise : scalar white noise (Pa) on the reading. Engage it
                         (`Barometer("baro", pressure_noise_sigma=50.0)`)
                         to give the EKF an auto-built R for this sensor.

    Rate (Hz):
        rate : measurement rate. `None` (default) ⇒ a fresh reading every
               tick. Set it (`Barometer("baro", rate=10.0)`) to model a
               slow sensor: the Sim publishes a new reading once per
               1/rate window and holds it in between, and the EKF folds
               each in exactly once. Pure metadata — the tick stays a
               smooth function.
    """

    requires_fields = [FluidField]

    rate: float = Parameter(None)

    pressure_noise = WhiteNoise("R1", sigma=0.0)

    pressure = Output()

    def update(self, ctx) -> PartUpdate:
        zero_v = Vec3[PartFrame].constant((0.0, 0.0, 0.0))
        fluid = ctx.field(FluidField).value_at_sym(
            ctx.position[WorldFrame], ctx.t)
        reading = Scalar(fluid.pressure) + self.pressure_noise
        return PartUpdate(
            wrench=Wrench(force=zero_v, torque=zero_v),
            outputs={"pressure": reading},
            rates={"pressure": self.rate},
        )

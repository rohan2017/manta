"""Stock parts. Each part is a Python class subclassing `Part`, declares
its parameters at class scope, and implements `update(ctx)` to contribute
a wrench to the craft.

Public surface re-exports the part classes for ergonomic imports::

    from manta_next.parts import Part, Parameter, Mass
"""

from .base import Part, Parameter, Input, Output, State, PartUpdate
from .wrench import Wrench
from .structure.mass import Mass
from .structure.point_buoy import PointBuoy
from .articulation.spinning_rotor import SpinningRotor
from .articulation.flywheel_motor import FlywheelMotor
from .sensor.imu import IMU
from .sensor.position_sensor import PositionSensor
from .actuation.thruster import Thruster

__all__ = [
    "Part", "Parameter", "Input", "Output", "State", "PartUpdate",
    "Wrench",
    "Mass", "PointBuoy",
    "SpinningRotor", "FlywheelMotor",
    "IMU", "PositionSensor",
    "Thruster",
]

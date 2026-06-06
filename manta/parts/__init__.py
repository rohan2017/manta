"""Stock parts. Each part is a Python class subclassing `Part`, declares
its parameters at class scope, and implements `update(ctx)` to contribute
a wrench to the craft.

Public surface re-exports the part classes for ergonomic imports::

    from manta.parts import Part, Parameter, Mass
"""

from .base import (
    CompositePart, Input, Output, Parameter, Part, PartUpdate, RootPart, State,
)
from ..ir.wrench import Wrench
from .structure.mass import Mass
from .structure.point_buoy import PointBuoy
from .structure.collider import Collider
from .articulation.joint import ArticulatedJoint, RevoluteJoint
from .sensor.dvl import DVL
from .sensor.imu import IMU
from .sensor.magnetometer import Magnetometer
from .sensor.position_sensor import PositionSensor
from .actuation.thruster import Thruster
from .aero.drag_surface import DragSurface
from .aero.naca_airfoil import Naca00xx
from .attachment.tether_endpoint import TetherEndpoint

__all__ = [
    "Part", "CompositePart", "RootPart",
    "Parameter", "Input", "Output", "State", "PartUpdate",
    "Wrench",
    "Mass", "PointBuoy", "Collider",
    "ArticulatedJoint", "RevoluteJoint",
    "IMU", "DVL", "Magnetometer", "PositionSensor",
    "Thruster",
    "DragSurface", "Naca00xx",
    "TetherEndpoint",
]

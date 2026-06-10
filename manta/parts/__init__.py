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
from .articulation.joint import ArticulatedJoint, PrismaticJoint, RevoluteJoint
from .sensor.dvl import DVL
from .sensor.imu import IMU
from .sensor.magnetometer import Magnetometer
from .sensor.position_sensor import PositionSensor
from .sensor.camera import BBoxCamera, CentroidCamera, ProjectiveCamera
from .actuation.thruster import Thruster
from .aero.drag_surface import DragSurface
from .disturbance.process_noise import ProcessNoise
from .aero.naca_airfoil import Naca00xx
from .attachment.tether_endpoint import TetherEndpoint
from .attachment.trajectory_endpoint import (
    LinearTrajectory, TrajectoryEndpoint, TrajectorySample, hover,
)
from .field_source import (
    FieldSource, GravitySource, MagneticSource, OpticalSource,
)

__all__ = [
    "Part", "CompositePart", "RootPart",
    "Parameter", "Input", "Output", "State", "PartUpdate",
    "Wrench",
    "Mass", "PointBuoy", "Collider",
    "ArticulatedJoint", "PrismaticJoint", "RevoluteJoint",
    "IMU", "DVL", "Magnetometer", "PositionSensor",
    "ProjectiveCamera", "BBoxCamera", "CentroidCamera",
    "Thruster",
    "DragSurface", "Naca00xx",
    "ProcessNoise",
    "TetherEndpoint",
    "TrajectoryEndpoint", "TrajectorySample", "LinearTrajectory", "hover",
    "FieldSource", "GravitySource", "MagneticSource", "OpticalSource",
]

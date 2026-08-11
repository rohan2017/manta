"""Stock parts. Each part is a Python class subclassing `Part`, declares
its parameters at class scope, and implements `update(ctx)` to contribute
a wrench to the craft.

Public surface re-exports the part classes for ergonomic imports::

    from manta.parts import Part, Parameter, Mass
"""

from ._declarations import (
    Input, Noise, Output, Parameter, PartUpdate, RandomWalkNoise, State,
    WhiteNoise, unit_axis,
)
from .base import CompositePart, Part, RootPart
from ..ir.wrench import Wrench
from .structure.mass import Mass
from .structure.point_buoy import PointBuoy
from .structure.collider import Collider
from .articulation.joint import (ArticulatedJoint, PrismaticJoint,
                                 RevoluteDOF, RevoluteJoint)
from .articulation.motor import Motor
from .thermal.thermal_mass import ThermalMass
from .sensor.imu import IMU
from .sensor.antenna import Antenna
from .sensor.magnetometer import Magnetometer
from .sensor.position_sensor import PositionSensor
from .sensor.velocity_sensor import VelocitySensor
from .sensor.barometer import Barometer
from .sensor.camera import BBoxCamera, CentroidCamera, ProjectiveCamera
from .actuation.thruster import Thruster
from .aero.added_mass import AddedMass
from .aero.drag_surface import DragSurface
from .aero.fossen_damping import FossenDamping
from .aero.rotational_drag import RotationalDrag
from .disturbance.process_noise import ProcessNoise
from .aero.aerofoil import Aerofoil, naca
from .aero.control_surface import ControlSurface
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
    "Noise", "WhiteNoise", "RandomWalkNoise", "unit_axis",
    "Wrench",
    "Mass", "PointBuoy", "Collider", "ThermalMass",
    "ArticulatedJoint", "Motor", "PrismaticJoint", "RevoluteDOF",
    "RevoluteJoint",
    "Antenna", "IMU", "VelocitySensor", "Magnetometer", "PositionSensor", "Barometer",
    "ProjectiveCamera", "BBoxCamera", "CentroidCamera",
    "Thruster",
    "AddedMass", "DragSurface", "FossenDamping", "RotationalDrag", "Aerofoil", "naca", "ControlSurface",
    "ProcessNoise",
    "TetherEndpoint",
    "TrajectoryEndpoint", "TrajectorySample", "LinearTrajectory", "hover",
    "FieldSource", "GravitySource", "MagneticSource", "OpticalSource",
]

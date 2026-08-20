"""Stock parts. Each part is a Python class subclassing `Part`, declares
its parameters at class scope, and implements `update(ctx)` to contribute
a wrench to the craft.

Public surface re-exports the part classes for ergonomic imports::

    from manta.parts import Part, Parameter, Mass
"""

from ._declarations import (
    Input,
    Noise,
    Output,
    Parameter,
    PartUpdate,
    RandomWalkNoise,
    State,
    WhiteNoise,
    unit_axis,
)
from .base import CompositePart, Part, PartRole, RootPart
from ..ir.wrench import Wrench
from .structure.mass import Mass
from .structure.point_buoy import PointBuoy
from .structure.displacement_hull import DisplacementHull, HullSample
from .structure.collider import Collider
from .articulation.joint import (
    ArticulatedJoint,
    PrismaticJoint,
    RevoluteDOF,
    RevoluteJoint,
)
from .articulation.motor import Motor
from .thermal.thermal_mass import ThermalMass
from .electrical import (
    ConstantCurrentLoad,
    ConstantPowerLoad,
    Contactor,
    DCConverter,
    DCSource,
    ElectricalBus,
    ElectricalLoad,
    ElectricalNode,
    ElectricalPort,
    ExternalDCSupply,
    Fuse,
    ResistiveLoad,
    ConstantPowerElectronicsLoad,
    PoweredControlSurface,
    PoweredDuctedPropeller,
    PoweredLoadMixin,
    PoweredMotor,
    PoweredThruster,
)
from .sensor.imu import IMU
from .sensor.heading_sensor import HeadingSensor
from .sensor.antenna import Antenna
from .sensor.magnetometer import Magnetometer
from .sensor.position_sensor import PositionSensor
from .sensor.velocity_sensor import VelocitySensor
from .sensor.barometer import Barometer
from .sensor.camera import BBoxCamera, CentroidCamera, ProjectiveCamera
from .actuation.thruster import Thruster
from .actuation.ducted_propeller import DuctedPropeller
from .aero.added_mass import AddedMass
from .aero.drag_surface import DragSurface
from .aero.fossen_damping import FossenDamping
from .aero.rotational_drag import RotationalDrag
from .disturbance.process_noise import ProcessNoise
from .disturbance.wrench_process_noise import WrenchProcessNoise
from .aero.aerofoil import Aerofoil, naca
from .aero.control_surface import ControlSurface
from .attachment.tether_endpoint import TetherEndpoint
from .attachment.trajectory_endpoint import (
    LinearTrajectory,
    TrajectoryEndpoint,
    TrajectorySample,
    hover,
)
from .field_source import (
    FieldSource,
    GravitySource,
    MagneticSource,
    OpticalSource,
)

__all__ = [
    "Part",
    "PartRole",
    "CompositePart",
    "RootPart",
    "Parameter",
    "Input",
    "Output",
    "State",
    "PartUpdate",
    "Noise",
    "WhiteNoise",
    "RandomWalkNoise",
    "unit_axis",
    "Wrench",
    "Mass",
    "PointBuoy",
    "Collider",
    "DisplacementHull",
    "HullSample",
    "ThermalMass",
    "ElectricalNode",
    "ElectricalLoad",
    "DCSource",
    "ElectricalBus",
    "DCConverter",
    "Contactor",
    "Fuse",
    "ResistiveLoad",
    "ConstantCurrentLoad",
    "ConstantPowerLoad",
    "ElectricalPort",
    "ExternalDCSupply",
    "PoweredLoadMixin",
    "PoweredMotor",
    "PoweredThruster",
    "PoweredDuctedPropeller",
    "PoweredControlSurface",
    "ConstantPowerElectronicsLoad",
    "ArticulatedJoint",
    "Motor",
    "PrismaticJoint",
    "RevoluteDOF",
    "RevoluteJoint",
    "Antenna",
    "HeadingSensor",
    "IMU",
    "VelocitySensor",
    "Magnetometer",
    "PositionSensor",
    "Barometer",
    "ProjectiveCamera",
    "BBoxCamera",
    "CentroidCamera",
    "DuctedPropeller",
    "Thruster",
    "AddedMass",
    "DragSurface",
    "FossenDamping",
    "RotationalDrag",
    "Aerofoil",
    "naca",
    "ControlSurface",
    "ProcessNoise",
    "WrenchProcessNoise",
    "TetherEndpoint",
    "TrajectoryEndpoint",
    "TrajectorySample",
    "LinearTrajectory",
    "hover",
    "FieldSource",
    "GravitySource",
    "MagneticSource",
    "OpticalSource",
]

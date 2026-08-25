"""Stock parts. Each part is a Python class subclassing `Part`, declares
its parameters at class scope, and implements `update(ctx)` to contribute
a wrench to the craft.

Public surface re-exports the part classes for ergonomic imports::

    from manta.parts import Part, Parameter, Mass
"""

from ..ir.wrench import Wrench
from ._declarations import (
    GaussMarkovNoise,
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
from .actuation.ducted_propeller import DuctedPropeller
from .actuation.thruster import Thruster
from .aero.added_mass import AddedMass
from .aero.aerofoil import Aerofoil, naca
from .aero.control_surface import ControlSurface
from .aero.drag_surface import DragSurface
from .aero.fossen_damping import FossenDamping
from .aero.projected_velocity_damping import ProjectedVelocityDamping
from .aero.rotational_drag import RotationalDrag
from .articulation.joint import (
    ArticulatedJoint,
    PrismaticJoint,
    RevoluteDOF,
    RevoluteJoint,
)
from .articulation.motor import Motor
from .attachment.tether_endpoint import TetherEndpoint
from .attachment.trajectory_endpoint import (
    LinearTrajectory,
    TrajectoryEndpoint,
    TrajectorySample,
    hover,
)
from .base import CompositePart, Part, PartRole, RootPart
from .disturbance.constant_wrench import ConstantWrench
from .disturbance.process_noise import ProcessNoise
from .disturbance.wrench_process_noise import WrenchProcessNoise
from .electrical import (
    ConstantCurrentLoad,
    ConstantPowerElectronicsLoad,
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
    PoweredControlSurface,
    PoweredDuctedPropeller,
    PoweredLoadMixin,
    PoweredMotor,
    PoweredThruster,
    ResistiveLoad,
)
from .field_source import (
    FieldSource,
    GravitySource,
    MagneticSource,
    OpticalSource,
)
from .sensor.antenna import Antenna
from .sensor.barometer import Barometer
from .sensor.camera import BBoxCamera, CentroidCamera, ProjectiveCamera
from .sensor.heading_sensor import HeadingSensor
from .sensor.imu import IMU
from .sensor.magnetometer import Magnetometer
from .sensor.model_force import ModelForce
from .sensor.position_sensor import PositionSensor
from .sensor.velocity_sensor import VelocitySensor
from .structure.collider import Collider
from .structure.displacement_hull import DisplacementHull, HullSample
from .structure.mass import Mass
from .structure.point_buoy import PointBuoy
from .thermal.thermal_mass import ThermalMass

__all__ = [
    "IMU",
    "AddedMass",
    "Aerofoil",
    "Antenna",
    "ArticulatedJoint",
    "BBoxCamera",
    "Barometer",
    "CentroidCamera",
    "Collider",
    "CompositePart",
    "ConstantCurrentLoad",
    "ConstantPowerElectronicsLoad",
    "ConstantPowerLoad",
    "ConstantWrench",
    "Contactor",
    "ControlSurface",
    "DCConverter",
    "DCSource",
    "DisplacementHull",
    "DragSurface",
    "DuctedPropeller",
    "ElectricalBus",
    "ElectricalLoad",
    "ElectricalNode",
    "ElectricalPort",
    "ExternalDCSupply",
    "FieldSource",
    "FossenDamping",
    "Fuse",
    "GaussMarkovNoise",
    "GravitySource",
    "HeadingSensor",
    "HullSample",
    "Input",
    "LinearTrajectory",
    "MagneticSource",
    "Magnetometer",
    "Mass",
    "ModelForce",
    "Motor",
    "Noise",
    "OpticalSource",
    "Output",
    "Parameter",
    "Part",
    "PartRole",
    "PartUpdate",
    "PointBuoy",
    "PositionSensor",
    "PoweredControlSurface",
    "PoweredDuctedPropeller",
    "PoweredLoadMixin",
    "PoweredMotor",
    "PoweredThruster",
    "PrismaticJoint",
    "ProcessNoise",
    "ProjectedVelocityDamping",
    "ProjectiveCamera",
    "RandomWalkNoise",
    "ResistiveLoad",
    "RevoluteDOF",
    "RevoluteJoint",
    "RootPart",
    "RotationalDrag",
    "State",
    "TetherEndpoint",
    "ThermalMass",
    "Thruster",
    "TrajectoryEndpoint",
    "TrajectorySample",
    "VelocitySensor",
    "WhiteNoise",
    "Wrench",
    "WrenchProcessNoise",
    "hover",
    "naca",
    "unit_axis",
]

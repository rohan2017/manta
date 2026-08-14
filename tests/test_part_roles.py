"""Semantic part roles are explicit model-selection metadata."""

from manta.parts import (
    Antenna, Barometer, ControlSurface, DuctedPropeller, IMU, Magnetometer,
    Mass, Motor, PartRole, PositionSensor, ProjectiveCamera, Thruster,
    VelocitySensor,
)


def test_stock_parts_declare_reduction_roles_at_their_family_boundary():
    assert Mass.role is PartRole.DYNAMICS
    for part_type in (Thruster, DuctedPropeller, ControlSurface, Motor):
        assert part_type.role is PartRole.ACTUATOR
    for part_type in (
        Antenna, Barometer, IMU, Magnetometer, PositionSensor,
        ProjectiveCamera, VelocitySensor,
    ):
        assert part_type.role is PartRole.SENSOR


def test_roles_are_strings_in_serializable_model_metadata():
    assert PartRole.ACTUATOR.value == "actuator"
    assert str(PartRole.SENSOR.value) == "sensor"

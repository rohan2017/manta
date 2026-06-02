"""Observability analysis — the symbolic F/H rank test correctly flags
which states a sensor set can and can't see.

These pin down the diagnosis behind the submarine heading drift: GPS + DVL
+ gyro leaves absolute yaw (orientation) unobservable; a magnetometer
(compass) restores full observability. The point of the tool is to surface
that at setup instead of as a silent estimate drift.
"""

import numpy as np

from manta import Craft, EKF, TargetNumpy, World
from manta.estimation import observability
from manta.fields import FluidField, GravityField, MagField
from manta.parts import (
    DVL, DragSurface, IMU, Magnetometer, Mass, PointBuoy, PositionSensor,
    Thruster,
)


def _sub_world():
    """The submarine demo's configuration (hull drag + a damping fin, buoy
    above CoM)."""
    s = Craft("sub")
    s.add(Mass("hull", mass=120.0, moi=(3, 12, 12)))
    s.add(PointBuoy("buoy", volume=120.0 / 1025.0, transform=(0, 0, 0.15)))
    s.add(DragSurface.isotropic_quadratic("drag", area=0.09,
                                          drag_coefficient=0.5))
    s.add(DragSurface.isotropic_quadratic("fin", area=0.35,
                                          drag_coefficient=1.1,
                                          transform=(-1.3, 0, 0)))
    s.add(Thruster("prop", force=(1, 0, 0), transform=(-1, 0, 0)))
    s.add(Thruster("yaw", torque=(0, 0, 1)))
    s.add(IMU("imu", gyro_noise_sigma=0.01, accel_noise_sigma=0.05))
    s.add(DVL("dvl"))
    s.add(PositionSensor("gps", position_noise_sigma=0.1))
    s.add(Magnetometer("mag", B_noise_sigma=1e-6))
    w = (World()
         .add_field(GravityField().add_uniform((0, 0, -9.81)))
         .add_field(FluidField().add_uniform(density=1025.0))
         .add_field(MagField().add_uniform((2e-5, 0.0, -4e-5))))
    w.add_craft(s, position=(0, 0, -8.0))
    return w


def _excited():
    """A turning, accelerating operating point (heading becomes observable
    through the maneuver — see test_heading_observability_is_excitation_dependent)."""
    a, ax = 0.4, np.array([1, 1, 0.0]) / np.sqrt(2)
    return (
        {"sub": {"orientation": [np.cos(a / 2), *(np.sin(a / 2) * ax)],
                 "velocity": (1.2, 0.1, 0.0),
                 "angular_velocity": (0.05, -0.03, 0.25)}},
        {"prop.throttle": 200.0, "yaw.throttle": 20.0})


def test_full_suite_is_observable():
    ekf = TargetNumpy(EKF(_sub_world()))
    rep = ekf.observability()
    assert rep.observable
    assert rep.rank == rep.tangent_dim
    assert rep.unobservable == []


def test_gps_dvl_gyro_leaves_heading_unobservable_at_rest():
    """The submarine drift, diagnosed: at rest / straight cruise, dropping
    the compass makes absolute yaw (orientation) fall out of the observable
    subspace — so it dead-reckons and drifts."""
    ekf = TargetNumpy(EKF(_sub_world()))
    rep = ekf.observability(
        sensors=["imu.gyro", "dvl.velocity", "gps.position"])
    assert not rep.observable
    assert rep.rank == rep.tangent_dim - 1          # exactly heading lost
    assert "sub.orientation" in {name for name, _ in rep.unobservable}


def test_compass_restores_observability_at_rest():
    """A magnetometer makes heading observable everywhere — including at
    rest, with no maneuvering required."""
    ekf = TargetNumpy(EKF(_sub_world()))
    rep = ekf.observability(
        sensors=["imu.gyro", "dvl.velocity", "gps.position", "mag.B"])
    assert rep.observable


def test_heading_observability_is_excitation_dependent():
    """Without a compass, heading is observable *through a maneuver*: at a
    turning + accelerating operating point GPS + DVL + gyro recover full
    rank, even though they don't at rest. That intermittency is the real
    nuance — heading drifts on straight legs and re-locks when turning."""
    ekf = TargetNumpy(EKF(_sub_world()))
    state, inputs = _excited()
    sensors = ["imu.gyro", "dvl.velocity", "gps.position"]
    assert not ekf.observability(sensors=sensors).observable        # at rest
    assert ekf.observability(state=state, inputs=inputs,
                             sensors=sensors).observable             # excited


def test_gyro_only_loses_position_and_attitude():
    ekf = TargetNumpy(EKF(_sub_world()))
    rep = ekf.observability(sensors=["imu.gyro"])
    assert not rep.observable
    flagged = {name for name, _ in rep.unobservable}
    assert {"sub.position", "sub.orientation", "sub.velocity"} <= flagged


def test_no_sensors_is_fully_unobservable():
    ekf = TargetNumpy(EKF(_sub_world()))
    rep = ekf.observability(sensors=[])
    assert rep.rank == 0
    assert not rep.observable


def test_report_is_readable():
    ekf = TargetNumpy(EKF(_sub_world()))
    text = ekf.observability(
        sensors=["imu.gyro", "dvl.velocity", "gps.position"]).summary()
    assert "NOT fully observable" in text
    assert "sub.orientation" in text


def test_full_suite_observable_label():
    ekf = TargetNumpy(EKF(_sub_world()))
    assert "fully observable" in ekf.observability().summary()

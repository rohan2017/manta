"""Observability analysis — the symbolic F/H rank test correctly flags
which states a sensor set can and can't see.

These pin down the diagnosis behind the submarine heading drift: GPS + DVL
+ gyro leaves absolute yaw (orientation) unobservable; a magnetometer
(compass) restores full observability. The point of the tool is to surface
that at setup instead of as a silent estimate drift.
"""

import numpy as np

from manta import Craft, EKF, World
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
    s.add(DVL("dvl", velocity_noise_sigma=0.01))
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
    ekf = EKF(_sub_world())
    rep = ekf.observability()
    assert rep.observable
    assert rep.rank == rep.tangent_dim
    assert rep.unobservable == []


def test_gps_dvl_gyro_leaves_heading_unobservable_at_rest():
    """The submarine drift, diagnosed: at rest / straight cruise, dropping
    the compass makes absolute yaw (orientation) fall out of the observable
    subspace — so it dead-reckons and drifts."""
    ekf = EKF(_sub_world())
    rep = ekf.observability(
        sensors=["imu.gyro", "dvl.velocity", "gps.position"])
    assert not rep.observable
    assert rep.rank == rep.tangent_dim - 1          # exactly heading lost
    assert "sub.orientation" in {name for name, _ in rep.unobservable}


def test_compass_restores_observability_at_rest():
    """A magnetometer makes heading observable everywhere — including at
    rest, with no maneuvering required."""
    ekf = EKF(_sub_world())
    rep = ekf.observability(
        sensors=["imu.gyro", "dvl.velocity", "gps.position", "mag.B"])
    assert rep.observable


def test_heading_observability_is_excitation_dependent():
    """Without a compass, heading is observable *through a maneuver*: at a
    turning + accelerating operating point GPS + DVL + gyro recover full
    rank, even though they don't at rest. That intermittency is the real
    nuance — heading drifts on straight legs and re-locks when turning."""
    ekf = EKF(_sub_world())
    state, inputs = _excited()
    sensors = ["imu.gyro", "dvl.velocity", "gps.position"]
    assert not ekf.observability(sensors=sensors).observable        # at rest
    assert ekf.observability(state=state, inputs=inputs,
                             sensors=sensors).observable             # excited


def test_gyro_only_loses_position_and_attitude():
    ekf = EKF(_sub_world())
    rep = ekf.observability(sensors=["imu.gyro"])
    assert not rep.observable
    flagged = {name for name, _ in rep.unobservable}
    assert {"sub.position", "sub.orientation", "sub.velocity"} <= flagged


def test_no_sensors_is_fully_unobservable():
    ekf = EKF(_sub_world())
    rep = ekf.observability(sensors=[])
    assert rep.rank == 0
    assert not rep.observable


def test_report_is_readable():
    ekf = EKF(_sub_world())
    text = ekf.observability(
        sensors=["imu.gyro", "dvl.velocity", "gps.position"]).summary()
    assert "NOT fully observable" in text
    assert "sub.orientation" in text


def test_full_suite_observable_label():
    ekf = EKF(_sub_world())
    assert "fully observable" in ekf.observability().summary()


def test_observable_basis_shape():
    rep = EKF(_sub_world()).observability(
        sensors=["imu.gyro", "dvl.velocity", "gps.position"])
    assert rep.basis.shape == (rep.tangent_dim, rep.rank)


def test_sigma_horizon_grades_heading_by_sensor_set():
    """The covariance-horizon analysis agrees with the rank test where the
    rank test is right: without a compass the orientation slot's worst
    direction (yaw) stays at its prior over the horizon; adding the
    compass collapses it."""
    w = _sub_world()
    ekf = EKF(w)
    kw = dict(horizon=2.0, dt=0.02, P0=0.1)
    blind = ekf.sigma_horizon(sensors=["imu.gyro", "dvl.velocity",
                                       "gps.position"], **kw)
    sighted = ekf.sigma_horizon(sensors=["imu.gyro", "dvl.velocity",
                                         "gps.position", "mag.B"], **kw)
    o = "sub.orientation"
    assert blind.sigmaT(o) > 0.9 * blind.sigma0(o)          # yaw untouched
    assert sighted.sigmaT(o) < 0.3 * sighted.sigma0(o)      # compass pins it
    # position is directly measured in both — it must converge regardless.
    assert blind.sigmaT("sub.position") < 0.3 * blind.sigma0("sub.position")


def test_sigma_horizon_report_shapes_and_summary():
    ekf = EKF(_sub_world())
    rep = ekf.sigma_horizon(horizon=1.0, dt=0.02,
                            P0=np.full(ekf.spec.tangent_dim, 0.1))
    n = ekf.spec.tangent_dim
    assert rep.P_final.shape == (n, n)
    assert set(rep.sigmas) == {s.name for s in ekf.spec.slots}
    assert all(len(v) == len(rep.times) for v in rep.sigmas.values())
    text = rep.summary()
    assert "σ-horizon" in text and "sub.orientation" in text


def test_trajectory_observability_reveals_heading_through_motion():
    """Single-point observability says heading is unobservable from
    GPS+DVL+gyro at rest; over a moving trajectory the DVL/GPS pairing
    recovers it. The trajectory tool captures that, the local one can't."""
    from manta.estimation import observability_trajectory
    sens = ["imu.gyro", "dvl.velocity", "gps.position"]
    w = _sub_world()
    at_rest = observability(EKF(w), sensors=sens)
    moving = observability_trajectory(
        w, dt=0.02, steps=300, sensors=sens, control={"prop.throttle": 600.0})
    assert not at_rest.observable                      # heading hidden at rest
    assert "sub.orientation" in {n for n, _ in at_rest.unobservable}
    assert moving.observable                            # recovered by motion
    assert moving.rank > at_rest.rank

"""Explicit local-frame physics without an environment or scheduler wrapper."""
import casadi as ca
import numpy as np
import pytest
from manta import Craft, INS, IMUPreintegrator, NavigationFrame, PreintegratedIMUPacket, TargetNumpy, World
from manta.fields import GravityField, PointMassGravity
from manta.parts import ConstantBiasIMU, Mass
from manta.types import Vec3, WorldFrame


@pytest.mark.parametrize("spin", [0., 2e-4, -1e-4])
def test_stationary_local_frame_with_spatial_gravitation(spin):
    anchor = np.array([2e6, 0., 2e6])
    omega = np.array([0., 0., spin])
    frame = NavigationFrame(frame_id="site", epoch="0", angular_velocity=omega,
                            origin_from_rotation_center=anchor, gravity_convention="gravitation")
    # Planet center relative to our fixed Cartesian origin. Rotation is
    # supplied separately, so centrifugal acceleration is not in this field.
    gravity = GravityField().add(PointMassGravity(position=tuple(-anchor), GM=4e12, eps=0))
    craft = Craft("robot")
    craft.add(Mass("body", mass=1, moi=(1, 1, 1)))
    craft.add(ConstantBiasIMU("imu", gyro_noise_sigma=1e-5, accel_noise_sigma=.001))
    world = World().add_field(gravity)
    world.add_craft(craft)
    ins = TargetNumpy(INS(world, imu="imu", sensors=[], navigation_frame=frame,
                          propagation="preintegrated", accel_integration="endpoint",
                          covariance="conventional"))
    pre = TargetNumpy(IMUPreintegrator(sample_rate_hz=100, gyro_integration="endpoint",
                                      accel_integration="endpoint", gyro_noise_density=1e-6,
                                      accel_noise_density=.0001))
    gravitation = -4e12*anchor/np.linalg.norm(anchor)**3
    g = gravity.value_at_sym(Vec3[WorldFrame].constant((0, 0, 0)), ca.DM(0))
    np.testing.assert_allclose(np.asarray(ca.evalf(g.mx)).ravel(), gravitation, atol=1e-15)
    accel = -gravitation + np.cross(omega, np.cross(omega, anchor))
    for k in range(100):
        values = pre.step(.01, accel=accel, gyro=omega, end_accel=accel, end_gyro=omega,
                          accel_bias=[0, 0, 0], gyro_bias=[0, 0, 0])
        if k % 2:
            ins.predict(PreintegratedIMUPacket(values, start_time=ins.time))
            pre.reset()
    state = ins.state_dict()["robot"]
    np.testing.assert_allclose(state["position"], 0, atol=1e-12)
    np.testing.assert_allclose(state["velocity"], 0, atol=1e-12)
    np.testing.assert_allclose(state["orientation"], [1, 0, 0, 0], atol=1e-12)

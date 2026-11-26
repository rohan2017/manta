"""Explicit acquisition/batching loop: shared boundaries, aiding and C parity."""
import numpy as np
import pytest
from manta import Craft, INS, IMUPreintegrator, PreintegratedIMUPacket, TargetNumpy, World
from manta.fields import GravityField
from manta.parts import ConstantBiasIMU, Mass, VelocitySensor


def run_intervals(predict_every, *, native=False):
    craft = Craft("vehicle")
    craft.add(Mass("body", mass=1, moi=(1, 1, 1)))
    craft.add(ConstantBiasIMU("imu", gyro_noise_sigma=1e-4, accel_noise_sigma=.01))
    craft.add(VelocitySensor("dvl", velocity_noise_sigma=.02, mount_offset=(.2, .1, 0)))
    world = World().add_field(GravityField(g=(0, 0, -9.81)))
    world.add_craft(craft)
    ins = TargetNumpy(INS(world, imu="imu", sensors=["dvl.velocity"],
                          propagation="preintegrated", accel_integration="endpoint",
                          covariance="conventional"))
    pre = TargetNumpy(IMUPreintegrator(sample_rate_hz=100, gyro_integration="endpoint",
                                      accel_integration="endpoint", gyro_noise_density=1e-5,
                                      accel_noise_density=.001))
    ins.reset(t=10.)
    if native:
        pre.compile_functions(("step",), optimization="balanced", max_instructions=20000)
        ins.compile_functions(("predict",), optimization="balanced", max_instructions=20000)
        ins.compile_sensor_updates(["dvl.velocity"], optimization="balanced", max_instructions=20000)
    start, count = ins.time, 0
    left_accel, left_gyro = [0, 0, 9.81], [.01, 0, 0]
    for k in range(1, 11):
        t = 10 + k*.01
        accel, gyro = [.01*k, 0, 9.81], [.01, 0, .002*k]
        bias = ins.state_dict()["vehicle"]
        values = pre.step(.01, t=t-.01, accel=left_accel, gyro=left_gyro,
                          end_accel=accel, end_gyro=gyro,
                          accel_bias=bias["imu.accel_bias"], gyro_bias=bias["imu.gyro_bias"])
        count += 1
        # Aiding and end-of-run both close a partial packet explicitly.
        if count == predict_every or k % 3 == 0 or k == 10:
            packet = PreintegratedIMUPacket(values, start_time=start)
            inputs = ins.preintegrated_inputs(packet)
            ins.predict(packet)
            if k % 3 == 0:
                ins.update("dvl.velocity", [.002*k, 0, 0], u=inputs)
            pre.reset()
            start, count = t, 0
        left_accel, left_gyro = accel, gyro
    return ins


def test_partial_packets_at_aiding_match_one_interval_predictions():
    a, b = run_intervals(1), run_intervals(4)
    assert a.time == b.time == 10.1
    np.testing.assert_allclose(a.x, b.x, atol=2e-8)
    np.testing.assert_allclose(a.P, b.P, atol=2e-8)


@pytest.mark.cpp
def test_explicit_driver_native_kernels_match_interpreted():
    a, b = run_intervals(2), run_intervals(2, native=True)
    np.testing.assert_allclose(a.x, b.x, atol=1e-12)
    np.testing.assert_allclose(a.P, b.P, atol=1e-12)

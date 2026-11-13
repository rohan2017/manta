"""Public streaming API: observed boundaries, corrections, and native parity."""

import numpy as np
import pytest

from manta import Craft, INSStream, World
from manta.fields import GravityField
from manta.parts import ConstantBiasIMU, Mass, VelocitySensor


def stream(**kwargs):
    craft = Craft("vehicle")
    craft.add(Mass("body", mass=1, moi=(1, 1, 1)))
    craft.add(ConstantBiasIMU("imu", gyro_noise_sigma=1e-4, accel_noise_sigma=0.01))
    craft.add(
        VelocitySensor("dvl", velocity_noise_sigma=0.02, mount_offset=(0.2, 0.1, 0))
    )
    world = World().add_field(GravityField(g=(0, 0, -9.81)))
    world.add_craft(craft)
    return INSStream(
        world, imu="imu", sample_rate_hz=100, sensors=["dvl.velocity"], **kwargs
    )


def test_flush_and_aiding_at_observed_boundary_match_one_interval_packets():
    a, b = stream(predict_every=1), stream(predict_every=4)
    assert a.filter.module.metadata["covariance"] == "conventional"
    before = b.x.copy()
    for k in range(11):
        for s in (a, b):
            s.update_imu([0.01 * k, 0, 9.81], [0.01, 0, 0.002 * k], t=10 + k * 0.01)
        if k == 0:
            np.testing.assert_array_equal(b.x, before)
        if k and k % 3 == 0:
            for s in (a, b):
                s.update("dvl.velocity", [0.002 * k, 0, 0], t=10 + k * 0.01)
    a.flush()
    b.flush()
    assert a.time == b.time == 10.1
    np.testing.assert_allclose(a.x, b.x, atol=2e-8)
    np.testing.assert_allclose(a.P, b.P, atol=2e-8)


def test_invalid_times_are_rejected_without_consuming_pending_samples():
    s = stream(predict_every=4)
    with pytest.raises(ValueError, match="endpoint"):
        s.update("dvl.velocity", [0, 0, 0], t=0)
    s.update_imu([0, 0, 9.81], [0, 0, 0], t=0)
    with pytest.raises(ValueError, match="integrated"):
        s.update("dvl.velocity", [0, 0, 0], t=0)
    s.update_imu([0, 0, 9.81], [0, 0, 0], t=0.01)
    x, P = s.x.copy(), s.P.copy()
    for t in (0.01, 0.005, float("nan")):
        with pytest.raises(ValueError):
            s.update_imu([0, 0, 9.81], [0, 0, 0], t=t)
    for t in (0.005, 0.02, float("nan")):
        with pytest.raises(ValueError, match="endpoint"):
            s.update("dvl.velocity", [0, 0, 0], t=t)
    np.testing.assert_array_equal(s.x, x)
    np.testing.assert_array_equal(s.P, P)
    s.update("dvl.velocity", [0, 0, 0], t=0.01)
    assert s.time == 0.01
    s.reset()
    assert s.time is None
    assert not s.flush()
    s.update_imu([0, 0, 9.81], [0, 0, 0], t=100)
    assert s.time == 100


@pytest.mark.cpp
def test_stream_native_kernels_match_interpreted():
    a, b = stream(predict_every=2), stream(predict_every=2)
    b.compile(sensors=["dvl.velocity"], optimization="balanced")
    for k in range(5):
        for s in (a, b):
            s.update_imu([0.01, 0, 9.81], [0.001, 0.002, 0.003], t=k * 0.01)
            if k:
                s.update("dvl.velocity", [0.01, 0.02, 0], t=k * 0.01)
    np.testing.assert_allclose(a.x, b.x, atol=1e-12)
    np.testing.assert_allclose(a.P, b.P, atol=1e-12)

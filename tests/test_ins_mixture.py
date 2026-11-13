"""Regression checks of the retained INS posterior and its runtime boundary."""

from dataclasses import replace
import numpy as np
import pytest
from manta import INS, GaussianSumINS, Craft, World, TargetNumpy
from manta.fields import GravityField
from manta.parts import Mass, ConstantBiasIMU, VelocitySensor


def component(**kwargs):
    c = Craft("craft")
    c.add(Mass("mass", mass=1.0, moi=(1.0, 1.0, 1.0)))
    c.add(ConstantBiasIMU("imu", gyro_noise_sigma=1e-5, accel_noise_sigma=1e-4))
    c.add(VelocitySensor("dvl", velocity_noise_sigma=0.002))
    w = World("mixture_regression").add_field(GravityField(g=(0.0, 0.0, -9.81)))
    w.add_craft(c)
    return INS(
        w,
        imu="imu",
        sensors=["dvl.velocity"],
        covariance="geometric",
        gates=None,
        expand=True,
        **kwargs,
    )


def inputs():
    return {"imu.accel": [0.0, 0.0, 9.81], "imu.gyro": [0.0, 0.0, 0.0]}


def test_split_preserves_physical_mean_and_covariance():
    f = component()
    mixture = GaussianSumINS(f, order=5, split_after_s=0.2)
    P = np.diag(
        np.r_[
            [0.01] * 3,
            np.radians([0.1, 0.1, 5.0]),
            [0.001] * 3,
            [1e-6] * 3,
            [0.015] * 3,
        ]
        ** 2
    )
    # Correlated tilt and accelerometer bias must survive the split too.
    P[3, 13] = P[13, 3] = 0.4 * np.sqrt(P[3, 3] * P[13, 13])
    mixture.reset(P=P)
    x0, P0 = mixture.x, mixture.P
    mixture._x, mixture._P = map(
        np.asarray, mixture._kernel("split")(mixture._x, mixture._P)
    )
    mixture._log_weights = np.log(mixture._split_weights)
    mixture._changed()
    np.testing.assert_allclose(mixture.x, x0, atol=2e-10, rtol=1e-7)
    scale = np.sqrt(np.outer(np.diag(P0), np.diag(P0)))
    np.testing.assert_allclose((mixture.P - P0) / scale, 0.0, atol=2e-5)


def test_checkpoint_retains_components_for_future_updates():
    f = component()
    mixture = GaussianSumINS(f, order=3, split_after_s=0.2)
    for _ in range(3):
        mixture.predict(0.1, u=inputs())
        mixture.update("dvl.velocity", [0.001, 0.0, 0.0], u=inputs())
    assert mixture.component_count == 9
    checkpoint = mixture.checkpoint()
    mixture.predict(0.1, u=inputs())
    mixture.update("dvl.velocity", [0.0, 0.001, 0.0], u=inputs())
    expected = mixture.checkpoint()
    mixture.restore(checkpoint)
    mixture.predict(0.1, u=inputs())
    mixture.update("dvl.velocity", [0.0, 0.001, 0.0], u=inputs())
    actual = mixture.checkpoint()
    np.testing.assert_array_equal(actual.component_x, expected.component_x)
    np.testing.assert_array_equal(actual.component_P, expected.component_P)
    np.testing.assert_array_equal(actual.log_weights, expected.log_weights)
    with pytest.raises(ValueError):
        mixture.restore(replace(checkpoint, log_weights=np.zeros(9)))
    np.testing.assert_array_equal(mixture.checkpoint().component_x, actual.component_x)


def test_rejected_observation_does_not_reweight_or_split_mixture():
    f = component()
    mixture = GaussianSumINS(f, order=3, split_after_s=0.0, gates={"dvl.velocity": 9.0})
    before = mixture.checkpoint()
    result = mixture.update("dvl.velocity", [100.0, 100.0, 100.0], u=inputs())
    assert not result.accepted
    after = mixture.checkpoint()
    np.testing.assert_array_equal(before.component_x, after.component_x)
    np.testing.assert_array_equal(before.component_P, after.component_P)
    np.testing.assert_array_equal(before.log_weights, after.log_weights)
    assert mixture.component_count == 1


def test_single_component_update_matches_existing_manta_kernel():
    f = component()
    single = TargetNumpy(f)
    mixture = GaussianSumINS(f, order=3, split_after_s=100.0)
    for _ in range(3):
        single.predict(0.1, u=inputs())
        mixture.predict(0.1, u=inputs())
        expected = single.update("dvl.velocity", [0.001, 0.0, 0.0], u=inputs())
        actual = mixture.update("dvl.velocity", [0.001, 0.0, 0.0], u=inputs())
        np.testing.assert_allclose(mixture._x[:, 0], single.x, atol=1e-14)
        np.testing.assert_allclose(mixture._P, single.P, atol=1e-14)
        assert actual.nis == pytest.approx(expected.nis, rel=1e-12)


def test_public_compilation_and_covariance_override_match_single_filter():
    f = component()
    single = TargetNumpy(f)
    mixture = GaussianSumINS(f, order=3, split_after_s=100.0)
    assert (
        mixture.compile_sensor_updates(
            "dvl.velocity",
            covariance="model",
            optimization="O1",
            max_instructions=150000,
        )
        is mixture
    )
    assert (
        mixture.compile_sensor_updates(
            ["dvl.velocity"],
            covariance="per_sample",
            optimization="O1",
            max_instructions=150000,
        )
        is mixture
    )
    R = np.diag([0.001, 0.002, 0.003]) ** 2
    expected = single.update("dvl.velocity", [0.001, 0.0, 0.0], R=R, u=inputs())
    actual = mixture.update("dvl.velocity", [0.001, 0.0, 0.0], R=R, u=inputs())
    assert actual.covariance_overridden
    np.testing.assert_allclose(
        actual.innovation_covariance, expected.innovation_covariance
    )
    np.testing.assert_allclose(mixture._x[:, 0], single.x, atol=1e-14)
    np.testing.assert_allclose(mixture._P, single.P, atol=1e-14)

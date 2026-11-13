"""Physical position output and navigation-origin invariance of finite INS.

The independent physical startup prior is intentionally broad in heading. A
translated world coordinate system must not change pressure information or
turn stationary DVL observations into horizontal movement.
"""

import numpy as np
import pytest

from manta import INS, Craft, TargetNumpy, World
from manta.fields import FluidField, GravityField
from manta.parts import Barometer, ConstantBiasIMU, Mass, VelocitySensor


def _build(mode="geometric"):
    craft = Craft("vehicle")
    craft.add(Mass("body", mass=1, moi=(1, 1, 1)))
    craft.add(ConstantBiasIMU(
        "imu", gyro_noise_sigma=0.0005817764, accel_noise_sigma=0.0075
    ))
    craft.add(VelocitySensor("dvl", velocity_noise_sigma=0.001))
    craft.add(Barometer("pressure", pressure_noise_sigma=100))
    world = World("position_coordinates")
    world.add_field(GravityField(g=(0, 0, -9.81)))
    world.add_field(FluidField().add_flat_ocean())
    world.add_craft(craft, position=(0, 0, -0.093))
    return INS(
        world, imu="imu", sensors=["dvl.velocity", "pressure.pressure"],
        covariance=mode, gates=None, expand=mode == "geometric",
    )


@pytest.fixture(scope="module")
def ins():
    return _build()


def _indices(spec, name, *, tangent=False):
    slot = spec.slot(f"vehicle.{name}")
    offset = slot.tangent_offset if tangent else slot.ambient_offset
    dimension = slot.tangent_dim if tangent else slot.ambient_dim
    return np.arange(offset, offset + dimension)


def _reset(ins, position, *, yaw=0.0, velocity=(0, 0, 0)):
    physical_spec = ins.module().port("prior_x").spec
    sigma = np.zeros(physical_spec.tangent_dim)
    for name, value in (
        ("position", 2.0), ("velocity", 2.0),
        ("orientation", np.deg2rad([20, 20, 45])),
        ("imu.gyro_bias", 0.000145444), ("imu.accel_bias", 0.014709975),
    ):
        sigma[_indices(physical_spec, name, tangent=True)] = value
    runtime = TargetNumpy(ins)
    physical_P = np.diag(sigma**2)
    runtime.reset({"vehicle": {
        "position": position,
        "orientation": [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)],
        "velocity": velocity,
        "imu.gyro_bias": [0, 0, 0], "imu.accel_bias": [0, 0, 0],
    }}, P=physical_P)
    return runtime, physical_P


@pytest.mark.parametrize("mode", ["geometric", "nonlinear"])
def test_finite_prior_reports_ordinary_physical_position_moments(mode):
    ins = _build(mode)
    p0 = np.array([2.787, 5.460, -0.093])
    runtime, physical_P = _reset(ins, p0)
    pa = _indices(ins.spec, "position")
    pi = _indices(ins.spec, "position", tangent=True)
    np.testing.assert_allclose(runtime.state_dict()["vehicle"]["position"], p0,
                               rtol=0, atol=1e-14)
    np.testing.assert_allclose(runtime.P[np.ix_(pi, pi)],
                               physical_P[np.ix_(pi, pi)], rtol=0, atol=1e-13)
    # Independently push a symmetric Gaussian quadrature through the actual
    # finite retraction: the position moments must agree with exported x/P,
    # even with broad, correlated attitude and bias coordinates.
    eigenvalues, eigenvectors = np.linalg.eigh(runtime.P)
    root = eigenvectors @ np.diag(np.sqrt(np.maximum(eigenvalues, 0)))
    dimension = ins.spec.tangent_dim
    points = np.sqrt(dimension) * np.hstack((root, -root))
    physical = np.asarray(ins.spec.plus.map(2 * dimension)(runtime.x, points))[pa]
    assert np.all(np.isfinite(physical))
    np.testing.assert_allclose(physical.mean(axis=1), p0, rtol=0, atol=1e-13)
    residual = physical - p0[:, None]
    np.testing.assert_allclose(residual @ residual.T / (2 * dimension),
                               runtime.P[np.ix_(pi, pi)], rtol=0, atol=1e-12)
    assert ins.module().metadata["error_model"].endswith("_v4")


def test_pressure_information_is_invariant_to_horizontal_origin(ins):
    records = []
    for p0 in ([0, 0, -0.093], [2.787, 5.460, -0.093], [12.787, 0.460, -0.093]):
        runtime, _ = _reset(ins, p0)
        checkpoint = runtime.checkpoint()
        pressure = -runtime.update("pressure.pressure", [0]).innovation[0]
        runtime.restore(checkpoint)
        result = runtime.update("pressure.pressure", [pressure + 500])
        correction = runtime.x[_indices(ins.spec, "position")] - p0
        np.testing.assert_allclose(correction[:2], 0, rtol=0, atol=1e-13)
        records.append((pressure, result.nis, result.innovation_covariance, correction))
    for actual in records[1:]:
        for value, expected in zip(actual, records[0]):
            np.testing.assert_allclose(value, expected, rtol=0, atol=1e-12)


def _stationary_or_constant_velocity(ins, p0, *, yaw=0.0, velocity=(0, 0, 0)):
    c, s = np.cos(yaw), np.sin(yaw)
    rotation = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    runtime, _ = _reset(ins, p0, yaw=yaw, velocity=rotation @ velocity)
    positions, velocities, nis = [], [], []
    for step in range(201):
        if step % 20 == 0:
            nis.append(runtime.update("dvl.velocity", velocity).nis)
        positions.append(runtime.x[_indices(ins.spec, "position")])
        velocities.append(runtime.x[_indices(ins.spec, "velocity")])
        if step < 200:
            runtime.predict(0.01, u={
                "vehicle.imu.accel": [0, 0, 9.81], "vehicle.imu.gyro": [0, 0, 0],
            })
    return np.asarray(positions), np.asarray(velocities), np.asarray(nis), rotation


def test_stationary_position_does_not_move_with_coordinate_origin(ins):
    first, _, _, _ = _stationary_or_constant_velocity(ins, [0, 0, -0.093])
    shift = np.array([2.787, 5.460, 0])
    translated, _, _, _ = _stationary_or_constant_velocity(ins, shift + [0, 0, -0.093])
    np.testing.assert_allclose(first[:, :2], 0, rtol=0, atol=1e-12)
    np.testing.assert_allclose(translated, first + shift, rtol=0, atol=1e-12)


def test_finite_navigation_is_equivariant_to_yaw_about_fixed_position(ins):
    p0 = np.array([2.787, 5.460, -0.093])
    initial_body_velocity = np.array([0.2, -0.1, 0])
    p, v, nis, _ = _stationary_or_constant_velocity(
        ins, p0, velocity=initial_body_velocity
    )
    changed_p, changed_v, changed_nis, rotation = _stationary_or_constant_velocity(
        ins, p0, yaw=np.deg2rad(-115), velocity=initial_body_velocity
    )
    np.testing.assert_allclose(changed_p, (p - p0) @ rotation.T + p0,
                               rtol=0, atol=1e-10)
    np.testing.assert_allclose(changed_v, v @ rotation.T, rtol=0, atol=1e-10)
    np.testing.assert_allclose(changed_nis, nis, rtol=0, atol=1e-10)

"""Strapdown INS transform and model-force disturbance observation."""

import logging

import casadi as ca
import numpy as np

from manta import INS, Craft, NoiseFit, Prior, TargetNumpy, Window, World
from manta.estimation import nees, observability_trajectory
from manta.fields import CraftWindBubble, FluidField, GravityField
from manta.parts import IMU, DragSurface, Mass, ModelForce


def _world(*, lever=(0.0, 0.0, 0.0), angular_velocity=(0.0, 0.0, 0.0),
           wind=False):
    craft = Craft("craft")
    craft.add(Mass("body", mass=2.0, moi=(1.0, 1.0, 1.0)))
    imu = IMU(
        "imu", mount_offset=lever,
        accel_noise_sigma=0.01, gyro_noise_sigma=0.001,
        accel_bias_sigma=1e-4, gyro_bias_sigma=1e-5,
    )
    craft.add(imu)
    craft.add(ModelForce(
        "model_force", imu=imu, mount_offset=lever,
        model_error_sigma=0.5,
    ))
    world = World("ins_test").add_field(GravityField(g=(0.0, 0.0, -9.81)))
    if wind:
        craft.add(DragSurface("drag", force=(-1.0, -1.0, -1.0)))
        fluid = FluidField().add_uniform(density=1.0)
        fluid.add(CraftWindBubble(craft, sigma=1e-3))
        world.add_field(fluid)
    world.add_craft(craft, angular_velocity=angular_velocity)
    return world


def _ins(world):
    return INS(
        world, imu="craft.imu",
        sensors=["craft.model_force.specific_force"],
    )


def test_ins_state_has_navigation_biases_but_no_angular_velocity():
    ins = _ins(_world())
    names = {slot.name for slot in ins.spec.slots}
    assert {
        "craft.position", "craft.orientation", "craft.velocity",
        "craft.imu.gyro_bias", "craft.imu.accel_bias",
    } <= names
    assert "craft.angular_velocity" not in names
    assert ins.module().metadata["prediction_inputs"] == (
        "craft.imu.accel", "craft.imu.gyro")


def test_model_force_residual_directly_observes_accel_bias():
    ins = _ins(_world())
    x0 = ins.module().state.field("x").init
    H = np.asarray(ins.sys.sensors[
        "craft.model_force.specific_force"].H_fn(
            x0, ins.sys.u_defaults, 0.0, 0.0), dtype=float)
    bias = ins.spec.slot("craft.imu.accel_bias")
    H_bias = H[:, bias.tangent_offset:bias.tangent_offset + 3]
    # Innovation is r = z - h, hence dr/db_a = -H_bias = -I.
    np.testing.assert_allclose(H_bias, np.eye(3), atol=1e-12)


def test_strapdown_stationary_sample_holds_navigation_state():
    runtime = TargetNumpy(_ins(_world()))
    u = {"imu.accel": (0.0, 0.0, 9.81), "imu.gyro": (0.0, 0.0, 0.0)}
    for _ in range(100):
        runtime.predict(0.01, u=u)
    state = runtime.state_dict()["craft"]
    np.testing.assert_allclose(state["position"], np.zeros(3), atol=1e-10)
    np.testing.assert_allclose(state["velocity"], np.zeros(3), atol=1e-10)
    np.testing.assert_allclose(state["orientation"], (1, 0, 0, 0), atol=1e-10)


def test_centripetal_lever_arm_does_not_accelerate_craft_origin():
    # At r=10 cm and omega=3 rad/s the sensor sees -omega^2 r = -0.9 m/s2.
    # Correcting the rigid lever term keeps the craft origin stationary.
    runtime = TargetNumpy(_ins(_world(
        lever=(0.1, 0.0, 0.0), angular_velocity=(0.0, 0.0, 3.0))))
    u = {"imu.accel": (-0.9, 0.0, 9.81), "imu.gyro": (0.0, 0.0, 3.0)}
    for _ in range(100):
        runtime.predict(0.01, u=u)
    state = runtime.state_dict()["craft"]
    np.testing.assert_allclose(state["position"], np.zeros(3), atol=1e-8)
    np.testing.assert_allclose(state["velocity"], np.zeros(3), atol=1e-8)


def test_autodiff_F_matches_finite_difference_oracle():
    ins = _ins(_world())
    spec = ins.spec
    x = np.asarray(ins.module().state.field("x").init, dtype=float)
    u = ins.sys.resolve_u({
        "imu.accel": (0.4, -0.2, 9.9),
        "imu.gyro": (0.1, -0.05, 0.2),
    })
    dt = 0.02
    F = np.asarray(ins.sys.F_fn(x, u, dt, 0.0), dtype=float)
    nominal = np.asarray(ins.sys.predict_fn(x, u, dt, 0.0), dtype=float).ravel()
    xa = ca.MX.sym("xa", spec.ambient_dim)
    xb = ca.MX.sym("xb", spec.ambient_dim)
    boxminus = ca.Function("boxminus", [xa, xb], [spec.boxminus_sym(xa, xb)])
    eps = 1e-6
    numeric = np.zeros_like(F)
    for j in range(spec.tangent_dim):
        delta = np.zeros(spec.tangent_dim)
        delta[j] = eps
        perturbed = spec.boxplus_num(x, delta)
        predicted = np.asarray(
            ins.sys.predict_fn(perturbed, u, dt, 0.0), dtype=float).ravel()
        numeric[:, j] = np.asarray(
            boxminus(predicted, nominal), dtype=float).ravel() / eps
    np.testing.assert_allclose(F, numeric, rtol=2e-5, atol=2e-7)


def test_model_force_exposes_wind_state_to_observability():
    ins = _ins(_world(wind=True))
    assert "craft_wind.wind" in {slot.name for slot in ins.spec.slots}
    report = ins.observability(inputs={
        "imu.accel": (0.0, 0.0, 9.81),
        "imu.gyro": (0.0, 0.0, 0.0),
    })
    assert "craft.model_force.specific_force" in report.sensors


def test_rho_is_artifact_metadata_and_runtime_diagnostic():
    ins = _ins(_world())
    expected = {"craft.model_force.specific_force": 0.02}
    assert dict(ins.rho_by_sensor) == expected
    assert TargetNumpy(ins).rho_by_sensor == expected


def test_rho_has_no_hard_coded_warning_threshold(caplog):
    world = _world()
    model_force = next(
        part for part in world.crafts[0].parts if isinstance(part, ModelForce))
    model_force.model_error_sigma = 0.01  # rho = 1: intentionally arbitrary
    with caplog.at_level(logging.INFO, logger="manta.codegen.numpy._filter"):
        TargetNumpy(_ins(world))
    rho_records = [record for record in caplog.records
                   if "noise ratio rho" in record.message]
    assert rho_records
    assert all(record.levelno == logging.INFO for record in rho_records)


def test_noisefit_accepts_ins_prediction_and_measurement_sources():
    world = _world()
    ins = _ins(world)
    fit = NoiseFit(
        world,
        noise={"model_force.model_error": Prior(mean=0.5, sigma=1.0)},
        estimator=ins,
    )
    K = 4
    window = Window(
        x0={},
        z={
            "imu.accel": np.tile((0.0, 0.0, 9.81), (K, 1)),
            "imu.gyro": np.zeros((K, 3)),
        },
        dt=0.02,
    )
    _x0, U, Z, count = fit._window_arrays(window)
    assert count == K
    assert U.shape == (6, K)
    assert Z.shape == (3, K)


def test_trajectory_observability_and_nees_accept_ins_estimator():
    world = _world()
    ins = _ins(world)
    observable = observability_trajectory(
        world, dt=0.02, steps=2, estimator=ins)
    consistent = nees(
        world, dt=0.02, steps=2, runs=1, warmup=0, estimator=ins)
    assert observable.sensors == ["craft.model_force.specific_force"]
    assert consistent.samples == 2

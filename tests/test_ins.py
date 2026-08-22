"""Strapdown INS transform and model-force disturbance observation."""

import logging
import warnings

import casadi as ca
import numpy as np
import pytest

from manta import INS, Craft, NoiseFit, Prior, TargetNumpy, Window, World
from manta.estimation import nees, observability_trajectory
from manta.estimation.ins import (
    MODEL_FORCE_RHO_CEILING,
    MODEL_FORCE_RHO_WARNING,
)
from manta.fields import CraftWindBubble, FluidField, GravityField
from manta.fit import (
    AxisFitEvidence,
    FitAcceptanceCriteria,
    FitEvidence,
    HeldOutWindow,
    ProcessNoiseModel,
)
from manta.parts import IMU, DragSurface, Mass, ModelForce


def _evidence(*, white_sigma=0.5, gm=(0.2, 2.0), bias=(0.0, 0.0, 0.0),
              channel="craft.imu.accel", criteria=None):
    """Hand-built held-out evidence on the IMU accelerometer channel:
    a white floor per axis, an optional Gauss–Markov (σ, τ) component, and
    a per-axis bias. `criteria` lets a test force the acceptance decision
    either way; the decision itself is always the criteria's."""
    axes = []
    for axis, b in zip("xyz", bias):
        if gm is None:
            model = ProcessNoiseModel("white", white_sigma)
            fallback, reason = True, "fitted correlation time below dt"
        else:
            model = ProcessNoiseModel("gauss_markov", gm[0], gm[1])
            fallback, reason = False, None
        axes.append(AxisFitEvidence(
            axis=axis, sample_count=600, residual_bias=b,
            residual_bias_stderr=0.01, residual_rms=white_sigma,
            lag_one_autocorrelation=0.5, lag_count=20, fitted_tau=2.0,
            fitted_correlated_fraction=0.5, correlation_chi2=40.0,
            correlation_chi2_limit=9.21, white_floor_fraction=0.04, noise_model=model,
            white_sigma=white_sigma, autocorrelation_rmse=0.04,
            white_fallback=fallback, white_fallback_reason=reason))
    return FitEvidence.evaluate(
        channel=channel, held_out=HeldOutWindow(2, 600, 0.02, ("w0", "w1")),
        axes=axes, criteria=criteria)


DEFAULT_EVIDENCE = _evidence()


def _world(*, lever=(0.0, 0.0, 0.0), angular_velocity=(0.0, 0.0, 0.0),
           wind=False, evidence=DEFAULT_EVIDENCE):
    craft = Craft("craft")
    craft.add(Mass("body", mass=2.0, moi=(1.0, 1.0, 1.0)))
    imu = IMU(
        "imu", mount_offset=lever,
        accel_noise_sigma=0.01, gyro_noise_sigma=0.001,
        accel_bias_sigma=1e-4, gyro_bias_sigma=1e-5,
    )
    craft.add(imu)
    craft.add(ModelForce(
        "model_force", imu=imu, mount_offset=lever, evidence=evidence,
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
        # The evidence's Gauss–Markov model error is filter state.
        "craft.model_force.model_error_correlated_x",
        "craft.model_force.model_error_correlated_y",
        "craft.model_force.model_error_correlated_z",
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


def _world_with_rho(rho: float):
    # accel_noise_sigma = 0.01; the white model-error floor is evidence.
    return _world(evidence=_evidence(white_sigma=0.01 / rho))


def test_model_force_error_model_is_built_from_the_evidence():
    evidence = _evidence(white_sigma=0.4, gm=(0.3, 1.5), bias=(0.1, -0.2, 0.05))
    world = _world(evidence=evidence)
    part = next(p for p in world.crafts[0].parts if isinstance(p, ModelForce))
    assert part.evidence is evidence
    assert part.white_sigmas == (0.4, 0.4, 0.4)
    assert part.correlated_sigmas == (0.3, 0.3, 0.3)
    assert part.correlated_taus == (1.5, 1.5, 1.5)
    assert part.residual_bias == (0.1, -0.2, 0.05)
    ins = _ins(world)
    # The held-out bias is a deterministic correction of h, not a state.
    # At the initial state the craft is in free fall (specific force 0),
    # accel_bias = 0 and the correlated slots start at 0, so h(x0) = bias.
    sys = ins.sys
    sm = sys.sensors["craft.model_force.specific_force"]
    h_fn = ca.Function("h", [sys.x_sym, sys.u_sym, sys.dt_sym, sys.t_sym],
                       [sm.h_sym])
    x0 = ins.module().state.field("x").init
    h = np.asarray(h_fn(x0, sys.u_defaults, 0.0, 0.0), dtype=float).ravel()
    np.testing.assert_allclose(h, (0.1, -0.2, 0.05), atol=1e-12)
    assert "craft.model_force.residual_bias" not in {
        slot.name for slot in ins.spec.slots}
    metadata = ins.module().metadata
    # The transform snapshots the world, so the artifact carries an equal
    # (value-identical, hence identically hashed) copy of the evidence.
    assert metadata["model_force_evidence"][
        "craft.model_force.specific_force"] == evidence
    assert ins.evidence_by_sensor["craft.model_force.specific_force"] == evidence


def test_ins_refuses_model_force_without_evidence():
    with pytest.raises(ValueError, match="carries no fit evidence"):
        _ins(_world(evidence=None))


def test_ins_refuses_model_force_whose_evidence_is_not_accepted():
    rejected = _evidence(criteria=FitAcceptanceCriteria(min_samples=10_000))
    assert not rejected.accepted
    with pytest.raises(ValueError, match=r"not accepted; failed: min_samples"):
        _ins(_world(evidence=rejected))


def test_model_force_refuses_a_random_walk_model_error():
    axes = list(_evidence().axes)
    axes[1] = AxisFitEvidence(
        axis="y", sample_count=600, residual_bias=0.0,
        residual_bias_stderr=0.01, residual_rms=0.5,
        lag_one_autocorrelation=0.9, lag_count=20, fitted_tau=50.0,
        fitted_correlated_fraction=0.9, correlation_chi2=40.0,
        correlation_chi2_limit=9.21, white_floor_fraction=0.04,
        noise_model=ProcessNoiseModel("random_walk", 0.1), white_sigma=0.5,
        autocorrelation_rmse=0.04, white_fallback=False,
        white_fallback_reason=None)
    evidence = FitEvidence.evaluate(
        channel="craft.imu.accel",
        held_out=HeldOutWindow(1, 600, 0.02, ("w",)), axes=axes)
    with pytest.raises(ValueError, match="random-walk model error"):
        _world(evidence=evidence)


def test_model_force_refuses_hand_set_error_alongside_evidence():
    craft = Craft("craft")
    imu = IMU("imu", accel_noise_sigma=0.01)
    craft.add(imu)
    with pytest.raises(TypeError, match="cannot be set alongside evidence"):
        ModelForce("model_force", imu=imu, evidence=_evidence(),
                   model_error_sigma=0.3)
    with pytest.raises(ValueError, match="not the colocated IMU"):
        ModelForce("model_force", imu=imu,
                   evidence=_evidence(channel="craft.other.accel"))


def test_rho_above_the_documented_ceiling_is_refused_at_construction():
    assert 0.0 < MODEL_FORCE_RHO_WARNING < MODEL_FORCE_RHO_CEILING
    with pytest.raises(ValueError, match="rho.*MODEL_FORCE_RHO_CEILING"):
        _ins(_world_with_rho(1.0))
    with pytest.raises(ValueError, match="rho.*MODEL_FORCE_RHO_CEILING"):
        _ins(_world_with_rho(MODEL_FORCE_RHO_CEILING * 1.01))


def test_rho_between_warning_and_ceiling_warns_by_value(caplog):
    rho = 0.5 * (MODEL_FORCE_RHO_WARNING + MODEL_FORCE_RHO_CEILING)
    with pytest.warns(RuntimeWarning, match=f"rho={rho:.4g}"):
        ins = _ins(_world_with_rho(rho))
    metadata = ins.module().metadata
    assert metadata["rho_ceiling"] == MODEL_FORCE_RHO_CEILING
    assert metadata["rho_warning"] == MODEL_FORCE_RHO_WARNING
    assert metadata["rho_warned_sensors"] == (
        "craft.model_force.specific_force",)
    assert metadata["rho_by_sensor"]["craft.model_force.specific_force"] \
        == pytest.approx(rho)
    with caplog.at_level(logging.INFO, logger="manta.codegen.numpy._filter"):
        TargetNumpy(ins)
    rho_records = [record for record in caplog.records
                   if "noise ratio rho" in record.message]
    assert [record.levelno for record in rho_records] == [logging.WARNING]


def test_rho_below_the_warning_level_is_an_info_diagnostic(caplog):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ins = _ins(_world())            # rho = 0.02
    assert ins.module().metadata["rho_warned_sensors"] == ()
    with caplog.at_level(logging.INFO, logger="manta.codegen.numpy._filter"):
        TargetNumpy(ins)
    rho_records = [record for record in caplog.records
                   if "noise ratio rho" in record.message]
    assert [record.levelno for record in rho_records] == [logging.INFO]


def test_noisefit_accepts_ins_prediction_and_measurement_sources():
    world = _world()
    ins = _ins(world)
    fit = NoiseFit(
        world,
        noise={"model_force.model_error_x": Prior(mean=0.5, sigma=1.0),
               "model_force.model_error_correlated_x": Prior(sigma=1.0)},
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
    assert [c.alias for c in fit.channels] == [
        "craft.model_force.model_error_x",
        "craft.model_force.model_error_correlated_x"]


def test_trajectory_observability_and_nees_accept_ins_estimator():
    world = _world()
    ins = _ins(world)
    observable = observability_trajectory(
        world, dt=0.02, steps=2, estimator=ins)
    consistent = nees(
        world, dt=0.02, steps=2, runs=1, warmup=0, estimator=ins)
    assert observable.sensors == ["craft.model_force.specific_force"]
    assert consistent.samples == 2

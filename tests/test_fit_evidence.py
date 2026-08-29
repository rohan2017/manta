"""Typed held-out fit evidence (`manta.fit._evidence`).

The doctrine's channel: held-out residual bias and time-correlated process
covariance are artifact evidence — typed, validated, canonically hashed,
with an acceptance decision the caller cannot set. Covered here: the
artifact's validation and hashing, the pipeline's recovery of a known
bias / τ / σ² from synthetic residuals (and its recorded white fallback),
the untouched-acceptance-set guard, and the consumer refusals.
"""

import copy

import numpy as np
import pytest

from manta import Craft, Sim, TargetNumpy, Window, World
from manta.fields import GravityField
from manta.fit import (
    AcceptanceCheck,
    AxisFitEvidence,
    FitAcceptanceCriteria,
    FitDefaultFill,
    FitDerivationReport,
    FitEvidence,
    FitEvidenceBinding,
    HeldOutWindow,
    ProcessNoiseModel,
    held_out_evidence,
    hold_out,
    window_digest,
)
from manta.model import canonical_derivation_bytes
from manta.parts import IMU, Mass, Thruster

DT = 0.01


def _axis(name="x", *, bias=0.0, kind="gauss_markov", white=0.05,
          sigma=0.2, tau=0.5, samples=2000, acf_rmse=0.03):
    if kind == "gauss_markov":
        model, fallback, reason = (ProcessNoiseModel(kind, sigma, tau), False,
                                   None)
    elif kind == "white":
        model, fallback, reason = (ProcessNoiseModel(kind, white), True,
                                   "fitted correlation time below dt")
    else:
        model, fallback, reason = ProcessNoiseModel(kind, sigma), False, None
    return AxisFitEvidence(
        axis=name, sample_count=samples, residual_bias=bias,
        residual_bias_stderr=0.01, residual_rms=0.25,
        lag_one_autocorrelation=0.9, lag_count=20, fitted_tau=tau,
        fitted_correlated_fraction=0.9, correlation_chi2=40.0,
        correlation_chi2_limit=9.21, white_floor_fraction=0.04, noise_model=model, white_sigma=white,
        autocorrelation_rmse=acf_rmse, white_fallback=fallback,
        white_fallback_reason=reason)


def _held(n=2):
    return HeldOutWindow(n, 2000, DT, tuple(f"w{i}" for i in range(n)))


# ---------------------------------------------------------------------------
# Artifact validation
# ---------------------------------------------------------------------------

def test_process_noise_model_vocabulary_and_tau_rules():
    assert ProcessNoiseModel("white", 0.1).tau is None
    assert ProcessNoiseModel("gauss_markov", 0.1, 2.0).tau == 2.0
    assert ProcessNoiseModel("random_walk", 0.1).kind == "random_walk"
    with pytest.raises(ValueError, match="kind"):
        ProcessNoiseModel("pink", 0.1)
    with pytest.raises(ValueError, match="requires tau"):
        ProcessNoiseModel("gauss_markov", 0.1)
    with pytest.raises(ValueError, match="takes no tau"):
        ProcessNoiseModel("white", 0.1, 1.0)
    with pytest.raises(ValueError, match="tau"):
        ProcessNoiseModel("gauss_markov", 0.1, 0.0)
    with pytest.raises(ValueError, match="sigma"):
        ProcessNoiseModel("white", -0.1)
    with pytest.raises(ValueError, match="finite"):
        ProcessNoiseModel("white", float("nan"))


def test_held_out_window_names_every_window():
    with pytest.raises(ValueError, match="every held-out window"):
        HeldOutWindow(2, 100, DT, ("only-one",))
    with pytest.raises(ValueError, match="duplicate"):
        HeldOutWindow(2, 100, DT, ("a", "a"))
    with pytest.raises(ValueError, match="dt"):
        HeldOutWindow(1, 100, 0.0, ("a",))
    assert _held().duration_s == pytest.approx(20.0)


def test_binding_refuses_dataset_role_overlap():
    with pytest.raises(ValueError, match="overlap between training and acceptance"):
        FitEvidenceBinding(
            fitted_model_id="fitted-model",
            fitted_artifact_id="fitted-artifact",
            source_model_id="source-model",
            source_artifact_id="source-artifact",
            configuration_id="configuration", profile_id="profile",
            training_window_digests=("same",),
            selection_window_digests=(),
            acceptance_window_digests=("same",),
            channel_shape=(3,), channel_rate_hz=100.0,
            channel_contract_id="channel-contract")


def test_axis_evidence_fallback_is_never_silent():
    with pytest.raises(ValueError, match="white_fallback_reason"):
        AxisFitEvidence(
            axis="x", sample_count=10, residual_bias=0.0,
            residual_bias_stderr=0.0, residual_rms=0.1,
            lag_one_autocorrelation=0.0, lag_count=5, fitted_tau=0.001,
            fitted_correlated_fraction=0.1, correlation_chi2=1.0,
            correlation_chi2_limit=9.21, white_floor_fraction=0.04,
            noise_model=ProcessNoiseModel("white", 0.1), white_sigma=0.1,
            autocorrelation_rmse=0.0, white_fallback=True,
            white_fallback_reason=None)
    with pytest.raises(ValueError, match="white noise_model"):
        AxisFitEvidence(
            axis="x", sample_count=10, residual_bias=0.0,
            residual_bias_stderr=0.0, residual_rms=0.1,
            lag_one_autocorrelation=0.0, lag_count=5, fitted_tau=1.0,
            fitted_correlated_fraction=0.5, correlation_chi2=40.0,
            correlation_chi2_limit=9.21, white_floor_fraction=0.04,
            noise_model=ProcessNoiseModel("gauss_markov", 0.1, 1.0),
            white_sigma=0.1, autocorrelation_rmse=0.0, white_fallback=True,
            white_fallback_reason="because")
    with pytest.raises(ValueError, match="must equal white_sigma"):
        AxisFitEvidence(
            axis="x", sample_count=10, residual_bias=0.0,
            residual_bias_stderr=0.0, residual_rms=0.1,
            lag_one_autocorrelation=0.0, lag_count=5, fitted_tau=None,
            fitted_correlated_fraction=0.0, correlation_chi2=0.0,
            correlation_chi2_limit=9.21, white_floor_fraction=0.04,
            noise_model=ProcessNoiseModel("white", 0.2), white_sigma=0.1,
            autocorrelation_rmse=0.0, white_fallback=False,
            white_fallback_reason=None)


def test_accepted_is_derived_from_declared_thresholds_never_caller_set():
    axes = (_axis("x"), _axis("y"), _axis("z"))
    evidence = FitEvidence.evaluate(channel="c.imu.accel", held_out=_held(),
                                    axes=axes)
    assert evidence.accepted
    assert evidence.criteria == FitAcceptanceCriteria()
    assert {c.criterion for c in evidence.checks} == {
        "min_samples", "max_bias_ratio", "max_autocorrelation_rmse"}
    # The same axes under stricter thresholds are rejected, with the
    # failing numbers recorded.
    strict = FitEvidence.evaluate(
        channel="c.imu.accel", held_out=_held(), axes=axes,
        criteria=FitAcceptanceCriteria(max_autocorrelation_rmse=0.01,
                                       max_residual_rms=0.2))
    assert not strict.accepted
    assert {(c.criterion, c.axis) for c in strict.failed_checks} == {
        ("max_autocorrelation_rmse", a) for a in "xyz"} | {
        ("max_residual_rms", a) for a in "xyz"}
    assert "FAILED max_residual_rms[x]: 0.25" in strict.summary()
    # Caller-set acceptance is refused in both directions.
    with pytest.raises(ValueError, match="derived from the acceptance"):
        FitEvidence(channel="c.imu.accel", held_out=_held(), axes=axes,
                    criteria=strict.criteria, checks=strict.checks,
                    accepted=True)
    with pytest.raises(ValueError, match="derived from the acceptance"):
        FitEvidence(channel="c.imu.accel", held_out=_held(), axes=axes,
                    criteria=evidence.criteria, checks=evidence.checks,
                    accepted=False)
    forged = tuple(AcceptanceCheck(c.criterion, c.axis, c.value, c.limit, True)
                   for c in strict.checks)
    with pytest.raises(ValueError, match="criteria evaluated on the axes"):
        FitEvidence(channel="c.imu.accel", held_out=_held(), axes=axes,
                    criteria=strict.criteria, checks=forged, accepted=True)
    # Bias relative to the modelled σ is the bias criterion.
    biased = FitEvidence.evaluate(
        channel="c.imu.accel", held_out=_held(),
        axes=(_axis("x", bias=0.5), _axis("y"), _axis("z")))
    assert not biased.accepted
    assert biased.failed_checks[0].criterion == "max_bias_ratio"
    assert biased.axis("x").bias_ratio == pytest.approx(
        0.5 / np.hypot(0.05, 0.2))


def test_criteria_are_validated():
    with pytest.raises(ValueError, match="max_bias_ratio"):
        FitAcceptanceCriteria(max_bias_ratio=-1.0)
    with pytest.raises(ValueError, match="min_samples"):
        FitAcceptanceCriteria(min_samples=0)
    with pytest.raises(ValueError, match="max_residual_rms"):
        FitAcceptanceCriteria(max_residual_rms=0.0)


def test_default_fills_are_identity_provenance_not_acceptance_checks():
    fill = FitDefaultFill(
        dataset_role="acceptance",
        window_digest="w0",
        source="model_control_default",
        name="c.thruster.throttle",
        shape=(),
        values=(0.0,),
    )
    held = HeldOutWindow(1, 2000, DT, ("w0",))
    plain = FitEvidence.evaluate(
        channel="c.imu.accel", held_out=held,
        axes=(_axis("x"), _axis("y"), _axis("z")),
    )
    filled = FitEvidence.evaluate(
        channel="c.imu.accel", held_out=held, axes=plain.axes,
        default_fills=(fill,),
    )
    assert plain.accepted and filled.accepted
    assert all(check.criterion != "default_fill" for check in filled.checks)
    assert canonical_derivation_bytes({"evidence": plain}) != \
        canonical_derivation_bytes({"evidence": filled})
    report = FitDerivationReport(
        "parameter_fit", "src", 1.0, (), filled, default_fills=(fill,)
    )
    assert report.default_fills == (fill,)
    with pytest.raises(ValueError, match="must match its evidence"):
        FitDerivationReport("parameter_fit", "src", 1.0, (), filled)
    with pytest.raises(ValueError, match="must be finite"):
        FitDefaultFill(
            dataset_role="training",
            window_digest="w1",
            source="model_initial_state",
            name="c.position",
            shape=(1,),
            values=(float("nan"),),
        )


# ---------------------------------------------------------------------------
# Canonical hashing through ModelArtifact
# ---------------------------------------------------------------------------

def _model():
    craft = Craft("c")
    craft.add(Mass("body", mass=1.0, moi=(0.1, 0.1, 0.1)))
    world = World("evidence").add_field(GravityField(g=(0.0, 0.0, -9.81)))
    world.add_craft(craft)
    return Sim(world).model


def test_evidence_hashes_by_value_through_the_artifact_identity():
    def evidence(bias=0.0, **kw):
        return FitEvidence.evaluate(
            channel="c.imu.accel", held_out=_held(),
            axes=(_axis("x", bias=bias, **kw), _axis("y"), _axis("z")))

    def report(ev):
        return FitDerivationReport("parameter_fit", "src", 1.0, (), ev)

    first = _model().with_derivation("fit", report(evidence()))
    second = _model().with_derivation("fit", report(evidence()))
    assert first.artifact_id == second.artifact_id
    assert first.artifact_id != first.model_id
    assert canonical_derivation_bytes({"fit": report(evidence())}) \
        == canonical_derivation_bytes({"fit": report(evidence())})
    # Every evidence field is identity: bias, tau, sigma, the held-out set.
    for changed in (
        evidence(bias=0.01),
        evidence(tau=0.6),
        evidence(sigma=0.21),
        evidence(kind="white"),
    ):
        assert _model().with_derivation("fit", report(changed)).artifact_id \
            != first.artifact_id
    other_set = FitEvidence.evaluate(channel="c.imu.accel",
                                     held_out=_held(3),
                                     axes=evidence().axes)
    assert _model().with_derivation("fit", report(other_set)).artifact_id \
        != first.artifact_id
    # A report with no evidence is a distinct, unaccepted identity.
    bare = _model().with_derivation("fit", report(None))
    assert bare.artifact_id != first.artifact_id
    assert not bare.derivation["fit"].accepted
    assert first.derivation["fit"].accepted


def test_derivation_report_has_no_untyped_form():
    with pytest.raises(TypeError, match="FitEvidence"):
        FitDerivationReport("parameter_fit", "src", 1.0, (),
                            {"accepted": True})


# ---------------------------------------------------------------------------
# The pipeline on synthetic residuals with known bias / tau / sigma
# ---------------------------------------------------------------------------

def _imu_world():
    craft = Craft("c")
    craft.add(Mass("body", mass=1.0, moi=(0.1, 0.1, 0.1)))
    craft.add(IMU("imu", accel_noise_sigma=0.05))
    world = World("synthetic").add_field(GravityField(g=(0.0, 0.0, -9.81)))
    world.add_craft(craft, position=(0.0, 0.0, 20.0))
    return world


def _controlled_imu_world():
    world = _imu_world()
    world.crafts[0].add(Thruster("thruster", force_quad=(0.0, 0.0, 1.0)))
    return world


def _windows(world, *, n_win, K, rng, bias, white, gm_sigma, tau):
    """Windows whose `imu.accel` trace is the model's own mean prediction
    plus a known per-axis bias + Gauss–Markov + white error."""
    sim = TargetNumpy(Sim(world))
    phi = np.exp(-DT / tau) if tau is not None else 0.0
    windows = []
    e = np.zeros(3)
    for _ in range(n_win):
        x0 = copy.deepcopy(sim.state)
        Z = []
        for _ in range(K):
            sim.step(DT)
            pred = np.array(sim.outputs()["c"]["imu.accel"]).ravel()
            if tau is not None:
                e = phi * e + np.sqrt(1 - phi * phi) * rng.normal(
                    0.0, gm_sigma, 3)
            Z.append(pred + bias + e + rng.normal(0.0, white, 3))
        windows.append(Window(x0=x0, z={"imu.accel": np.array(Z)}, dt=DT))
    return windows


def test_pipeline_recovers_bias_tau_and_sigma_from_synthetic_residuals():
    """60 s of a τ = 0.5 s process: the sample autocorrelation pins τ to
    roughly ±30 % and the variances to ±20 %; the biases to a few
    standard errors. The large deliberate biases fail the default bias
    criterion (checked separately), so the acceptance threshold is
    declared loose here."""
    rng = np.random.default_rng(7)
    bias = np.array([0.3, -0.1, 0.0])
    windows = _windows(_imu_world(), n_win=4, K=1500, rng=rng, bias=bias,
                       white=0.05, gm_sigma=0.2, tau=0.5)
    evidence = held_out_evidence(
        _imu_world(), windows, sensor="imu.accel",
        criteria=FitAcceptanceCriteria(max_bias_ratio=5.0))
    assert evidence.channel == "c.imu.accel"
    assert evidence.held_out.window_count == 4
    assert evidence.held_out.sample_count == 6000
    assert evidence.held_out.window_digests == tuple(
        window_digest(w) for w in windows)
    assert evidence.accepted, evidence.summary()
    for ax, b in zip(evidence.axes, bias):
        assert not ax.white_fallback
        assert ax.correlation_chi2 > 10 * ax.correlation_chi2_limit
        assert ax.noise_model.kind == "gauss_markov"
        assert abs(ax.residual_bias - b) < 3.5 * ax.residual_bias_stderr + 0.01
        assert abs(ax.noise_model.tau - 0.5) / 0.5 < 0.4, ax
        assert abs(ax.noise_model.sigma - 0.2) / 0.2 < 0.25, ax
        assert abs(ax.white_sigma - 0.05) / 0.05 < 0.5, ax
        assert ax.autocorrelation_rmse < 0.05
    assert not held_out_evidence(
        _imu_world(), windows, sensor="imu.accel").accepted


def test_pipeline_records_the_white_fallback_with_its_reason():
    rng = np.random.default_rng(9)
    windows = _windows(_imu_world(), n_win=2, K=400, rng=rng,
                       bias=np.zeros(3), white=0.05, gm_sigma=0.0, tau=None)
    evidence = held_out_evidence(_imu_world(), windows, sensor="imu.accel")
    assert evidence.accepted, evidence.summary()
    for ax in evidence.axes:
        assert ax.white_fallback
        assert ax.noise_model == ProcessNoiseModel("white", ax.white_sigma)
        assert ("not significant" in ax.white_fallback_reason
                or "below the sample interval" in ax.white_fallback_reason
                or "no positive correlated variance" in ax.white_fallback_reason)
        assert ax.correlation_chi2_limit == pytest.approx(9.2103, abs=1e-3)
        assert ax.fitted_tau is not None        # the fit itself is recorded
        assert abs(ax.white_sigma - 0.05) / 0.05 < 0.15
        assert ax.autocorrelation_rmse < 0.08


def test_pipeline_records_every_missing_state_and_control_model_default():
    world = _controlled_imu_world()
    complete = _windows(
        world, n_win=1, K=100, rng=np.random.default_rng(19),
        bias=np.zeros(3), white=0.05, gm_sigma=0.0, tau=None,
    )[0]
    partial = Window(x0={}, u={}, z=complete.z, dt=complete.dt)
    evidence = held_out_evidence(
        world, [partial], sensor="imu.accel", lag_count=10
    )
    explicit = held_out_evidence(
        world,
        [Window(
            x0=complete.x0,
            u={"thruster.throttle": 0.0},
            z=complete.z,
            dt=complete.dt,
        )],
        sensor="imu.accel",
        lag_count=10,
    )

    assert evidence.accepted == explicit.accepted
    assert evidence.checks == explicit.checks
    fills = evidence.default_fills
    assert {fill.dataset_role for fill in fills} == {"acceptance"}
    assert {fill.window_digest for fill in fills} == {window_digest(partial)}
    controls = [
        fill for fill in fills if fill.source == "model_control_default"
    ]
    states = [fill for fill in fills if fill.source == "model_initial_state"]
    assert [(fill.name, fill.shape, fill.values) for fill in controls] == [
        ("c.thruster.throttle", (), (0.0,))
    ]
    assert {fill.name for fill in states} == {
        "c.angular_velocity", "c.orientation", "c.position", "c.velocity"
    }
    assert tuple(fills) == tuple(sorted(
        fills,
        key=lambda fill: (
            fill.dataset_role, fill.window_digest, fill.source, fill.name
        ),
    ))


def test_pipeline_rejects_a_biased_model_by_the_declared_criteria():
    rng = np.random.default_rng(2)
    windows = _windows(_imu_world(), n_win=2, K=400, rng=rng,
                       bias=np.array([0.0, 0.0, 0.2]), white=0.05,
                       gm_sigma=0.0, tau=None)
    evidence = held_out_evidence(_imu_world(), windows, sensor="imu.accel")
    assert not evidence.accepted
    assert [(c.criterion, c.axis) for c in evidence.failed_checks] == [
        ("max_bias_ratio", "z")]
    loose = held_out_evidence(
        _imu_world(), windows, sensor="imu.accel",
        criteria=FitAcceptanceCriteria(max_bias_ratio=10.0))
    assert loose.accepted
    assert loose.criteria.max_bias_ratio == 10.0


def test_pipeline_refuses_training_windows_short_windows_and_mixed_dt():
    rng = np.random.default_rng(1)
    windows = _windows(_imu_world(), n_win=3, K=100, rng=rng,
                       bias=np.zeros(3), white=0.05, gm_sigma=0.0, tau=None)
    training, held = hold_out(windows, fraction=0.3)
    assert len(training) == 2 and len(held) == 1 and held[0] is windows[-1]
    with pytest.raises(ValueError, match="training window"):
        held_out_evidence(_imu_world(), held, sensor="imu.accel",
                          training=[window_digest(w) for w in windows])
    with pytest.raises(ValueError, match="needs more than 150"):
        held_out_evidence(_imu_world(), held, sensor="imu.accel",
                          lag_count=150)
    mixed = [held[0], Window(x0=held[0].x0, z=held[0].z, dt=2 * DT)]
    with pytest.raises(ValueError, match="share dt"):
        held_out_evidence(_imu_world(), mixed, sensor="imu.accel")
    with pytest.raises(ValueError, match="no z trace"):
        held_out_evidence(_imu_world(), [Window(x0={}, z={
            "imu.gyro": np.zeros((100, 3))}, dt=DT)], sensor="imu.accel")
    with pytest.raises(ValueError, match="both sides"):
        hold_out(windows[:1])


def test_window_digest_is_a_content_identity():
    rng = np.random.default_rng(4)
    a, b = _windows(_imu_world(), n_win=2, K=30, rng=rng, bias=np.zeros(3),
                    white=0.05, gm_sigma=0.0, tau=None)
    assert window_digest(a) == window_digest(copy.deepcopy(a))
    assert window_digest(a) != window_digest(b)
    assert window_digest(a) != window_digest(Window(x0=a.x0, z=a.z, dt=2 * DT))

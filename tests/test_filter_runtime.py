"""Deployable filter-runtime contracts shared by EKF, UKF, and INS.

Every model-derived estimator emits the same-shaped filter Module, so the
held-state, checkpoint, gating, R-override, and generated-C++ contracts are
pinned once here over all of them — INS in both its raw-IMU and
preintegrated-packet propagation modes included, so the strapdown path is
inside the compiled-parity net rather than asserted from header text alone.

The shared world carries a first-order Gauss–Markov noise channel (the
position sensor's correlated drift) so the `exp(-dt/τ)` kernel term and its
process noise are exercised by every estimator here, including the C++
compile-and-run parity test; the inertial worlds additionally mount a
`ModelForce` built from Gauss–Markov fit evidence, the doctrine's
model-aided INS path.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from manta import (
    EKF,
    INS,
    UKF,
    Craft,
    FilterCheckpoint,
    IMUPreintegrator,
    TargetCpp,
    TargetNumpy,
    UpdateResult,
    World,
)
from manta.fields import GravityField
from manta.fit import (
    AxisFitEvidence,
    FitEvidence,
    FitEvidenceBinding,
    HeldOutWindow,
    ProcessNoiseModel,
)
from manta.ir.frames import PartFrame, WorldFrame
from manta.ir.types import Vec3
from manta.ir.wrench import Wrench
from manta.parts import (
    IMU,
    GaussMarkovNoise,
    Mass,
    ModelForce,
    Output,
    Part,
    PartUpdate,
    PositionSensor,
    VelocitySensor,
    WhiteNoise,
)

DRIFT_TAU = 4.0
DRIFT_SIGMA = 0.2
DRIFT_X0 = (0.3, -0.2, 0.1)


class DriftingPositionSensor(Part):
    """GPS with a Gauss–Markov correlated error on top of white noise."""

    position_noise = WhiteNoise("R3", frame=WorldFrame, sigma=0.0)
    position_drift = GaussMarkovNoise("R3", frame=WorldFrame, sigma=0.0,
                                      tau=DRIFT_TAU)
    position = Output()

    def update(self, ctx):
        zero = Vec3[PartFrame].constant((0.0, 0.0, 0.0))
        return PartUpdate(
            wrench=Wrench(force=zero, torque=zero),
            outputs={"position": ctx.position[WorldFrame]
                     + self.position_drift + self.position_noise})


def _model_force_evidence():
    axes = [AxisFitEvidence(
        axis=axis, sample_count=600, residual_bias=bias,
        residual_bias_stderr=0.01, residual_rms=0.5,
        lag_one_autocorrelation=0.6, lag_count=20, fitted_tau=2.0,
        fitted_correlated_fraction=0.4, correlation_chi2=40.0,
        correlation_chi2_limit=9.21, white_floor_fraction=0.04,
        noise_model=ProcessNoiseModel("gauss_markov", 0.3, 2.0),
        white_sigma=0.4, autocorrelation_rmse=0.04, white_fallback=False,
        white_fallback_reason=None)
        for axis, bias in zip("xyz", (0.02, -0.01, 0.03))]
    held = HeldOutWindow(2, 600, 0.02, ("w0", "w1"))
    binding = FitEvidenceBinding(
        fitted_model_id="test-fitted-model",
        fitted_artifact_id="test-fitted-artifact",
        source_model_id="test-source-model",
        source_artifact_id="test-source-artifact",
        configuration_id="test-configuration", profile_id="test-profile",
        training_window_digests=(), selection_window_digests=(),
        acceptance_window_digests=held.window_digests,
        channel_shape=(3,), channel_rate_hz=None,
        channel_contract_id="test-imu-accel-contract")
    return FitEvidence.evaluate(
        channel="drone.imu.accel", held_out=held, axes=axes,
        binding=binding)


def _world(*, inertial: bool = False, mount_sigma: float = 0.0):
    craft = Craft("drone")
    craft.add(Mass("body", mass=1.0, moi=(0.1, 0.1, 0.1)))
    factor = np.zeros((6, 6))
    factor[:3, :3] = np.diag((0.3, 0.2, 0.1))
    craft.add(DriftingPositionSensor(
        "gps",
        position_noise_sigma=0.5,
        position_drift_sigma=DRIFT_SIGMA,
        mount_uncertainty_sigma=mount_sigma,
        mount_uncertainty_sqrt=tuple(factor.reshape(-1)),
    ))
    if inertial:
        imu = IMU("imu", accel_noise_sigma=0.01, gyro_noise_sigma=0.001,
                  accel_bias_sigma=1e-4, gyro_bias_sigma=1e-5)
        craft.add(imu)
        craft.add(ModelForce("model_force", imu=imu,
                             evidence=_model_force_evidence()))
    world = World(name="filter_runtime")
    world.add_field(GravityField(g=(0.0, 0.0, -9.81)))
    world.add_craft(craft, position=(0.0, 0.0, 5.0),
                    **{"gps.position_drift": DRIFT_X0})
    return world


_INS_SENSORS = ["gps.position", "model_force.specific_force"]


def _ekf(**options):
    return EKF(_world(), **options)


def _ukf(**options):
    return UKF(_world(), **options)


def _ins_raw(**options):
    return INS(_world(inertial=True), imu="imu", sensors=_INS_SENSORS,
               **options)


def _ins_preintegrated(**options):
    return INS(_world(inertial=True), imu="imu", sensors=_INS_SENSORS,
               propagation="preintegrated", **options)


def _ekf_schmidt(**options):
    return EKF(_world(mount_sigma=1.0), **options)


def _ins_preintegrated_schmidt(**options):
    return INS(
        _world(inertial=True, mount_sigma=1.0),
        imu="imu",
        sensors=_INS_SENSORS,
        propagation="preintegrated",
        **options,
    )


ESTIMATORS = [
    pytest.param(_ekf, id="EKF"),
    pytest.param(_ukf, id="UKF"),
    pytest.param(_ins_raw, id="INS-raw"),
    pytest.param(_ins_preintegrated, id="INS-preintegrated"),
]

CPP_ESTIMATORS = [
    *ESTIMATORS,
    pytest.param(_ekf_schmidt, id="EKF-Schmidt"),
    pytest.param(_ins_preintegrated_schmidt, id="INS-preintegrated-Schmidt"),
]


def _advance(filt, dt: float) -> None:
    """Advance any filter view by `dt` honouring its propagation contract."""
    if filt.module.metadata.get("propagation") == "preintegrated":
        packet = TargetNumpy(IMUPreintegrator()).step(
            dt, accel=(0.0, 0.0, 9.81), gyro=(0.0, 0.0, 0.0),
            accel_bias=(0.0, 0.0, 0.0), gyro_bias=(0.0, 0.0, 0.0))
        filt.predict_preintegrated(packet)
    else:
        filt.predict(dt)


@pytest.mark.parametrize("estimator", ESTIMATORS)
def test_checkpoint_restore_is_complete_and_owned(estimator):
    filt = TargetNumpy(estimator())
    _advance(filt, 0.2)
    checkpoint = filt.checkpoint()
    assert isinstance(checkpoint, FilterCheckpoint)
    assert checkpoint.time == pytest.approx(0.2)

    expected_x, expected_P = checkpoint.x.copy(), checkpoint.P.copy()
    _advance(filt, 0.1)
    filt.update("gps.position", [1.0, 0.0, 5.0])
    filt.restore(checkpoint)
    np.testing.assert_array_equal(filt.x, expected_x)
    np.testing.assert_array_equal(filt.P, expected_P)
    assert filt.time == pytest.approx(0.2)

    # A caller mutating its old snapshot cannot mutate a restored runtime.
    with pytest.raises(ValueError, match="read-only"):
        checkpoint.x[:] = 99.0
    with pytest.raises(ValueError, match="read-only"):
        checkpoint.P[:] = 99.0
    np.testing.assert_array_equal(filt.x, expected_x)
    np.testing.assert_array_equal(filt.P, expected_P)


def test_restore_rejects_bad_checkpoint_without_partial_mutation():
    filt = TargetNumpy(EKF(_world()))
    before = filt.checkpoint()
    bad = FilterCheckpoint(before.x, -np.eye(before.P.shape[0]), before.time,
                           before.artifact_id)
    with pytest.raises(ValueError, match="positive semidefinite"):
        filt.restore(bad)
    np.testing.assert_array_equal(filt.x, before.x)
    np.testing.assert_array_equal(filt.P, before.P)
    assert filt.time == before.time


def test_restore_accepts_psd_eigensolver_roundoff_at_covariance_scale():
    filt = TargetNumpy(EKF(_world()))
    before = filt.checkpoint()
    covariance = np.eye(before.P.shape[0])
    covariance[0, 0] = 1.0e6
    covariance[-1, -1] = -1.0e-10
    checkpoint = FilterCheckpoint(
        before.x, covariance, before.time, before.artifact_id
    )

    filt.restore(checkpoint)
    np.testing.assert_array_equal(filt.P, covariance)


def test_restore_still_rejects_materially_indefinite_scaled_covariance():
    filt = TargetNumpy(EKF(_world()))
    before = filt.checkpoint()
    covariance = np.eye(before.P.shape[0])
    covariance[0, 0] = 1.0e6
    covariance[-1, -1] = -1.0e-3
    checkpoint = FilterCheckpoint(
        before.x, covariance, before.time, before.artifact_id
    )

    with pytest.raises(ValueError, match="positive semidefinite"):
        filt.restore(checkpoint)


@pytest.mark.parametrize("estimator", ESTIMATORS)
def test_same_order_replays_bit_for_bit(estimator):
    filt = TargetNumpy(estimator(gates=20.0))
    initial = filt.checkpoint()

    def sequence():
        first = filt.update("gps.position", [0.1, 0.0, 5.0])
        assert filt.time == 0.0  # updates never advance logical time
        _advance(filt, 0.05)
        second = filt.update("gps.position", [0.2, 0.0, 5.0],
                             R=np.eye(3) * 0.2)
        return filt.checkpoint(), first, second

    end_a, first_a, second_a = sequence()
    filt.restore(initial)
    end_b, first_b, second_b = sequence()
    np.testing.assert_array_equal(end_a.x, end_b.x)
    np.testing.assert_array_equal(end_a.P, end_b.P)
    assert end_a.time == end_b.time
    assert first_a.nis == first_b.nis
    assert second_a.nis == second_b.nis


@pytest.mark.parametrize("estimator", ESTIMATORS)
def test_gate_configuration_fails_locally(estimator):
    with pytest.raises(ValueError, match="finite and > 0"):
        estimator(gates=0.0)
    with pytest.raises(ValueError, match="finite and > 0"):
        estimator(gates=float("nan"))
    with pytest.raises(KeyError, match="unknown sensor"):
        estimator(gates={"missing": 10.0})


@pytest.mark.parametrize("estimator", ESTIMATORS)
def test_update_reports_innovation_nis_and_gate_disposition(estimator):
    filt = TargetNumpy(estimator(gates={"gps.position": 9.0}))
    x0, P0 = filt.x.copy(), filt.P.copy()
    rejected = filt.update("gps.position", [20.0, 0.0, 5.0])
    assert isinstance(rejected, UpdateResult)
    assert rejected.sensor == "drone.gps.position"
    assert rejected.innovation.shape == (3,)
    assert rejected.innovation_covariance.shape == (3, 3)
    assert rejected.nis > rejected.gate == 9.0
    assert not rejected.accepted
    np.testing.assert_array_equal(filt.x, x0)
    np.testing.assert_array_equal(filt.P, P0)

    accepted = filt.update("gps.position", [0.1, 0.0, 5.0])
    assert accepted.accepted
    assert accepted.nis <= accepted.gate
    assert not accepted.covariance_overridden


@pytest.mark.parametrize("estimator", ESTIMATORS)
def test_per_sample_R_override_is_typed_validated_and_effective(estimator):
    low = TargetNumpy(estimator())
    high = TargetNumpy(estimator())
    z = np.array([2.0, 0.0, 5.0])
    low_result = low.update("gps.position", z, R=np.eye(3) * 0.01)
    high_result = high.update("gps.position", z, R=np.eye(3) * 100.0)
    assert low_result.covariance_overridden
    assert high_result.covariance_overridden
    assert low_result.innovation_covariance[0, 0] \
        < high_result.innovation_covariance[0, 0]
    # Low measurement variance trusts z more strongly.
    assert abs(low.state_dict()["drone"]["position"][0]) \
        > abs(high.state_dict()["drone"]["position"][0])

    with pytest.raises(ValueError, match="shape"):
        low.update("gps.position", z, R=np.eye(2))
    with pytest.raises(ValueError, match="symmetric"):
        low.update("gps.position", z,
                   R=np.array([[1.0, 1.0, 0.0],
                               [0.0, 1.0, 0.0],
                               [0.0, 0.0, 1.0]]))
    with pytest.raises(ValueError, match="positive definite"):
        low.update("gps.position", z, R=np.zeros((3, 3)))


@pytest.mark.parametrize("operation", ("reset", "update"))
@pytest.mark.parametrize("scale", (0.0, 1e-8, 1.0, 1e8))
@pytest.mark.parametrize("within_tolerance", (True, False))
def test_covariance_symmetry_tolerance_is_elementwise(
    operation, scale, within_tolerance
):
    filt = TargetNumpy(_ekf())
    before = filt.checkpoint()
    dim = before.P.shape[0] if operation == "reset" else 3
    matrix = np.eye(dim) * max(1.0, 2 * scale)
    # An unrelated large variance must not hide an asymmetric pair. The
    # absolute floor also permits roundoff on nominally zero correlations.
    matrix[-1, -1] = 1e12
    matrix[0, 1] = scale
    matrix[1, 0] = scale + (0.5 if within_tolerance else 2.0) * (
        1e-12 + 1e-10 * scale
    )

    def apply():
        if operation == "reset":
            filt.reset(P=matrix)
        else:
            filt.update("gps.position", [0.0, 0.0, 5.0], R=matrix)

    if within_tolerance:
        apply()
    else:
        with pytest.raises(ValueError, match="symmetric"):
            apply()
        np.testing.assert_array_equal(filt.x, before.x)
        np.testing.assert_array_equal(filt.P, before.P)
        assert filt.time == before.time


@pytest.mark.parametrize("bad", (np.nan, np.inf, -np.inf))
@pytest.mark.parametrize("operation", ("reset", "update"))
def test_nonfinite_covariance_rejection_is_atomic(operation, bad):
    filt = TargetNumpy(_ekf())
    before = filt.checkpoint()
    dim = before.P.shape[0] if operation == "reset" else 3
    matrix = np.eye(dim)
    matrix[0, 0] = bad
    with pytest.raises(ValueError, match="non-finite"):
        if operation == "reset":
            filt.reset(P=matrix)
        else:
            filt.update("gps.position", [0.0, 0.0, 5.0], R=matrix)
    np.testing.assert_array_equal(filt.x, before.x)
    np.testing.assert_array_equal(filt.P, before.P)
    assert filt.time == before.time


@pytest.mark.parametrize("estimator", ESTIMATORS)
def test_reset_covariance_remains_owned(estimator):
    filt = TargetNumpy(estimator())
    supplied = filt.P
    filt.reset(P=supplied)
    expected = filt.P
    supplied[:] = np.nan
    np.testing.assert_array_equal(filt.P, expected)


@pytest.mark.parametrize("estimator", (_ekf, _ins_raw, _ins_preintegrated))
def test_schmidt_mount_posterior_is_retained_with_per_sample_R(estimator):
    baseline_transform = estimator()
    if estimator is _ekf:
        uncertain_transform = EKF(_world(mount_sigma=1.0))
    else:
        uncertain_transform = INS(
            _world(inertial=True, mount_sigma=1.0),
            imu="imu",
            sensors=_INS_SENSORS,
            propagation=(
                "preintegrated"
                if estimator is _ins_preintegrated
                else "raw"
            ),
        )
    baseline = TargetNumpy(baseline_transform)
    uncertain = TargetNumpy(uncertain_transform)
    supplied = np.eye(3) * 0.01
    z = np.array([0.0, 0.0, 5.0])
    base_result = baseline.update("gps.position", z, R=supplied)
    uncertain_result = uncertain.update("gps.position", z, R=supplied)
    np.testing.assert_allclose(
        uncertain_result.innovation_covariance
        - base_result.innovation_covariance,
        np.diag((0.3**2, 0.2**2, 0.1**2)),
        atol=1e-12,
    )
    assert uncertain.P_consider is not None


def test_ukf_refuses_static_mount_posterior_until_schmidt_is_supported():
    with pytest.raises(NotImplementedError, match="Schmidt"):
        UKF(_world(mount_sigma=1.0))


def test_mount_posterior_covariance_uses_state_dependent_sensor_jacobian():
    craft = Craft("vehicle")
    craft.add(Mass("body", mass=1.0, moi=(0.1, 0.1, 0.1)))
    factor = np.zeros((6, 6))
    factor[3:, 3:] = np.eye(3) * 0.1
    craft.add(VelocitySensor(
        "dvl",
        velocity_noise_sigma=0.01,
        mount_uncertainty_sigma=1.0,
        mount_uncertainty_sqrt=tuple(factor.reshape(-1)),
    ))
    world = World("mount_jacobian").add_field(GravityField.none())
    world.add_craft(craft)
    transform = EKF(world, sensors=["dvl.velocity"])
    runtime = TargetNumpy(transform)
    checkpoint = runtime.checkpoint()
    x = checkpoint.x.copy()
    slot = transform.sys.spec.slot("vehicle.velocity")
    x[
        slot.ambient_offset : slot.ambient_offset + slot.ambient_dim
    ] = (2.0, 0.0, 0.0)
    runtime.restore(FilterCheckpoint(
        x=x,
        P=np.zeros_like(checkpoint.P),
        time=checkpoint.time,
        artifact_id=checkpoint.artifact_id,
        P_consider=checkpoint.P_consider,
    ))
    result = runtime.update(
        "dvl.velocity", (2.0, 0.0, 0.0), R=np.eye(3) * 1e-4
    )
    # J_orientation = -skew(v) at the identity mount. Rotation about the
    # measured velocity axis is invisible; the other two 0.1-rad posterior
    # directions each contribute (2 m/s * 0.1 rad)^2.
    np.testing.assert_allclose(
        np.diag(result.innovation_covariance),
        (1e-4, 1e-4 + 0.04, 1e-4 + 0.04),
        atol=1e-12,
    )


def test_schmidt_mount_uncertainty_is_not_averaged_away_across_updates():
    craft = Craft("vehicle")
    craft.add(Mass("body", mass=1.0, moi=(0.1, 0.1, 0.1)))
    factor = np.zeros((6, 6))
    factor[:3, :3] = np.eye(3) * 0.3
    craft.add(PositionSensor(
        "gps",
        position_noise_sigma=0.01,
        mount_uncertainty_sigma=1.0,
        mount_uncertainty_sqrt=tuple(factor.reshape(-1)),
    ))
    world = World("static_mount").add_field(GravityField.none())
    world.add_craft(craft)
    transform = EKF(world, sensors=["gps.position"])
    runtime = TargetNumpy(transform)
    runtime.reset(P=np.eye(transform.spec.tangent_dim))

    for _ in range(100):
        runtime.update("gps.position", (0.0, 0.0, 0.0))

    position = transform.spec.slot("vehicle.position")
    position_variance = np.diag(runtime.P)[
        position.tangent_offset : position.tangent_offset + 3
    ]
    # Repeated readings can remove white sensor noise, but not one unknown
    # mount translation that remains the same for every reading.
    assert np.all(position_variance > 0.08)
    assert runtime.P_consider is not None
    assert np.linalg.norm(runtime.P_consider) > 0.1


def test_schmidt_restore_rejects_impossible_joint_covariance_atomically():
    runtime = TargetNumpy(_ekf_schmidt())
    before = runtime.checkpoint()
    invalid = FilterCheckpoint(
        x=before.x,
        P=before.P,
        time=before.time,
        artifact_id=before.artifact_id,
        P_consider=np.full_like(before.P_consider, 10.0),
    )

    with pytest.raises(ValueError, match="joint covariance"):
        runtime.restore(invalid)

    after = runtime.checkpoint()
    np.testing.assert_array_equal(after.x, before.x)
    np.testing.assert_array_equal(after.P, before.P)
    np.testing.assert_array_equal(after.P_consider, before.P_consider)
    assert after.time == before.time


@pytest.mark.parametrize("estimator", ESTIMATORS)
def test_gauss_markov_channel_is_filter_state_with_exact_transition(estimator):
    """The correlated drift is a tracked slot; one predict applies
    `φ = exp(-dt/τ)` to its mean and `φ²·P + (1-φ²)·σ²` to its variance —
    the exact discrete Gauss–Markov recursion, not an Euler step."""
    transform = estimator()
    slot = transform.spec.slot("drone.gps.position_drift")
    filt = TargetNumpy(transform)
    p0 = 0.05
    filt.reset(P=np.eye(transform.spec.tangent_dim) * p0)
    before = np.array(filt.state_dict()["drone"]["gps.position_drift"])
    np.testing.assert_allclose(before, DRIFT_X0)
    dt = 0.2
    _advance(filt, dt)
    phi = np.exp(-dt / DRIFT_TAU)
    after = np.array(filt.state_dict()["drone"]["gps.position_drift"])
    np.testing.assert_allclose(after, phi * before, rtol=1e-9, atol=1e-12)
    block = slice(slot.tangent_offset, slot.tangent_offset + 3)
    np.testing.assert_allclose(
        filt.P[block, block],
        np.eye(3) * (phi * phi * p0 + (1.0 - phi * phi) * DRIFT_SIGMA ** 2),
        rtol=1e-6, atol=1e-9)
    if transform.module().metadata.get("estimator") == "ins":
        evidence = transform.module().metadata["model_force_evidence"]
        assert evidence["drone.model_force.specific_force"].accepted
        assert "drone.model_force.model_error_correlated_x" in {
            s.name for s in transform.spec.slots}


@pytest.mark.parametrize("estimator", ESTIMATORS)
def test_module_exposes_compatible_and_extended_update_entries(estimator):
    module = estimator(gates=7.5).module()
    methods = {entry.method for entry in module.entry_points}
    assert "update_drone_gps_position" in methods
    assert "update_diagnostic_drone_gps_position" in methods
    assert "update_with_R_drone_gps_position" in methods
    assert module.port("R_drone_gps_position").shape == (3, 3)
    assert module.metadata["nis_gates"]["drone.gps.position"] == 7.5


# `@FIRST_SPAN@` / `@SECOND_SPAN@` bind the packet span for a preintegrated
# INS (its predict contract is dt == packet duration); `@SPAN_PROBE@` drives a
# deliberate mismatch through the generated kernel to pin its fail-loud
# behaviour (a NaN navigation state, never a mis-scaled one).
CPP_HARNESS = r"""
#include "filter.hpp"
#include <cmath>
#include <cstdio>

int main() {
    manta_gen::Filter filter;
    manta_gen::Filter::Cov P0 = manta_gen::Filter::Cov::Identity() * 0.1;
    filter.reset(manta_gen::Filter::State{}, P0);
    manta_gen::Filter::Inputs u;
    Eigen::Matrix3d R = Eigen::Matrix3d::Identity() * 0.04;
    auto out = filter.update_with_R_drone_gps_position(
        Eigen::Vector3d(0.2, 0.0, 5.0), R, u);
    auto x = filter.state();
    std::printf("accepted %.17g\n", out.accepted_drone_gps_position);
    std::printf("nis %.17g\n", out.nis_drone_gps_position);
    std::printf("innovation %.17g %.17g %.17g\n",
        out.innovation_drone_gps_position[0],
        out.innovation_drone_gps_position[1],
        out.innovation_drone_gps_position[2]);
    std::printf("x %.17g %.17g %.17g\n",
        x.drone_position[0], x.drone_position[1], x.drone_position[2]);
    std::printf("p00 %.17g\n", filter.covariance()(0, 0));
    auto checkpoint = filter.checkpoint();
    @FIRST_SPAN@
    filter.predict(u, 0.2);
    auto advanced = filter.state();
    std::printf("advanced_time %.17g\n", filter.time());
    std::printf("advanced_x %.17g %.17g %.17g\n",
        advanced.drone_position[0], advanced.drone_position[1],
        advanced.drone_position[2]);
    std::printf("advanced_p00 %.17g\n", filter.covariance()(0, 0));
    // The Gauss–Markov drift slot: exp(-dt/tau) transition + its Q.
    std::printf("advanced_drift %.17g %.17g %.17g\n",
        advanced.drone_gps_position_drift[0],
        advanced.drone_gps_position_drift[1],
        advanced.drone_gps_position_drift[2]);
    std::printf("advanced_pdrift %.17g %.17g %.17g\n",
        filter.covariance()(@DRIFT@, @DRIFT@),
        filter.covariance()(@DRIFT@ + 1, @DRIFT@ + 1),
        filter.covariance()(@DRIFT@ + 2, @DRIFT@ + 2));
    filter.restore(checkpoint);
    std::printf("restored_time %.17g\n", filter.time());
    @SECOND_SPAN@
    filter.predict(u, 0.1, 4.0);
    std::printf("resynced_time %.17g\n", filter.time());
    @SPAN_PROBE@
    return 0;
}
"""

_SPAN_PROBE = r"""
    filter.restore(checkpoint);
    u.drone_imu_preintegrated_duration = 0.05;
    filter.predict(u, 0.1);
    std::printf("mismatch_finite %d\n",
        std::isfinite(filter.state().drone_position[0]) ? 1 : 0);
"""


@pytest.mark.parametrize("estimator", CPP_ESTIMATORS)
def test_extended_update_numpy_cpp_parity(estimator, tmp_path: Path):
    cxx = next((c for c in ("c++", "g++", "clang++") if shutil.which(c)),
               None)
    cc = next((c for c in ("cc", "gcc", "clang") if shutil.which(c)), None)
    eigen = next((p for p in ("/usr/include/eigen3",
                              "/usr/local/include/eigen3")
                  if Path(p, "Eigen", "Dense").exists()), None)
    if cxx is None or cc is None or eigen is None:
        pytest.skip("C/C++ compiler (cc + c++) and Eigen headers are "
                    "required for the compile-and-run parity test")

    transform = estimator(gates=9.0)
    preintegrated = transform.module().metadata.get("propagation") == \
        "preintegrated"
    generated = TargetCpp(transform, tmp_path,
                          class_name="Filter", basename="filter")
    k_obj, w_obj = tmp_path / "kernels.o", tmp_path / "wrapper.o"
    commands = (
        [cc, "-c", "-O2", str(generated.kernels_c), "-o", str(k_obj)],
        [cxx, "-c", "-std=c++17", "-O2", f"-I{eigen}", f"-I{tmp_path}",
         str(generated.wrapper_cpp), "-o", str(w_obj)],
    )
    for command in commands:
        proc = subprocess.run(command, capture_output=True, text=True, check=False)
        assert proc.returncode == 0, proc.stderr
    source, binary = tmp_path / "main.cpp", tmp_path / "harness"
    drift = transform.spec.slot("drone.gps.position_drift").tangent_offset
    source.write_text(
        CPP_HARNESS
        .replace("@DRIFT@", str(drift))
        .replace("@FIRST_SPAN@", "u.drone_imu_preintegrated_duration = 0.2;"
                 if preintegrated else "")
        .replace("@SECOND_SPAN@", "u.drone_imu_preintegrated_duration = 0.1;"
                 if preintegrated else "")
        .replace("@SPAN_PROBE@", _SPAN_PROBE if preintegrated else ""))
    proc = subprocess.run(
        [cxx, "-std=c++17", "-O2", f"-I{eigen}", f"-I{tmp_path}",
         str(source), str(w_obj), str(k_obj), "-o", str(binary)],
        capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stderr
    proc = subprocess.run([str(binary)], capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stderr
    cpp = {line.split()[0]: np.asarray([float(v) for v in line.split()[1:]])
           for line in proc.stdout.splitlines()}

    runtime = TargetNumpy(estimator(gates=9.0))
    runtime.reset(P=np.eye(runtime.spec.tangent_dim) * 0.1)
    out = runtime.update("gps.position", [0.2, 0.0, 5.0],
                         R=np.eye(3) * 0.04)
    np.testing.assert_allclose(cpp["accepted"], [float(out.accepted)],
                               atol=1e-12)
    np.testing.assert_allclose(cpp["nis"], [out.nis], rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(cpp["innovation"], out.innovation, atol=1e-10)
    np.testing.assert_allclose(
        cpp["x"], runtime.state_dict()["drone"]["position"], atol=1e-9)
    np.testing.assert_allclose(cpp["p00"], [runtime.P[0, 0]], atol=1e-9)
    # The prediction itself (strapdown or dynamics) agrees between backends.
    if preintegrated:
        runtime.predict(0.2, u={"imu.preintegrated.duration": 0.2})
    else:
        runtime.predict(0.2)
    np.testing.assert_allclose(cpp["advanced_time"], [0.2], atol=1e-12)
    np.testing.assert_allclose(
        cpp["advanced_x"], runtime.state_dict()["drone"]["position"],
        rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(
        cpp["advanced_p00"], [runtime.P[0, 0]], rtol=1e-9, atol=1e-9)
    drift_block = slice(drift, drift + 3)
    np.testing.assert_allclose(
        cpp["advanced_drift"],
        runtime.state_dict()["drone"]["gps.position_drift"],
        rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(
        cpp["advanced_pdrift"], np.diag(runtime.P)[drift_block],
        rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(cpp["restored_time"], [0.0], atol=1e-12)
    np.testing.assert_allclose(cpp["resynced_time"], [4.1], atol=1e-12)
    if preintegrated:
        assert cpp["mismatch_finite"] == [0.0], \
            "generated C++ accepted a packet consumed over the wrong span"

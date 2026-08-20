"""Deployable filter-runtime contracts shared by EKF and UKF."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from manta import (Craft, EKF, FilterCheckpoint, TargetCpp, TargetNumpy, UKF,
                   UpdateResult, World)
from manta.fields import GravityField
from manta.parts import Mass, PositionSensor


def _world():
    craft = Craft("drone")
    craft.add(Mass("body", mass=1.0, moi=(0.1, 0.1, 0.1)))
    craft.add(PositionSensor("gps", position_noise_sigma=0.5))
    world = World(name="filter_runtime")
    world.add_field(GravityField(g=(0.0, 0.0, -9.81)))
    world.add_craft(craft, position=(0.0, 0.0, 5.0))
    return world


@pytest.mark.parametrize("estimator", [EKF, UKF])
def test_checkpoint_restore_is_complete_and_owned(estimator):
    filt = TargetNumpy(estimator(_world()))
    filt.predict(0.2)
    checkpoint = filt.checkpoint()
    assert isinstance(checkpoint, FilterCheckpoint)
    assert checkpoint.time == pytest.approx(0.2)

    expected_x, expected_P = checkpoint.x.copy(), checkpoint.P.copy()
    filt.predict(0.1)
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


@pytest.mark.parametrize("estimator", [EKF, UKF])
def test_same_order_replays_bit_for_bit(estimator):
    filt = TargetNumpy(estimator(_world(), gates=20.0))
    initial = filt.checkpoint()

    def sequence():
        first = filt.update("gps.position", [0.1, 0.0, 5.0])
        assert filt.time == 0.0  # updates never advance logical time
        filt.predict(0.05)
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


@pytest.mark.parametrize("estimator", [EKF, UKF])
def test_gate_configuration_fails_locally(estimator):
    with pytest.raises(ValueError, match="finite and > 0"):
        estimator(_world(), gates=0.0)
    with pytest.raises(ValueError, match="finite and > 0"):
        estimator(_world(), gates=float("nan"))
    with pytest.raises(KeyError, match="unknown sensor"):
        estimator(_world(), gates={"missing": 10.0})


@pytest.mark.parametrize("estimator", [EKF, UKF])
def test_update_reports_innovation_nis_and_gate_disposition(estimator):
    filt = TargetNumpy(estimator(_world(), gates={"gps.position": 9.0}))
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


@pytest.mark.parametrize("estimator", [EKF, UKF])
def test_per_sample_R_override_is_typed_validated_and_effective(estimator):
    world = _world()
    low = TargetNumpy(estimator(world))
    high = TargetNumpy(estimator(world))
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


@pytest.mark.parametrize("estimator", [EKF, UKF])
def test_module_exposes_compatible_and_extended_update_entries(estimator):
    module = estimator(_world(), gates=7.5).module()
    methods = {entry.method for entry in module.entry_points}
    assert "update_drone_gps_position" in methods
    assert "update_diagnostic_drone_gps_position" in methods
    assert "update_with_R_drone_gps_position" in methods
    assert module.port("R_drone_gps_position").shape == (3, 3)
    assert module.metadata["nis_gates"]["drone.gps.position"] == 7.5


CPP_HARNESS = r"""
#include "filter.hpp"
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
    filter.predict(u, 0.2);
    std::printf("advanced_time %.17g\n", filter.time());
    filter.restore(checkpoint);
    std::printf("restored_time %.17g\n", filter.time());
    filter.predict(u, 0.1, 4.0);
    std::printf("resynced_time %.17g\n", filter.time());
    return 0;
}
"""


@pytest.mark.parametrize("estimator", [EKF, UKF])
def test_extended_update_numpy_cpp_parity(estimator, tmp_path: Path):
    cxx = next((c for c in ("c++", "g++", "clang++") if shutil.which(c)),
               None)
    cc = next((c for c in ("cc", "gcc", "clang") if shutil.which(c)), None)
    eigen = next((p for p in ("/usr/include/eigen3",
                              "/usr/local/include/eigen3")
                  if Path(p, "Eigen", "Dense").exists()), None)
    if cxx is None or cc is None or eigen is None:
        pytest.skip("C/C++ compiler and Eigen are required")

    generated = TargetCpp(estimator(_world(), gates=9.0), tmp_path,
                          class_name="Filter", basename="filter")
    k_obj, w_obj = tmp_path / "kernels.o", tmp_path / "wrapper.o"
    commands = (
        [cc, "-c", "-O2", str(generated.kernels_c), "-o", str(k_obj)],
        [cxx, "-c", "-std=c++17", "-O2", f"-I{eigen}", f"-I{tmp_path}",
         str(generated.wrapper_cpp), "-o", str(w_obj)],
    )
    for command in commands:
        proc = subprocess.run(command, capture_output=True, text=True)
        assert proc.returncode == 0, proc.stderr
    source, binary = tmp_path / "main.cpp", tmp_path / "harness"
    source.write_text(CPP_HARNESS)
    proc = subprocess.run(
        [cxx, "-std=c++17", "-O2", f"-I{eigen}", f"-I{tmp_path}",
         str(source), str(w_obj), str(k_obj), "-o", str(binary)],
        capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    proc = subprocess.run([str(binary)], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    cpp = {line.split()[0]: np.asarray([float(v) for v in line.split()[1:]])
           for line in proc.stdout.splitlines()}

    runtime = TargetNumpy(estimator(_world(), gates=9.0))
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
    np.testing.assert_allclose(cpp["advanced_time"], [0.2], atol=1e-12)
    np.testing.assert_allclose(cpp["restored_time"], [0.0], atol=1e-12)
    np.testing.assert_allclose(cpp["resynced_time"], [4.1], atol=1e-12)

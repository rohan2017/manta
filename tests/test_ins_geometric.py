"""Contracts of the analytic finite-chart INS and explicit graph expansion."""

import itertools

import numpy as np
import pytest

from examples.qualification.earth_ins import build, prior
from manta import TargetNumpy


def test_joint_acceptance_cannot_conceal_heading_failure():
    from examples.qualification.earth_ins import final_consistency_checks

    checks = final_consistency_checks(
        {"attitude_bias_anees": 9.97, "physical_heading_anees": 1.724}, 64
    )
    assert checks["attitude_bias_anees"]["pass"]
    assert not checks["physical_heading_anees"]["pass"]


def test_turn_preserves_physical_bias_means_without_sigma_point_prediction():
    ins = build(covariance="geometric", expand=True)
    runtime = TargetNumpy(ins)
    runtime.reset(P=prior(ins, 1e-8))
    u = ins.sys.u_defaults.copy()
    u[ins.sys._input_slices[ins.sys.accel_input]] = [0, 0, 9.81]
    u[ins.sys._input_slices[ins.sys.gyro_input]] = np.asarray(
        ins.navigation_frame.angular_velocity
    ) + [0, 0.025, 0.02]
    x, P = runtime.x.copy(), runtime.P.copy()
    predicted = ins.module().functions["predict"](x, P, u, 0.01, 0)
    nodes, weights = np.polynomial.hermite.hermgauss(7)
    combinations = list(itertools.product(range(7), repeat=3))
    probabilities = np.array([np.prod(weights[list(c)]) for c in combinations])
    probabilities /= np.pi**1.5
    offsets = np.array([nodes[list(c)] * np.sqrt(2) for c in combinations]).T
    point = ins.spec.plus.map(len(combinations))

    def physical_bias_mean(x, P):
        delta = np.zeros((15, len(combinations)))
        delta[3:6] = np.linalg.cholesky(np.asarray(P)[3:6, 3:6]) @ offsets
        values = np.asarray(point(x, delta))
        # Bias enters affinely; the remaining zero-mean Euclidean errors
        # integrate out. This quadrature is independent of the analytic
        # second-order moment formula in the predictor.
        return values[10:16] @ probabilities

    np.testing.assert_allclose(
        physical_bias_mean(predicted[0], predicted[1]),
        physical_bias_mean(x, P),
        rtol=0,
        atol=2e-12,
    )


@pytest.mark.parametrize("covariance", ["linearized", "geometric", "nonlinear"])
def test_expansion_preserves_entry_contracts_and_noise_overrides(covariance):
    plain = build(covariance=covariance)
    expanded = build(covariance=covariance, expand=True)
    first, second = plain.module(), expanded.module()
    assert first.artifact_id != second.artifact_id
    assert second.metadata["expanded_filter_kernels"] is True
    assert first.entry_points == second.entry_points
    rng = np.random.default_rng(439)
    for name in first.functions:
        if not (name == "predict" or name.startswith(("predict_", "update_"))):
            continue
        f, g = first.functions[name], second.functions[name]
        assert g.is_a("SXFunction")
        values = []
        for index in range(f.n_in()):
            label, shape = f.name_in(index), f.size_in(index)
            if label == "x":
                value = first.state.field("x").init
            elif label == "P":
                value = first.state.field("P").init
            elif label == "u":
                value = plain.sys.u_defaults.copy()
                value[plain.sys._input_slices[plain.sys.accel_input]] = [0, 0, 9.81]
                value[plain.sys._input_slices[plain.sys.gyro_input]] = [
                    0.01,
                    0.02,
                    0.03,
                ]
            elif label in ("Q", "R"):
                value = np.eye(shape[0]) * 1e-6
            elif label == "dt":
                value = 0.01
            elif label == "t":
                value = 0.0
            else:
                value = rng.normal(size=shape) * 1e-4
            values.append(value)
        for expected, actual in zip(f(*values), g(*values)):
            assert np.all(np.isfinite(np.asarray(actual)))
            np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-10)

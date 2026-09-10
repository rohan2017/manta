"""Covariance contracts behind Earth-relative INS weak-bias debugging."""

import numpy as np
import pytest

from examples.qualification.earth_ins import build, normalized_nees, prior
from examples.qualification.earth_ins_debug import jacobian_audit, stationary_audit


@pytest.fixture(scope="module")
def ins():
    return build()


def test_earth_ins_process_noise_sensor_and_reset_jacobians(ins):
    # Independent central differences at a moving, tilted, biased state.
    for name, error in jacobian_audit(ins).items():
        assert error < 2e-8, (name, error)


def test_stationary_heading_bias_family_remains_locally_unobservable(ins):
    # Do not "fix" consistency by inventing heading sensitivity in F or H.
    report = stationary_audit(ins)
    for row in report["family"]:
        assert row["stationary_state_max_error"] < 1e-14
        assert row["F_direction_max_error"] < 1e-14
        assert row["H_direction_max_error"] < 1e-14


@pytest.mark.parametrize("rate", [100, 200, 500])
def test_raw_sample_noise_density_is_converted_once(rate):
    ins = build(rate=rate)
    x = np.array(ins.module().state.field("x").init, dtype=float)
    u = ins.sys.resolve_u(
        {
            "imu.gyro": ins.navigation_frame.angular_velocity,
            "imu.accel": (0, 0, 9.81),
        }
    )
    dt = 1 / rate
    zero = np.zeros((ins.spec.tangent_dim, ins.spec.tangent_dim))
    _, covariance = ins.module().functions["predict"](x, zero, u, dt, 0)
    covariance = np.asarray(covariance)
    # White density² / dt is sample variance; integration gives density²*dt.
    # Earth's sub-microradian sample rotation only contributes O((Omega*dt)²).
    for name, density in (("orientation", 1e-7), ("velocity", 1e-5)):
        slot = ins.spec.slot(f"craft.{name}")
        diagonal = np.diag(covariance)[slot.tangent_offset : slot.tangent_offset + 3]
        np.testing.assert_allclose(diagonal, density**2 * dt, rtol=1e-10, atol=0)


def test_nees_is_invariant_to_physical_units_and_does_not_regularize():
    covariance = np.array([[1.0, 0.75], [0.75, 1.0]])
    error = np.array([0.3, -0.7])
    expected = error @ np.linalg.solve(covariance, error)
    units = np.array([1e4, 1e-8])
    assert normalized_nees(
        error * units, covariance * np.outer(units, units)
    ) == pytest.approx(expected)
    with pytest.raises(np.linalg.LinAlgError):
        normalized_nees(error, np.array([[1.0, 2.0], [2.0, 1.0]]))


def test_heading_prior_control_preserves_bias_and_noise_contract(ins):
    original = prior(ins, 1e-5)
    changed = prior(ins, 1e-5, yaw_sigma_deg=0.1)
    yaw = ins.spec.slot("craft.orientation").tangent_offset + 2
    expected = original.copy()
    expected[yaw, yaw] = np.radians(0.1) ** 2
    np.testing.assert_array_equal(changed, expected)
    with pytest.raises(ValueError, match="yaw_sigma_deg"):
        prior(ins, 1e-5, yaw_sigma_deg=0)

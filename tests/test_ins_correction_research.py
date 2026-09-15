"""Independent mathematical references for opt-in correction experiments."""

import casadi as ca
import numpy as np
import pytest

from manta.estimation._ins_correction import (
    conditional_statistics,
    conditional_update,
    damped_iterated_update,
    posterior_linearized_update,
)


class Euclidean:
    slots = ()

    def __init__(self, n):
        self.tangent_dim = n

    @staticmethod
    def boxplus_sym(x, delta):
        return x+delta


def kernel(n, measurement, *, iterations=3, method="iterated"):
    x = ca.MX.sym("x", n)
    P = ca.MX.sym("P", n, n)
    z = ca.MX.sym("z", measurement.size_out(0)[0])
    R = ca.MX.sym("R", z.numel(), z.numel())
    d = ca.MX.sym("d", n)
    h = measurement(x+d)
    function = ca.Function("measurement_at_error", [x, d], [h, ca.jacobian(h, d)])
    arguments = (x, P, R, z, Euclidean(n), lambda value: function(x, value))
    if method == "iterated":
        result = damped_iterated_update(*arguments, iterations=iterations)
    elif method == "conditional":
        result = conditional_update(*arguments, axes=[0])
    elif method == "posterior":
        result = posterior_linearized_update(*arguments, axes=[0], iterations=iterations)
    else:
        raise ValueError(method)
    return ca.Function("iterated_correction", [x, P, z, R], result)


@pytest.mark.parametrize("method", ["iterated", "conditional", "posterior"])
def test_linear_update_preserves_cross_covariance_and_counts_measurement_once(method):
    x = ca.MX.sym("x", 3)
    H = np.array(((1., 2., 0.), (0., 1., 3.)))
    observation = ca.Function("linear_h", [x], [H@x])
    # Exact zero-variance direction; no inverse/floor may be inserted for P.
    P = np.array(((4., .3, 0.), (.3, 2., 0.), (0., 0., 0.)))
    mean, z, R = np.array((1., 2., 3.)), np.array((5.5, 11.2)), np.eye(2)*.1
    K = P@H.T@np.linalg.inv(H@P@H.T+R)
    expected_x = mean+K@(z-H@mean)
    A = np.eye(3)-K@H
    expected_P = A@P@A.T+K@R@K.T
    for count in (1, 3, 6):
        result = kernel(3, observation, iterations=count, method=method)(mean, P, z, R)
        np.testing.assert_allclose(np.array(result[0]).ravel(), expected_x, atol=1e-13)
        np.testing.assert_allclose(result[1], expected_P, atol=1e-13)


def test_nonlinear_gyrocompass_map_matches_independent_scalar_reference():
    x = ca.MX.sym("yaw")
    omega = 7.2921159e-5*np.cos(np.radians(37.78))
    h = ca.Function("gyrocompass", [x], [omega*ca.vertcat(ca.sin(x), ca.cos(x))])
    truth = np.radians(15.)
    variance = 1e-8**2+1e-7**2/300
    observed = omega*np.array((np.sin(truth), np.cos(truth)))
    result = kernel(1, h, iterations=4)(0, np.radians(5.)**2, observed, np.eye(2)*variance)
    # Solve the exact scalar stationarity equation using independent Newton
    # arithmetic. A 1-D posterior grid gives the same answer to this tolerance.
    reference = truth
    for _ in range(10):
        derivative = reference/np.radians(5.)**2+omega**2/variance*np.sin(reference-truth)
        hessian = 1/np.radians(5.)**2+omega**2/variance*np.cos(reference-truth)
        reference -= derivative/hessian
    np.testing.assert_allclose(float(result[0]), reference, atol=1e-10)
    assert abs(float(result[0])-truth) < np.radians(.001)
    assert float(result[1]) > 0


def test_damping_does_not_accept_increased_exact_map_cost():
    x = ca.MX.sym("x")
    h = ca.Function("exponential", [x], [ca.exp(x)])
    prior_variance, noise, observed = 4., .01, 20.
    result = kernel(1, h, iterations=3)(0., prior_variance, observed, noise)
    estimate = float(result[0])
    cost = estimate**2/prior_variance+(observed-np.exp(estimate))**2/noise
    initial_cost = (observed-1)**2/noise
    assert cost < initial_cost
    assert np.isfinite(float(result[1])) and float(result[1]) > 0


def test_conditional_quadratic_moments_against_exact_gaussian_integrals():
    x = ca.MX.sym("x", 3)
    P = ca.MX.sym("P", 3, 3)
    d = ca.MX.sym("d", 3)
    h = ca.vertcat(d[0]**2+d[1], d[0]+d[2])
    evaluate = ca.Function("quadratic", [d], [h, ca.jacobian(h, d)])
    values = conditional_statistics(x, P, ca.DM([[1., 0., 0.]]), evaluate)
    fn = ca.Function("statistics", [x, P], values[:3])
    mean = np.array((.4, -.2, .3))
    cov = np.array(((.3, .04, -.02), (.04, .5, .03), (-.02, .03, .7)))
    actual_mean, actual_cross, actual_var = fn(mean, cov)
    expected_mean = (mean[0]**2+cov[0, 0]+mean[1], mean[0]+mean[2])
    expected_cross = np.column_stack((2*mean[0]*cov[:, 0]+cov[:, 1], cov[:, 0]+cov[:, 2]))
    expected_var = np.array((
        (4*mean[0]**2*cov[0, 0]+2*cov[0, 0]**2+cov[1, 1]+4*mean[0]*cov[0, 1],
         2*mean[0]*(cov[0, 0]+cov[0, 2])+cov[0, 1]+cov[1, 2]),
        (2*mean[0]*(cov[0, 0]+cov[0, 2])+cov[0, 1]+cov[1, 2],
         cov[0, 0]+cov[2, 2]+2*cov[0, 2])))
    np.testing.assert_allclose(np.array(actual_mean).ravel(), expected_mean, atol=1e-14)
    np.testing.assert_allclose(actual_cross, expected_cross, atol=1e-14)
    np.testing.assert_allclose(actual_var, expected_var, atol=1e-14)


def test_single_posterior_iteration_is_conditional_statistical_update():
    x = ca.MX.sym("x", 3)
    h = ca.Function("quadratic", [x], [ca.vertcat(x[0]**2+x[1], x[0]+x[2])])
    prior = np.array(((.3, .04, -.02), (.04, .5, .03), (-.02, .03, .7)))
    inputs = (np.array((.4, -.2, .3)), prior, np.array((1., 2.)), np.eye(2)*.1)
    statistical = kernel(3, h, method="conditional")(*inputs)
    posterior = kernel(3, h, method="posterior", iterations=1)(*inputs)
    for first, second in zip(statistical, posterior):
        np.testing.assert_allclose(first, second, atol=1e-13)
    iterated = kernel(3, h, method="posterior", iterations=4)(*inputs)
    assert np.isfinite(iterated[0]).all()
    assert np.linalg.eigvalsh(np.array(iterated[1])).min() > 0

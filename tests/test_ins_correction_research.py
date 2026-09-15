"""Independent mathematical references for opt-in correction experiments."""

import casadi as ca
import numpy as np

from manta.estimation._ins_correction import damped_iterated_update


class Euclidean:
    slots = ()

    def __init__(self, n):
        self.tangent_dim = n

    @staticmethod
    def boxplus_sym(x, delta):
        return x+delta


def kernel(n, measurement, *, iterations=3):
    x = ca.MX.sym("x", n)
    P = ca.MX.sym("P", n, n)
    z = ca.MX.sym("z", measurement.size_out(0)[0])
    R = ca.MX.sym("R", z.numel(), z.numel())
    d = ca.MX.sym("d", n)
    h = measurement(x+d)
    function = ca.Function("measurement_at_error", [x, d], [h, ca.jacobian(h, d)])
    result = damped_iterated_update(x, P, R, z, Euclidean(n),
                                    lambda value: function(x, value), iterations=iterations)
    return ca.Function("iterated_correction", [x, P, z, R], result)


def test_linear_update_preserves_cross_covariance_and_counts_measurement_once():
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
        result = kernel(3, observation, iterations=count)(mean, P, z, R)
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

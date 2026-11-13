"""Point-sample covariance checked against derivatives of independent raw data."""

import numpy as np
import pytest

from manta import IMUPreintegrator, TargetNumpy
from tests.test_ins_gyro_endpoints import difference, exp, mul, rotate


def reference(raw, intervals, bias):
    q, v, p = np.array([1., 0, 0, 0]), np.zeros(3), np.zeros(3)
    for index, h in enumerate(intervals):
        left, right = raw[index:index+2]
        a = rotate(q, left[3:]-bias[3:])
        p += v*h + .5*a*h*h
        v += a*h
        q = mul(mul(q, exp(.5*h*(left[:3]-bias[:3]))),
                exp(.5*h*(right[:3]-bias[:3])))
    return q, v, p


def integrate(raw, intervals, *, rate=100, sg=.003, sa=.01, bias=None):
    bias = np.zeros(6) if bias is None else bias
    runtime = TargetNumpy(IMUPreintegrator(
        gyro_integration="endpoint", sample_rate_hz=rate,
        gyro_noise_density=sg, accel_noise_density=sa))
    for index, h in enumerate(intervals):
        packet = runtime.step(h, accel=raw[index, 3:], gyro=raw[index, :3],
            end_gyro=raw[index+1, :3], accel_bias=bias[3:], gyro_bias=bias[:3])
    return packet


@pytest.mark.parametrize("nonuniform", [False, True])
def test_joint_covariance_and_bias_from_all_acquisitions(nonuniform):
    rng = np.random.default_rng(9195)
    raw = rng.normal(size=(9, 6))*.4
    raw[:, 5] += 9.81
    h = np.full(len(raw)-1, .01)
    if nonuniform:
        h += rng.uniform(-.002, .002, len(h))
    rate, sg, sa = 100, .003, .01
    bias = np.array([.03, -.02, .01, -.04, .07, -.05])
    nominal = reference(raw, h, bias)
    packet = integrate(raw, h, rate=rate, sg=sg, sa=sa, bias=bias)
    for name, expected in zip(("delta_orientation", "delta_velocity", "delta_position"), nominal, strict=True):
        np.testing.assert_allclose(packet[name], expected, atol=1e-13, rtol=1e-12)

    # Differentiate every original acquisition, rather than repeating the
    # recurrence's covariance recursion. Interior gyro samples are reused.
    eps = 1e-5
    J = np.zeros((9, raw.size))
    for index in range(raw.size):
        plus, minus = raw.copy(), raw.copy()
        plus.flat[index] += eps
        minus.flat[index] -= eps
        J[:, index] = (difference(reference(plus, h, bias), nominal)
                       - difference(reference(minus, h, bias), nominal))/(2*eps)
    sigmas = np.tile(np.r_[np.full(3, sg), np.full(3, sa)]*np.sqrt(rate), len(raw))
    G = J*sigmas
    np.testing.assert_allclose(packet["covariance"].reshape(9, 9, order="F"),
                               G @ G.T, atol=2e-12, rtol=2e-6)
    for name, columns in [("delta_start_gyro_cross_covariance", slice(0, 3)),
                          ("delta_end_gyro_cross_covariance", slice(-6, -3))]:
        np.testing.assert_allclose(packet[name].reshape(9, 3, order="F"),
                                   G[:, columns], atol=2e-11, rtol=2e-6)
    np.testing.assert_array_equal(packet["start_end_gyro_correlation"], 0)
    np.testing.assert_allclose(packet["end_gyro_noise_sigma"], sg*np.sqrt(rate))
    np.testing.assert_array_equal(packet["start_gyro"], raw[0, :3])
    np.testing.assert_array_equal(packet["end_gyro"], raw[-1, :3])
    Jb = np.zeros((9, 6))
    for index in range(6):
        perturb = np.eye(6)[index]*eps
        Jb[:, index] = (difference(reference(raw, h, bias+perturb), nominal)
                        - difference(reference(raw, h, bias-perturb), nominal))/(2*eps)
    np.testing.assert_allclose(packet["bias_jacobian"].reshape(9, 6, order="F"),
                               Jb, atol=2e-10, rtol=2e-6)


def test_shared_noise_variance_uses_actual_interval_weights_fixed_sample_sigma():
    intervals = np.array([.008, .012, .011, .009, .010])
    raw = np.zeros((len(intervals)+1, 6))
    density, rate = .003, 100
    packet = integrate(raw, intervals, rate=rate, sg=density)
    weights = .5*np.r_[intervals[0], intervals[:-1]+intervals[1:], intervals[-1]]
    expected = density**2*rate*np.sum(weights**2)
    np.testing.assert_allclose(packet["covariance"].reshape(9, 9, order="F")[:3, :3],
                               expected*np.eye(3), atol=1e-18)


def test_linear_rate_does_not_leak_gravity():
    h = .01
    t = np.arange(101)*h
    angle = .1*t + .3*t*t
    raw = np.zeros((len(t), 6))
    raw[:, 0] = .1 + .6*t
    for index, a in enumerate(angle):
        raw[index, 3:] = rotate(exp([-a, 0, 0]), [0, 0, 9.81])
    packet = integrate(raw, np.full(len(t)-1, h), sg=0, sa=0)
    np.testing.assert_allclose(packet["delta_orientation"], exp([angle[-1], 0, 0]), atol=1e-13)
    np.testing.assert_allclose(packet["delta_velocity"], [0, 0, 9.81*t[-1]], atol=1e-12)


@pytest.mark.parametrize("options", [
    {"gyro_integration": "unknown"},
    {"gyro_integration": "endpoint"},
    {"gyro_integration": "endpoint", "sample_rate_hz": 0},
    {"gyro_integration": "endpoint", "sample_rate_hz": float("nan")},
    {"sample_rate_hz": 100},
])
def test_invalid_sampling_contract(options):
    with pytest.raises(ValueError):
        IMUPreintegrator(**options)

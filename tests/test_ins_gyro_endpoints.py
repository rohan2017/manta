"""Independent raw-sample derivatives and signal convergence, not self replay."""

from itertools import pairwise

import numpy as np

from manta import IMUPreintegrator
from manta.estimation._ins_gyro_endpoints import gyro_endpoint_update


def mul(a, b):
    return np.r_[a[0]*b[0]-a[1:] @ b[1:],
                 a[0]*b[1:]+b[0]*a[1:]+np.cross(a[1:], b[1:])]


def exp(w):
    w = np.asarray(w)
    angle = np.linalg.norm(w)
    return np.r_[np.cos(angle/2), .5*np.sinc(angle/(2*np.pi))*w]


def rotate(q, v):
    v = np.asarray(v)
    return v + 2*np.cross(q[1:], np.cross(q[1:], v)+q[0]*v)


def reference(raw, h, bias=None):
    bias = np.zeros(6) if bias is None else bias
    q, v, p = np.array([1., 0, 0, 0]), np.zeros(3), np.zeros(3)
    for left, right in pairwise(raw):
        a = rotate(q, left[3:]-bias[3:])
        p += v*h+.5*a*h*h
        v += a*h
        q = mul(mul(q, exp(.5*h*(left[:3]-bias[:3]))),
                exp(.5*h*(right[:3]-bias[:3])))
    return q, v, p


def difference(value, nominal):
    q = mul(nominal[0]*[1, -1, -1, -1], value[0])
    return np.r_[2*q[1:], value[1]-nominal[1], value[2]-nominal[2]]


def integrate(raw, h, *, density=.003, accel_density=.01, bias=None):
    bias = np.zeros(6) if bias is None else bias
    block = IMUPreintegrator(gyro_noise_density=density, accel_noise_density=accel_density)
    fn, x = gyro_endpoint_update(block), block.x0.copy()
    for left, right in pairwise(raw):
        # Original inputs are accel, gyro, accel bias, gyro bias.
        inputs = np.r_[left[3:], left[:3], bias[3:], bias[:3], right[:3]]
        x = np.asarray(fn(x, inputs, h, 0)[0]).ravel()
    result, off = {}, 0
    for field in block.outputs:
        result[field.name] = x[off:off+field.dim]
        off += field.dim
    return result


def test_covariance_and_bias_match_independent_raw_derivatives():
    rng = np.random.default_rng(9195)
    raw = rng.normal(size=(9, 6))*.4
    raw[:, 5] += 9.81
    h, sg, sa = .002, .003, .01
    bias = np.array([.03, -.02, .01, -.04, .07, -.05])
    nominal = reference(raw, h, bias)
    packet = integrate(raw, h, density=sg, accel_density=sa, bias=bias)
    for field, value in zip(("delta_orientation", "delta_velocity", "delta_position"), nominal, strict=True):
        np.testing.assert_allclose(packet[field], value, atol=1e-14, rtol=1e-12)
    eps = 1e-5
    J = np.zeros((9, raw.size))
    for i in range(raw.size):
        plus, minus = raw.copy(), raw.copy()
        plus.flat[i] += eps
        minus.flat[i] -= eps
        J[:, i] = (difference(reference(plus, h, bias), nominal)
                   - difference(reference(minus, h, bias), nominal))/(2*eps)
    sigmas = np.tile(np.r_[np.full(3, sg), np.full(3, sa)]/np.sqrt(h), len(raw))
    G = J*sigmas
    np.testing.assert_allclose(packet["covariance"].reshape(9, 9, order="F"),
                               G @ G.T, atol=2e-14, rtol=1e-6)
    np.testing.assert_allclose(packet["delta_start_gyro_cross_covariance"].reshape(9, 3, order="F"),
                               G[:, :3], atol=2e-12, rtol=1e-6)
    np.testing.assert_allclose(packet["delta_end_gyro_cross_covariance"].reshape(9, 3, order="F"),
                               G[:, -6:-3], atol=2e-12, rtol=1e-6)
    Jb = np.zeros((9, 6))
    for i in range(6):
        perturb = np.eye(6)[i]*eps
        Jb[:, i] = (difference(reference(raw, h, bias+perturb), nominal)
                    - difference(reference(raw, h, bias-perturb), nominal))/(2*eps)
    np.testing.assert_allclose(packet["bias_jacobian"].reshape(9, 6, order="F"),
                               Jb, atol=2e-11, rtol=1e-6)


def test_shared_interior_noise_is_not_halved():
    raw = np.zeros((21, 6))
    h, density = .002, .003
    packet = integrate(raw, h, density=density)
    # Endpoint weights h/2, 19 interior weights h, with independent samples.
    expected = density**2*h*(20-.5)
    C = packet["covariance"].reshape(9, 9, order="F")
    np.testing.assert_allclose(C[:3, :3], expected*np.eye(3), atol=1e-18)
    assert C[0, 0] > 1.9*(density**2*h*20/2)


def test_linear_axis_rate_is_exact_and_preserves_left_force_quadrature():
    h, n = .002, 100
    t = np.arange(n+1)*h
    angle = .3*t+.7*t*t
    raw = np.zeros((n+1, 6))
    raw[:, 0] = .3+1.4*t
    for i in range(n+1):
        raw[i, 3:] = rotate(exp([-angle[i], 0, 0]), [0, 0, 9.81])
    packet = integrate(raw, h, density=0, accel_density=0)
    np.testing.assert_allclose(packet["delta_orientation"], exp([angle[-1], 0, 0]), atol=1e-14)
    np.testing.assert_allclose(packet["delta_velocity"], [0, 0, 9.81*t[-1]], atol=1e-13)
    np.testing.assert_allclose(packet["delta_position"], [0, 0, .5*9.81*t[-1]**2], atol=1e-14)


def test_noncommuting_smooth_signal_has_second_order_convergence():
    def samples(h):
        t = np.arange(round(.4/h)+1)*h
        return np.c_[.3+2*t, .5*np.sin(4*t), -.2+t*t, np.zeros((len(t), 3))]
    nominal = reference(samples(.000025), .000025)
    errors = []
    for h in (.01, .005, .0025):
        errors.append(np.linalg.norm(difference(reference(samples(h), h), nominal)[:3]))
    assert 3.95 < errors[0]/errors[1] < 4.05
    assert 3.95 < errors[1]/errors[2] < 4.05

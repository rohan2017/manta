"""Independent acquisition derivatives and continuous-motion integration oracles."""

import numpy as np
import pytest
from tests._rotation_oracle import Rotation
from manta import IMUPreintegrator, TargetNumpy
from manta.estimation._ins_gyro_endpoints import compose_endpoint_packets


def integrate(raw, dt=0.01, *, sg=0.001, sa=0.01, bias=None):
    bias = np.zeros(6) if bias is None else bias
    block = TargetNumpy(
        IMUPreintegrator(
            gyro_integration="endpoint",
            accel_integration="endpoint",
            sample_rate_hz=1 / dt,
            gyro_noise_density=sg,
            accel_noise_density=sa,
        )
    )
    for left, right in zip(raw[:-1], raw[1:]):
        packet = block.step(
            dt,
            accel=left[3:],
            gyro=left[:3],
            end_accel=right[3:],
            end_gyro=right[:3],
            gyro_bias=bias[:3],
            accel_bias=bias[3:],
        )
    return packet


def reference(raw, dt, bias):
    R = Rotation.identity()
    v = np.zeros(3)
    p = np.zeros(3)
    for left, right in zip(raw[:-1] - bias, raw[1:] - bias):
        w0, w1, a0, a1 = left[:3], right[:3], left[3:], right[3:]
        middle = R * Rotation.from_rotvec((0.75 * w0 + 0.25 * w1) * dt / 2)
        final = R * Rotation.from_rotvec(
            (w0 + w1) * dt / 2 + np.cross(w0, w1) * dt * dt / 12
        )
        f0, fm, f1 = R.apply(a0), middle.apply((a0 + a1) / 2), final.apply(a1)
        p += v * dt + dt * dt * (f0 + 2 * fm) / 6
        v += dt * (f0 + 4 * fm + f1) / 6
        R = final
    return R, v, p


def difference(value, base):
    return np.r_[
        (base[0].inv() * value[0]).as_rotvec(), value[1] - base[1], value[2] - base[2]
    ]


def test_point_accel_covariance_and_bias_include_every_shared_acquisition():
    rng = np.random.default_rng(7541)
    raw = rng.normal(size=(9, 6)) * 0.4
    raw[:, 5] += 9.81
    dt = 0.01
    bias = np.array([0.02, -0.03, 0.01, 0.04, 0.07, -0.01])
    base = reference(raw, dt, bias)
    packet = integrate(raw, dt, bias=bias)
    eps = 1e-5
    J = np.zeros((9, raw.size))
    for i in range(raw.size):
        plus, minus = raw.copy(), raw.copy()
        plus.flat[i] += eps
        minus.flat[i] -= eps
        J[:, i] = (
            difference(reference(plus, dt, bias), base)
            - difference(reference(minus, dt, bias), base)
        ) / (2 * eps)
    G = J * np.tile(np.r_[[0.001] * 3, [0.01] * 3] / np.sqrt(dt), len(raw))
    np.testing.assert_allclose(
        np.asarray(packet["covariance"]).reshape(9, 9, order="F"),
        G @ G.T,
        atol=1e-12,
        rtol=1e-6,
    )
    for name, indices in [
        ("start_gyro", slice(0, 3)),
        ("start_accel", slice(3, 6)),
        ("end_gyro", slice(-6, -3)),
        ("end_accel", slice(-3, None)),
    ]:
        np.testing.assert_allclose(
            np.asarray(packet[f"delta_{name}_cross_covariance"]).reshape(
                9, 3, order="F"
            ),
            G[:, indices],
            atol=1e-11,
            rtol=1e-6,
        )
    Jb = np.column_stack(
        [
            (
                difference(reference(raw, dt, bias + d), base)
                - difference(reference(raw, dt, bias - d), base)
            )
            / (2 * eps)
            for d in np.eye(6) * eps
        ]
    )
    np.testing.assert_allclose(
        np.asarray(packet["bias_jacobian"]).reshape(9, 6, order="F"),
        Jb,
        atol=1e-10,
        rtol=1e-6,
    )


def test_accel_endpoint_linear_force_integrates_exactly():
    t = np.arange(101) * 0.01
    raw = np.zeros((len(t), 6))
    raw[:, 3] = 0.7 + 2 * t
    p = integrate(raw)
    np.testing.assert_allclose(p["delta_velocity"], [1.7, 0, 0], atol=1e-13)
    np.testing.assert_allclose(p["delta_position"], [0.35 + 1 / 3, 0, 0], atol=1e-13)


def test_accel_endpoint_shared_noise_is_not_halved():
    p = integrate(np.zeros((21, 6)))
    C = np.asarray(p["covariance"]).reshape(9, 9, order="F")
    np.testing.assert_allclose(
        C[3:6, 3:6], np.eye(3) * 0.01**2 * 0.01 * 19.5, atol=1e-17
    )


def test_accel_endpoint_packet_composition_matches_acquisitions():
    rng = np.random.default_rng(713)
    raw = rng.normal(size=(21, 6)) * 0.5
    raw[:, 5] += 9.81
    whole = integrate(raw)
    a, b, c = (integrate(raw[i : j + 1]) for i, j in [(0, 1), (1, 7), (7, 20)])
    for joined in [
        compose_endpoint_packets(compose_endpoint_packets(a, b), c),
        compose_endpoint_packets(a, compose_endpoint_packets(b, c)),
    ]:
        for key in whole:
            np.testing.assert_allclose(
                joined[key], whole[key], atol=2e-12, rtol=1e-9, err_msg=key
            )


@pytest.mark.parametrize("outer_rate", [5, 10, 100])
def test_full_ins_preserves_shared_accel_noise_across_packet_boundaries(outer_rate):
    from manta import INS, IMUPreintegrator, PreintegratedIMUPacket, TargetNumpy, Craft, World
    from manta.parts import Mass, ConstantBiasIMU
    from manta.fields import GravityField

    craft = Craft("c")
    craft.add(Mass("mass", mass=1, moi=(1, 1, 1)))
    craft.add(ConstantBiasIMU("imu", gyro_noise_sigma=0, accel_noise_sigma=0.1))
    world = World("endpoint_noise").add_field(GravityField(g=(0, 0, -9.81)))
    world.add_craft(craft)
    rt = TargetNumpy(INS(world, imu="imu", sensors=[], propagation="preintegrated",
                         accel_integration="endpoint", covariance="conventional"))
    pre = TargetNumpy(IMUPreintegrator(sample_rate_hz=100, gyro_integration="endpoint",
                                      accel_integration="endpoint", accel_noise_density=.01))
    rt.reset(P=np.zeros((15, 15)))
    start = 0.
    for i in range(1, 21):
        values = pre.step(.01, accel=[0, 0, 0], gyro=[0, 0, 0],
                          end_accel=[0, 0, 0], end_gyro=[0, 0, 0],
                          accel_bias=[0, 0, 0], gyro_bias=[0, 0, 0])
        if i % (100 // outer_rate) == 0 or i == 20:
            rt.predict(PreintegratedIMUPacket(values, start_time=start))
            start = rt.time
            pre.reset()
    slot = rt.spec.slot("c.velocity")
    i = slot.tangent_offset
    np.testing.assert_allclose(
        rt.P[i : i + 3, i : i + 3], np.eye(3) * 0.01**2 * 0.01 * 19.5, atol=1e-15
    )


def test_model_force_uses_shared_acceleration_noise_once():
    from manta import Craft, INS, IMUPreintegrator, PreintegratedIMUPacket, TargetNumpy, World
    from manta.fields import GravityField
    from manta.parts import ConstantBiasIMU, Mass, ModelForce
    from tests.test_ins import _evidence

    craft = Craft("craft")
    craft.add(Mass("body", mass=1, moi=(1, 1, 1)))
    imu = ConstantBiasIMU("imu", gyro_noise_sigma=0, accel_noise_sigma=0.1)
    craft.add(imu)
    craft.add(
        ModelForce("model_force", imu=imu, evidence=_evidence(white_sigma=0.2, gm=None))
    )
    world = World().add_field(GravityField(g=(0, 0, -9.81)))
    world.add_craft(craft)
    ins = TargetNumpy(INS(world, imu="imu", sensors=["model_force.specific_force"],
                          propagation="preintegrated", accel_integration="endpoint",
                          covariance="conventional"))
    pre = TargetNumpy(IMUPreintegrator(sample_rate_hz=100, gyro_integration="endpoint",
                                      accel_integration="endpoint", accel_noise_density=.01))
    ins.reset(P=np.zeros((15, 15)))
    values = pre.step(.01, accel=[0, 0, 0], gyro=[0, 0, 0],
                      end_accel=[0, 0, 0], end_gyro=[0, 0, 0],
                          accel_bias=[0, 0, 0], gyro_bias=[0, 0, 0])
    packet = PreintegratedIMUPacket(values, start_time=0.)
    inputs = ins.preintegrated_inputs(packet)
    ins.predict(packet)
    # v error = h/2*(n0+n1), residual error = n1+independent model error.
    # The held-out residual variance .2² already contains n1's variance .1².
    prior_variance = 0.01**2 * 0.1**2 / 2
    cross_covariance = 0.01 * 0.1**2 / 2
    slot = ins.spec.slot("craft.velocity")
    selection = slice(slot.tangent_offset, slot.tangent_offset + 3)
    np.testing.assert_allclose(
        ins.P[selection, selection], np.eye(3) * prior_variance, atol=1e-18
    )
    result = ins.update("model_force.specific_force", [0, 0, 0], u=inputs)
    np.testing.assert_allclose(
        result.innovation_covariance, np.eye(3) * 0.2**2, atol=1e-15
    )
    np.testing.assert_allclose(
        ins.P[selection, selection],
        np.eye(3) * (prior_variance - cross_covariance**2 / 0.2**2),
        atol=1e-18,
    )

"""Finite-error, physical-prior, and covariance contracts of nonlinear INS."""

import casadi as ca
import numpy as np
import pytest

from examples.qualification.earth_ins import build, prior
from manta import TargetNumpy
from manta.estimation._ins_error import INSStateSpec
from manta.estimation._ins_moments import psd_root
from manta.estimation._kalman import schmidt_update
from manta.ir._rotation import quat_to_rotmat, so3_exp


@pytest.fixture(scope="module")
def ins():
    return build(covariance="nonlinear")


@pytest.fixture(scope="module")
def chart():
    base = build()
    return INSStateSpec(
        base.spec,
        craft="craft",
        imu="craft.imu",
        rotation_body_from_imu=np.asarray(
            quat_to_rotmat(so3_exp(ca.DM([0.2, -0.4, 0.7])))
        ),
        reference_specific_force=[1.3, -2.4, 9.2],
        reference_angular_velocity=[1e-5, -4e-5, 5e-5],
    )


def test_finite_chart_and_reset_with_arbitrary_gravity_and_mount(chart):
    rng = np.random.default_rng(39105)
    x = np.zeros(16)
    x[3] = 1
    x = chart.product_spec.boxplus_num(x, rng.normal(size=15) * 0.3)
    d = rng.normal(size=15) * 0.15
    shifted = chart.boxplus_num(x, d)
    np.testing.assert_allclose(
        np.asarray(chart.boxminus_sym(shifted, x)).ravel(), d, atol=1e-13
    )
    eps = 1e-6
    derivative = np.column_stack(
        [
            (
                np.asarray(
                    chart.boxminus_sym(chart.boxplus_num(x, d + eps * e), shifted)
                ).ravel()
                - np.asarray(
                    chart.boxminus_sym(chart.boxplus_num(x, d - eps * e), shifted)
                ).ravel()
            )
            / (2 * eps)
            for e in np.eye(15)
        ]
    )
    np.testing.assert_allclose(
        derivative, chart.exact_reset(x, d), atol=2e-9, rtol=1e-7
    )
    for e in np.eye(15):
        finite = (chart.boxplus_num(x, eps * e) - chart.boxplus_num(x, -eps * e)) / (
            2 * eps
        )
        physical = (
            chart.product_spec.boxplus_num(x, eps * e)
            - chart.product_spec.boxplus_num(x, -eps * e)
        ) / (2 * eps)
        np.testing.assert_allclose(finite, physical, atol=2e-9)


def test_finite_gravity_and_body_velocity_are_affine_in_chart(chart):
    x = np.zeros(16)
    x[3] = 1
    x[7:10] = [0.5, -0.4, 0.3]
    x = chart.product_spec.boxplus_num(x, np.arange(15) * 0.01)
    d = np.linspace(-0.2, 0.3, 15)
    shifted = chart.boxplus_num(x, d)
    R = np.asarray(quat_to_rotmat(x[3:7]))
    Rt = np.asarray(quat_to_rotmat(shifted[3:7]))
    M = np.asarray(chart.mount)
    g = np.asarray(chart.gravity).ravel()
    up = np.asarray(chart.up).ravel()
    tilt = d[3:6] - up * np.dot(up, d[3:6])
    actual = M.T @ Rt.T @ g + shifted[13:16] - (M.T @ R.T @ g + x[13:16])
    expected = d[12:15] - M.T @ R.T @ np.cross(tilt, g)
    np.testing.assert_allclose(actual, expected, atol=3e-14)
    earth = np.asarray(chart.earth_rate).ravel()
    gyro_residual = M.T @ Rt.T @ earth + shifted[10:13] - (M.T @ R.T @ earth + x[10:13])
    np.testing.assert_allclose(
        gyro_residual, d[9:12] - M.T @ R.T @ np.cross(d[3:6], earth), atol=1e-16
    )
    residual = Rt.T @ shifted[7:10] - R.T @ x[7:10]
    np.testing.assert_allclose(
        residual, R.T @ (d[6:9] - np.cross(d[3:6], x[7:10])), atol=3e-14
    )


def test_psd_square_root_preserves_zero_directions_and_physical_units():
    x = ca.MX.sym("P", 5, 5)
    fn = ca.Function("root", [x], [psd_root(x)])
    scale = np.array([1e4, 1e-9, 0, 1e-5, 1.0])
    basis = np.array([[1, 0], [0.4, 0.8], [0, 0], [0.2, -0.7], [0.5, 0.3]])
    p = np.outer(scale, scale) * (basis @ basis.T)
    root = np.asarray(fn(p))
    np.testing.assert_allclose(root @ root.T, p, atol=1e-16, rtol=1e-12)
    np.testing.assert_array_equal(root[2], np.zeros(5))
    np.testing.assert_array_equal(fn(np.zeros((5, 5))), np.zeros((5, 5)))
    invalid = np.diag([1.0, 1e-18, -1e-20, 1e-8, 1.0])
    assert not np.all(np.isfinite(fn(invalid)))


def test_prior_mapping_matches_independent_physical_samples(ins):
    module = ins.module()
    physical = ins.spec.product_spec
    x = np.asarray(module.port("prior_x").init)
    p = prior(ins, 1e-5)
    mean, cov = (np.asarray(v) for v in module.functions["initialize_prior"](x, p))
    mean = mean.ravel()
    # The gravity curvature has a nonzero mean; copying P and x is incorrect.
    assert mean[15] == pytest.approx(-9.81 * np.radians(0.1) ** 2, rel=0.002)
    rng = np.random.default_rng(61093)
    count = 30000
    delta = rng.normal(size=(15, count)) * np.sqrt(np.diag(p))[:, None]
    d = ca.MX.sym("d", 15)
    e = ins.spec.boxminus_sym(physical.boxplus_sym(ca.DM(x), d), ca.DM(mean))
    errors = np.asarray(ca.Function("sample_error", [d], [e]).map(count)(delta))
    np.testing.assert_allclose(
        np.mean(errors, axis=1) / np.sqrt(np.diag(cov)), 0, atol=0.025
    )
    # Fourth moments of tilt matter for the vertical bias variance. An axial
    # 15D unscented prior gives a substantially incorrect value here.
    assert np.var(errors[14]) == pytest.approx(cov[14, 14], rel=0.04)


def test_numpy_reset_and_checkpoint_use_physical_prior_once(ins):
    runtime = TargetNumpy(ins)
    p = prior(ins, 1e-5)
    x = np.asarray(ins.module().port("prior_x").init)
    expected = ins.module().functions["initialize_prior"](x, p)
    runtime.reset(P=p)
    np.testing.assert_allclose(runtime.x, np.asarray(expected[0]).ravel(), atol=0)
    np.testing.assert_allclose(runtime.P, expected[1], atol=0)
    checkpoint = runtime.checkpoint()
    runtime.reset(P=np.zeros_like(p))
    np.testing.assert_allclose(runtime.x, x, atol=1e-15)
    np.testing.assert_allclose(runtime.P, 0, atol=1e-30)
    runtime.restore(checkpoint)
    np.testing.assert_array_equal(runtime.x, checkpoint.x)
    np.testing.assert_array_equal(runtime.P, checkpoint.P)


def test_schmidt_cross_covariance_uses_same_full_reset(ins):
    spec = ins.spec
    n = spec.tangent_dim
    x = np.asarray(ins.module().port("prior_x").init)
    p = np.eye(n) * 0.02
    c = np.arange(n * 2).reshape(n, 2) * 1e-4
    h = np.zeros(3)
    H = np.zeros((3, n))
    H[:, 3:6] = np.eye(3)
    Hc = np.array([[0.1, 0], [0, 0.2], [0.1, -0.2]])
    R = np.eye(3) * 0.01
    z = np.array([0.2, -0.1, 0.3])
    result = schmidt_update(
        ca.MX(x),
        ca.MX(p),
        ca.MX(c),
        ca.MX.eye(2),
        ca.MX(h),
        ca.MX(H),
        ca.MX(Hc),
        ca.MX(R),
        ca.MX(z),
        spec,
    )
    f = ca.Function("schmidt_result", [], list(result))
    values = f()
    S = H @ p @ H.T + H @ c @ Hc.T + Hc @ c.T @ H.T + Hc @ Hc.T + R
    K = np.linalg.solve(S, (p @ H.T + c @ Hc.T).T).T
    correction = K @ z
    G = np.asarray(spec.exact_reset(x, correction))
    expected = G @ (c - K @ (H @ c + Hc))
    np.testing.assert_allclose(values["o2"], expected, atol=1e-13)


@pytest.mark.cpp
@pytest.mark.parametrize("propagation", ["raw", "preintegrated"])
@pytest.mark.parametrize("compiler_flags", [["-O1"], ["-O3", "-march=native"]])
def test_nonlinear_numpy_cpp_prior_predict_update_and_checkpoint(
    propagation, compiler_flags, tmp_path
):
    ins = build(
        covariance="nonlinear",
        propagation=propagation,
        mounted=propagation == "preintegrated",
    )
    import shutil
    import subprocess
    from pathlib import Path

    from manta import TargetCpp

    cc = shutil.which("cc")
    cxx = shutil.which("c++")
    eigen = Path("/usr/include/eigen3")
    if not cc or not cxx or not (eigen / "Eigen/Dense").exists():
        pytest.skip("C/C++ and Eigen are required")
    generated = TargetCpp(ins, tmp_path, class_name="Filter", basename="filter")
    p = prior(ins, 1e-5)
    omega = ins.navigation_frame.angular_velocity
    inputs = {
        "craft.imu.accel": np.array([0.01, -0.02, 9.81]),
        "craft.imu.gyro": np.asarray(omega),
    }
    if propagation == "preintegrated":
        from manta import IMUPreintegrator
        from manta.estimation.imu_preintegrator import frame_preintegrated_packet

        pre = TargetNumpy(
            IMUPreintegrator(gyro_noise_density=1e-7, accel_noise_density=1e-5)
        )
        packet = pre.step(
            0.01,
            accel=inputs["craft.imu.accel"],
            gyro=omega,
            accel_bias=np.zeros(3),
            gyro_bias=np.zeros(3),
        )
        packet = frame_preintegrated_packet(
            packet,
            end_accel=inputs["craft.imu.accel"],
            end_gyro=np.asarray(omega) + [0.001, -0.002, 0.003],
            end_gyro_noise_sigma=np.full(3, 1e-6),
        )
        inputs = {
            full: np.asarray(packet[name]).ravel()
            for name, full in ins.preintegration_input_map.items()
        }
    assignments = "".join(
        f"    u.{name.replace('.', '_')} "
        + (
            f"= {float(value[0])!r};\n"
            if len(value) == 1
            else "<< " + ",".join(repr(float(v)) for v in value) + ";\n"
        )
        for name, value in inputs.items()
    )
    diagonal = ",".join(repr(float(v)) for v in np.diag(p))
    source = tmp_path / "main.cpp"
    source.write_text(
        """#include "filter.hpp"
#include <cstdio>
int main() {
    manta_gen::Filter filter;
    auto P = manta_gen::Filter::PriorCov::Zero().eval();
    double diagonal[] = {"""
        + diagonal
        + """};
    for(int i=0;i<15;++i) P(i,i)=diagonal[i];
    filter.reset(manta_gen::Filter::State{},P);
    auto saved=filter.checkpoint();
    filter.initialize_prior(manta_gen::Filter::State{},manta_gen::Filter::PriorCov::Zero());
    filter.restore(saved);
    manta_gen::Filter::Inputs u;
"""
        + assignments
        + """
    filter.predict(u,.01);
    filter.update_craft_dvl_velocity(Eigen::Vector3d(.001,-.002,.0003),u);
    auto learned=filter.checkpoint();
    filter.reset(manta_gen::Filter::State{},P);
    filter.restore(learned);
    auto x=filter.state();
"""
        + "".join(
            f'    for(int i=0;i<{slot.ambient_dim};++i) std::printf("%.17g ",x.{slot.name.replace(".", "_")}[i]);\n'
            for slot in ins.spec.slots
        )
        + f'    for(int i=0;i<{ins.spec.tangent_dim};++i) for(int j=0;j<{ins.spec.tangent_dim};++j) std::printf("%.17g ",filter.covariance()(i,j));\n'
        + "    return 0;\n}\n"
    )
    for cmd in (
        [
            cc,
            *compiler_flags,
            "-c",
            str(generated.kernels_c),
            "-o",
            str(tmp_path / "kernels.o"),
        ],
        [
            cxx,
            *compiler_flags,
            "-std=c++17",
            f"-I{eigen}",
            f"-I{tmp_path}",
            str(source),
            str(generated.wrapper_cpp),
            str(tmp_path / "kernels.o"),
            "-o",
            str(tmp_path / "run"),
        ],
    ):
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=120, check=False
        )
        assert result.returncode == 0, result.stderr
    actual = subprocess.run(
        [str(tmp_path / "run")], capture_output=True, text=True, check=True, timeout=20
    )
    values = np.fromstring(actual.stdout, sep=" ")
    runtime = TargetNumpy(ins)
    runtime.reset(P=p)
    runtime.predict(0.01, u=inputs)
    runtime.update("dvl.velocity", [0.001, -0.002, 0.0003], u=inputs)
    na, n = ins.spec.ambient_dim, ins.spec.tangent_dim
    np.testing.assert_allclose(values[:na], runtime.x, atol=1e-13, rtol=1e-10)
    covariance = values[na:].reshape(n, n)
    scale = np.sqrt(np.diag(runtime.P))
    np.testing.assert_allclose(
        (covariance - runtime.P) / np.outer(scale, scale), 0, atol=2e-9
    )


def test_stationary_gyrocompass_family_transports_without_fake_information(ins):
    from examples.qualification.earth_ins_debug import jacobian_audit, stationary_audit

    for value in jacobian_audit(ins).values():
        assert value < 2e-8
    audit = stationary_audit(ins)
    np.testing.assert_allclose(
        audit["product_reset_heading_direction_mismatch"], 0, atol=1e-14
    )
    for row in audit["family"]:
        assert row["F_direction_max_error"] < 1e-14
        assert row["H_direction_max_error"] < 1e-14


def test_qualification_reuses_fresh_packet_boundary_as_next_start(monkeypatch):
    from examples.qualification import earth_ins_split as fixture

    original = fixture.frame_preintegrated_packet
    previous = [None, None]
    calls = 0

    def frame(packet, **kwargs):
        nonlocal calls
        seed = calls % 2
        calls += 1
        if previous[seed] is not None:
            np.testing.assert_array_equal(packet["start_gyro"], previous[seed])
        previous[seed] = np.asarray(kwargs["end_gyro"]).copy()
        framed = original(packet, **kwargs)
        np.testing.assert_array_equal(
            framed["delta_end_gyro_cross_covariance"], np.zeros(27)
        )
        np.testing.assert_array_equal(framed["start_end_gyro_correlation"], np.zeros(9))
        return framed

    monkeypatch.setattr(fixture, "frame_preintegrated_packet", frame)
    result = fixture.run(
        duration=10,
        seeds=2,
        covariance="nonlinear",
        gyro_density=0.001,
        bias_sigma=0.001,
    )
    assert calls == 200
    assert result["fixture_schema"] == 2


def test_noisy_zero_spin_keeps_unobservable_heading_uncertainty():
    from manta.codegen.numpy._compile import compile_functions

    ir = build(covariance="nonlinear", spin=0, gyro_density=0.001)
    module = ir.module()
    n = ir.spec.tangent_dim
    x = np.asarray(module.port("prior_x").init)
    p = prior(ir, 0.001)
    x, p = module.functions["initialize_prior"](x, p)
    sx = ca.MX.sym("x", ir.spec.ambient_dim)
    sp = ca.MX.sym("p", n, n)
    u = ca.MX.sym("u", len(ir.sys.u_defaults))
    z = ca.MX.sym("z", 3)
    predicted = module.functions["predict"](sx, sp, u, 0.01, 0)
    updated = module.functions["update_craft_dvl_velocity"](*predicted, z, u, 0)
    fn = ca.Function("zero_spin_confidence", [sx, sp, u, z], list(updated))
    fn = compile_functions(
        {"zero_spin_confidence": fn}, optimization="O1", max_instructions=30000
    )["zero_spin_confidence"]
    rng = np.random.default_rng(73119)
    bias = rng.normal(size=3) * 0.001
    for _ in range(6000):
        inputs = ir.sys.u_defaults.copy()
        inputs[ir.sys._input_slices[ir.sys.accel_input]] = [0, 0, 9.81] + rng.normal(
            size=3
        ) * 1e-4
        inputs[ir.sys._input_slices[ir.sys.gyro_input]] = (
            bias + rng.normal(size=3) * 0.01
        )
        x, p = fn(x, p, inputs, rng.normal(size=3) * 0.001)
    expected = np.radians(5.0) ** 2 + (0.001 * 60.0) ** 2 + 0.001**2 * 60.0
    assert float(p[5, 5]) == pytest.approx(expected, rel=0.03)


def test_nonlinear_displaced_imu_requires_measured_endpoints():
    with pytest.raises(NotImplementedError, match="timestamped gyro endpoints"):
        build(covariance="nonlinear", mounted=True)


def test_local_chart_refuses_to_wrap_broad_uncertainty(ins):
    x = np.asarray(ins.module().port("prior_x").init)
    delta = np.zeros(ins.spec.tangent_dim)
    delta[5] = np.pi + 0.01
    assert not np.all(np.isfinite(ins.spec.boxplus_num(x, delta)))
    runtime = TargetNumpy(ins)
    old = runtime.checkpoint()
    with pytest.raises(ValueError, match="non-finite"):
        runtime.reset(P=prior(ins, 1e-5, yaw_sigma_deg=120))
    np.testing.assert_array_equal(runtime.x, old.x)
    np.testing.assert_array_equal(runtime.P, old.P)


def test_packet_boundary_error_is_estimated_then_replaced():
    from manta import IMUPreintegrator
    from manta.estimation.imu_preintegrator import frame_preintegrated_packet

    ins = build(
        covariance="nonlinear",
        propagation="preintegrated",
        mounted=True,
        gyro_density=0.001,
    )
    runtime = TargetNumpy(ins)
    runtime.reset(P=prior(ins, 0.001))
    np.testing.assert_array_equal(runtime.x[-3:], 0)
    np.testing.assert_allclose(runtime.P[-3:, -3:], np.eye(3), atol=1e-14)
    assert runtime.P_consider is None
    assert runtime.module.port("prior_P").shape == (15, 15)
    R = np.asarray(quat_to_rotmat(runtime.x[3:7])) @ ins.sys.R_craft_from_sensor
    a = R.T @ np.array([0, 0, 9.81])
    g = R.T @ np.asarray(ins.navigation_frame.angular_velocity)
    pre = TargetNumpy(
        IMUPreintegrator(gyro_noise_density=0.001, accel_noise_density=1e-5)
    )
    packet = pre.step(
        0.01, accel=a, gyro=g, accel_bias=np.zeros(3), gyro_bias=np.zeros(3)
    )
    endpoint = g + [0.01, -0.02, 0.003]
    packet = frame_preintegrated_packet(packet, end_accel=a, end_gyro=endpoint)
    runtime.predict_preintegrated(packet)
    np.testing.assert_allclose(runtime.P[-3:, -3:], np.eye(3), atol=1e-12)
    runtime.update("dvl.velocity", np.zeros(3))
    assert np.linalg.norm(runtime.x[-3:]) > 0.01
    assert np.trace(runtime.P[-3:, -3:]) < 2.9
    saved = runtime.checkpoint()
    runtime.reset(P=prior(ins, 0.001))
    runtime.restore(saved)
    np.testing.assert_array_equal(runtime.x, saved.x)
    np.testing.assert_array_equal(runtime.P, saved.P)
    pre.reset()
    packet = pre.step(
        0.01, accel=a, gyro=endpoint, accel_bias=np.zeros(3), gyro_bias=np.zeros(3)
    )
    packet = frame_preintegrated_packet(packet, end_accel=a, end_gyro=g)
    runtime.predict_preintegrated(packet)
    np.testing.assert_allclose(runtime.x[-3:], 0, atol=1e-14)
    np.testing.assert_allclose(runtime.P[-3:, -3:], np.eye(3), atol=1e-12)


def test_packet_active_boundary_retains_static_schmidt_installation_covariance():
    from manta import INS, Craft, IMUPreintegrator, World
    from manta.fields import GravityField
    from manta.parts import ConstantBiasIMU, Mass, PositionSensor

    craft = Craft("craft")
    craft.add(Mass("mass", mass=1, moi=(1, 1, 1)))
    craft.add(ConstantBiasIMU("imu"))
    factor = np.zeros((6, 6))
    factor[:3, :3] = 0.3 * np.eye(3)
    craft.add(
        PositionSensor(
            "gps",
            position_noise_sigma=0.01,
            mount_uncertainty_sigma=1,
            mount_uncertainty_sqrt=tuple(factor.ravel()),
        )
    )
    world = World("nonlinear_schmidt").add_field(GravityField(g=(0, 0, -9.81)))
    world.add_craft(craft)
    ins = INS(
        world,
        imu="imu",
        sensors=["gps.position"],
        covariance="nonlinear",
        propagation="preintegrated",
    )
    runtime = TargetNumpy(ins)
    p0 = np.zeros((15, 15))
    p0[:3, :3] = np.eye(3)
    runtime.reset(P=p0)
    for _ in range(100):
        runtime.update("gps.position", np.zeros(3))
    assert np.all(np.diag(runtime.P)[:3] > 0.08)
    assert runtime.P_consider.shape == (18, 6)
    assert np.linalg.norm(runtime.P_consider) > 0.1
    pre = TargetNumpy(IMUPreintegrator())
    packet = pre.step(
        0.01,
        accel=[0, 0, 9.81],
        gyro=np.zeros(3),
        accel_bias=np.zeros(3),
        gyro_bias=np.zeros(3),
    )
    runtime.predict_preintegrated(packet)
    assert np.all(np.diag(runtime.P)[:3] > 0.08)
    np.testing.assert_allclose(runtime.P[-3:, -3:], np.eye(3), atol=1e-12)

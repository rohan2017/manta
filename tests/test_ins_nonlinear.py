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
def test_nonlinear_numpy_cpp_prior_predict_update_and_checkpoint(ins, tmp_path):
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
    diagonal = ",".join(repr(float(v)) for v in np.diag(p))
    source = tmp_path / "main.cpp"
    source.write_text(
        """#include "filter.hpp"
#include <cstdio>
int main() {
    manta_gen::Filter filter;
    auto P = manta_gen::Filter::Cov::Zero().eval();
    double diagonal[] = {"""
        + diagonal
        + """};
    for(int i=0;i<15;++i) P(i,i)=diagonal[i];
    filter.reset(manta_gen::Filter::State{},P);
    auto saved=filter.checkpoint();
    filter.initialize_prior(manta_gen::Filter::State{},manta_gen::Filter::Cov::Zero());
    filter.restore(saved);
    manta_gen::Filter::Inputs u;
    u.craft_imu_accel << 0.01,-0.02,9.81;
    u.craft_imu_gyro << """
        + ",".join(repr(float(v)) for v in omega)
        + """;
    filter.predict(u,.01);
    filter.update_craft_dvl_velocity(Eigen::Vector3d(.001,-.002,.0003),u);
    auto x=filter.state();
"""
        + "".join(
            f'    for(int i=0;i<{slot.ambient_dim};++i) std::printf("%.17g ",x.{slot.name.replace(".", "_")}[i]);\n'
            for slot in ins.spec.slots
        )
        + """
    for(int i=0;i<15;++i) for(int j=0;j<15;++j) std::printf("%.17g ",filter.covariance()(i,j));
    return 0;
}
"""
    )
    for cmd in (
        [cc, "-O1", "-c", str(generated.kernels_c), "-o", str(tmp_path / "kernels.o")],
        [
            cxx,
            "-O1",
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
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        assert result.returncode == 0, result.stderr
    actual = subprocess.run(
        [str(tmp_path / "run")], capture_output=True, text=True, check=True, timeout=20
    )
    values = np.fromstring(actual.stdout, sep=" ")
    runtime = TargetNumpy(ins)
    runtime.reset(P=p)
    u = {"imu.accel": [0.01, -0.02, 9.81], "imu.gyro": omega}
    runtime.predict(0.01, u=u)
    runtime.update("dvl.velocity", [0.001, -0.002, 0.0003], u=u)
    np.testing.assert_allclose(values[:16], runtime.x, atol=1e-13, rtol=1e-10)
    covariance = values[16:].reshape(15, 15)
    scale = np.sqrt(np.diag(runtime.P))
    np.testing.assert_allclose(
        (covariance - runtime.P) / np.outer(scale, scale), 0, atol=2e-9
    )

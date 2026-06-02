"""Mahony AHRS recurrence block — numpy behavior + C++ roundtrip.

Like Madgwick, Mahony adds zero backend code. These tests pin the
convention, the defining gyro-bias estimation, and a compiled Python↔C++
roundtrip of both the quaternion trajectory and the learned bias.
"""

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from manta import Mahony, TargetCpp, TargetNumpy
from manta.codegen.block import KIND_RECURRENCE, block_kind


def test_mahony_is_recurrence_block():
    f = Mahony(kp=0.5, ki=0.1)
    assert block_kind(f) == KIND_RECURRENCE
    assert [p.name for p in f.inputs] == ["gyro", "accel"]
    assert [p.name for p in f.outputs] == ["orientation"]
    assert [s.name for s in f.spec.slots] == ["orientation", "gyro_bias"]


def test_mahony_gyro_only_integration():
    """kp = ki = 0 ⇒ pure gyro integration about body z."""
    r = TargetNumpy(Mahony(kp=0.0, ki=0.0))
    dt, wz, T = 0.001, 2.0, 1.0
    for _ in range(int(T / dt)):
        q = r.step(dt, gyro=[0.0, 0.0, wz], accel=[0.0, 0.0, 1.0])["orientation"]
    theta = wz * T
    np.testing.assert_allclose(
        q, [np.cos(theta / 2), 0.0, 0.0, np.sin(theta / 2)], atol=1e-3)


def test_mahony_estimates_gyro_bias():
    """A constant rate-gyro offset, with a level accelerometer, is learned
    into `gyro_bias` (on the gravity-observable axes) so the attitude does
    not drift. Yaw-axis bias is unobservable from gravity and stays ~0."""
    r = TargetNumpy(Mahony(kp=1.0, ki=0.5))
    true_bias = np.array([0.02, -0.03, 0.0])
    for _ in range(40000):
        q = r.step(0.005, gyro=true_bias, accel=[0.0, 0.0, 1.0])["orientation"]
    b_est = np.asarray(r.state["gyro_bias"])
    # The estimate cancels the true bias on x/y (corrected gyro ≈ 0 there).
    np.testing.assert_allclose(b_est[:2], -true_bias[:2], atol=1e-3)
    q0, q1, q2, q3 = q
    g_pred = np.array([2 * (q1 * q3 - q0 * q2),
                       2 * (q0 * q1 + q2 * q3),
                       1 - 2 * q1 * q1 - 2 * q2 * q2])
    tilt = np.degrees(np.arccos(np.clip(g_pred @ [0, 0, 1], -1, 1)))
    assert tilt < 0.1


def test_mahony_no_bias_when_ki_zero():
    """ki = 0 disables the integral term ⇒ the bias state never moves."""
    r = TargetNumpy(Mahony(kp=1.0, ki=0.0))
    for _ in range(100):
        r.step(0.01, gyro=[0.1, 0.1, 0.1], accel=[0.0, 0.0, 9.81])
    assert np.allclose(np.asarray(r.state["gyro_bias"]), 0.0)


# --- C++ roundtrip ----------------------------------------------------------

_GYRO = [[0.1, -0.2, 0.05], [0.3, 0.1, -0.1], [-0.2, 0.2, 0.15],
         [0.0, -0.3, 0.2], [0.25, 0.1, -0.05]]
_ACCEL = [[0.0, 0.0, 9.81], [0.5, 0.0, 9.7], [0.3, -0.4, 9.6],
          [-0.2, 0.3, 9.75], [0.1, 0.1, 9.8]]
_DT = 0.02

_HARNESS = r"""
#include "mahony.hpp"
#include <cstdio>

int main() {
    manta_gen::Mahony f;
    const double g[5][3] = {{0.1,-0.2,0.05},{0.3,0.1,-0.1},{-0.2,0.2,0.15},
                            {0.0,-0.3,0.2},{0.25,0.1,-0.05}};
    const double a[5][3] = {{0.0,0.0,9.81},{0.5,0.0,9.7},{0.3,-0.4,9.6},
                            {-0.2,0.3,9.75},{0.1,0.1,9.8}};
    for (int i = 0; i < 5; ++i) {
        manta_gen::Mahony::Inputs u;
        u.gyro  << g[i][0], g[i][1], g[i][2];
        u.accel << a[i][0], a[i][1], a[i][2];
        auto o = f.step(u, 0.02, 0.0);
        std::printf("q %.17g %.17g %.17g %.17g\n",
                    o.orientation[0], o.orientation[1],
                    o.orientation[2], o.orientation[3]);
    }
    std::printf("b %.17g %.17g %.17g\n",
                f.state().gyro_bias[0], f.state().gyro_bias[1],
                f.state().gyro_bias[2]);
    return 0;
}
"""


def test_mahony_emits_cpp_files(tmp_path: Path):
    result = TargetCpp(Mahony(kp=0.5, ki=0.1), tmp_path, class_name="Mahony")
    for p in (result.kernels_c, result.kernels_h, result.wrapper_hpp,
              result.wrapper_cpp, result.cmakelists):
        assert p.exists(), p


def test_mahony_python_cpp_roundtrip(tmp_path: Path):
    cxx = next((c for c in ("c++", "g++", "clang++") if shutil.which(c)), None)
    cc = next((c for c in ("cc", "gcc", "clang") if shutil.which(c)), None)
    if cxx is None or cc is None:
        pytest.skip("no C/C++ compiler on PATH")
    eigen_inc = next((p for p in ("/usr/include/eigen3",
                                  "/usr/local/include/eigen3")
                      if Path(p, "Eigen", "Dense").exists()), None)
    if eigen_inc is None:
        pytest.skip("Eigen headers not found")

    f = Mahony(kp=0.5, ki=0.1)
    result = TargetCpp(f, tmp_path, class_name="Mahony")

    k_obj, w_obj = tmp_path / "k.o", tmp_path / "w.o"
    for cmd in (
        [cc, "-c", "-O2", "-fPIC", str(result.kernels_c), "-o", str(k_obj)],
        [cxx, "-c", "-std=c++17", "-O2", "-fPIC", f"-I{eigen_inc}",
         f"-I{tmp_path}", str(result.wrapper_cpp), "-o", str(w_obj)],
    ):
        p = subprocess.run(cmd, capture_output=True, text=True)
        assert p.returncode == 0, p.stderr

    h_src = tmp_path / "harness_main.cpp"
    h_src.write_text(_HARNESS)
    binary = tmp_path / "harness"
    p = subprocess.run(
        [cxx, "-std=c++17", "-O2", f"-I{eigen_inc}", f"-I{tmp_path}",
         str(h_src), str(w_obj), str(k_obj), "-o", str(binary)],
        capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    p = subprocess.run([str(binary)], capture_output=True, text=True)
    assert p.returncode == 0, p.stderr

    cpp_q, cpp_b = [], None
    for line in p.stdout.strip().splitlines():
        tok = line.split()
        if tok[0] == "q":
            cpp_q.append([float(x) for x in tok[1:]])
        elif tok[0] == "b":
            cpp_b = [float(x) for x in tok[1:]]

    r = TargetNumpy(f)
    np_q = [list(r.step(_DT, gyro=g, accel=a)["orientation"])
            for g, a in zip(_GYRO, _ACCEL)]
    np_b = list(np.asarray(r.state["gyro_bias"]))

    np.testing.assert_allclose(cpp_q, np_q, atol=1e-12)
    np.testing.assert_allclose(cpp_b, np_b, atol=1e-12)

"""Madgwick AHRS recurrence block — numpy behavior + C++ roundtrip.

Madgwick adds zero backend code (it reuses the generic `lower_recurrence`
path PID established). These tests pin the convention (gyro integration,
accelerometer convergence, quaternion stays unit) and prove the compiled
C++ `step()` reproduces the numpy quaternion trajectory exactly.
"""

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from manta import Madgwick, TargetCpp, TargetNumpy


def test_madgwick_is_recurrence_block():
    f = Madgwick(beta=0.1)
    from manta.ir.module import Hosting
    assert f.module().hosting is Hosting.HELD
    assert [p.name for p in f.inputs] == ["gyro", "accel"]
    assert [p.name for p in f.outputs] == ["orientation"]


def test_madgwick_gyro_only_integration():
    """beta = 0 ⇒ pure gyro integration: a constant body-z rate yields a
    rotation about z by ω·t."""
    r = TargetNumpy(Madgwick(beta=0.0))
    dt, wz, T = 0.001, 2.0, 1.0
    for _ in range(int(T / dt)):
        q = r.step(dt, gyro=[0.0, 0.0, wz], accel=[0.0, 0.0, 1.0])["orientation"]
    theta = wz * T
    np.testing.assert_allclose(
        q, [np.cos(theta / 2), 0.0, 0.0, np.sin(theta / 2)], atol=1e-3)


def test_madgwick_accel_convergence():
    """With gyro at rest, the estimate's predicted gravity converges to the
    measured accelerometer direction."""
    r = TargetNumpy(Madgwick(beta=0.5))
    acc = np.array([0.3, 0.0, 0.95]); acc /= np.linalg.norm(acc)
    for _ in range(20000):
        q = r.step(0.005, gyro=[0.0, 0.0, 0.0], accel=acc)["orientation"]
    q0, q1, q2, q3 = q
    g_pred = np.array([2 * (q1 * q3 - q0 * q2),
                       2 * (q0 * q1 + q2 * q3),
                       1 - 2 * q1 * q1 - 2 * q2 * q2])
    angle = np.degrees(np.arccos(np.clip(g_pred @ acc, -1, 1)))
    assert angle < 0.5
    assert np.linalg.norm(q) == pytest.approx(1.0, abs=1e-9)


def test_madgwick_quaternion_stays_unit():
    r = TargetNumpy(Madgwick(beta=0.2))
    rng = np.random.default_rng(0)
    for _ in range(500):
        q = r.step(0.01, gyro=rng.normal(0, 1, 3),
                   accel=rng.normal(0, 1, 3) + [0, 0, 9.81])["orientation"]
        assert np.linalg.norm(q) == pytest.approx(1.0, abs=1e-9)


# --- C++ roundtrip ----------------------------------------------------------

_GYRO = [[0.1, -0.2, 0.05], [0.3, 0.1, -0.1], [-0.2, 0.2, 0.15],
         [0.0, -0.3, 0.2], [0.25, 0.1, -0.05]]
_ACCEL = [[0.0, 0.0, 9.81], [0.5, 0.0, 9.7], [0.3, -0.4, 9.6],
          [-0.2, 0.3, 9.75], [0.1, 0.1, 9.8]]
_DT = 0.02

_HARNESS = r"""
#include "madgwick.hpp"
#include <cstdio>

int main() {
    manta_gen::Madgwick f;
    const double g[5][3] = {{0.1,-0.2,0.05},{0.3,0.1,-0.1},{-0.2,0.2,0.15},
                            {0.0,-0.3,0.2},{0.25,0.1,-0.05}};
    const double a[5][3] = {{0.0,0.0,9.81},{0.5,0.0,9.7},{0.3,-0.4,9.6},
                            {-0.2,0.3,9.75},{0.1,0.1,9.8}};
    for (int i = 0; i < 5; ++i) {
        manta_gen::Madgwick::Inputs u;
        u.gyro  << g[i][0], g[i][1], g[i][2];
        u.accel << a[i][0], a[i][1], a[i][2];
        auto o = f.step(u, 0.02, 0.0);
        std::printf("q %.17g %.17g %.17g %.17g\n",
                    o.orientation[0], o.orientation[1],
                    o.orientation[2], o.orientation[3]);
    }
    return 0;
}
"""


def test_madgwick_emits_cpp_files(tmp_path: Path):
    result = TargetCpp(Madgwick(beta=0.1), tmp_path, class_name="Madgwick")
    for p in (result.kernels_c, result.kernels_h, result.wrapper_hpp,
              result.wrapper_cpp, result.cmakelists):
        assert p.exists(), p


def test_madgwick_python_cpp_roundtrip(tmp_path: Path):
    cxx = next((c for c in ("c++", "g++", "clang++") if shutil.which(c)), None)
    cc = next((c for c in ("cc", "gcc", "clang") if shutil.which(c)), None)
    if cxx is None or cc is None:
        pytest.skip("no C/C++ compiler on PATH")
    eigen_inc = next((p for p in ("/usr/include/eigen3",
                                  "/usr/local/include/eigen3")
                      if Path(p, "Eigen", "Dense").exists()), None)
    if eigen_inc is None:
        pytest.skip("Eigen headers not found")

    f = Madgwick(beta=0.1)
    result = TargetCpp(f, tmp_path, class_name="Madgwick")

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
    cpp_q = [[float(x) for x in line.split()[1:]]
             for line in p.stdout.strip().splitlines()]

    r = TargetNumpy(f)
    np_q = [list(r.step(_DT, gyro=g, accel=a)["orientation"])
            for g, a in zip(_GYRO, _ACCEL)]

    np.testing.assert_allclose(cpp_q, np_q, atol=1e-12)

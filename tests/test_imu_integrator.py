"""IMU strapdown integrator recurrence block — numpy + C++ roundtrip.

The first recurrence block with more than one output port (position,
velocity, orientation), so the roundtrip also exercises the multi-output
unpack in the generic C++ wrapper. Zero backend code, as for the AHRS
filters. Numpy tests pin the mechanization against analytic motion.
"""

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from manta import IMUIntegrator, TargetCpp, TargetNumpy
from manta.codegen.block import KIND_RECURRENCE, block_kind


def test_imu_is_recurrence_block():
    f = IMUIntegrator()
    assert block_kind(f) == KIND_RECURRENCE
    assert [p.name for p in f.inputs] == ["accel", "gyro"]
    assert [p.name for p in f.outputs] == ["position", "velocity", "orientation"]


def test_imu_at_rest():
    """Level, +g accelerometer, no rotation ⇒ no motion."""
    r = TargetNumpy(IMUIntegrator())
    for _ in range(1000):
        s = r.step(0.001, accel=[0, 0, 9.81], gyro=[0, 0, 0])
    np.testing.assert_allclose(s["position"], 0.0, atol=1e-9)
    np.testing.assert_allclose(s["velocity"], 0.0, atol=1e-9)


def test_imu_free_fall():
    """Zero specific force ⇒ falls under gravity: p_z = −½gt², v_z = −gt."""
    r = TargetNumpy(IMUIntegrator())
    dt, T = 0.001, 1.0
    for _ in range(int(T / dt)):
        s = r.step(dt, accel=[0, 0, 0], gyro=[0, 0, 0])
    assert s["position"][2] == pytest.approx(-0.5 * 9.81 * T * T, abs=1e-6)
    assert s["velocity"][2] == pytest.approx(-9.81 * T, abs=1e-9)


def test_imu_constant_accel_exact():
    """Constant body-x specific force (level) ⇒ exact constant-accel motion."""
    r = TargetNumpy(IMUIntegrator())
    dt, T, a = 0.001, 1.0, 2.0
    for _ in range(int(T / dt)):
        s = r.step(dt, accel=[a, 0, 9.81], gyro=[0, 0, 0])
    assert s["position"][0] == pytest.approx(0.5 * a * T * T, abs=1e-9)
    assert s["velocity"][0] == pytest.approx(a * T, abs=1e-12)


def test_imu_yaw_rotation():
    """A constant yaw rate integrates to a z-axis quaternion; a level craft
    stays at rest while it spins."""
    r = TargetNumpy(IMUIntegrator())
    dt, T, wz = 0.001, 1.0, 1.0
    for _ in range(int(T / dt)):
        s = r.step(dt, accel=[0, 0, 9.81], gyro=[0, 0, wz])
    th = wz * T
    np.testing.assert_allclose(
        s["orientation"], [np.cos(th / 2), 0, 0, np.sin(th / 2)], atol=1e-9)
    np.testing.assert_allclose(s["position"], 0.0, atol=1e-9)


def test_imu_custom_initial_state():
    f = IMUIntegrator(p0=(1, 2, 3), v0=(0.5, 0, 0))
    r = TargetNumpy(f)
    assert np.allclose(r.state["position"], [1, 2, 3])
    assert np.allclose(r.state["velocity"], [0.5, 0, 0])


# --- C++ roundtrip ----------------------------------------------------------

_ACCEL = [[0.2, 0.0, 9.81], [0.5, -0.3, 9.6], [0.0, 0.4, 9.9],
          [-0.3, 0.1, 9.7], [0.1, 0.1, 9.8]]
_GYRO = [[0.0, 0.0, 0.5], [0.1, 0.2, 0.3], [-0.2, 0.1, 0.4],
         [0.0, -0.1, 0.2], [0.05, 0.05, 0.1]]
_DT = 0.02

_HARNESS = r"""
#include "imu.hpp"
#include <cstdio>

int main() {
    manta_gen::Imu f;
    const double a[5][3] = {{0.2,0.0,9.81},{0.5,-0.3,9.6},{0.0,0.4,9.9},
                            {-0.3,0.1,9.7},{0.1,0.1,9.8}};
    const double g[5][3] = {{0.0,0.0,0.5},{0.1,0.2,0.3},{-0.2,0.1,0.4},
                            {0.0,-0.1,0.2},{0.05,0.05,0.1}};
    for (int i = 0; i < 5; ++i) {
        manta_gen::Imu::Inputs u;
        u.accel << a[i][0], a[i][1], a[i][2];
        u.gyro  << g[i][0], g[i][1], g[i][2];
        auto o = f.step(u, 0.02, 0.0);
        std::printf("r %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g\n",
                    o.position[0], o.position[1], o.position[2],
                    o.velocity[0], o.velocity[1], o.velocity[2],
                    o.orientation[0], o.orientation[1],
                    o.orientation[2], o.orientation[3]);
    }
    return 0;
}
"""


def test_imu_emits_cpp_files(tmp_path: Path):
    result = TargetCpp(IMUIntegrator(), tmp_path, class_name="Imu")
    for p in (result.kernels_c, result.kernels_h, result.wrapper_hpp,
              result.wrapper_cpp, result.cmakelists):
        assert p.exists(), p


def test_imu_python_cpp_roundtrip(tmp_path: Path):
    cxx = next((c for c in ("c++", "g++", "clang++") if shutil.which(c)), None)
    cc = next((c for c in ("cc", "gcc", "clang") if shutil.which(c)), None)
    if cxx is None or cc is None:
        pytest.skip("no C/C++ compiler on PATH")
    eigen_inc = next((p for p in ("/usr/include/eigen3",
                                  "/usr/local/include/eigen3")
                      if Path(p, "Eigen", "Dense").exists()), None)
    if eigen_inc is None:
        pytest.skip("Eigen headers not found")

    f = IMUIntegrator()
    result = TargetCpp(f, tmp_path, class_name="Imu")

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
    cpp_rows = [[float(x) for x in line.split()[1:]]
                for line in p.stdout.strip().splitlines()]

    r = TargetNumpy(f)
    np_rows = []
    for a, g in zip(_ACCEL, _GYRO):
        s = r.step(_DT, accel=a, gyro=g)
        np_rows.append(list(s["position"]) + list(s["velocity"])
                       + list(s["orientation"]))

    np.testing.assert_allclose(cpp_rows, np_rows, atol=1e-12)

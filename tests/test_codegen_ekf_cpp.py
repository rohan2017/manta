"""End-to-end Python ↔ C++ roundtrip for `TargetCpp(EKF(...))`.

Emit a C++ EKF, compile it, run a predict + update loop in C++ and the
identical loop via `TargetNumpy(EKF(w))` with the *same* measurements, and
compare the final state and covariance. Proof that the exported estimator
is the same filter — same predict, same Joseph update, same manifold
boxplus — on both sides.

gyro + GPS updates only: their `h` doesn't depend on the inputs `u`, so the
C++ `update_*(z, u)` and the numpy sensor update agree regardless of `u`
(the accelerometer's `u`-dependence is exercised by the dynamics/predict).
"""

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from manta import EKF, Craft, TargetCpp, TargetNumpy, World
from manta.fields import GravityField
from manta.parts import IMU, Mass, PositionSensor, Thruster


def _world():
    c = Craft("drone")
    c.add(Mass("body", mass=1.5, moi=(0.05, 0.05, 0.08)))
    c.add(Thruster("t", force=(0.0, 0.0, 1.0), force_noise_sigma=0.2))
    c.add(IMU("g", gyro_noise_sigma=0.01))
    c.add(PositionSensor("gps", position_noise_sigma=0.05))
    w = World(name="hover_world")
    w.add_field(GravityField(g=(0.0, 0.0, -9.81)))
    w.add_craft(c, position=(0.0, 0.0, 5.0))
    return w, c


HARNESS_SRC = r"""
#include "droneekf.hpp"
#include <cstdio>

int main() {
    manta_gen::DroneEkf ekf;
    manta_gen::DroneEkf::State x0;
    x0.drone_position << 0.2, -0.1, 5.0;
    manta_gen::DroneEkf::Cov P0 = manta_gen::DroneEkf::Cov::Identity() * 0.1;
    ekf.reset(x0, P0);

    manta_gen::DroneEkf::Inputs u;
    u.drone_t_throttle = 1.5 * 9.81;

    for (int i = 0; i < 100; i++) {
        ekf.predict(u, 0.01);
        Eigen::Vector3d zg(0.0, 0.0, 0.0);
        ekf.update_drone_g_gyro(zg, u);
        if (i % 5 == 0) {
            Eigen::Vector3d zp(0.2 + 0.001 * i, -0.1, 5.0);
            ekf.update_drone_gps_position(zp, u);
        }
    }
    auto x = ekf.state();
    auto P = ekf.covariance();
    std::printf("pos %.17g %.17g %.17g\n",
                x.drone_position[0], x.drone_position[1], x.drone_position[2]);
    std::printf("ori %.17g %.17g %.17g %.17g\n",
                x.drone_orientation[0], x.drone_orientation[1],
                x.drone_orientation[2], x.drone_orientation[3]);
    std::printf("vel %.17g %.17g %.17g\n",
                x.drone_velocity[0], x.drone_velocity[1], x.drone_velocity[2]);
    std::printf("omg %.17g %.17g %.17g\n",
                x.drone_angular_velocity[0], x.drone_angular_velocity[1],
                x.drone_angular_velocity[2]);
    std::printf("trP %.17g\n", P.trace());
    std::printf("p00 %.17g\n", P(0, 0));
    std::printf("p67 %.17g\n", P(6, 7));
    return 0;
}
"""


def test_ekf_python_cpp_roundtrip(tmp_path: Path):
    cxx = next((c for c in ("c++", "g++", "clang++") if shutil.which(c)), None)
    cc = next((c for c in ("cc", "gcc", "clang") if shutil.which(c)), None)
    if cxx is None or cc is None:
        pytest.skip("no C/C++ compiler on PATH")
    eigen_inc = next((p for p in ("/usr/include/eigen3",
                                  "/usr/local/include/eigen3")
                      if Path(p, "Eigen", "Dense").exists()), None)
    if eigen_inc is None:
        pytest.skip("Eigen headers not found")

    w, c = _world()

    # ---- emit + compile ----
    result = TargetCpp(EKF(w), tmp_path, class_name="DroneEkf")
    k_obj, w_obj = tmp_path / "k.o", tmp_path / "w.o"
    for cmd in (
        [cc, "-c", "-O2", "-fPIC", str(result.kernels_c), "-o", str(k_obj)],
        [cxx, "-c", "-std=c++17", "-O2", "-fPIC", f"-I{eigen_inc}",
         f"-I{tmp_path}", str(result.wrapper_cpp), "-o", str(w_obj)],
    ):
        p = subprocess.run(cmd, capture_output=True, text=True, check=False)
        assert p.returncode == 0, p.stderr

    h_src = tmp_path / "harness_main.cpp"
    h_src.write_text(HARNESS_SRC)
    binary = tmp_path / "harness"
    p = subprocess.run(
        [cxx, "-std=c++17", "-O2", f"-I{eigen_inc}", f"-I{tmp_path}",
         str(h_src), str(w_obj), str(k_obj), "-o", str(binary)],
        capture_output=True, text=True, check=False)
    assert p.returncode == 0, p.stderr
    p = subprocess.run([str(binary)], capture_output=True, text=True, check=False)
    assert p.returncode == 0, p.stderr
    cpp = {l.split()[0]: [float(x) for x in l.split()[1:]]
           for l in p.stdout.strip().splitlines()}

    # ---- identical loop in numpy ----
    ekf = TargetNumpy(EKF(w))
    ekf.reset_from_model_record(
        {"drone": c.initial_state(position=(0.2, -0.1, 5.0))},
        P=np.eye(ekf.spec.tangent_dim) * 0.1)
    next(p for p in c.parts if p.name == "g")
    next(p for p in c.parts if p.name == "gps")
    thr = 1.5 * 9.81
    for i in range(100):
        ekf.predict(0.01, t=0.0, u={"t.throttle": thr})
        ekf.update("g.gyro", np.zeros(3))
        if i % 5 == 0:
            ekf.update("gps.position", np.array([0.2 + 0.001 * i, -0.1, 5.0]))

    est = ekf.state_dict()["drone"]
    P = ekf.P
    ATOL = 1e-7
    np.testing.assert_allclose(cpp["pos"], est["position"].ravel(), atol=ATOL)
    np.testing.assert_allclose(cpp["ori"], est["orientation"].ravel(), atol=ATOL)
    np.testing.assert_allclose(cpp["vel"], est["velocity"].ravel(), atol=ATOL)
    np.testing.assert_allclose(cpp["omg"], est["angular_velocity"].ravel(),
                               atol=ATOL)
    assert abs(cpp["trP"][0] - float(np.trace(P))) < 1e-6
    assert abs(cpp["p00"][0] - float(P[0, 0])) < 1e-7
    assert abs(cpp["p67"][0] - float(P[6, 7])) < 1e-7


def test_target_cpp_ekf_emits_files(tmp_path: Path):
    """Structural: lowering an EKF no longer raises and emits the project."""
    w, _ = _world()
    result = TargetCpp(EKF(w), tmp_path, class_name="DroneEkf")
    for p in (result.kernels_c, result.kernels_h, result.wrapper_hpp,
              result.wrapper_cpp, result.cmakelists):
        assert p.exists()


def _two_craft_world():
    def craft(n):
        c = Craft(n)
        c.add(Mass("body", mass=1.0, moi=(0.1, 0.1, 0.1)))
        c.add(Thruster("t", force=(0, 0, 1), force_noise_sigma=0.2))
        c.add(PositionSensor("gps", position_noise_sigma=0.05))
        return c
    a, b = craft("a"), craft("b")
    w = World(name="twocraft")
    w.add_field(GravityField(g=(0, 0, -9.81)))
    w.add_craft(a, position=(0, 0, 5))
    w.add_craft(b, position=(3, 0, 5))
    return w, a, b


TWO_HARNESS = r"""
#include "twoekf.hpp"
#include <cstdio>

int main() {
    manta_gen::TwoEkf ekf;
    manta_gen::TwoEkf::State x0;
    x0.a_position << 0.1, 0.0, 5.0;
    x0.b_position << 3.0, 0.0, 5.0;
    ekf.reset(x0, manta_gen::TwoEkf::Cov::Identity() * 0.1);
    manta_gen::TwoEkf::Inputs u; u.a_t_throttle = 9.81; u.b_t_throttle = 9.81;
    for (int i = 0; i < 80; i++) {
        ekf.predict(u, 0.01);
        if (i % 5 == 0) {
            ekf.update_a_gps_position(Eigen::Vector3d(0.1, 0.0, 5.0), u);
            ekf.update_b_gps_position(Eigen::Vector3d(3.0, 0.0, 5.0), u);
        }
    }
    auto x = ekf.state(); auto P = ekf.covariance();
    std::printf("ap %.17g %.17g %.17g\n", x.a_position[0], x.a_position[1], x.a_position[2]);
    std::printf("bp %.17g %.17g %.17g\n", x.b_position[0], x.b_position[1], x.b_position[2]);
    std::printf("trP %.17g\n", P.trace());
    std::printf("xblk %.17g\n", P(0, 12));   // cross-block: must stay 0
    return 0;
}
"""


def test_ekf_multicraft_roundtrip(tmp_path: Path):
    """Multi-craft covariance propagation (independent crafts) matches numpy,
    and the cross-block covariance stays exactly zero. The predict is one
    dense `F P Fᵀ + Q` kernel; because F is block-diagonal under the
    independent-subsystem partition, the off-block terms it computes are
    structural zeros — so cross-block P never becomes nonzero."""
    cxx = next((c for c in ("c++", "g++", "clang++") if shutil.which(c)), None)
    cc = next((c for c in ("cc", "gcc", "clang") if shutil.which(c)), None)
    if cxx is None or cc is None:
        pytest.skip("no C/C++ compiler on PATH")
    eigen_inc = next((p for p in ("/usr/include/eigen3",
                                  "/usr/local/include/eigen3")
                      if Path(p, "Eigen", "Dense").exists()), None)
    if eigen_inc is None:
        pytest.skip("Eigen headers not found")

    w, a, b = _two_craft_world()
    assert EKF(w).n_blocks == 2
    result = TargetCpp(EKF(w), tmp_path, class_name="TwoEkf")

    k_obj, w_obj = tmp_path / "k.o", tmp_path / "w.o"
    for cmd in (
        [cc, "-c", "-O2", "-fPIC", str(result.kernels_c), "-o", str(k_obj)],
        [cxx, "-c", "-std=c++17", "-O2", "-fPIC", f"-I{eigen_inc}",
         f"-I{tmp_path}", str(result.wrapper_cpp), "-o", str(w_obj)],
    ):
        p = subprocess.run(cmd, capture_output=True, text=True, check=False)
        assert p.returncode == 0, p.stderr
    h_src = tmp_path / "harness_main.cpp"
    h_src.write_text(TWO_HARNESS)
    binary = tmp_path / "harness"
    p = subprocess.run(
        [cxx, "-std=c++17", "-O2", f"-I{eigen_inc}", f"-I{tmp_path}",
         str(h_src), str(w_obj), str(k_obj), "-o", str(binary)],
        capture_output=True, text=True, check=False)
    assert p.returncode == 0, p.stderr
    p = subprocess.run([str(binary)], capture_output=True, text=True, check=False)
    assert p.returncode == 0, p.stderr
    cpp = {l.split()[0]: [float(x) for x in l.split()[1:]]
           for l in p.stdout.strip().splitlines()}

    ekf = TargetNumpy(EKF(w))
    ekf.reset_from_model_record(
        {"a": a.initial_state(position=(0.1, 0, 5)),
         "b": b.initial_state(position=(3, 0, 5))},
        P=np.eye(ekf.spec.tangent_dim) * 0.1)
    next(p for p in a.parts if p.name == "gps")
    next(p for p in b.parts if p.name == "gps")
    for i in range(80):
        ekf.predict(0.01, t=0.0,
                    u={"a.t.throttle": 9.81, "b.t.throttle": 9.81})
        if i % 5 == 0:
            ekf.update("a.gps.position", np.array([0.1, 0, 5.0]))
            ekf.update("b.gps.position", np.array([3.0, 0, 5.0]))
    est, P = ekf.state_dict(), ekf.P
    np.testing.assert_allclose(cpp["ap"], est["a"]["position"].ravel(), atol=1e-7)
    np.testing.assert_allclose(cpp["bp"], est["b"]["position"].ravel(), atol=1e-7)
    assert abs(cpp["trP"][0] - float(np.trace(P))) < 1e-6
    assert abs(cpp["xblk"][0]) < 1e-12 and abs(float(P[0, 12])) < 1e-12

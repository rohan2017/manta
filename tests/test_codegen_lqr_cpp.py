"""End-to-end Python ↔ C++ roundtrip for `TargetCpp(LQR(...))`.

Lower one LQR IR to both backends and check the C++ `control(x)` matches
`TargetNumpy(LQR).control(x)` at several states — the baked feed-forward
law (gain + operating point + manifold boxminus) is identical on both
sides.
"""

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from manta import Craft, LQR, TargetCpp, TargetNumpy, World
from manta.fields import GravityField
from manta.parts import Mass, Thruster

M, G = 2.0, 9.81


def _lqr_ir():
    c = Craft("c")
    c.add(Mass("body", mass=M))
    c.add(Thruster("tx", force=(1, 0, 0)))
    c.add(Thruster("ty", force=(0, 1, 0)))
    c.add(Thruster("tz", force=(0, 0, 1)))
    w = World(name="ff")
    w.add_field(GravityField(g=(0, 0, -G)))
    w.add_craft(c, position=(0, 0, 10))
    return LQR(w, x_ref={"c": {"position": (0, 0, 10), "velocity": (0, 0, 0)}},
               u_ref={"tz.throttle": M * G},
               track=["c.position", "c.velocity"],
               Q=np.diag([10, 10, 10, 1, 1, 1]), R=np.eye(3) * 0.1, dt=0.02)


HARNESS_SRC = r"""
#include "fflqr.hpp"
#include <cstdio>

int main() {
    manta_gen::FfLqr lqr;
    manta_gen::FfLqr::State x;
    x.c_position << 3.0, -2.0, 7.0;
    x.c_velocity << 0.1, 0.2, -0.3;
    auto u = lqr.control(x);
    std::printf("u %.17g %.17g %.17g\n",
                u.c_tx_throttle, u.c_ty_throttle, u.c_tz_throttle);

    manta_gen::FfLqr::State x2;          // at the setpoint, at rest
    x2.c_position << 0.0, 0.0, 10.0;
    auto u2 = lqr.control(x2);
    std::printf("u2 %.17g %.17g %.17g\n",
                u2.c_tx_throttle, u2.c_ty_throttle, u2.c_tz_throttle);

    manta_gen::FfLqr::State ref;         // retarget: State{} defaults to
    ref.c_position << 1.0, -1.0, 12.0;   // the built reference; move it
    auto u3 = lqr.control(x, ref);
    std::printf("u3 %.17g %.17g %.17g\n",
                u3.c_tx_throttle, u3.c_ty_throttle, u3.c_tz_throttle);
    return 0;
}
"""


def test_lqr_python_cpp_roundtrip(tmp_path: Path):
    cxx = next((c for c in ("c++", "g++", "clang++") if shutil.which(c)), None)
    cc = next((c for c in ("cc", "gcc", "clang") if shutil.which(c)), None)
    if cxx is None or cc is None:
        pytest.skip("no C/C++ compiler on PATH")
    eigen_inc = next((p for p in ("/usr/include/eigen3",
                                  "/usr/local/include/eigen3")
                      if Path(p, "Eigen", "Dense").exists()), None)
    if eigen_inc is None:
        pytest.skip("Eigen headers not found")

    lqr = _lqr_ir()
    result = TargetCpp(lqr, tmp_path, class_name="FfLqr")

    k_obj, w_obj = tmp_path / "k.o", tmp_path / "w.o"
    for cmd in (
        [cc, "-c", "-O2", "-fPIC", str(result.kernels_c), "-o", str(k_obj)],
        [cxx, "-c", "-std=c++17", "-O2", "-fPIC", f"-I{eigen_inc}",
         f"-I{tmp_path}", str(result.wrapper_cpp), "-o", str(w_obj)],
    ):
        p = subprocess.run(cmd, capture_output=True, text=True)
        assert p.returncode == 0, p.stderr

    h_src = tmp_path / "harness_main.cpp"
    h_src.write_text(HARNESS_SRC)
    binary = tmp_path / "harness"
    p = subprocess.run(
        [cxx, "-std=c++17", "-O2", f"-I{eigen_inc}", f"-I{tmp_path}",
         str(h_src), str(w_obj), str(k_obj), "-o", str(binary)],
        capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    p = subprocess.run([str(binary)], capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    cpp = {l.split()[0]: [float(x) for x in l.split()[1:]]
           for l in p.stdout.strip().splitlines()}

    lqr_np = TargetNumpy(lqr)
    order = ["c.tx.throttle", "c.ty.throttle", "c.tz.throttle"]
    u_np = lqr_np.control({"c": {"position": (3.0, -2.0, 7.0),
                                 "velocity": (0.1, 0.2, -0.3)}})
    u2_np = lqr_np.control({"c": {"position": (0.0, 0.0, 10.0),
                                  "velocity": (0.0, 0.0, 0.0)}})
    lqr_np.retarget({"c": {"position": (1.0, -1.0, 12.0)}})
    u3_np = lqr_np.control({"c": {"position": (3.0, -2.0, 7.0),
                                  "velocity": (0.1, 0.2, -0.3)}})
    np.testing.assert_allclose(cpp["u"], [u_np[k] for k in order], atol=1e-9)
    np.testing.assert_allclose(cpp["u2"], [u2_np[k] for k in order], atol=1e-9)
    np.testing.assert_allclose(cpp["u3"], [u3_np[k] for k in order], atol=1e-9)
    # Sanity: at the setpoint the command is the trim (0, 0, M·G).
    np.testing.assert_allclose(cpp["u2"], [0.0, 0.0, M * G], atol=1e-6)


def test_target_cpp_lqr_emits_files(tmp_path: Path):
    result = TargetCpp(_lqr_ir(), tmp_path, class_name="FfLqr")
    for p in (result.kernels_c, result.kernels_h, result.wrapper_hpp,
              result.wrapper_cpp, result.cmakelists):
        assert p.exists()

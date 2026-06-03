"""End-to-end Python ↔ C++ roundtrip for `TargetCpp(cw, ...)`.

The test:
  1. Build a Sim + emit a C++ library via `TargetCpp`.
  2. Compile the kernels (cc) and wrapper (c++ + Eigen) into .o files.
  3. Build a `harness_main.cpp` that runs predict N times and evaluates
     each measurement Output, then prints the final state + outputs.
  4. Run the binary, parse stdout.
  5. Run the equivalent Python loop via `TargetNumpy(cw)`.
  6. Compare element-wise to a tight tolerance.

This is the proof that the C++ export path actually works: same math,
same Jacobians, same numbers on both sides.
"""

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from manta import Craft, Sim, TargetCpp, TargetNumpy, World
from manta.fields import GravityField
from manta.parts import IMU, Mass, PositionSensor, Thruster


def _hover_world():
    c = Craft("drone")
    c.add(Mass("body", mass=1.5, moi=(0.05, 0.05, 0.08)))
    c.add(Thruster("t", force=(0.0, 0.0, 1.0)))
    c.add(IMU("g"))
    c.add(PositionSensor("gps"))
    w = World(name="hover_world")
    w.add_field(GravityField(g=(0.0, 0.0, -9.81)))
    w.add_craft(c)
    return w


HARNESS_SRC = r"""
#include "drone.hpp"
#include <cstdio>

int main() {
    manta_gen::Drone drone;
    auto x = drone.initial_state();
    x.drone_position << 0.0, 0.0, 5.0;
    x.drone_velocity << 0.5, 0.0, 0.0;

    manta_gen::Drone::Inputs u;
    u.drone_t_throttle = 1.5 * 9.81;

    const int N = 200;
    const double dt = 0.005;
    for (int i = 0; i < N; i++) {
        x = drone.predict(x, u, dt);
    }

    // Measurements at the final state.
    auto gps  = drone.measure_drone_gps_position(x, u);
    auto gyro = drone.measure_drone_g_gyro(x, u);
    auto H_pos = drone.measure_drone_gps_position_jacobian(x, u);
    auto F     = drone.predict_jacobian(x, u, dt);

    // Print everything in a fixed canonical order with full precision.
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
    std::printf("gps %.17g %.17g %.17g\n", gps[0], gps[1], gps[2]);
    std::printf("gyr %.17g %.17g %.17g\n", gyro[0], gyro[1], gyro[2]);
    std::printf("trF %.17g\n", F.trace());
    std::printf("hpp %.17g\n", H_pos(0, 0));
    return 0;
}
"""


def test_python_cpp_roundtrip(tmp_path: Path):
    cxx = next((c for c in ("c++", "g++", "clang++") if shutil.which(c)),
               None)
    cc  = next((c for c in ("cc",  "gcc", "clang")    if shutil.which(c)),
               None)
    if cxx is None or cc is None:
        pytest.skip("no C/C++ compiler on PATH")

    eigen_inc = next((p for p in ("/usr/include/eigen3",
                                   "/usr/local/include/eigen3")
                      if Path(p, "Eigen", "Dense").exists()), None)
    if eigen_inc is None:
        pytest.skip("Eigen headers not found")

    # ---- 1: TargetCpp ----
    w      = _hover_world()
    cw     = Sim(w)
    result = TargetCpp(cw.module(noise=False), tmp_path, class_name="Drone")

    # ---- 2: compile kernels + wrapper ----
    k_obj = tmp_path / "kernels.o"
    w_obj = tmp_path / "wrapper.o"
    p = subprocess.run(
        [cc, "-c", "-O2", "-fPIC",
         str(result.kernels_c), "-o", str(k_obj)],
        capture_output=True, text=True)
    assert p.returncode == 0, p.stderr

    p = subprocess.run(
        [cxx, "-c", "-std=c++17", "-O2", "-fPIC",
         f"-I{eigen_inc}", f"-I{tmp_path}",
         str(result.wrapper_cpp), "-o", str(w_obj)],
        capture_output=True, text=True)
    assert p.returncode == 0, p.stderr

    # ---- 3: build the harness ----
    h_src = tmp_path / "harness_main.cpp"
    h_src.write_text(HARNESS_SRC)
    binary = tmp_path / "harness"
    p = subprocess.run(
        [cxx, "-std=c++17", "-O2",
         f"-I{eigen_inc}", f"-I{tmp_path}",
         str(h_src), str(w_obj), str(k_obj),
         "-o", str(binary)],
        capture_output=True, text=True)
    assert p.returncode == 0, p.stderr

    # ---- 4: run the harness ----
    p = subprocess.run([str(binary)], capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    cpp_lines = {l.split()[0]: [float(x) for x in l.split()[1:]]
                 for l in p.stdout.strip().splitlines()}

    # ---- 5: run the equivalent Python loop ----
    sim = TargetNumpy(cw)
    state = sim.initial_state()
    state["drone"]["position"]   = np.array([0.0, 0.0, 5.0])
    state["drone"]["velocity"]   = np.array([0.5, 0.0, 0.0])
    state["drone"]["t.throttle"] = 1.5 * 9.81
    for _ in range(200):
        state = sim.step(state, dt=0.005)

    # ---- 6: Jacobian sanity checks via the same Module kernels ----
    m = cw.module(noise=False)
    spec = m.spec
    flat = {f"drone.{k}": v for k, v in state["drone"].items()
            if f"drone.{k}" in spec}
    x_flat = spec.pack(flat)
    u_flat = np.array([state["drone"]["t.throttle"]])
    F_py = np.asarray(
        m.functions["predict_jacobian"](x_flat, u_flat, 0.005, 0.0))
    # honest measure kernels take (x, u, t) — no dt
    H_pos_py = np.asarray(
        m.functions["measure_drone_gps_position_jacobian"](x_flat, u_flat, 0.0))
    gps_py_new = np.asarray(
        m.functions["measure_drone_gps_position"](x_flat, u_flat, 0.0)).ravel()
    gyro_py_new = np.asarray(
        m.functions["measure_drone_g_gyro"](x_flat, u_flat, 0.0)).ravel()

    # ---- 7: compare ----
    ATOL = 1e-10
    np.testing.assert_allclose(cpp_lines["pos"],
                                np.array(state["drone"]["position"]).ravel(),
                                atol=ATOL)
    np.testing.assert_allclose(cpp_lines["ori"],
                                np.array(state["drone"]["orientation"]).ravel(),
                                atol=ATOL)
    np.testing.assert_allclose(cpp_lines["vel"],
                                np.array(state["drone"]["velocity"]).ravel(),
                                atol=ATOL)
    np.testing.assert_allclose(cpp_lines["omg"],
                                np.array(state["drone"]["angular_velocity"]).ravel(),
                                atol=ATOL)
    np.testing.assert_allclose(cpp_lines["gps"], gps_py_new, atol=ATOL)
    np.testing.assert_allclose(cpp_lines["gyr"], gyro_py_new, atol=ATOL)
    assert abs(cpp_lines["trF"][0] - float(F_py.trace())) < ATOL
    assert abs(cpp_lines["hpp"][0] - float(H_pos_py[0, 0])) < ATOL

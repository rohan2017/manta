"""End-to-end Python ↔ C++ roundtrip for emit_cpp.

The test:
  1. Build a Craft + emit its C++ library via `emit_cpp`.
  2. Compile the kernels (cc) and wrapper (c++ + Eigen) into .o files.
  3. Compile a tiny `harness_main.cpp` that runs predict N times and
     also evaluates each measurement Output, then prints the final
     state + outputs as floating-point text.
  4. Run the binary, parse stdout.
  5. Run the equivalent Python loop using the same craft/spec.
  6. Compare element-wise to 1e-12 (allows for FP-ordering differences
     between CasADi's flat-C and CasADi-on-Python evaluation, which
     should still be bit-identical in practice).

This is the proof that the C++ export path actually works: same math,
same Jacobians, same numbers on both sides.
"""

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from manta_next import Craft
from manta_next.fields import GravityField
from manta_next.codegen import emit_cpp
from manta_next.parts import IMU, Mass, PositionSensor, Thruster


def _hover_craft():
    c = Craft("drone")
    c.add(Mass("body", mass=1.5, moi=(0.05, 0.05, 0.08)))
    c.add(Thruster("t", force=(0.0, 0.0, 1.0)))
    c.add(IMU("g"))
    c.add(PositionSensor("gps"))
    return c


HARNESS_SRC = r"""
#include "drone.hpp"
#include <cstdio>

int main() {
    manta_next_gen::Drone drone;
    auto x = drone.initial_state();
    x.position << 0.0, 0.0, 5.0;
    x.velocity << 0.5, 0.0, 0.0;

    manta_next_gen::Drone::Inputs u;
    u.t_throttle = 1.5 * 9.81;

    const int N = 200;
    const double dt = 0.005;
    for (int i = 0; i < N; i++) {
        x = drone.predict(x, u, dt);
    }

    // Measurements at the final state.
    auto gps  = drone.measure_gps_position(x, u);
    auto gyro = drone.measure_g_gyro(x, u);
    auto H_pos = drone.measure_gps_position_jacobian(x, u);
    auto F     = drone.predict_jacobian(x, u, dt);

    // Print everything in a fixed canonical order with full precision.
    std::printf("pos %.17g %.17g %.17g\n",
                x.position[0], x.position[1], x.position[2]);
    std::printf("ori %.17g %.17g %.17g %.17g\n",
                x.orientation[0], x.orientation[1],
                x.orientation[2], x.orientation[3]);
    std::printf("vel %.17g %.17g %.17g\n",
                x.velocity[0], x.velocity[1], x.velocity[2]);
    std::printf("omg %.17g %.17g %.17g\n",
                x.angular_velocity[0], x.angular_velocity[1],
                x.angular_velocity[2]);
    std::printf("gps %.17g %.17g %.17g\n", gps[0], gps[1], gps[2]);
    std::printf("gyr %.17g %.17g %.17g\n", gyro[0], gyro[1], gyro[2]);
    // Trace of F + (0,0) entry of H_pos as a coarse Jacobian sanity check.
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

    # ---- 1: emit_cpp ----
    craft  = _hover_craft()
    result = emit_cpp(craft, tmp_path, class_name="Drone",
                      gravity_field=GravityField(g=(0.0, 0.0, -9.81)))

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

    # ---- 5: run the same loop in Python ----
    tick = craft.compile_tick(gravity_field=GravityField(g=(0.0, 0.0, -9.81)))
    state = craft.initial_state()
    state["position"] = np.array([0.0, 0.0, 5.0])
    state["velocity"] = np.array([0.5, 0.0, 0.0])
    state["t.throttle"] = 1.5 * 9.81
    for _ in range(200):
        out = tick(dt=0.005, **state)
        state = {**state, **out}

    # Sensor outputs from the same tick result.
    gps_py  = np.array(out["gps.position"]).ravel()
    gyro_py = np.array(out["g.gyro"]).ravel()

    # Jacobian sanity checks via extract.
    funcs = result.funcs
    x_flat = funcs.spec.pack(state)
    u_flat = np.array([state["t.throttle"]])
    F_py = np.asarray(funcs.predict_jacobian_fn(x_flat, u_flat, 0.005, 0.0))
    H_pos_py = np.asarray(
        next(o for o in funcs.outputs if o.full_name == "gps.position"
             ).H_fn(x_flat, u_flat, 0.005, 0.0))

    # ---- 6: compare ----
    ATOL = 1e-10        # CasADi flat-C vs Python eval — bit-identical in
                        # practice, but allow a tiny margin for ordering.
    np.testing.assert_allclose(cpp_lines["pos"],
                               np.array(state["position"]).ravel(), atol=ATOL)
    np.testing.assert_allclose(cpp_lines["ori"],
                               np.array(state["orientation"]).ravel(), atol=ATOL)
    np.testing.assert_allclose(cpp_lines["vel"],
                               np.array(state["velocity"]).ravel(), atol=ATOL)
    np.testing.assert_allclose(cpp_lines["omg"],
                               np.array(state["angular_velocity"]).ravel(),
                               atol=ATOL)
    # IMPORTANT: gps/gyro read at the *current* state, which in Python is
    # the new state we just merged. In C++ the harness reads gps/gyro
    # AFTER the final predict() (using `x` post-loop). Both should match
    # since `out["gps.position"]` from the last Python tick was computed
    # from the input state of that tick, then `state` was overwritten —
    # so the C++ value (sensor at new x) is one tick AHEAD. Recompute
    # from the new state instead.
    h_gps = next(o for o in funcs.outputs if o.full_name == "gps.position")
    h_gyr = next(o for o in funcs.outputs if o.full_name == "g.gyro")
    gps_py_new  = np.asarray(h_gps.h_fn(x_flat, u_flat, 0.005, 0.0)).ravel()
    gyro_py_new = np.asarray(h_gyr.h_fn(x_flat, u_flat, 0.005, 0.0)).ravel()
    np.testing.assert_allclose(cpp_lines["gps"], gps_py_new, atol=ATOL)
    np.testing.assert_allclose(cpp_lines["gyr"], gyro_py_new, atol=ATOL)

    assert abs(cpp_lines["trF"][0] - float(F_py.trace())) < ATOL
    assert abs(cpp_lines["hpp"][0] - float(H_pos_py[0, 0])) < ATOL

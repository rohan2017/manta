"""Python ↔ C++ roundtrip for an ARTICULATED craft.

Same shape as test_codegen_roundtrip.py, but the craft carries both
joint types — a revolute reaction wheel (spinning, so the gyroscopic
coupling is live) and a prismatic slider (displaced, so the
configuration-dependent mass matrix is live). This proves the
joint-space block solve — the (3+K)×(3+K) `ca.solve`, the Hamel bias,
and the generalized momentum integrator — all lower through the C++
backend and reproduce the Python tick bit-for-bit-ish (1e-10).
"""

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from manta import Craft, Sim, TargetCpp, TargetNumpy, World
from manta.fields import GravityField
from manta.parts import Mass, PrismaticJoint, RevoluteJoint, Thruster


def _jointed_world():
    c = Craft("rover")
    c.add(Mass("body", mass=4.0, moi=(0.4, 0.5, 0.6)))
    c.add(Thruster("t", force=(0.0, 0.0, 1.0)))
    wheel = RevoluteJoint("wheel", mode="saturating", stall_torque=0.5,
                          axis=(0.0, 0.0, 1.0))
    wheel.add(Mass("disk", mass=0.3, moi=(0.004, 0.006, 0.008)))
    c.add(wheel)
    arm = PrismaticJoint("arm", mode="saturating", stall_force=2.0,
                         axis=(1.0, 0.0, 0.0))
    arm.add(Mass("tip", mass=0.2, moi=(1e-4, 1e-4, 1e-4)))
    c.add(arm)
    w = World(name="jointed_world")
    w.add_field(GravityField(g=(0.0, 0.0, -9.81)))
    w.add_craft(c)
    return w


HARNESS_SRC = r"""
#include "rover.hpp"
#include <cstdio>

int main() {
    manta_gen::Rover rover;
    auto x = rover.initial_state();
    x.rover_position << 0.0, 0.0, 5.0;
    x.rover_angular_velocity << 0.3, -0.2, 0.5;
    x.rover_wheel_rate = 40.0;
    x.rover_arm_displacement = 0.15;

    manta_gen::Rover::Inputs u;
    u.rover_t_throttle = 4.5 * 9.81;
    u.rover_wheel_torque_cmd = 0.2;
    u.rover_arm_force_cmd = 0.8;

    const int N = 200;
    const double dt = 0.002;
    for (int i = 0; i < N; i++) {
        x = rover.predict(x, u, dt);
    }

    auto F = rover.predict_jacobian(x, u, dt);

    std::printf("pos %.17g %.17g %.17g\n",
                x.rover_position[0], x.rover_position[1], x.rover_position[2]);
    std::printf("ori %.17g %.17g %.17g %.17g\n",
                x.rover_orientation[0], x.rover_orientation[1],
                x.rover_orientation[2], x.rover_orientation[3]);
    std::printf("vel %.17g %.17g %.17g\n",
                x.rover_velocity[0], x.rover_velocity[1], x.rover_velocity[2]);
    std::printf("omg %.17g %.17g %.17g\n",
                x.rover_angular_velocity[0], x.rover_angular_velocity[1],
                x.rover_angular_velocity[2]);
    std::printf("whl %.17g %.17g\n", x.rover_wheel_angle, x.rover_wheel_rate);
    std::printf("arm %.17g %.17g\n",
                x.rover_arm_displacement, x.rover_arm_rate);
    std::printf("trF %.17g\n", F.trace());
    return 0;
}
"""


def test_jointed_python_cpp_roundtrip(tmp_path: Path):
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

    w      = _jointed_world()
    cw     = Sim(w)
    result = TargetCpp(cw.module(noise=False), tmp_path, class_name="Rover")

    k_obj = tmp_path / "kernels.o"
    w_obj = tmp_path / "wrapper.o"
    # -O0 on purpose: the jointed predict_jacobian differentiates through
    # two 5×5 LU solves, producing a ~2 MB machine-generated kernels.c
    # that -O2 chews on for the better part of an hour. The roundtrip
    # checks NUMBERS, not speed — -O0 gives identical results in seconds.
    p = subprocess.run(
        [cc, "-c", "-O0", "-fPIC",
         str(result.kernels_c), "-o", str(k_obj)],
        capture_output=True, text=True)
    assert p.returncode == 0, p.stderr

    p = subprocess.run(
        [cxx, "-c", "-std=c++17", "-O2", "-fPIC",
         f"-I{eigen_inc}", f"-I{tmp_path}",
         str(result.wrapper_cpp), "-o", str(w_obj)],
        capture_output=True, text=True)
    assert p.returncode == 0, p.stderr

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

    p = subprocess.run([str(binary)], capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    cpp = {l.split()[0]: [float(x) for x in l.split()[1:]]
           for l in p.stdout.strip().splitlines()}

    sim = TargetNumpy(cw)
    sim.state["rover"]["position"]         = np.array([0.0, 0.0, 5.0])
    sim.state["rover"]["angular_velocity"] = np.array([0.3, -0.2, 0.5])
    sim.state["rover"]["wheel.rate"]       = 40.0
    sim.state["rover"]["arm.displacement"] = 0.15
    sim.state["rover"]["t.throttle"]       = 4.5 * 9.81
    sim.state["rover"]["wheel.torque_cmd"] = 0.2
    sim.state["rover"]["arm.force_cmd"]    = 0.8
    for _ in range(200):
        sim.step(0.002)

    m = cw.module(noise=False)
    spec = m.spec
    flat = {f"rover.{k}": v for k, v in sim.state["rover"].items()
            if f"rover.{k}" in spec}
    x_flat = spec.pack(flat)
    u_names = [n for n in ("rover.t.throttle", "rover.wheel.torque_cmd",
                           "rover.arm.force_cmd")]
    u_flat = np.array([sim.state["rover"][n.split(".", 1)[1]]
                       for n in u_names])
    F_py = np.asarray(
        m.functions["predict_jacobian"](x_flat, u_flat, 0.002, 0.0))

    st = sim.state["rover"]
    ATOL = 1e-10
    np.testing.assert_allclose(cpp["pos"], np.asarray(st["position"]).ravel(),
                               atol=ATOL)
    np.testing.assert_allclose(cpp["ori"], np.asarray(st["orientation"]).ravel(),
                               atol=ATOL)
    np.testing.assert_allclose(cpp["vel"], np.asarray(st["velocity"]).ravel(),
                               atol=ATOL)
    np.testing.assert_allclose(cpp["omg"],
                               np.asarray(st["angular_velocity"]).ravel(),
                               atol=ATOL)
    np.testing.assert_allclose(cpp["whl"],
                               [float(st["wheel.angle"]),
                                float(st["wheel.rate"])], atol=ATOL)
    np.testing.assert_allclose(cpp["arm"],
                               [float(st["arm.displacement"]),
                                float(st["arm.rate"])], atol=ATOL)
    assert abs(cpp["trF"][0] - float(F_py.trace())) < 1e-8

"""PID recurrence block — numpy behavior + Python ↔ C++ roundtrip.

PID is the first `recurrence`-kind block. These tests pin its control law
and anti-windup, and prove the generic `lower_recurrence` C++ path: lower
one PID to both backends and check the C++ `step()` reproduces the numpy
command trajectory *and* the final state, exactly, over several steps.
"""

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from manta import PID, TargetCpp, TargetNumpy
from manta.codegen.block import KIND_RECURRENCE, block_kind


def test_pid_is_recurrence_block():
    pid = PID(kp=1.0, ki=0.5, kd=0.1)
    assert block_kind(pid) == KIND_RECURRENCE
    assert [p.name for p in pid.inputs] == ["setpoint", "measurement"]
    assert [p.name for p in pid.outputs] == ["command"]


def test_pid_proportional_only():
    """ki = kd = 0 ⇒ command is exactly kp·(setpoint − measurement)."""
    r = TargetNumpy(PID(kp=3.0))
    out = r.step(0.01, setpoint=2.0, measurement=0.5)
    assert out["command"] == pytest.approx(3.0 * 1.5)


def test_pid_integral_accumulates_and_clamps():
    r = TargetNumpy(PID(kp=0.0, ki=1.0, kd=0.0, integral_limit=0.05))
    # error = 1.0 each step, dt = 0.1 ⇒ integral grows 0.1/step, clamps at 0.05.
    c0 = r.step(0.1, setpoint=1.0, measurement=0.0)["command"]
    assert c0 == pytest.approx(0.05)          # 0.1 clamped to 0.05
    c1 = r.step(0.1, setpoint=1.0, measurement=0.0)["command"]
    assert c1 == pytest.approx(0.05)          # stays clamped
    assert float(r.state()["integral"]) == pytest.approx(0.05)


def test_pid_derivative_zero_on_first_step():
    """The `primed` flag zeros the derivative on the very first step (no
    spurious kick), then it engages."""
    r = TargetNumpy(PID(kp=0.0, ki=0.0, kd=1.0))
    c0 = r.step(0.1, setpoint=0.0, measurement=10.0)["command"]  # kd only
    assert c0 == pytest.approx(0.0)                     # primed=0 ⇒ no kick
    c1 = r.step(0.1, setpoint=0.0, measurement=12.0)["command"]  # d=(12-10)/0.1
    assert c1 == pytest.approx(-1.0 * (12.0 - 10.0) / 0.1)


def test_pid_output_limit():
    r = TargetNumpy(PID(kp=100.0, output_limit=1.5))
    out = r.step(0.01, setpoint=1.0, measurement=0.0)
    assert out["command"] == pytest.approx(1.5)        # saturated


def test_pid_reset():
    r = TargetNumpy(PID(kp=0.0, ki=1.0))
    r.step(0.1, setpoint=1.0, measurement=0.0)
    assert float(r.state()["integral"]) != 0.0
    r.reset()
    assert float(r.state()["integral"]) == pytest.approx(0.0)
    assert float(r.state()["primed"]) == pytest.approx(0.0)


# --- C++ roundtrip ----------------------------------------------------------

_SP = [1.0, 1.0, 0.5, 0.5, -0.3]
_MS = [0.0, 0.2, 0.4, 0.45, 0.40]
_DT = 0.02

_HARNESS = r"""
#include "pid.hpp"
#include <cstdio>

int main() {
    manta_gen::Pid pid;
    const double sp[5] = {1.0, 1.0, 0.5, 0.5, -0.3};
    const double ms[5] = {0.0, 0.2, 0.4, 0.45, 0.40};
    for (int i = 0; i < 5; ++i) {
        manta_gen::Pid::Inputs u;
        u.setpoint = sp[i];
        u.measurement = ms[i];
        auto o = pid.step(u, 0.02, 0.0);
        std::printf("c %.17g\n", o.command);
    }
    std::printf("s %.17g %.17g %.17g\n",
                pid.state().integral, pid.state().prev_measurement,
                pid.state().primed);
    return 0;
}
"""


def test_pid_emits_cpp_files(tmp_path: Path):
    result = TargetCpp(PID(kp=2.0, ki=5.0, kd=0.5, integral_limit=2.0),
                       tmp_path, class_name="Pid")
    for p in (result.kernels_c, result.kernels_h, result.wrapper_hpp,
              result.wrapper_cpp, result.cmakelists):
        assert p.exists(), p


def test_pid_python_cpp_roundtrip(tmp_path: Path):
    cxx = next((c for c in ("c++", "g++", "clang++") if shutil.which(c)), None)
    cc = next((c for c in ("cc", "gcc", "clang") if shutil.which(c)), None)
    if cxx is None or cc is None:
        pytest.skip("no C/C++ compiler on PATH")
    eigen_inc = next((p for p in ("/usr/include/eigen3",
                                  "/usr/local/include/eigen3")
                      if Path(p, "Eigen", "Dense").exists()), None)
    if eigen_inc is None:
        pytest.skip("Eigen headers not found")

    pid = PID(kp=2.0, ki=5.0, kd=0.5, integral_limit=2.0)
    result = TargetCpp(pid, tmp_path, class_name="Pid")

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

    cpp_cmds, cpp_state = [], None
    for line in p.stdout.strip().splitlines():
        tok = line.split()
        if tok[0] == "c":
            cpp_cmds.append(float(tok[1]))
        elif tok[0] == "s":
            cpp_state = [float(x) for x in tok[1:]]

    r = TargetNumpy(pid)
    np_cmds = [r.step(_DT, setpoint=sp, measurement=ms)["command"]
               for sp, ms in zip(_SP, _MS)]
    st = r.state()
    np_state = [float(st["integral"]), float(st["prev_measurement"]),
                float(st["primed"])]

    np.testing.assert_allclose(cpp_cmds, np_cmds, atol=1e-12)
    np.testing.assert_allclose(cpp_state, np_state, atol=1e-12)

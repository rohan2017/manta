"""Smoke tests for the top-level `TargetCpp(cw, ...)` pipeline."""

import shutil
import subprocess
from pathlib import Path

from manta import Craft, Sim, TargetCpp, World
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
    return Sim(w)


def test_target_cpp_produces_expected_files(tmp_path: Path):
    result = TargetCpp(_hover_world(), tmp_path, class_name="Drone")
    expected = {
        result.kernels_c, result.kernels_h,
        result.wrapper_hpp, result.wrapper_cpp,
        result.cmakelists,
    }
    for p in expected:
        assert p.exists(), p
    assert result.class_name == "Drone"
    # A small funcs summary (world_name / dims) is carried for cross-checking.
    assert result.funcs.world_name == "hover_world"
    assert result.funcs.ambient_dim == 13
    assert result.funcs.tangent_dim == 12


def test_target_cpp_custom_basename(tmp_path: Path):
    result = TargetCpp(_hover_world(), tmp_path,
                       class_name="Drone", basename="my_robot")
    assert result.kernels_c.name == "my_robot_kernels.c"
    assert result.wrapper_hpp.name == "my_robot.hpp"
    cmake_text = result.cmakelists.read_text()
    assert "my_robot_kernels.c" in cmake_text
    assert "my_robot.cpp" in cmake_text


def test_target_cpp_rejects_non_module():
    """TargetCpp lowers a Module (or a transform exposing .module());
    anything else raises TypeError."""
    import pytest
    with pytest.raises(TypeError, match="Module or a transform"):
        TargetCpp(object(), "/tmp/whatever", class_name="X")


def _make_simple_craft():
    c = Craft("d"); c.add(Mass("body", mass=1.0))
    return c


def _syntax_check(result, tmp_path: Path) -> None:
    """When a toolchain is available, prove the wrapper actually compiles."""
    cxx = next((c for c in ("c++", "g++", "clang++") if shutil.which(c)), None)
    eigen_inc = next((p for p in ("/usr/include/eigen3",
                                  "/usr/local/include/eigen3")
                      if Path(p, "Eigen", "Dense").exists()), None)
    if cxx is None or eigen_inc is None:
        return
    p = subprocess.run(
        [cxx, "-std=c++17", "-fsyntax-only", f"-I{eigen_inc}",
         f"-I{tmp_path}", str(result.wrapper_cpp)],
        capture_output=True, text=True)
    assert p.returncode == 0, p.stderr


def test_scalar_promoted_parameter_emits_scalar_ref(tmp_path: Path):
    """A size-1 PARAMETER port is declared `const double&` — the kernel
    call must take its address (like a scalar MEASUREMENT's `&z`), never
    `.data()` on a double."""
    c = Craft("d")
    c.add(Mass("body", mass=1.5))
    c.add(Thruster("t", force=(0.0, 0.0, 1.0)))
    w = World(name="scalar_world")
    w.add_field(GravityField(g=(0.0, 0.0, -9.81)))
    w.add_craft(c)
    sim = Sim(w, parameters=["body.mass"])
    assert sim.module().port("params").size == 1
    result = TargetCpp(sim, tmp_path, class_name="ScalarDrone")
    assert "const double& params" in result.wrapper_hpp.read_text()
    cpp = result.wrapper_cpp.read_text()
    assert "&params" in cpp and "params.data()" not in cpp
    _syntax_check(result, tmp_path)


def test_size1_noise_port_emits_scalar_ref(tmp_path: Path):
    """Same for a total-noise-dim-1 Module (no stock part declares a lone
    scalar channel, so the typed Module is built by hand)."""
    import casadi as ca
    from manta.ir.module import (
        EntryPoint, Module, Port, PortField, PortRef, Role, StateLayout,
    )
    n, p = ca.MX.sym("n"), ca.MX.sym("p")
    fn = ca.Function("eval", [n, p], [ca.vertcat(n + p, n * p)])
    m = Module(
        name="scalar_noise",
        state=StateLayout(),
        ports=(
            Port("noise", Role.NOISE, (1,),
                 fields=(PortField("w", 1, sigma=0.1),)),
            Port("params", Role.PARAMETER, (1,),
                 fields=(PortField("k", 1, default=2.0),)),
            Port("y", Role.MEASUREMENT, (2,)),
        ),
        functions={"eval": fn},
        entry_points=(EntryPoint("eval", "eval",
                                 (PortRef("noise"), PortRef("params")),
                                 returns=("y",)),))
    result = TargetCpp(m, tmp_path, class_name="ScalarNoise")
    assert "const double& noise" in result.wrapper_hpp.read_text()
    cpp = result.wrapper_cpp.read_text()
    assert "&noise" in cpp and "noise.data()" not in cpp
    assert "&params" in cpp and "params.data()" not in cpp
    _syntax_check(result, tmp_path)

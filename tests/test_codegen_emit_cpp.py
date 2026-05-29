"""Smoke tests for the top-level `TargetCpp(cw, ...)` pipeline."""

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
    # WorldFunctions are carried in the result for cross-checking.
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


def test_target_cpp_rejects_non_world_ir():
    """TargetCpp(<not a Sim>) raises a helpful TypeError."""
    import pytest
    from manta.estimation import EKF
    w = World().add_field(GravityField(g=(0,0,-9.81)))
    w.add_craft(_make_simple_craft())
    ekf = EKF(w)   # not a Sim
    with pytest.raises(TypeError, match="Sim"):
        TargetCpp(ekf, "/tmp/whatever", class_name="X")


def _make_simple_craft():
    c = Craft("d"); c.add(Mass("body", mass=1.0))
    return c

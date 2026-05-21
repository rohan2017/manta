"""Smoke tests for the top-level emit_cpp pipeline."""

from pathlib import Path

import pytest

from manta_next import Craft
from manta_next.codegen import emit_cpp
from manta_next.parts import IMU, Mass, PositionSensor, Thruster


def _hover_craft():
    c = Craft("drone")
    c.add(Mass("body", mass=1.5, moi=(0.05, 0.05, 0.08)))
    c.add(Thruster("t", force=(0.0, 0.0, 1.0)))
    c.add(IMU("g"))
    c.add(PositionSensor("gps"))
    return c


def test_emit_cpp_produces_expected_files(tmp_path: Path):
    result = emit_cpp(_hover_craft(), tmp_path, class_name="Drone")
    expected = {
        result.kernels_c, result.kernels_h,
        result.wrapper_hpp, result.wrapper_cpp,
        result.cmakelists,
    }
    for p in expected:
        assert p.exists(), p
    assert result.class_name == "Drone"
    # CraftFunctions are carried in the result for cross-checking.
    assert result.funcs.craft_name == "drone"
    assert result.funcs.ambient_dim == 13
    assert result.funcs.tangent_dim == 12


def test_emit_cpp_custom_basename(tmp_path: Path):
    result = emit_cpp(_hover_craft(), tmp_path,
                      class_name="Drone", basename="my_robot")
    assert result.kernels_c.name == "my_robot_kernels.c"
    assert result.wrapper_hpp.name == "my_robot.hpp"
    cmake_text = result.cmakelists.read_text()
    assert "my_robot_kernels.c" in cmake_text
    assert "my_robot.cpp" in cmake_text

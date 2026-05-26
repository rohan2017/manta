"""Tests for the Eigen-shaped C++ wrapper emission (world-tick basis)."""

import re
import shutil
import subprocess
from pathlib import Path

import pytest

from manta_next import Craft, World
from manta_next.fields import GravityField
from manta_next.codegen.extract import extract
from manta_next.codegen.kernels import emit_kernels
from manta_next.codegen.wrapper import emit_wrapper
from manta_next.parts import IMU, Mass, PositionSensor, Thruster


def _hover_world():
    c = Craft("drone")
    c.add(Mass("body", mass=1.5, moi=(0.05, 0.05, 0.08)))
    c.add(Thruster("t", force=(0.0, 0.0, 1.0)))
    c.add(IMU("g"))
    c.add(PositionSensor("gps"))
    w = World(name="hover_world")
    w.add_field(GravityField(g=(0.0, 0.0, -9.81)))
    w.add_craft(c)
    return w.compile()


# ---------------------------------------------------------------------------
# Files exist and reference the expected symbols
# ---------------------------------------------------------------------------

def test_wrapper_files_exist(tmp_path: Path):
    cw = _hover_world()
    funcs = extract(cw)
    emit_kernels(funcs, tmp_path, basename="drone")
    paths = emit_wrapper(funcs, cw.world, tmp_path,
                          class_name="Drone", basename="drone")
    assert paths["hpp"].exists()
    assert paths["cpp"].exists()


def test_wrapper_hpp_declares_typed_state_and_methods(tmp_path: Path):
    cw = _hover_world()
    funcs = extract(cw)
    emit_kernels(funcs, tmp_path, basename="drone")
    paths = emit_wrapper(funcs, cw.world, tmp_path,
                          class_name="Drone", basename="drone")
    hpp = paths["hpp"].read_text()

    # Rigid-body slots, owner-prefixed.
    for fld, ty in [("drone_position",         "Eigen::Vector3d"),
                    ("drone_orientation",      "Eigen::Vector4d"),
                    ("drone_velocity",         "Eigen::Vector3d"),
                    ("drone_angular_velocity", "Eigen::Vector3d")]:
        assert re.search(rf"{ty}\s+{fld}\b", hpp), (
            f"missing {ty} {fld} in:\n{hpp}")

    # Thrust input.
    assert "double drone_t_throttle" in hpp

    # Methods.
    assert "predict(const State&" in hpp
    assert "predict_jacobian" in hpp
    assert "measure_drone_gps_position" in hpp
    assert "measure_drone_g_gyro" in hpp
    assert "measure_drone_gps_position_jacobian" in hpp
    assert "measure_drone_g_gyro_jacobian" in hpp


# ---------------------------------------------------------------------------
# End-to-end compile of kernels + wrapper
# ---------------------------------------------------------------------------

@pytest.fixture
def cxx_compiler() -> str | None:
    for c in ("c++", "g++", "clang++"):
        if shutil.which(c):
            return c
    return None


@pytest.fixture
def eigen_include() -> str | None:
    for p in ("/usr/include/eigen3", "/usr/local/include/eigen3"):
        if Path(p, "Eigen", "Dense").exists():
            return p
    return None


def test_wrapper_compiles_against_kernels(tmp_path: Path,
                                          cxx_compiler: str | None,
                                          eigen_include: str | None):
    if cxx_compiler is None:
        pytest.skip("no C++ compiler on PATH")
    if eigen_include is None:
        pytest.skip("Eigen headers not found")

    cw = _hover_world()
    funcs = extract(cw)
    kpaths = emit_kernels(funcs, tmp_path, basename="drone")
    wpaths = emit_wrapper(funcs, cw.world, tmp_path,
                          class_name="Drone", basename="drone")

    obj_c = tmp_path / "kernels.o"
    obj_cxx = tmp_path / "wrapper.o"

    p1 = subprocess.run(
        ["cc", "-c", "-O2", "-Wall", "-Wno-unused-parameter",
         str(kpaths["c"]), "-o", str(obj_c)],
        capture_output=True, text=True)
    assert p1.returncode == 0, p1.stderr

    p2 = subprocess.run(
        [cxx_compiler, "-c", "-std=c++17", "-O2", "-Wall",
         f"-I{eigen_include}", f"-I{tmp_path}",
         str(wpaths["cpp"]), "-o", str(obj_cxx)],
        capture_output=True, text=True)
    assert p2.returncode == 0, p2.stderr

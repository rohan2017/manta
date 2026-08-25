"""Tests for `manta.codegen.kernels` — flat-C kernel emission."""

import re
from pathlib import Path

import pytest

from manta import Craft, Sim, World
from manta.codegen.cpp.kernels import emit_kernel_list
from manta.fields import GravityField
from manta.parts import IMU, Mass, PositionSensor, Thruster


def _hover_world() -> "Sim":
    c = Craft("drone")
    c.add(Mass("body", mass=1.5, moi=(0.05, 0.05, 0.08)))
    c.add(Thruster("t", force=(0.0, 0.0, 1.0)))
    c.add(IMU("g"))
    c.add(PositionSensor("gps"))
    w = World(name="hover_world")
    w.add_field(GravityField(g=(0.0, 0.0, -9.81)))
    w.add_craft(c)
    return Sim(w)


def _sim_kernels(m):
    """The flat-C kernels a Sim Module emits — every baked ca.Function it
    carries (predict + step + Jacobian F + per-sensor h/H)."""
    return list(m.functions.values())


def test_emit_kernels_produces_c_and_h(tmp_path: Path):
    m = _hover_world().deploy_module()
    paths = emit_kernel_list(_sim_kernels(m), tmp_path,
                             basename=m.name)
    assert paths["c"].exists()
    assert paths["h"].exists()
    assert paths["c"].stat().st_size > 1000
    assert paths["h"].stat().st_size > 100


def test_kernel_header_declares_every_function(tmp_path: Path):
    m = _hover_world().deploy_module()
    paths = emit_kernel_list(_sim_kernels(m), tmp_path,
                             basename=m.name)
    header_text = paths["h"].read_text()

    expected_names = [fn.name() for fn in _sim_kernels(m)]
    for name in expected_names:
        pattern = rf"\b{re.escape(name)}\b"
        assert re.search(pattern, header_text), (
            f"function {name!r} missing from generated header.\n"
            f"Header text:\n{header_text[:1000]}…")


def test_kernel_c_compiles_with_cc(tmp_path: Path):
    """Smoke-test that the emitted C is well-formed by invoking the
    system C compiler. Skips if no `cc` on PATH."""
    import shutil
    import subprocess
    cc = shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")
    if cc is None:
        pytest.skip("no C compiler on PATH")

    m = _hover_world().deploy_module()
    paths = emit_kernel_list(_sim_kernels(m), tmp_path,
                             basename=m.name)

    obj = tmp_path / "kernels.o"
    proc = subprocess.run(
        [cc, "-c", "-O2", "-Wall", "-Wno-unused-parameter",
         str(paths["c"]), "-o", str(obj)],
        capture_output=True, text=True, check=False)
    assert proc.returncode == 0, (
        f"compile failed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")
    assert obj.exists()

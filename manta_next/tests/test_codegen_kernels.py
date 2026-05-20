"""Tests for manta_next.codegen.kernels — flat-C kernel emission."""

import os
import re
from pathlib import Path

import pytest

from manta_next import Craft
from manta_next.codegen.extract import extract
from manta_next.codegen.kernels import emit_kernels, kernel_function_names
from manta_next.parts import IMU, Mass, PositionSensor, Thruster


def _hover_craft():
    c = Craft("drone")
    c.add(Mass("body", mass=1.5, moi=(0.05, 0.05, 0.08)))
    c.add(Thruster.linear("t", max_thrust=1.0))
    c.add(IMU("g"))
    c.add(PositionSensor("gps"))
    return c


def test_emit_kernels_produces_c_and_h(tmp_path: Path):
    funcs = extract(_hover_craft())
    paths = emit_kernels(funcs, tmp_path)
    assert paths["c"].exists()
    assert paths["h"].exists()
    assert paths["c"].stat().st_size > 1000     # actually emitted real code
    assert paths["h"].stat().st_size > 100


def test_kernel_header_declares_every_function(tmp_path: Path):
    funcs = extract(_hover_craft())
    paths = emit_kernels(funcs, tmp_path)
    header_text = paths["h"].read_text()

    expected_names = kernel_function_names(funcs).values()
    for name in expected_names:
        # CasADi's header declares each function with extern C int.
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

    funcs = extract(_hover_craft())
    paths = emit_kernels(funcs, tmp_path)

    obj = tmp_path / "kernels.o"
    proc = subprocess.run(
        [cc, "-c", "-O2", "-Wall", "-Wno-unused-parameter",
         str(paths["c"]), "-o", str(obj)],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, (
        f"compile failed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")
    assert obj.exists()

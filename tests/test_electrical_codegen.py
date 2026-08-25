"""Generated C++ parity for a dynamic electrical network."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from manta import Craft, Sim, TargetCpp, TargetNumpy, World
from manta.fields import GravityField
from manta.parts import Contactor, DCSource, Mass, ResistiveLoad


def _electrical_world():
    craft = Craft("rig")
    craft.add(Mass("body", mass=1.0, moi=(1.0, 1.0, 1.0)))
    source = craft.add(DCSource(
        "source", open_circuit_voltage=12.0, rail_voltage=12.0,
        source_resistance=0.1, capacitance=2.0, current_limit=20.0))
    rail = craft.add(Contactor(
        "rail", rail_voltage=10.0, capacitance=1.5,
        series_resistance=0.2, input_current_limit=10.0, closed=0.0))
    load = craft.add(ResistiveLoad("load", resistance=5.0))
    source.connect(rail).connect(load)
    world = World(name="electrical_codegen").add_field(GravityField.none())
    world.add_craft(craft)
    return world


HARNESS = r"""
#include "electricalrig.hpp"
#include <cstdio>

int main() {
    manta_gen::ElectricalRig rig;
    auto x = rig.initial_state();
    manta_gen::ElectricalRig::Inputs u;
    u.rig_source_enabled = 1.0;
    u.rig_rail_closed = 0.0;
    u.rig_load_enabled = 1.0;
    for (int i = 0; i < 1000; ++i) {
        x = rig.predict(x, u, 0.001);
    }
    auto voltage = rig.measure_rig_rail_voltage(x, u);
    auto power = rig.measure_rig_load_input_power(x, u);
    std::printf("state %.17g\n", x.rig_rail_rail_voltage);
    std::printf("voltage %.17g\n", voltage);
    std::printf("power %.17g\n", power);
    return 0;
}
"""


@pytest.mark.cpp
def test_electrical_python_cpp_roundtrip(tmp_path: Path):
    cxx = next((name for name in ("c++", "g++", "clang++")
                if shutil.which(name)), None)
    cc = next((name for name in ("cc", "gcc", "clang")
               if shutil.which(name)), None)
    if cxx is None or cc is None:
        pytest.skip("no C/C++ compiler on PATH")
    eigen = next((path for path in ("/usr/include/eigen3",
                                    "/usr/local/include/eigen3")
                  if Path(path, "Eigen", "Dense").exists()), None)
    if eigen is None:
        pytest.skip("Eigen headers not found")

    transform = Sim(_electrical_world())
    generated = TargetCpp(
        transform.deploy_module(), tmp_path, class_name="ElectricalRig")
    # Electrical diagnostics are scalar Outputs, so each generated
    # measurement Jacobian is 1×N.  Eigen rejects an explicitly ColMajor
    # fixed row vector; keep this assertion beside the compile/run proof.
    assert "Eigen::RowMajor" in generated.wrapper_cpp.read_text()

    kernel_object = tmp_path / "kernels.o"
    wrapper_object = tmp_path / "wrapper.o"
    command = [cc, "-c", "-O2", "-fPIC", str(generated.kernels_c),
               "-o", str(kernel_object)]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    command = [cxx, "-c", "-std=c++17", "-O2", "-fPIC",
               f"-I{eigen}", f"-I{tmp_path}", str(generated.wrapper_cpp),
               "-o", str(wrapper_object)]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr

    harness_source = tmp_path / "harness.cpp"
    harness_source.write_text(HARNESS)
    executable = tmp_path / "harness"
    command = [cxx, "-std=c++17", "-O2", f"-I{eigen}", f"-I{tmp_path}",
               str(harness_source), str(wrapper_object), str(kernel_object),
               "-o", str(executable)]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    result = subprocess.run([str(executable)], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    cpp = {line.split()[0]: float(line.split()[1])
           for line in result.stdout.splitlines()}

    sim = TargetNumpy(transform)
    for _ in range(1000):
        sim.step(0.001)
    state = float(sim.state["rig"]["rail.rail_voltage"])
    # The C++ calls measurement kernels against the final state.  NumPy's
    # ``outputs()`` intentionally holds the pre-commit readings from the last
    # step, so use the same final-state analytical measurements here.
    voltage = state
    power = state**2 / 5.0
    assert cpp["state"] == pytest.approx(state, abs=1e-11)
    assert cpp["voltage"] == pytest.approx(voltage, abs=1e-11)
    assert cpp["power"] == pytest.approx(power, abs=1e-11)

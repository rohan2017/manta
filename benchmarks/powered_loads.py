"""Representative powered-actuator and hotel-load benchmark.

Run from a checkout with::

    PYTHONPATH=. python benchmarks/powered_loads.py --steps 10000
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from manta import Craft, Sim, TargetNumpy, World
from manta.fields import FluidField, GravityField
from manta.parts import (
    ConstantPowerElectronicsLoad,
    DCConverter,
    DCSource,
    ElectricalBus,
    Mass,
    PoweredControlSurface,
    PoweredThruster,
)


def representative_world() -> World:
    craft = Craft("mako")
    craft.add(Mass("body", mass=55.0, moi=(8.0, 12.0, 12.0)))
    source = craft.add(DCSource(
        "pack", open_circuit_voltage=50.4, rail_voltage=50.4,
        capacitance=5.0, source_resistance=0.05, current_limit=100.0,
        brownout_voltage=34.0, recovery_voltage=38.0))
    bus = craft.add(ElectricalBus(
        "main_bus", rail_voltage=48.0, capacitance=0.5,
        series_resistance=0.02, input_current_limit=80.0,
        brownout_voltage=32.0, recovery_voltage=36.0))
    compute_reg = craft.add(DCConverter(
        "compute_reg", output_voltage=12.0, rail_voltage=12.0,
        capacitance=0.2, efficiency=0.92,
        output_current_limit=15.0, output_power_limit=160.0,
        input_power_limit=180.0, brownout_voltage=8.0,
        recovery_voltage=10.0))
    computer = craft.add(ConstantPowerElectronicsLoad(
        "computer", power=100.0, current_limit=14.0,
        brownout_voltage=8.0, recovery_voltage=10.0,
        voltage_floor=1.0))
    source.connect(bus)
    bus.connect(compute_reg).connect(computer)

    for index, offset in enumerate((-0.25, 0.25)):
        thruster = craft.add(PoweredThruster(
            f"thruster_{index}", force=(400.0, 0.0, 0.0),
            mount_offset=(0.0, offset, 0.0), rated_voltage=48.0,
            rated_mechanical_power=1500.0, conversion_efficiency=0.82,
            power_exponent=2.0, brownout_voltage=32.0,
            recovery_voltage=36.0))
        bus.connect(thruster)

    for index in range(4):
        fin = craft.add(PoweredControlSurface(
            f"fin_{index}", area=0.08, chord=0.18,
            rated_voltage=48.0, rated_mechanical_power=30.0,
            conversion_efficiency=0.75, idle_power=0.5,
            brownout_voltage=32.0, recovery_voltage=36.0))
        bus.connect(fin)

    world = (World(name="powered_load_benchmark")
             .add_field(GravityField(g=(0.0, 0.0, 0.0)))
             .add_field(FluidField().add_uniform(density=1025.0)))
    commands = {"computer.enabled": 1.0}
    for index in range(2):
        commands[f"thruster_{index}.throttle"] = 0.35
    for index in range(4):
        commands[f"fin_{index}.deflection_cmd"] = 0.08 if index % 2 else -0.08
    world.add_craft(craft, **commands)
    return world


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--dt", type=float, default=0.001)
    args = parser.parse_args()
    if args.steps <= 0 or args.dt <= 0.0:
        parser.error("--steps and --dt must be positive")

    started = time.perf_counter()
    runtime = TargetNumpy(Sim(representative_world()))
    compile_seconds = time.perf_counter() - started
    for _ in range(100):
        runtime.step(args.dt)
    started = time.perf_counter()
    for _ in range(args.steps):
        runtime.step(args.dt)
    elapsed = time.perf_counter() - started

    outputs = runtime.outputs()["mako"]
    residuals = [abs(float(np.asarray(value).item()))
                 for name, value in outputs.items()
                 if name.endswith("energy_residual")]
    print(f"compile_seconds={compile_seconds:.6f}")
    print(f"steps={args.steps} dt={args.dt:.9g}")
    print(f"step_seconds={elapsed:.6f}")
    print(f"steps_per_second={args.steps / elapsed:.1f}")
    print(f"maximum_final_energy_residual={max(residuals):.3e}")


if __name__ == "__main__":
    main()

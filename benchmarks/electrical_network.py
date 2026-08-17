"""Representative lumped-DC compile and simulation benchmark.

Run from a checkout with:

    PYTHONPATH=. python benchmarks/electrical_network.py --steps 10000 --dt 0.001

The source is a 50.4 V 12-series-equivalent terminal feeding two independent
regulators and compute/payload loads. Battery cell physics belongs to A2; this
workload exercises the A1 network topology and compiler.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from manta import Craft, Sim, TargetNumpy, World
from manta.parts import (
    ConstantCurrentLoad,
    ConstantPowerLoad,
    DCConverter,
    DCSource,
    ElectricalBus,
    Mass,
)


def representative_world() -> World:
    craft = Craft("vehicle")
    craft.add(Mass("body", mass=50.0, moi=(10.0, 10.0, 10.0)))
    source = craft.add(DCSource(
        "pack_terminal", open_circuit_voltage=50.4, rail_voltage=50.4,
        source_resistance=0.06, capacitance=5.0, current_limit=80.0,
        brownout_voltage=34.0, recovery_voltage=38.0))
    bus = craft.add(ElectricalBus(
        "main_bus", rail_voltage=49.0, capacitance=0.5,
        series_resistance=0.01, input_current_limit=70.0,
        brownout_voltage=32.0, recovery_voltage=36.0))
    compute_regulator = craft.add(DCConverter(
        "compute_regulator", output_voltage=12.0, rail_voltage=12.0,
        capacitance=0.2, dropout_voltage=1.0, efficiency=0.92,
        output_current_limit=15.0, output_power_limit=160.0,
        input_power_limit=180.0, brownout_voltage=9.0,
        recovery_voltage=10.0))
    sensor_regulator = craft.add(DCConverter(
        "sensor_regulator", output_voltage=24.0, rail_voltage=24.0,
        capacitance=0.2, dropout_voltage=1.5, efficiency=0.90,
        output_current_limit=8.0, output_power_limit=150.0,
        input_power_limit=170.0, brownout_voltage=18.0,
        recovery_voltage=20.0))
    computer = craft.add(ConstantPowerLoad(
        "computer", power=120.0, current_limit=14.0, voltage_floor=1.0,
        brownout_voltage=8.0, recovery_voltage=10.0))
    sonar = craft.add(ConstantPowerLoad(
        "sonar", power=100.0, current_limit=6.0, voltage_floor=2.0,
        brownout_voltage=16.0, recovery_voltage=20.0))
    hotel = craft.add(ConstantCurrentLoad(
        "hotel", current=1.5, brownout_voltage=30.0,
        recovery_voltage=34.0))

    source.connect(bus)
    bus.connect(compute_regulator).connect(computer)
    bus.connect(sensor_regulator).connect(sonar)
    bus.connect(hotel)
    world = World(name="electrical_benchmark")
    world.add_craft(craft)
    return world


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--dt", type=float, default=0.001)
    args = parser.parse_args()
    if args.steps <= 0 or args.dt <= 0.0:
        parser.error("--steps and --dt must be positive")

    started = time.perf_counter()
    transform = Sim(representative_world())
    runtime = TargetNumpy(transform)
    compile_seconds = time.perf_counter() - started

    for _ in range(100):
        runtime.step(args.dt)
    started = time.perf_counter()
    for _ in range(args.steps):
        runtime.step(args.dt)
    elapsed = time.perf_counter() - started

    outputs = runtime.outputs()["vehicle"]
    residual_names = [name for name in outputs
                      if name.endswith(("kcl_residual", "energy_residual"))]
    maximum_residual = max(
        abs(float(np.asarray(outputs[name]).item()))
        for name in residual_names)
    voltages = {
        name: float(value)
        for name, value in runtime.state["vehicle"].items()
        if name.endswith("rail_voltage")
    }
    print(f"compile_seconds={compile_seconds:.6f}")
    print(f"steps={args.steps} dt={args.dt:.9g}")
    print(f"step_seconds={elapsed:.6f}")
    print(f"steps_per_second={args.steps / elapsed:.1f}")
    print(f"maximum_final_residual={maximum_residual:.3e}")
    print("final_voltages=" + ",".join(
        f"{name}:{voltage:.6f}" for name, voltage in voltages.items()))


if __name__ == "__main__":
    main()

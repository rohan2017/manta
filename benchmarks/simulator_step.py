"""Small repeatable benchmark for the numpy simulator's per-tick surface.

Run from the repository root:

    python benchmarks/simulator_step.py --steps 5000

Construction and native compilation are deliberately outside the timed
region. The workload is a wide truth plant — one craft with many sensor
ports — stepped at a fixed dt with a fresh noise draw each tick, which
exercises the runtime's argument gather, kernel call, and output scatter
rather than the compiled dynamics alone. This is the shape of a Shiver
fleet-simulation tick, where that Python glue, not the kernel, bounded the
achievable real-time factor (2026-08).
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from manta import Craft, Sim, TargetNumpy, World
from manta.fields import GravityField
from manta.parts import IMU, Mass, PositionSensor, VelocitySensor


def make_sim(sensors: int, *, compile_kernels: bool):
    craft = Craft("vehicle")
    craft.add(Mass("body", mass=20.0, moi=(2.0, 2.0, 3.0)))
    for index in range(sensors):
        craft.add(PositionSensor(f"gps{index}", position_noise_sigma=0.5))
        craft.add(VelocitySensor(f"dvl{index}", velocity_noise_sigma=0.05))
        craft.add(IMU(f"imu{index}", accel_noise_sigma=0.01,
                      gyro_noise_sigma=0.001))
    world = World(name="simulator_benchmark").add_field(GravityField.none())
    world.add_craft(craft, position=(0.0, 0.0, 5.0))
    sim = TargetNumpy(Sim(world))
    if compile_kernels:
        sim._enable_compile(optimization="O1")
    return sim


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=5_000)
    parser.add_argument("--sensors", type=int, default=8)
    parser.add_argument("--no-compile", action="store_true")
    args = parser.parse_args()
    if args.steps <= 0 or args.sensors <= 0:
        parser.error("--steps and --sensors must be positive")

    sim = make_sim(args.sensors, compile_kernels=not args.no_compile)
    dt = 0.002
    for _ in range(50):
        sim.step(dt)
    start = time.perf_counter()
    for _ in range(args.steps):
        sim.step(dt)
    elapsed = time.perf_counter() - start
    per_step_us = 1e6 * elapsed / args.steps
    print(
        f"{args.steps} steps, {4 * args.sensors} sensor ports: "
        f"{per_step_us:.1f} us/step, "
        f"{(args.steps * dt) / elapsed:.2f}x real time at dt={dt}"
    )
    readings = sim.outputs()
    assert readings and all(
        np.all(np.isfinite(np.asarray(v))) for group in readings.values()
        for v in group.values())


if __name__ == "__main__":
    main()

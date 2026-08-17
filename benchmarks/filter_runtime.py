"""Small repeatable benchmark for the estimator's live update surface.

Run from the repository root:

    python benchmarks/filter_runtime.py --filter ekf --iterations 10000

Construction/code generation is deliberately outside the timed region. The
workload alternates a 100 Hz prediction and GPS update with per-sample R,
which exercises diagnostics and gating rather than a bare CasADi kernel.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from manta import Craft, EKF, TargetNumpy, UKF, World
from manta.parts import Mass, PositionSensor


def make_runtime(kind: str):
    craft = Craft("vehicle")
    craft.add(Mass("body", mass=20.0, moi=(2.0, 2.0, 3.0)))
    craft.add(PositionSensor("gps", position_noise_sigma=0.5))
    world = World(name="filter_benchmark")
    world.add_craft(craft, position=(0.0, 0.0, 5.0))
    transform = EKF if kind == "ekf" else UKF
    return TargetNumpy(transform(world, gates={"gps.position": 16.3}))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--filter", choices=("ekf", "ukf"), default="ekf")
    parser.add_argument("--iterations", type=int, default=10_000)
    args = parser.parse_args()
    if args.iterations <= 0:
        parser.error("--iterations must be positive")

    runtime = make_runtime(args.filter)
    z = np.array([0.1, -0.1, 5.0])
    R = np.eye(3) * 0.16
    checkpoint = runtime.checkpoint()
    started = time.perf_counter()
    accepted = 0
    for _ in range(args.iterations):
        accepted += runtime.update("gps.position", z, R=R).accepted
        runtime.predict(0.01)
    elapsed = time.perf_counter() - started
    final = runtime.checkpoint()
    runtime.restore(checkpoint)
    print(f"filter={args.filter} iterations={args.iterations}")
    print(f"elapsed_s={elapsed:.6f}")
    print(f"cycles_per_s={args.iterations / elapsed:.1f}")
    print(f"accepted={accepted}")
    print(f"final_time={final.time:.6f}")


if __name__ == "__main__":
    main()

"""Representative simulation-only 12S battery benchmark.

Run from a checkout with:

    PYTHONPATH=. python benchmarks/battery_pack.py --steps 100000 --dt 0.01
"""

from __future__ import annotations

import argparse
import time

from manta.simulation import (
    BatteryCell,
    BatteryCellFaults,
    BatteryStepInput,
    PassiveBalancer,
    SeriesBatteryPack,
)


def representative_pack() -> SeriesBatteryPack:
    return SeriesBatteryPack(
        [BatteryCell(
            usable_capacity_ah=30.0, internal_resistance=0.0025,
            self_discharge_current=1e-5,
            self_discharge_log_sigma=0.1) for _ in range(12)],
        initial_soc=tuple(0.92 - 0.002 * index for index in range(12)),
        balancers=tuple(PassiveBalancer(index, resistance=100.0,
                                        current_limit=0.1)
                        for index in range(12)),
        seed=7,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument("--dt", type=float, default=0.01)
    args = parser.parse_args()
    if args.steps <= 0 or args.dt <= 0.0:
        parser.error("--steps and --dt must be positive")
    pack = representative_pack()
    inputs = BatteryStepInput(
        requested_series_current=8.0,
        cell_temperatures=(298.15,) * 12,
        cell_faults=(BatteryCellFaults(),) * 12,
        balance_enabled=(True, True, True) + (False,) * 9,
    )
    for _ in range(100):
        pack.step(args.dt, inputs)
    started = time.perf_counter()
    maximum_residual = 0.0
    for _ in range(args.steps):
        telemetry = pack.step(args.dt, inputs)
        maximum_residual = max(maximum_residual,
                               abs(telemetry.energy_residual))
    elapsed = time.perf_counter() - started
    print(f"steps={args.steps} dt={args.dt:.9g}")
    print(f"step_seconds={elapsed:.6f}")
    print(f"steps_per_second={args.steps / elapsed:.1f}")
    print(f"maximum_energy_residual={maximum_residual:.3e}")
    print(f"series_current={telemetry.series_current:.6f}")
    print(f"final_pack_voltage={telemetry.terminal_voltage:.6f}")
    print(f"final_soc_imbalance={telemetry.soc_imbalance:.9f}")


if __name__ == "__main__":
    main()

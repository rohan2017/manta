"""Measure the public INS runtime, including validation and input packing.

Run from Manta's root with PYTHONPATH=. and single-threaded BLAS:
    python benchmarks/ins_runtime.py --output /tmp/ins-runtime.json

Uses the existing synthetic Earth INS qualification fixture: raw linearized
INS at 100 Hz with a DVL correction at 10 Hz and per-sample R. It excludes
Shiver, preintegration, transport and other vehicle processes. Construction,
compilation, warmup, reset and memory sampling are outside the timed regions.
Use cProfile separately; do not compare its times with quiet runs.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import casadi
import numpy as np

from examples.qualification.earth_ins import build, prior
from manta import TargetNumpy


def resident_kib() -> int | None:
    """Current Linux process RSS, including retained construction objects."""
    status = Path("/proc/self/status")
    if not status.exists():
        return None
    for line in status.read_text().splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1])
    return None


def run(cycles: int, batches: int) -> dict:
    ins = build(covariance="linearized", expand=True, propagation="raw")
    runtime = TargetNumpy(ins)
    runtime.compile_functions(
        tuple(runtime.module.functions), optimization="O1", max_instructions=250000
    )
    u = {
        ins.sys.accel_input: np.array([0., 0., 9.81]),
        ins.sys.gyro_input: np.asarray(ins.navigation_frame.angular_velocity),
    }
    z = np.zeros(3)
    R = np.eye(3) * 1e-6
    P = prior(ins, 1e-8)

    def exercise(count: int) -> None:
        for index in range(count):
            runtime.predict(.01, u=u)
            if (index + 1) % 10 == 0:
                runtime.update("dvl.velocity", z, R=R, u=u)

    runtime.reset(P=P)
    exercise(1000)
    samples = []
    final = None
    for _ in range(batches):
        runtime.reset(P=P)
        rss_before = resident_kib()
        cpu_start = time.process_time()
        wall_start = time.perf_counter()
        exercise(cycles)
        wall = time.perf_counter() - wall_start
        cpu = time.process_time() - cpu_start
        samples.append({"cpu_s": cpu, "wall_s": wall,
                        "rss_before_kib": rss_before,
                        "rss_after_kib": resident_kib()})
        observed = (runtime.x, runtime.P)
        for value in observed:
            if not np.isfinite(value).all():
                raise ValueError("nonfinite final state or covariance")
        if final is not None:
            for actual, expected in zip(observed, final):
                np.testing.assert_array_equal(actual, expected)
        final = observed
    assert final is not None
    return {
        "scope": "synthetic raw linearized INS public runtime, expanded O1; "
                 "100 Hz prediction, 10 Hz DVL with R override; excludes "
                 "Shiver, preintegration and I/O; faster than real time",
        "host": platform.platform(), "python": platform.python_version(),
        "numpy": np.__version__, "casadi": casadi.__version__,
        "cycles_per_batch": cycles, "updates_per_batch": cycles // 10,
        "simulated_s_per_batch": cycles / 100,
        "median_cpu_s": float(np.median([s["cpu_s"] for s in samples])),
        "median_wall_s": float(np.median([s["wall_s"] for s in samples])),
        "samples": samples,
        "final_x": final[0].tolist(), "final_P": final[1].tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cycles", type=int, default=20000)
    parser.add_argument("--batches", type=int, default=7)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.cycles <= 0 or args.batches <= 0:
        parser.error("--cycles and --batches must be positive")
    result = run(args.cycles, args.batches)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items()
                      if key not in {"final_x", "final_P", "samples"}}, indent=2))


if __name__ == "__main__":
    main()

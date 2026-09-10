"""Desktop runtime and generated-source cost for the synthetic INS fixture."""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

from manta import IMUPreintegrator, TargetCpp, TargetNumpy

from .earth_ins import build


def run(output: Path, iterations: int = 2000):
    records = []
    for mode in ("raw", "preintegrated"):
        for active in (False, True):
            ir = build(propagation=mode, frame_enabled=active)
            tick = time.perf_counter()
            runtime = TargetNumpy(ir, compile=True, max_instructions=30000)
            build_seconds = time.perf_counter() - tick
            accel = (0, 0, 9.81)
            gyro = build().navigation_frame.angular_velocity
            accumulator = TargetNumpy(IMUPreintegrator())
            for _ in range(10):
                packet = accumulator.step(
                    0.01,
                    accel=accel,
                    gyro=gyro,
                    accel_bias=(0, 0, 0),
                    gyro_bias=(0, 0, 0),
                )
            generated = TargetCpp(
                ir, output / f"{mode}-{active}", class_name="EarthIns"
            )
            tick = time.perf_counter()
            for _ in range(iterations):
                if mode == "raw":
                    runtime.predict(0.01, u={"imu.accel": accel, "imu.gyro": gyro})
                else:
                    runtime.predict_preintegrated(packet)
            records.append(
                {
                    "propagation": mode,
                    "rotating_frame": active,
                    "runtime_predict_us": (time.perf_counter() - tick)
                    / iterations
                    * 1e6,
                    "build_or_cache_load_s": build_seconds,
                    "generated_c_bytes": generated.kernels_c.stat().st_size,
                    "generated_wrapper_bytes": generated.wrapper_cpp.stat().st_size,
                    "input_doubles": len(ir.sys.u_defaults),
                    "state_doubles": ir.spec.ambient_dim,
                    "platform": platform.machine(),
                }
            )
    return records


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    records = run(args.output / "generated")
    (args.output / "benchmark.json").write_text(json.dumps(records, indent=2) + "\n")
    print(json.dumps(records), flush=True)


if __name__ == "__main__":
    main()

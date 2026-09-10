"""Run the actual INS state kernel for 6/12/24 hours at IMU cadence.

A generated fold removes Python call overhead without changing the one-step
recurrence. This is an oracle numerical test, not statistical qualification.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import casadi as ca
import numpy as np

from manta import IMUPreintegrator, TargetNumpy
from manta.codegen.numpy._compile import compile_functions

from .earth_ins import build


def run(rate=100, propagation="raw"):
    ir = build(rate=rate, propagation=propagation)
    frame = ir.navigation_frame
    runtime = TargetNumpy(ir)
    a = (0, 0, 9.81)
    w = frame.angular_velocity
    if propagation == "raw":
        dt = 1 / rate
        inputs = {"imu.accel": a, "imu.gyro": w}
    else:
        # Ten physical samples in every companion packet.
        pre = TargetNumpy(IMUPreintegrator())
        for _ in range(10):
            p = pre.step(
                1 / rate, accel=a, gyro=w, accel_bias=(0, 0, 0), gyro_bias=(0, 0, 0)
            )
        dt = 10 / rate
        inputs = runtime.preintegrated_inputs(p)
    u = ir.sys.resolve_u(inputs)
    x = ca.MX.sym("x", ir.spec.ambient_dim)
    step = ca.Function("oracle_step", [x], [ir.sys.predict_fn(x, u, dt, 0)])
    # Fold one minute of exactly the same generated recurrence.
    fold = step.fold(round(60 / dt))
    tick = time.perf_counter()
    native = compile_functions(
        {"minute": fold}, optimization="O1", max_instructions=20000
    )["minute"]
    build_s = time.perf_counter() - tick
    state = runtime.x
    records = []
    tick = time.perf_counter()
    for minute in range(24 * 60):
        state = native(state)
        if minute + 1 in (360, 720, 1440):
            values = np.asarray(state).reshape(-1)

            def value(name, values=values):
                s = ir.spec.slot("craft." + name)
                return values[s.ambient_offset : s.ambient_offset + s.ambient_dim]

            records.append(
                {
                    "hours": (minute + 1) / 60,
                    "position_norm_m": float(np.linalg.norm(value("position"))),
                    "velocity_norm_m_s": float(np.linalg.norm(value("velocity"))),
                    "quaternion_norm_error": float(
                        abs(np.linalg.norm(value("orientation")) - 1)
                    ),
                    "quaternion_error": float(
                        np.linalg.norm(value("orientation") - np.array([1, 0, 0, 0]))
                    ),
                }
            )
    return {
        "rate": rate,
        "propagation": propagation,
        "build_s": build_s,
        "run_s": time.perf_counter() - tick,
        "records": records,
    }


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--rate", type=int, default=100)
    p.add_argument("--propagation", choices=["raw", "preintegrated"], default="raw")
    args = p.parse_args()
    report = run(args.rate, args.propagation)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()

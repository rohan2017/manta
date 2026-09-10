"""Native kernel timings for linearized and nonlinear INS on the current host."""

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np

from manta import IMUPreintegrator, TargetNumpy
from manta.codegen.numpy._compile import compile_functions
from manta.estimation.imu_preintegrator import frame_preintegrated_packet
from manta.ir._rotation import quat_to_rotmat

from .earth_ins import build, prior


def run(repetitions=1000, *, expand=False):
    rows = []
    for propagation in ("raw", "preintegrated"):
        for mode in ("linearized", "geometric", "nonlinear"):
            ins = build(
                covariance=mode,
                expand=expand,
                propagation=propagation,
                mounted=propagation == "preintegrated",
            )
            module = ins.module()
            runtime = TargetNumpy(ins)
            runtime.reset(P=prior(ins, 1e-8))
            rotation = (
                np.asarray(quat_to_rotmat(runtime.x[3:7])) @ ins.sys.R_craft_from_sensor
            )
            accel = rotation.T @ np.array([0.0, 0.0, 9.81])
            gyro = rotation.T @ np.asarray(ins.navigation_frame.angular_velocity)
            u = ins.sys.u_defaults.copy()
            if propagation == "raw":
                u[ins.sys._input_slices[ins.sys.accel_input]] = accel
                u[ins.sys._input_slices[ins.sys.gyro_input]] = gyro
                dt = 0.01
            else:
                pre = TargetNumpy(
                    IMUPreintegrator(gyro_noise_density=1e-7, accel_noise_density=1e-5)
                )
                for _ in range(10):
                    packet = pre.step(
                        0.01,
                        accel=accel,
                        gyro=gyro,
                        accel_bias=np.zeros(3),
                        gyro_bias=np.zeros(3),
                    )
                packet = frame_preintegrated_packet(
                    packet,
                    end_accel=accel,
                    end_gyro=gyro,
                )
                for name, full in ins.preintegration_input_map.items():
                    u[ins.sys._input_slices[full]] = np.asarray(packet[name]).ravel()
                dt = 0.1
            started = time.perf_counter()
            functions = {
                "predict": module.functions["predict"],
                "update": module.functions["update_craft_dvl_velocity"],
            }
            if expand:
                functions = {name: fn.expand() for name, fn in functions.items()}
            native = compile_functions(
                functions,
                optimization="O1",
                max_instructions=250000 if expand else 50000,
                timeout_s=120,
            )
            compile_s = time.perf_counter() - started
            state = [runtime.x, runtime.P]
            if runtime.P_consider is not None:
                state.append(runtime.P_consider)
            row = {
                "propagation": propagation,
                "covariance": mode,
                "state_dimension": ins.spec.tangent_dim,
                "compile_or_cache_load_s": compile_s,
            }
            for name, args in (
                ("predict", state + [u, dt, 0]),
                ("update", state + [np.zeros(3), u, 0]),
            ):
                reference = module.functions[
                    "predict" if name == "predict" else "update_craft_dvl_velocity"
                ](*args)
                actual = native[name](*args)
                for index, (expected, observed) in enumerate(zip(reference, actual)):
                    expected, observed = np.asarray(expected), np.asarray(observed)
                    if not np.all(np.isfinite(observed)):
                        raise ValueError(f"nonfinite native {name} output {index}")
                    if index == 1:
                        # Compare covariance in correlation units so large
                        # position variances cannot hide tiny bias errors.
                        scale = np.sqrt(np.maximum(np.diag(expected), 0))
                        denominator = np.outer(scale, scale)
                        denominator[denominator == 0] = 1
                        expected, observed = (
                            expected / denominator,
                            observed / denominator,
                        )
                    np.testing.assert_allclose(
                        observed, expected, rtol=1e-9, atol=1e-11
                    )
                fn = native[name].map(repetitions)
                fn(*args)
                samples = []
                for _ in range(7):
                    started = time.perf_counter()
                    fn(*args)
                    samples.append((time.perf_counter() - started) * 1e6 / repetitions)
                row[name + "_median_us"] = float(np.median(samples))
            row["native_output_parity"] = "pass"
            rows.append(row)
            print(json.dumps(row), flush=True)
    return {
        "host": platform.platform(),
        "machine": platform.machine(),
        "repetitions": repetitions,
        "expanded_kernels": expand,
        "scope": "native double-precision kernels, O1, CasADi map loop; excludes compilation, Python runtime validation, framing and I/O",
        "rows": rows,
    }


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--expand", action="store_true")
    a = p.parse_args()
    r = run(expand=a.expand)
    a.output.write_text(json.dumps(r, indent=2) + "\n")


if __name__ == "__main__":
    main()

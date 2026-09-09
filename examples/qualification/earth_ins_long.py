"""Long-duration native INS numerical checks with a stationary Earth oracle.

The IMU readings here are noiseless. Declared nonzero sensor covariance and a
calibrated physical prior exercise the estimator's covariance recursion; this
is a numerical durability check, not Monte Carlo statistical acceptance.
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
from manta.estimation.imu_preintegrator import frame_preintegrated_packet
from manta.ir._rotation import quat_to_rotmat

from .earth_ins import build, prior


def run(*, hours=24, rate=1000, packet_samples=100, mounted=True):
    started = time.perf_counter()
    ins = build(
        covariance="nonlinear", propagation="preintegrated", rate=rate, mounted=mounted
    )
    module, spec = ins.module(), ins.spec
    dt = packet_samples / rate
    x0 = np.asarray(module.port("prior_x").init)
    R = np.asarray(quat_to_rotmat(x0[3:7])) @ ins.sys.R_craft_from_sensor
    a = R.T @ np.array([0.0, 0.0, 9.81])
    g = R.T @ np.asarray(ins.navigation_frame.angular_velocity)
    pre = TargetNumpy(
        IMUPreintegrator(gyro_noise_density=1e-7, accel_noise_density=1e-5)
    )
    for _ in range(packet_samples):
        packet = pre.step(
            1 / rate, accel=a, gyro=g, accel_bias=np.zeros(3), gyro_bias=np.zeros(3)
        )
    packet = frame_preintegrated_packet(packet, end_accel=a, end_gyro=g)
    u = ins.sys.u_defaults.copy()
    for name, full in ins.preintegration_input_map.items():
        u[ins.sys._input_slices[full]] = np.asarray(packet[name]).ravel()
    na, n = spec.ambient_dim, spec.tangent_dim
    state = ca.MX.sym("state", na + n * n)
    x, p = state[:na], ca.reshape(state[na:], n, n)
    pred = module.functions["predict"](x, p, u, dt, 0)
    updated = module.functions["update_craft_dvl_velocity"](*pred, np.zeros(3), u, 0)
    step = ca.Function(
        "long_ins_step", [state], [ca.vertcat(updated[0], ca.vec(updated[1]))]
    )
    # Fixed inputs and gravity make this a legitimate constant-input fold.
    chunk = 1000
    folded = step.fold(chunk)
    native = compile_functions(
        {"chunk": folded, "step": step}, optimization="O1", max_instructions=50000
    )
    initial = module.functions["initialize_prior"](x0, prior(ins, 1e-8))
    current = np.r_[
        np.asarray(initial[0]).ravel(), np.asarray(initial[1]).ravel(order="F")
    ]
    interpreted = np.asarray(step(current)).ravel()
    generated = np.asarray(native["step"](current)).ravel()
    np.testing.assert_allclose(generated, interpreted, atol=1e-12, rtol=1e-10)
    # Also check the deterministic mechanization without aiding or covariance.
    physical_step = ca.Function(
        "oracle_step", [ins.sys.x_sym], [ins.sys.predict_fn(ins.sys.x_sym, u, dt, 0)]
    )
    oracle_fold = compile_functions(
        {"oracle": physical_step.fold(chunk)}, optimization="O1", max_instructions=50000
    )["oracle"]
    oracle = np.r_[x0, np.zeros(3)]
    extrema = {
        "quaternion_norm_error": 0.0,
        "scaled_covariance_asymmetry": 0.0,
        "minimum_correlation_eigenvalue": float("inf"),
    }
    records = []
    total = round(hours * 3600 / dt)
    if total % chunk:
        raise ValueError("duration must contain a whole number of chunks")
    for k in range(chunk, total + 1, chunk):
        current = np.asarray(native["chunk"](current)).ravel()
        oracle = np.asarray(oracle_fold(oracle)).ravel()
        if not np.all(np.isfinite(current)) or not np.all(np.isfinite(oracle)):
            raise AssertionError(f"nonfinite state at {k * dt} seconds")
        p = current[na:].reshape((n, n), order="F")
        scale = np.sqrt(np.diag(p))
        correlation = p / np.outer(scale, scale)
        extrema["quaternion_norm_error"] = max(
            extrema["quaternion_norm_error"], abs(np.linalg.norm(current[3:7]) - 1)
        )
        extrema["scaled_covariance_asymmetry"] = max(
            extrema["scaled_covariance_asymmetry"],
            float(np.max(np.abs(correlation - correlation.T))),
        )
        extrema["minimum_correlation_eigenvalue"] = min(
            extrema["minimum_correlation_eigenvalue"],
            float(np.linalg.eigvalsh(correlation).min()),
        )
        if k * dt in (6 * 3600, 12 * 3600, 24 * 3600) or k == total:
            records.append(
                {
                    "hours": k * dt / 3600,
                    **extrema,
                    "oracle_position_norm_m": float(np.linalg.norm(oracle[:3])),
                    "oracle_velocity_norm_mps": float(np.linalg.norm(oracle[7:10])),
                    "oracle_quaternion_error": float(
                        np.linalg.norm(oracle[3:7] - x0[3:7])
                    ),
                    "aided_position_norm_m": float(np.linalg.norm(current[:3])),
                    "aided_velocity_norm_mps": float(np.linalg.norm(current[7:10])),
                    "yaw_sigma_deg": float(np.degrees(np.sqrt(p[5, 5]))),
                    "elapsed_s": time.perf_counter() - started,
                }
            )
            print(json.dumps(records[-1]), flush=True)
    assert extrema["quaternion_norm_error"] < 1e-12
    assert extrema["scaled_covariance_asymmetry"] < 1e-12
    assert extrema["minimum_correlation_eigenvalue"] >= -1e-12
    assert np.linalg.norm(oracle[7:10]) < 1e-7
    assert np.linalg.norm(oracle[3:7] - x0[3:7]) < 1e-9
    return {
        "acceptance": "pass",
        "scope": "stationary numerical durability; not statistical acceptance",
        "rate": rate,
        "packet_samples": packet_samples,
        "mounted": mounted,
        "records": records,
        "artifact_id": module.artifact_id,
    }


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--hours", type=float, default=24)
    parser.add_argument("--rate", type=int, default=1000)
    parser.add_argument("--packet-samples", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    args = vars(parser.parse_args())
    output = args.pop("output")
    report = run(**args)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()

"""Matched-cadence raw/packet INS consistency control.

Both paths receive the same 100 Hz IMU samples and independent 10 Hz DVL
samples by default. Truth is stationary with a colocated IMU and constant
physical biases. Packet boundary Schmidt covariance is retained. This is
a statistical fixture, not a sensor-driver packet framing qualification.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import casadi as ca
import numpy as np

from manta import IMUPreintegrator
from manta.codegen.numpy._compile import compile_functions
from manta.estimation import chi2_quantile
from manta.ir._rotation import quat_to_rotmat

from .earth_ins import build, normalized_nees, prior


def run(
    *,
    propagation="preintegrated",
    bias_sigma=1e-5,
    gyro_density=1e-7,
    duration=300,
    seed=82719,
    seeds=16,
    packet_samples=10,
):
    rate = 100
    if not isinstance(seeds, int) or seeds < 2:
        raise ValueError("seeds must be an integer >= 2")
    if (
        not isinstance(packet_samples, int)
        or packet_samples <= 0
        or 1000 % packet_samples
    ):
        raise ValueError("packet_samples must be a positive divisor of 1000")
    ticks = round(duration * rate)
    if not np.isfinite(duration) or duration < 10 or ticks % 1000:
        raise ValueError("duration must be a positive multiple of 10 seconds")
    started = time.perf_counter()
    dt = 1 / rate
    ins = build(propagation=propagation, gyro_density=gyro_density)
    spec = ins.spec
    module = ins.module()
    n, na = spec.tangent_dim, spec.ambient_dim
    x0 = np.asarray(module.state.field("x").init)
    p0 = prior(ins, bias_sigma)
    xp = ca.MX.sym("x", na)
    pp = ca.MX.sym("P", n, n)
    cc = ca.MX.sym("C", n, 3)
    u = ca.MX.sym("u", len(ins.sys.u_defaults))
    z = ca.MX.sym("z", 3)
    pre = propagation == "preintegrated"
    pred = module.functions["predict"](
        xp, pp, *([cc] if pre else []), u, dt * packet_samples if pre else dt, 0
    )
    up = module.functions["update_diagnostic_craft_dvl_velocity"](*pred, z, u, 0)
    step = ca.Function(
        "packet_qualify",
        [xp, pp, cc, u, z],
        [up[0], up[1], up[2] if pre else cc, up[-2], up[-1]],
    )
    funcs = {"packet_qualify": step}
    if not pre:
        funcs["raw_qualify"] = ca.Function(
            "raw_qualify", [xp, pp, u], [pred[0], pred[1]]
        )
    native = compile_functions(funcs, optimization="O1", max_instructions=50000)
    step = native["packet_qualify"].map(seeds)
    raw_predict = native["raw_qualify"].map(seeds) if not pre else None
    if pre:
        block = IMUPreintegrator(
            gyro_noise_density=gyro_density, accel_noise_density=1e-5
        )
        accum = compile_functions(
            {"accumulate": block.update_fn}, optimization="O1", max_instructions=50000
        )["accumulate"].map(seeds)
        accum_zero = np.tile(block.x0[:, None], (1, seeds))
        accum_x = accum_zero.copy()
        packet_slices = {}
        offset = 0
        for field in block.outputs:
            packet_slices[field.name] = slice(offset, offset + field.dim)
            offset += field.dim

    def ambient(name):
        slot = spec.slot(name)
        return slice(slot.ambient_offset, slot.ambient_offset + slot.ambient_dim)

    orientation = ambient("craft.orientation")
    gyro_bias = ambient("craft.imu.gyro_bias")
    accel_bias = ambient("craft.imu.accel_bias")
    rng = np.random.default_rng(seed)
    samples = rng.normal(size=(n, seeds)) * np.sqrt(np.diag(p0))[:, None]
    physical = getattr(spec, "product_spec", spec)
    truth = np.column_stack(
        [physical.boxplus_num(x0, samples[:, i]) for i in range(seeds)]
    )
    truth[ambient("craft.position")] = 0
    truth[ambient("craft.velocity")] = 0
    omega = np.asarray(ins.navigation_frame.angular_velocity)
    gyro = np.column_stack(
        [
            np.asarray(quat_to_rotmat(ca.DM(truth[orientation, i]))).T @ omega
            + truth[gyro_bias, i]
            for i in range(seeds)
        ]
    )
    accel = np.column_stack(
        [
            np.asarray(quat_to_rotmat(ca.DM(truth[orientation, i]))).T @ [0, 0, 9.81]
            + truth[accel_bias, i]
            for i in range(seeds)
        ]
    )
    x = np.tile(x0[:, None], (1, seeds))
    covariance = np.tile(p0, (1, seeds))
    cross = np.zeros((n, 3 * seeds))
    xt = ca.MX.sym("truth", na)
    error = ca.Function("error", [xp, xt], [spec.boxminus_sym(xt, xp)]).map(seeds)
    selected = []
    for name in ("craft.orientation", "craft.imu.gyro_bias", "craft.imu.accel_bias"):
        slot = spec.slot(name)
        selected.extend(
            range(slot.tangent_offset, slot.tangent_offset + slot.tangent_dim)
        )
    heading = spec.slot("craft.orientation").tangent_offset + 2
    records = []
    rejected = 0
    for k in range(ticks):
        a = accel + rng.normal(size=(3, seeds)) * 1e-5 / np.sqrt(dt)
        g = gyro + rng.normal(size=(3, seeds)) * gyro_density / np.sqrt(dt)
        if pre:
            accum_x, y = accum(accum_x, np.vstack([a, g, np.zeros((6, seeds))]), dt, 0)
            if (k + 1) % packet_samples:
                continue
            packet = np.asarray(y)
            inputs = np.tile(ins.sys.u_defaults[:, None], (1, seeds))
            for name, full in ins.preintegration_input_map.items():
                inputs[ins.sys._input_slices[full]] = packet[packet_slices[name]]
            accum_x = accum_zero.copy()
        else:
            inputs = np.tile(ins.sys.u_defaults[:, None], (1, seeds))
            inputs[ins.sys._input_slices[ins.sys.accel_input]] = a
            inputs[ins.sys._input_slices[ins.sys.gyro_input]] = g
            if (k + 1) % packet_samples:
                x, covariance = raw_predict(x, covariance, inputs)
                continue
        x, covariance, cross, nis, accepted = step(
            x, covariance, cross, inputs, rng.normal(size=(3, seeds)) * 0.001
        )
        rejected += int(np.count_nonzero(np.asarray(accepted) < 0.5))
        if (k + 1) % 1000 == 0:
            e = np.asarray(error(x, truth))
            cov = np.asarray(covariance)
            nees = []
            for i in range(seeds):
                marginal = cov[:, i * n : (i + 1) * n][np.ix_(selected, selected)]
                nees.append(normalized_nees(e[selected, i], marginal))
            records.append(
                {
                    "t": (k + 1) * dt,
                    "yaw_rmse_deg": float(
                        np.degrees(np.sqrt(np.mean(e[heading] ** 2)))
                    ),
                    "yaw_sigma_deg": float(
                        np.degrees(
                            np.sqrt(
                                np.mean(
                                    [
                                        cov[heading, i * n + heading]
                                        for i in range(seeds)
                                    ]
                                )
                            )
                        )
                    ),
                    "attitude_bias_anees": float(np.mean(nees)),
                    "anis": float(np.mean(nis)),
                }
            )
    bounds = [chi2_quantile(9 * seeds, p) / seeds for p in (0.025, 0.975)]
    return {
        "acceptance": "pass"
        if bounds[0] <= records[-1]["attitude_bias_anees"] <= bounds[1]
        else "fail",
        "acceptance_scope": "final attitude/bias ANEES; not release",
        "truth_motion": "stationary, colocated IMU, constant biases",
        "physical_prior": "product SO(3) attitude and independent additive bias samples",
        "error_coordinates": getattr(spec, "error_model", "product_manifold"),
        "propagation": propagation,
        "gyro_density": gyro_density,
        "gyro_bias_prior_sigma": bias_sigma,
        "rate": rate,
        "aiding_rate": rate / packet_samples,
        "packet_samples": packet_samples,
        "duration": duration,
        "seed": seed,
        "seeds": seeds,
        "rejected_updates": rejected,
        "attitude_bias_anees_95_percent_bounds": bounds,
        "records": records,
        "module_artifact_id": module.artifact_id,
        "elapsed_s": time.perf_counter() - started,
    }


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument(
        "--propagation", choices=("raw", "preintegrated"), default="preintegrated"
    )
    parser.add_argument("--bias-sigma", type=float, default=1e-5)
    parser.add_argument("--gyro-density", type=float, default=1e-7)
    parser.add_argument("--duration", type=float, default=300)
    parser.add_argument("--seed", type=int, default=82719)
    parser.add_argument("--seeds", type=int, default=16)
    parser.add_argument("--packet-samples", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    args = vars(parser.parse_args())
    output = args.pop("output")
    report = run(**args)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps({"acceptance": report["acceptance"], **report["records"][-1]}),
        flush=True,
    )
    raise SystemExit(0 if report["acceptance"] == "pass" else 2)


if __name__ == "__main__":
    main()

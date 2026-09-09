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
from manta.estimation.imu_preintegrator import frame_preintegrated_packet
from manta.ir._rotation import quat_to_rotmat

from .earth_ins import SPIN, build, normalized_nees, prior


def run(
    *,
    propagation="preintegrated",
    bias_sigma=1e-5,
    gyro_density=1e-7,
    duration=300,
    seed=82719,
    seeds=16,
    packet_samples=10,
    covariance="linearized",
    aiding_samples=10,
    mounted=False,
    rate=100,
    latitude=37.78,
    spin=SPIN,
):
    if not isinstance(rate, int) or rate <= 0:
        raise ValueError("rate must be a positive integer")
    if not isinstance(seeds, int) or seeds < 2:
        raise ValueError("seeds must be an integer >= 2")
    if (
        not isinstance(packet_samples, int)
        or packet_samples <= 0
        or 1000 % packet_samples
    ):
        raise ValueError("packet_samples must be a positive divisor of 1000")
    if (
        not isinstance(aiding_samples, int)
        or aiding_samples <= 0
        or aiding_samples % packet_samples
    ):
        raise ValueError("aiding_samples must be a positive multiple of packet_samples")
    ticks = round(duration * rate)
    if not np.isfinite(duration) or duration < 10 or ticks % 1000:
        raise ValueError("duration must be a positive multiple of 10 seconds")
    started = time.perf_counter()
    dt = 1 / rate
    ins = build(
        propagation=propagation,
        gyro_density=gyro_density,
        covariance=covariance,
        mounted=mounted,
        rate=rate,
        latitude=latitude,
        spin=spin,
    )
    spec = ins.spec
    module = ins.module()
    n, na = spec.tangent_dim, spec.ambient_dim
    x0 = np.asarray(
        module.port("prior_x").init
        if "initialize_prior" in module.functions
        else module.state.field("x").init
    )
    p0 = prior(ins, bias_sigma)
    xp = ca.MX.sym("x", na)
    pp = ca.MX.sym("P", n, n)
    has_consider = any(field.name == "P_consider" for field in module.state.fields)
    nc = module.state.field("P_consider").shape[1] if has_consider else 0
    cc = ca.MX.sym("C", n, nc)
    u = ca.MX.sym("u", len(ins.sys.u_defaults))
    z = ca.MX.sym("z", 3)
    pre = propagation == "preintegrated"
    pred = module.functions["predict"](
        xp,
        pp,
        *([cc] if has_consider else []),
        u,
        dt * packet_samples if pre else dt,
        0,
    )
    up = module.functions["update_diagnostic_craft_dvl_velocity"](*pred, z, u, 0)
    step = ca.Function(
        "packet_qualify",
        [xp, pp, cc, u, z],
        [up[0], up[1], up[2] if has_consider else cc, up[-2], up[-1]],
    )
    funcs = {"packet_qualify": step}
    funcs["predict_qualify"] = ca.Function(
        "predict_qualify",
        [xp, pp, cc, u],
        [pred[0], pred[1], pred[2] if has_consider else cc],
    )
    native = compile_functions(funcs, optimization="O1", max_instructions=50000)
    step = native["packet_qualify"].map(seeds)
    predict_only = native["predict_qualify"].map(seeds)
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
    samples = rng.normal(size=(len(p0), seeds)) * np.sqrt(np.diag(p0))[:, None]
    physical = (
        module.port("prior_x").spec if "initialize_prior" in module.functions else spec
    )
    truth = np.column_stack(
        [physical.boxplus_num(x0, samples[:, i]) for i in range(seeds)]
    )
    if physical.ambient_dim != spec.ambient_dim:
        truth = np.column_stack(
            [spec.pack_projected(physical.to_nested(truth[:, i])) for i in range(seeds)]
        )
    truth[ambient("craft.position")] = 0
    truth[ambient("craft.velocity")] = 0
    omega = np.asarray(ins.navigation_frame.angular_velocity)
    gyro = np.column_stack(
        [
            (
                np.asarray(quat_to_rotmat(ca.DM(truth[orientation, i])))
                @ ins.sys.R_craft_from_sensor
            ).T
            @ omega
            + truth[gyro_bias, i]
            for i in range(seeds)
        ]
    )
    accel = np.column_stack(
        [
            (
                np.asarray(quat_to_rotmat(ca.DM(truth[orientation, i])))
                @ ins.sys.R_craft_from_sensor
            ).T
            @ [0, 0, 9.81]
            + truth[accel_bias, i]
            for i in range(seeds)
        ]
    )
    if "initialize_prior" in module.functions:
        mapped = module.functions["initialize_prior"](x0, p0)
        x0, p0 = np.asarray(mapped[0]).ravel(), np.asarray(mapped[1])
    x = np.tile(x0[:, None], (1, seeds))
    covariance = np.tile(p0, (1, seeds))
    cross = np.zeros((n, nc * seeds))
    xt = ca.MX.sym("truth", na)
    error = ca.Function("error", [xp, xt], [spec.boxminus_sym(xt, xp)]).map(seeds)
    rotation = quat_to_rotmat(xp[orientation])
    heading_value = ca.atan2(rotation[1, 0], rotation[0, 0])
    heading_delta = ca.MX.sym("heading_delta", n)
    heading_gradient = ca.substitute(
        ca.jacobian(
            ca.substitute(heading_value, xp, spec.boxplus_sym(xp, heading_delta)),
            heading_delta,
        ),
        heading_delta,
        ca.MX.zeros(n),
    )
    heading_fn = ca.Function(
        "physical_heading", [xp], [heading_value, heading_gradient]
    )
    truth_heading = np.array([float(heading_fn(truth[:, i])[0]) for i in range(seeds)])
    selected = []
    for name in ("craft.orientation", "craft.imu.gyro_bias", "craft.imu.accel_bias"):
        slot = spec.slot(name)
        selected.extend(
            range(slot.tangent_offset, slot.tangent_offset + slot.tangent_dim)
        )
    heading = spec.slot("craft.orientation").tangent_offset + 2
    records = []
    rejected = 0
    # Independent acquisition and aiding streams keep samples identical when
    # changing packet length. Look ahead to the real right boundary; the same
    # sample becomes the next interval's left acquisition.
    a_seed, g_seed, d_seed = np.random.SeedSequence(seed).spawn(3)
    a_rng, g_rng, d_rng = (np.random.default_rng(s) for s in (a_seed, g_seed, d_seed))

    def sample():
        return (
            accel + a_rng.normal(size=(3, seeds)) * 1e-5 / np.sqrt(dt),
            gyro + g_rng.normal(size=(3, seeds)) * gyro_density / np.sqrt(dt),
        )

    next_a, next_g = sample()
    for k in range(ticks):
        a, g = next_a, next_g
        next_a, next_g = sample()
        if pre:
            accum_x, y = accum(accum_x, np.vstack([a, g, np.zeros((6, seeds))]), dt, 0)
            if (k + 1) % packet_samples:
                continue
            packet = np.asarray(y)
            framed = [
                frame_preintegrated_packet(
                    {name: packet[sl, i] for name, sl in packet_slices.items()},
                    end_accel=next_a[:, i],
                    end_gyro=next_g[:, i],
                    end_gyro_noise_sigma=np.full(3, gyro_density / np.sqrt(dt)),
                )
                for i in range(seeds)
            ]
            inputs = np.tile(ins.sys.u_defaults[:, None], (1, seeds))
            for name, full in ins.preintegration_input_map.items():
                inputs[ins.sys._input_slices[full]] = np.column_stack(
                    [p[name] for p in framed]
                )
            accum_x = accum_zero.copy()
        else:
            inputs = np.tile(ins.sys.u_defaults[:, None], (1, seeds))
            inputs[ins.sys._input_slices[ins.sys.accel_input]] = a
            inputs[ins.sys._input_slices[ins.sys.gyro_input]] = g

        if (k + 1) % aiding_samples:
            x, covariance, cross = predict_only(x, covariance, cross, inputs)
            continue
        x, covariance, cross, nis, accepted = step(
            x, covariance, cross, inputs, d_rng.normal(size=(3, seeds)) * 0.001
        )
        rejected += int(np.count_nonzero(np.asarray(accepted) < 0.5))
        if (k + 1) % 1000 == 0:
            boundary_name = module.metadata.get("gyro_boundary_error_state")
            if boundary_name and gyro_density > 0:
                truth[ambient(boundary_name)] = -(next_g - gyro) / (
                    gyro_density / np.sqrt(dt)
                )
            e = np.asarray(error(x, truth))
            cov = np.asarray(covariance)
            nees = []
            marginal_nees = [[], [], []]
            for i in range(seeds):
                marginal = cov[:, i * n : (i + 1) * n][np.ix_(selected, selected)]
                nees.append(normalized_nees(e[selected, i], marginal))
                for j in range(3):
                    sl = slice(3 * j, 3 * j + 3)
                    marginal_nees[j].append(
                        normalized_nees(e[selected, i][sl], marginal[sl, sl])
                    )
            physical_heading, physical_heading_variance = [], []
            for i in range(seeds):
                value, gradient = heading_fn(x[:, i])
                difference = truth_heading[i] - float(value)
                physical_heading.append(
                    np.arctan2(np.sin(difference), np.cos(difference))
                )
                gradient = np.asarray(gradient).ravel()
                physical_heading_variance.append(
                    float(gradient @ cov[:, i * n : (i + 1) * n] @ gradient)
                )
            physical_heading = np.asarray(physical_heading)
            physical_heading_variance = np.asarray(physical_heading_variance)
            standardized_heading = physical_heading / np.sqrt(physical_heading_variance)
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
                    "physical_heading_rmse_deg": float(
                        np.degrees(np.sqrt(np.mean(physical_heading**2)))
                    ),
                    "physical_heading_sigma_deg": float(
                        np.degrees(np.sqrt(np.mean(physical_heading_variance)))
                    ),
                    "physical_heading_anees": float(np.mean(standardized_heading**2)),
                    "physical_heading_coverage": {
                        str(k): float(np.mean(np.abs(standardized_heading) <= k))
                        for k in (1, 2, 3)
                    },
                    "marginal_anees": {
                        name: float(np.mean(values))
                        for name, values in zip(
                            ("orientation", "gyro_bias", "accel_bias"), marginal_nees
                        )
                    },
                    "physical_accel_bias_error_mean": np.mean(
                        np.asarray(x)[accel_bias] - truth[accel_bias], axis=1
                    ).tolist(),
                    "physical_gyro_bias_error_mean": np.mean(
                        np.asarray(x)[gyro_bias] - truth[gyro_bias], axis=1
                    ).tolist(),
                    "anis": float(np.mean(nis)),
                }
            )
            if boundary_name:
                slot = spec.slot(boundary_name)
                joint_indices = selected + list(
                    range(slot.tangent_offset, slot.tangent_offset + 3)
                )
                joint_nees = [
                    normalized_nees(
                        e[joint_indices, i],
                        cov[:, i * n : (i + 1) * n][
                            np.ix_(joint_indices, joint_indices)
                        ],
                    )
                    for i in range(seeds)
                ]
                records[-1]["attitude_bias_boundary_anees"] = float(np.mean(joint_nees))
                records[-1]["boundary_error_mean"] = np.mean(
                    e[slot.tangent_offset : slot.tangent_offset + 3], axis=1
                ).tolist()
    bounds = [chi2_quantile(9 * seeds, p) / seeds for p in (0.025, 0.975)]
    return {
        "acceptance": "pass"
        if bounds[0] <= records[-1]["attitude_bias_anees"] <= bounds[1]
        else "fail",
        "acceptance_scope": "final attitude/bias ANEES; not release",
        "truth_motion": "stationary, known rigid IMU installation, constant biases",
        "acquisition_contract": "fresh right boundary reused as next left sample",
        "random_stream_contract": "SeedSequence(seed).spawn(3): accel,gyro,DVL; independent physical prior stream",
        "fixture_schema": 2,
        "mounted": mounted,
        "latitude": latitude,
        "spin": spin,
        "physical_prior": "product SO(3) attitude and independent additive bias samples",
        "error_coordinates": getattr(spec, "error_model", "product_manifold"),
        "propagation": propagation,
        "covariance_model": module.metadata["covariance"],
        "gyro_density": gyro_density,
        "gyro_bias_prior_sigma": bias_sigma,
        "rate": rate,
        "aiding_rate": rate / aiding_samples,
        "packet_samples": packet_samples,
        "duration": duration,
        "seed": seed,
        "seeds": seeds,
        "rejected_updates": rejected,
        "attitude_bias_anees_95_percent_bounds": bounds,
        "heading_anees_95_percent_bounds": [
            chi2_quantile(seeds, p) / seeds for p in (0.025, 0.975)
        ],
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
    parser.add_argument("--aiding-samples", type=int, default=10)
    parser.add_argument(
        "--covariance", choices=("linearized", "nonlinear"), default="linearized"
    )
    parser.add_argument("--mounted", action="store_true")
    parser.add_argument("--rate", type=int, default=100)
    parser.add_argument("--latitude", type=float, default=37.78)
    parser.add_argument("--spin", type=float, default=SPIN)
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

"""Reproducible synthetic INS gyrocompass qualification (no vehicle dynamics).

White-noise densities are converted to per-sample sigmas exactly once. The
ordinary ConstantBiasIMU contract gives bias uncertainty without invented RW.
A stationary, independently sampled DVL constrains velocity; gyro is only a
process input. This analytical fixture complements the rotating-Earth plant
oracle in tests/test_ins_navigation_frame.py.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import casadi as ca
import numpy as np

from manta import INS, Craft, NavigationFrame, TargetNumpy, World
from manta.estimation import chi2_quantile, observability
from manta.estimation import sigma_horizon as covariance_horizon
from manta.fields import GravityField
from manta.ir._rotation import quat_mul, quat_to_rotmat, so3_exp
from manta.parts import ConstantBiasIMU, Mass, VelocitySensor

SPIN = 7.2921159e-5


def final_consistency_checks(record, seeds):
    """Keep heading and joint consistency visible in the top-level verdict."""
    checks = {}
    for metric, dof in (
        ("attitude_bias_anees", 9),
        ("physical_heading_anees", 1),
        ("attitude_bias_boundary_anees", 12),
    ):
        if metric not in record:
            continue
        bounds = [chi2_quantile(dof * seeds, q) / seeds for q in (0.025, 0.975)]
        value = record[metric]
        checks[metric] = {
            "value": value,
            "bounds": bounds,
            "pass": bounds[0] <= value <= bounds[1],
        }
    return checks


def build(
    latitude=37.78,
    *,
    rate=100,
    spin=SPIN,
    gyro_density=1e-7,
    accel_density=1e-5,
    propagation="raw",
    frame_enabled=True,
    covariance="linearized",
    expand=False,
    mounted=False,
):
    # Cartesian Earth-axis projection, independent of INS implementation.
    phi = np.radians(latitude)
    east = np.array([0.0, 1.0, 0.0])
    up = np.array([np.cos(phi), 0, np.sin(phi)])
    axes = np.column_stack([east, np.cross(up, east), up])
    frame = NavigationFrame(
        frame_id=f"synthetic/{latitude}",
        epoch="1",
        angular_velocity=axes.T @ np.array([0.0, 0.0, spin]),
        origin_from_rotation_center=axes.T @ (6378137 * up),
        gravity_convention="effective",
    )
    c = Craft("craft")
    c.add(Mass("mass", mass=1, moi=(1, 1, 1)))
    c.add(
        ConstantBiasIMU(
            "imu",
            gyro_noise_sigma=gyro_density * np.sqrt(rate),
            accel_noise_sigma=accel_density * np.sqrt(rate),
            mount_offset=(0.3, -0.2, 0.1) if mounted else (0, 0, 0),
            mount_orientation=tuple(
                np.asarray(so3_exp(ca.DM([0.1, 0.2, -0.3]))).ravel()
            )
            if mounted
            else (1, 0, 0, 0),
        )
    )
    c.add(VelocitySensor("dvl", velocity_noise_sigma=0.001))
    w = World("earth_ins_qualification").add_field(GravityField(g=(0, 0, -9.81)))
    w.add_craft(
        c,
        orientation=tuple(np.asarray(so3_exp(ca.DM([0.2, -0.15, 0.7]))).ravel())
        if mounted
        else (1, 0, 0, 0),
    )
    return INS(
        w,
        imu="imu",
        sensors=["dvl.velocity"],
        navigation_frame=frame if frame_enabled else None,
        propagation=propagation,
        gates=None,
        covariance=covariance,
        expand=expand,
    )


def prior(ins, bias_sigma, *, yaw_sigma_deg=5.0):
    if not np.isfinite(yaw_sigma_deg) or yaw_sigma_deg <= 0:
        raise ValueError("yaw_sigma_deg must be finite and positive")
    spec = getattr(
        getattr(ins.spec, "product_spec", ins.spec), "physical_spec", ins.spec
    )
    p = np.zeros(spec.tangent_dim)
    for slot in spec.slots:
        value = {
            "position": 0.01,
            "velocity": 0.001,
            "orientation": np.radians(0.1),
            "imu.gyro_bias": bias_sigma,
            "imu.accel_bias": 1e-5,
        }[slot.name.split(".", 1)[1]]
        p[slot.tangent_offset : slot.tangent_offset + slot.tangent_dim] = value**2
    orientation = spec.slot("craft.orientation").tangent_offset
    p[orientation + 2] = np.radians(yaw_sigma_deg) ** 2
    return np.diag(p)


def normalized_nees(error, covariance):
    """Whiten after unit scaling; attitude and bias variances differ greatly.

    No covariance regularization: an indefinite matrix must still fail.
    This is algebraically e.T @ solve(P, e), with a better scaled solve.
    """
    scale = np.sqrt(np.diag(covariance))
    correlation = covariance / np.outer(scale, scale)
    whitened = np.linalg.solve(np.linalg.cholesky(correlation), error / scale)
    return float(whitened @ whitened)


def run(
    *,
    latitude=37.78,
    rate=100,
    duration=300,
    seeds=16,
    spin=SPIN,
    gyro_density=1e-7,
    bias_sigma=1e-8,
    yaw_sigma_deg=5.0,
    seed=82719,
    sigma_horizon=False,
    motion=False,
    initialize_prior=None,
    covariance="linearized",
    expand=False,
    assumed_gyro_density=None,
    mounted=False,
):
    if not isinstance(seeds, int) or seeds < 2:
        raise ValueError("seeds must be an integer >= 2")
    if not isinstance(rate, int) or rate <= 0:
        raise ValueError("rate must be a positive integer")
    if not np.isfinite(duration) or duration < 10:
        raise ValueError("duration must be finite and at least 10 seconds")
    started = time.perf_counter()
    ins = build(
        latitude,
        rate=rate,
        spin=spin,
        gyro_density=gyro_density
        if assumed_gyro_density is None
        else assumed_gyro_density,
        covariance=covariance,
        expand=expand,
        mounted=mounted,
    )
    if mounted and motion:
        raise ValueError("this fixture has no mounted angular-acceleration truth model")
    module = ins.module()
    runtime = TargetNumpy(ins)
    P0 = prior(ins, bias_sigma, yaw_sigma_deg=yaw_sigma_deg)
    x0 = runtime.x.copy()
    if "initialize_prior" in module.functions:
        x0 = np.asarray(module.port("prior_x").init).copy()

        def initialize_prior(ir, mean, covariance):
            mapped = ir.module().functions["initialize_prior"](mean, covariance)
            return np.asarray(mapped[0]).ravel(), np.asarray(mapped[1])

    n = ins.spec.tangent_dim
    na = len(x0)
    dt = 1 / rate
    # Invoke the emitted kernels in a seed batch; no alternate covariance math.
    xp = ca.MX.sym("x", na)
    pp = ca.MX.sym("P", n, n)
    u = ca.MX.sym("u", len(ins.sys.u_defaults))
    z = ca.MX.sym("z", 3)
    predict = module.functions["predict"]
    update = module.functions["update_diagnostic_craft_dvl_velocity"]
    predicted = predict(xp, pp, u, dt, 0.0)
    updated = update(*predicted, z, u, 0.0)
    from manta.codegen.numpy._compile import compile_functions

    step = ca.Function(
        "qualify", [xp, pp, u, z], [updated[0], updated[1], updated[4], updated[5]]
    )
    step = compile_functions(
        {"qualify": step}, optimization="O1", max_instructions=30000
    )["qualify"]
    fn = step.map(seeds)
    rng = np.random.default_rng(seed)
    samples = rng.normal(size=(n, seeds)) * np.sqrt(np.diag(P0))[:, None]
    physical_spec = getattr(ins.spec, "product_spec", ins.spec)
    true_x = np.asarray(physical_spec.boxplus_sym(ca.DM(x0), ca.DM.zeros(n)))[:, 0]
    truth = np.column_stack(
        [
            np.asarray(physical_spec.boxplus_sym(ca.DM(true_x), ca.DM(samples[:, i])))[
                :, 0
            ]
            for i in range(seeds)
        ]
    )
    # Position and velocity are physically zero in this stationary fixture.
    # Their small priors are conservative; this ANEES scores attitude/bias only.
    for name in ("craft.position", "craft.velocity"):
        slot = ins.spec.slot(name)
        truth[slot.ambient_offset : slot.ambient_offset + slot.ambient_dim] = 0
    selected = []
    marginal_indices = {}
    for name in ["craft.orientation", "craft.imu.gyro_bias", "craft.imu.accel_bias"]:
        sl = ins.spec.slot(name)
        indices = list(range(sl.tangent_offset, sl.tangent_offset + sl.tangent_dim))
        marginal_indices[name] = indices
        selected.extend(indices)
    qi = ins.spec.slot("craft.orientation").ambient_offset
    gi = ins.spec.slot("craft.imu.gyro_bias").ambient_offset
    ai = ins.spec.slot("craft.imu.accel_bias").ambient_offset
    gyro = np.column_stack(
        [
            (
                np.asarray(quat_to_rotmat(ca.DM(truth[qi : qi + 4, i])))
                @ ins.sys.R_craft_from_sensor
            ).T
            @ np.array(ins.navigation_frame.angular_velocity)
            + truth[gi : gi + 3, i]
            for i in range(seeds)
        ]
    )
    accel = np.column_stack(
        [
            (
                np.asarray(quat_to_rotmat(ca.DM(truth[qi : qi + 4, i])))
                @ ins.sys.R_craft_from_sensor
            ).T
            @ np.array([0, 0, 9.81])
            + truth[ai : ai + 3, i]
            for i in range(seeds)
        ]
    )
    if initialize_prior is not None:
        x0, P0 = initialize_prior(ins, x0, P0)
    x = np.tile(x0[:, None], (1, seeds))
    P = np.tile(P0, (1, seeds))
    records = []
    rejected = 0
    ui_a = ins.sys._input_slices[ins.sys.accel_input]
    ui_g = ins.sys._input_slices[ins.sys.gyro_input]
    xt = ca.MX.sym("truth", na)
    error = ca.Function("error", [xp, xt], [ins.spec.boxminus_sym(xt, xp)]).map(seeds)
    # Physical heading and its differential covariance are independent of the
    # finite error chart used for the multivariate consistency score.
    rotation = quat_to_rotmat(xp[qi : qi + 4])
    heading = ca.atan2(rotation[1, 0], rotation[0, 0])
    delta = ca.MX.sym("heading_delta", n)
    shifted = ins.spec.boxplus_sym(xp, delta)
    heading_delta = ca.substitute(heading, xp, shifted)
    heading_jacobian = ca.substitute(
        ca.jacobian(heading_delta, delta), delta, ca.MX.zeros(n)
    )
    heading_fn = ca.Function("physical_heading", [xp], [heading, heading_jacobian]).map(
        seeds
    )
    true_heading = np.asarray(heading_fn(truth)[0]).ravel()
    if motion:
        body_rate = ca.MX.sym("relative_body_rate", 3)
        q_truth = xt[qi : qi + 4]
        rotation_truth = quat_to_rotmat(q_truth)
        earth = ca.DM(ins.navigation_frame.angular_velocity)
        inertial_body_rate = body_rate + rotation_truth.T @ earth
        next_truth = ca.MX(xt)
        next_truth[qi : qi + 4] = quat_mul(
            so3_exp(-earth * dt), quat_mul(q_truth, so3_exp(inertial_body_rate * dt))
        )
        # Analytic left-held acquisition contract, independent of INS functions.
        motion_fn = ca.Function(
            "rotation_truth",
            [xt, body_rate],
            [
                next_truth,
                inertial_body_rate + xt[gi : gi + 3],
                rotation_truth.T @ ca.DM([0, 0, 9.81]) + xt[ai : ai + 3],
            ],
        ).map(seeds)
    physical_moments_fn = None
    if hasattr(ins.spec, "product_spec"):
        from manta.estimation._kalman import sigma_deltas, unscented_weights, ut_predict

        wm, wc, spread = unscented_weights(n, 1.0, 2.0, 0.0)
        points = [ins.spec.boxplus_sym(xp, d) for d in sigma_deltas(pp, spread, n)]
        physical_mean, physical_cov = ut_predict(
            points, ca.MX.zeros(n, n), wm, wc, physical_spec, 3
        )
        physical_error = physical_spec.boxminus_sym(xt, physical_mean)
        physical_moments_fn = ca.Function(
            "physical_moments",
            [xp, pp, xt],
            [physical_mean, physical_cov, physical_error],
        ).map(seeds)
    every = max(1, int(rate * 10))
    for k in range(round(duration * rate)):
        if motion:
            time_s = k * dt
            body_rate = [
                0.03 * np.sin(0.13 * time_s),
                0.025 * np.cos(0.17 * time_s),
                0.02 + 0.01 * np.sin(0.07 * time_s),
            ]
            next_truth, gyro, accel = motion_fn(truth, body_rate)
            gyro, accel = np.asarray(gyro), np.asarray(accel)
        inputs = np.tile(ins.sys.u_defaults[:, None], (1, seeds))
        inputs[ui_a] = accel + rng.normal(size=(3, seeds)) * 1e-5 * np.sqrt(rate)
        inputs[ui_g] = gyro + rng.normal(size=(3, seeds)) * gyro_density * np.sqrt(rate)
        x, P, nis, accepted = fn(x, P, inputs, rng.normal(size=(3, seeds)) * 0.001)
        if motion:
            truth = np.asarray(next_truth)
            true_heading = np.asarray(heading_fn(truth)[0]).ravel()
        rejected += int(np.count_nonzero(np.asarray(accepted) < 0.5))
        if (k + 1) % every == 0:
            errors = np.asarray(error(x, truth))
            cov = np.asarray(P)
            estimated_heading, heading_jac = heading_fn(x)
            angle_difference = true_heading - np.asarray(estimated_heading).ravel()
            yaw = np.arctan2(np.sin(angle_difference), np.cos(angle_difference))
            heading_jac = np.asarray(heading_jac)
            sig = np.array(
                [
                    np.sqrt(
                        (
                            heading_jac[:, i * n : (i + 1) * n]
                            @ cov[:, i * n : (i + 1) * n]
                            @ heading_jac[:, i * n : (i + 1) * n].T
                        ).item()
                    )
                    for i in range(seeds)
                ]
            )
            nees = []
            marginals = {name: [] for name in marginal_indices}
            for i in range(seeds):
                pi = cov[:, i * n : (i + 1) * n]
                ps = pi[np.ix_(selected, selected)]
                e = errors[selected, i]
                nees.append(normalized_nees(e, ps))
                for name, indices in marginal_indices.items():
                    marginals[name].append(
                        normalized_nees(
                            errors[indices, i], pi[np.ix_(indices, indices)]
                        )
                    )
            records.append(
                {
                    "t": (k + 1) / rate,
                    "yaw_rmse_deg": float(np.degrees(np.sqrt(np.mean(yaw * yaw)))),
                    "physical_heading_anees": float(np.mean((yaw / sig) ** 2)),
                    "yaw_sigma_deg": float(np.degrees(np.sqrt(np.mean(sig * sig)))),
                    "heading_coverage": {
                        str(width): float(np.mean(np.abs(yaw) <= width * sig))
                        for width in (1, 2, 3)
                    },
                    "attitude_bias_anees": float(np.mean(nees)),
                    "dof": len(selected),
                    "marginal_anees": {
                        name: {"value": float(np.mean(values)), "dof": 3}
                        for name, values in marginals.items()
                    },
                    "anis": float(np.mean(np.asarray(nis))),
                    "gyro_bias_rmse_rad_s": float(
                        np.sqrt(
                            np.mean(
                                (np.asarray(x)[gi : gi + 3] - truth[gi : gi + 3]) ** 2
                            )
                        )
                    ),
                    "accel_bias_mean_error_m_s2": np.mean(
                        np.asarray(x)[ai : ai + 3] - truth[ai : ai + 3], axis=1
                    ).tolist(),
                    "accel_bias_rms_error_m_s2": np.sqrt(
                        np.mean(
                            (np.asarray(x)[ai : ai + 3] - truth[ai : ai + 3]) ** 2,
                            axis=1,
                        )
                    ).tolist(),
                    "accel_bias_rms_sigma_m_s2": np.sqrt(
                        np.mean(
                            [
                                np.diag(cov[:, i * n : (i + 1) * n])[
                                    marginal_indices["craft.imu.accel_bias"]
                                ]
                                for i in range(seeds)
                            ],
                            axis=0,
                        )
                    ).tolist(),
                    "covariance_symmetry_error": float(
                        max(
                            np.max(
                                np.abs(
                                    cov[:, i * n : (i + 1) * n]
                                    - cov[:, i * n : (i + 1) * n].T
                                )
                            )
                            for i in range(seeds)
                        )
                    ),
                    "covariance_min_eigenvalue": float(
                        min(
                            np.linalg.eigvalsh(cov[:, i * n : (i + 1) * n]).min()
                            for i in range(seeds)
                        )
                    ),
                }
            )
            if physical_moments_fn is not None:
                pm, pc, pe = (
                    np.asarray(value) for value in physical_moments_fn(x, P, truth)
                )
                pnees = [
                    normalized_nees(
                        pe[selected, i],
                        pc[:, i * n : (i + 1) * n][np.ix_(selected, selected)],
                    )
                    for i in range(seeds)
                ]
                records[-1]["physical_moment_anees"] = float(np.mean(pnees))
                records[-1]["physical_accel_bias_mean_error_m_s2"] = np.mean(
                    pm[ai : ai + 3] - truth[ai : ai + 3], axis=1
                ).tolist()
                records[-1]["physical_accel_bias_rms_error_m_s2"] = np.sqrt(
                    np.mean((pm[ai : ai + 3] - truth[ai : ai + 3]) ** 2, axis=1)
                ).tolist()
                records[-1]["physical_accel_bias_rms_sigma_m_s2"] = np.sqrt(
                    np.mean(
                        [
                            np.diag(pc[:, i * n : (i + 1) * n])[
                                marginal_indices["craft.imu.accel_bias"]
                            ]
                            for i in range(seeds)
                        ],
                        axis=0,
                    )
                ).tolist()
    report = observability(
        ins,
        inputs={
            "imu.accel": (0, 0, 9.81),
            "imu.gyro": ins.navigation_frame.angular_velocity,
        },
        dt=dt,
    )
    bounds = [chi2_quantile(len(selected) * seeds, p) / seeds for p in (0.025, 0.975)]
    horizon_report = None
    if sigma_horizon:
        report_horizon = covariance_horizon(
            ins,
            horizon=duration,
            dt=dt,
            P0=P0,
            control={
                "imu.accel": (0, 0, 9.81),
                "imu.gyro": ins.navigation_frame.angular_velocity,
            },
        )
        horizon_report = {
            "times": report_horizon.times.tolist(),
            "sigmas": {
                key: value.tolist() for key, value in report_horizon.sigmas.items()
            },
            "summary": report_horizon.summary(),
        }
    thresholds = {}
    for degrees in (10, 5, 1):
        thresholds[str(degrees)] = next(
            (row["t"] for row in records if row["yaw_sigma_deg"] <= degrees), None
        )
    return {
        "sigma_horizon": horizon_report,
        "module_artifact_id": module.artifact_id,
        "heading_sigma_threshold_times_s": thresholds,
        "acceptance_scope": "final heading and joint ANEES; not release",
        "consistency_checks": final_consistency_checks(records[-1], seeds),
        "acceptance": (
            "pass"
            if all(
                c["pass"] for c in final_consistency_checks(records[-1], seeds).values()
            )
            else "fail"
        ),
        "attitude_bias_anees_95_percent_bounds": bounds,
        "anis_95_percent_bounds": [
            chi2_quantile(3 * seeds, p) / seeds for p in (0.025, 0.975)
        ],
        "rejected_updates": rejected,
        "latitude": latitude,
        "rate": rate,
        "duration": duration,
        "seeds": seeds,
        "seed": seed,
        "spin": spin,
        "motion": motion,
        "mounted": mounted,
        "gyro_density": gyro_density,
        "assumed_gyro_density": gyro_density
        if assumed_gyro_density is None
        else assumed_gyro_density,
        "gyro_bias_prior_sigma": bias_sigma,
        "initial_yaw_sigma_deg": yaw_sigma_deg,
        "error_coordinates": getattr(ins.spec, "error_model", "product_manifold"),
        "physical_prior": "Independent SO(3) attitude and additive physical bias samples",
        "covariance_model": covariance,
        "rank": report.rank,
        "tangent_dim": report.tangent_dim,
        "records": records,
        "elapsed_s": time.perf_counter() - started,
    }


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--duration", type=float, default=300)
    p.add_argument("--rate", type=int, default=100)
    p.add_argument("--seeds", type=int, default=16)
    p.add_argument("--seed", type=int, default=82719)
    p.add_argument("--latitude", type=float, default=37.78)
    p.add_argument("--spin", type=float, default=SPIN)
    p.add_argument("--gyro-density", type=float, default=1e-7)
    p.add_argument("--assumed-gyro-density", type=float)
    p.add_argument("--bias-sigma", type=float, default=1e-8)
    p.add_argument("--yaw-sigma-deg", type=float, default=5.0)
    p.add_argument("--sigma-horizon", action="store_true")
    p.add_argument("--motion", action="store_true")
    p.add_argument("--expand", action="store_true")
    p.add_argument("--mounted", action="store_true")
    p.add_argument(
        "--covariance",
        choices=("linearized", "geometric", "nonlinear"),
        default="linearized",
    )
    args = p.parse_args()
    values = vars(args).copy()
    output = values.pop("output")
    report = run(**values)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps({"acceptance": report["acceptance"], **report["records"][-1]}),
        flush=True,
    )
    if report["acceptance"] != "pass":
        raise SystemExit(2)


if __name__ == "__main__":
    main()

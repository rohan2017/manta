"""Deterministic audits for weak-bias Earth-relative INS inconsistency.

The linearization-switch experiment deliberately controls the reference state;
its covariance trace is a diagnostic, not a realizable estimator or truth run.
No measurement, covariance floor, or alternative estimator is added to INS.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import casadi as ca
import numpy as np

from manta.codegen.numpy._compile import compile_functions
from manta.estimation._kalman import _reset_jacobian_np
from manta.ir._rotation import quat_to_rotmat, so3_exp_np

from .earth_ins import build, prior


def state_slice(ins, name, *, tangent=False):
    slot = ins.spec.slot(f"craft.{name}")
    start = slot.tangent_offset if tangent else slot.ambient_offset
    dim = slot.tangent_dim if tangent else slot.ambient_dim
    return slice(start, start + dim)


def stationary_state(ins, heading_deg):
    """An exact stationary solution with identical readings for every heading.

    Only this synthetic fixture has uniform effective gravity, a root-mounted
    IMU, and constant biases. This is not a global symmetry of moving INS.
    """
    x = np.array(ins.module().state.field("x").init, dtype=float)
    q = so3_exp_np(np.array([0.0, 0.0, np.radians(heading_deg)]))
    r = np.asarray(quat_to_rotmat(ca.DM(q)))
    w = np.array(ins.navigation_frame.angular_velocity)
    x[state_slice(ins, "orientation")] = q
    x[state_slice(ins, "imu.gyro_bias")] = w - r.T @ w
    return x


def stationary_input(ins):
    return ins.sys.resolve_u(
        {
            "imu.gyro": ins.navigation_frame.angular_velocity,
            "imu.accel": (0, 0, 9.81),
        }
    )


def heading_tangent(ins, x):
    """Derivative of the exact stationary family, per radian of heading."""
    r = np.asarray(quat_to_rotmat(ca.DM(x[state_slice(ins, "orientation")])))
    w = np.array(ins.navigation_frame.angular_velocity)
    vertical = np.array([0.0, 0.0, 1.0])
    direction = np.zeros(ins.spec.tangent_dim)
    direction[state_slice(ins, "orientation", tangent=True)] = vertical
    direction[state_slice(ins, "imu.gyro_bias", tangent=True)] = r.T @ np.cross(
        vertical, w
    )
    return direction


def stationary_audit(ins):
    u = stationary_input(ins)
    rows = []
    for heading in (-10, 0, 5, 10):
        x = stationary_state(ins, heading)
        direction = heading_tangent(ins, x)
        f = np.asarray(ins.sys.F_fn(x, u, 0.01, 0))
        h = np.asarray(ins.sys.sensors["craft.dvl.velocity"].H_fn(x, u, 0, 0))
        predicted = np.asarray(ins.sys.predict_fn(x, u, 0.01, 0)).ravel()
        rows.append(
            {
                "heading_deg": heading,
                "stationary_state_max_error": float(np.max(np.abs(predicted - x))),
                "F_direction_max_error": float(
                    np.max(np.abs(f @ direction - direction))
                ),
                "H_direction_max_error": float(np.max(np.abs(h @ direction))),
            }
        )
    x0, x1 = (stationary_state(ins, angle) for angle in (0, 5))
    correction = np.asarray(ins.spec.boxminus_sym(ca.DM(x1), ca.DM(x0))).ravel()
    reset = _reset_jacobian_np(ins.spec, correction)
    mismatch = reset @ heading_tangent(ins, x0) - heading_tangent(ins, x1)
    return {
        "family": rows,
        "product_reset_heading_direction_mismatch": mismatch.tolist(),
        "finite_heading_bias_curvature_rad_s": (
            x1[state_slice(ins, "imu.gyro_bias")]
            - np.radians(5)
            * heading_tangent(ins, x0)[state_slice(ins, "imu.gyro_bias", tangent=True)]
        ).tolist(),
    }


def jacobian_audit(ins):
    """Central differences at a moving, tilted state with nonzero biases."""
    spec, sys = ins.spec, ins.sys
    n = spec.tangent_dim
    x = np.array(ins.module().state.field("x").init, dtype=float)
    delta = np.zeros(n)
    delta[state_slice(ins, "orientation", tangent=True)] = (0.2, -0.1, 0.7)
    delta[state_slice(ins, "velocity", tangent=True)] = (0.3, -0.2, 0.1)
    delta[state_slice(ins, "imu.gyro_bias", tangent=True)] = (1e-5, -2e-5, 3e-5)
    delta[state_slice(ins, "imu.accel_bias", tangent=True)] = (1e-3, -2e-3, 3e-3)
    x = spec.boxplus_num(x, delta)
    u = sys.resolve_u({"imu.gyro": (0.1, -0.05, 0.2), "imu.accel": (0.4, -0.2, 9.9)})
    dt = 0.02
    sm = sys.sensors["craft.dvl.velocity"]
    h_fn = ca.Function(
        "audit_h", [sys.x_sym, sys.u_sym, sys.dt_sym, sys.t_sym], [sm.h_sym]
    )
    noisy_fn = ca.Function(
        "audit_noisy",
        [sys.x_sym, sys.u_sym, sys.dt_sym, sys.t_sym, sys.n_sym],
        [sys.x_new_noisy],
    )
    a, b = (ca.MX.sym(name, spec.ambient_dim) for name in ("a", "b"))
    minus = ca.Function("audit_minus", [a, b], [spec.boxminus_sym(a, b)])
    nominal = sys.predict_fn(x, u, dt, 0)
    eps = 1e-6
    numeric_f, numeric_h = np.zeros((n, n)), np.zeros((3, n))
    for column in range(n):
        step = np.zeros(n)
        step[column] = eps
        xp, xm = (spec.boxplus_num(x, d) for d in (step, -step))
        numeric_f[:, column] = np.asarray(
            minus(sys.predict_fn(xp, u, dt, 0), nominal)
            - minus(sys.predict_fn(xm, u, dt, 0), nominal)
        ).ravel() / (2 * eps)
        numeric_h[:, column] = np.asarray(
            h_fn(xp, u, 0, 0) - h_fn(xm, u, 0, 0)
        ).ravel() / (2 * eps)
    nn = sys.n_sym.numel()
    numeric_l = np.zeros((n, nn))
    for column in range(nn):
        step = np.zeros(nn)
        step[column] = eps
        numeric_l[:, column] = np.asarray(
            minus(noisy_fn(x, u, dt, 0, step), nominal)
            - minus(noisy_fn(x, u, dt, 0, -step), nominal)
        ).ravel() / (2 * eps)
    shifted = spec.boxplus_num(x, delta)
    numeric_reset = np.zeros((n, n))
    for column in range(n):
        step = np.zeros(n)
        step[column] = eps
        numeric_reset[:, column] = np.asarray(
            minus(spec.boxplus_num(x, delta + step), shifted)
            - minus(spec.boxplus_num(x, delta - step), shifted)
        ).ravel() / (2 * eps)
    return {
        "F_central_difference_max_error": float(
            np.max(np.abs(numeric_f - np.asarray(sys.F_fn(x, u, dt, 0))))
        ),
        "H_central_difference_max_error": float(
            np.max(np.abs(numeric_h - np.asarray(sm.H_fn(x, u, 0, 0))))
        ),
        "L_central_difference_max_error": float(
            np.max(np.abs(numeric_l - np.asarray(sys.L_fn(x, u, dt, 0))))
        ),
        "reset_central_difference_max_error": float(
            np.max(np.abs(numeric_reset - _reset_jacobian_np(spec, delta)))
        ),
    }


def linearization_probe(ins, *, seconds=60):
    """Same zero-innovation data, references fixed or alternated every 10 s.

    Holding the state explicitly isolates covariance bookkeeping. The reset
    is the existing product-manifold injection reset; it is not claimed to
    transport a nonlinear posterior exactly between the two hypotheses.
    """
    spec = ins.spec
    x0, x1 = (stationary_state(ins, angle) for angle in (0, 5))
    u = stationary_input(ins)
    module = ins.module()
    x, p = (
        ca.MX.sym("x", spec.ambient_dim),
        ca.MX.sym("p", spec.tangent_dim, spec.tangent_dim),
    )
    predicted = module.functions["predict"](x, p, u, 0.01, 0)
    updated = module.functions["update_diagnostic_craft_dvl_velocity"](
        *predicted, [0, 0, 0], u, 0
    )
    step = ca.Function("linearization_probe", [x, p], [updated[1], updated[2]])
    step = compile_functions(
        {"linearization_probe": step}, optimization="O1", max_instructions=30000
    )["linearization_probe"]
    yaw = state_slice(ins, "orientation", tangent=True).stop - 1
    results = {}
    for switch in (False, True):
        covariance = prior(ins, 1e-5)
        at_second = False
        state = x0
        max_innovation = 0.0
        for k in range(round(seconds * 100)):
            if switch and k and k % 1000 == 0:
                at_second = not at_second
                next_state = x1 if at_second else x0
                correction = np.asarray(
                    spec.boxminus_sym(ca.DM(next_state), ca.DM(state))
                ).ravel()
                reset = _reset_jacobian_np(spec, correction)
                covariance = reset @ covariance @ reset.T
                state = next_state
            covariance, innovation = step(state, covariance)
            max_innovation = max(max_innovation, float(np.max(np.abs(innovation))))
        results["alternating" if switch else "fixed"] = {
            "yaw_sigma_deg": float(np.degrees(np.sqrt(float(covariance[yaw, yaw])))),
            "max_innovation": max_innovation,
        }
    return results


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    ins = build()
    report = {
        "scope": "debugging evidence, not release qualification",
        "jacobians": jacobian_audit(ins),
        "stationary": stationary_audit(ins),
        "linearization_probe_60s": linearization_probe(ins),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

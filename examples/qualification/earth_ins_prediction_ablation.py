"""Isolate prediction cost from the accepted finite INS error coordinates.

Research runner only: retain the v2 prior, measurement update, reset and
active packet boundary, but replace sigma-point prediction with its first
derivative. No public estimator mode or default changes. A passing individual
trial batch is not qualification of this alternative.
"""

from __future__ import annotations

import argparse
import json
from contextlib import contextmanager
from pathlib import Path

import casadi as ca

from manta.estimation import _ins_moments, chi2_quantile
from manta.estimation._assembly import _q_auto
from manta.estimation._kalman import symmetrize


def analytic_prediction(sys, spec, P, C, *, process_noise=True, extra_Q=None):
    """First-order propagation in the v2 chart, with exact joint noise law."""
    # The finite chart deliberately has the physical chart's zero-error
    # differential, so the existing F and L are also its local derivatives.
    F = sys.F_sym
    covariance = F @ P @ F.T
    if process_noise:
        covariance += _q_auto(sys)
    if sys.propagation == "preintegrated":
        if not hasattr(sys, "boundary_state_name"):
            raise ValueError("this ablation requires the active packet boundary")
        residual = ca.MX.sym("residual", 12)
        output = sys.packet_residual_fn(
            sys.x_sym,
            sys.u_sym,
            sys.dt_sym,
            sys.t_sym,
            ca.MX.zeros(sys.n_sym.numel()),
            residual,
        )
        # At the nominal point both charts have the same differential. Use
        # the physical chart here so code generation does not repeatedly
        # evaluate the finite swing/twist inverse merely to cancel it at zero.
        error = spec.product_spec.boxminus_sym(output, sys.x_new)
        G = ca.substitute(ca.jacobian(error, residual), residual, ca.MX.zeros(12))
        root = _ins_moments.psd_root(sys.boundary_joint_covariance_sym)[3:, 3:]
        noise_root = G @ root
        covariance += noise_root @ noise_root.T
    if extra_Q is not None:
        covariance += extra_Q
    return sys.x_new, symmetrize(covariance), None if C is None else F @ C


@contextmanager
def predictor_override(*, expand=False):
    original = _ins_moments.predict_moments

    def prediction(sys, spec, P, C, **kwargs):
        outputs = analytic_prediction(sys, spec, P, C, **kwargs)
        if not expand:
            return outputs
        inputs = [sys.x_sym, sys.u_sym, sys.dt_sym, sys.t_sym, P]
        if C is not None:
            inputs.append(C)
        if kwargs.get("extra_Q") is not None:
            inputs.append(kwargs["extra_Q"])
        function = ca.Function(
            "analytic_expanded", inputs, [v for v in outputs if v is not None]
        ).expand()
        result = list(function(*inputs))
        return tuple(result) if C is not None else (*result, None)

    _ins_moments.predict_moments = prediction
    try:
        yield
    finally:
        _ins_moments.predict_moments = original


def main():
    from .earth_ins_split import run

    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bias-sigma", type=float, default=1e-5)
    parser.add_argument("--gyro-density", type=float, default=1e-7)
    parser.add_argument("--seeds", type=int, default=16)
    parser.add_argument("--seed", type=int, default=82719)
    parser.add_argument("--duration", type=int, default=300)
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument(
        "--expand",
        action="store_true",
        help="inline scalar expressions before native code generation",
    )
    args = parser.parse_args()
    with predictor_override(expand=args.expand):
        if args.benchmark:
            from .earth_ins_covariance_benchmark import run as benchmark

            report = benchmark(expand=args.expand)
        else:
            report = run(
                covariance="nonlinear",
                mounted=True,
                bias_sigma=args.bias_sigma,
                gyro_density=args.gyro_density,
                seeds=args.seeds,
                seed=args.seed,
                duration=args.duration,
            )
    report["prediction_ablation"] = "first_order_prediction_with_v2_prior_and_reset"
    report["release_qualified"] = False
    report["expanded_prediction"] = args.expand
    if args.benchmark:
        for row in report["rows"]:
            if row["covariance"] == "nonlinear":
                row["covariance"] = "finite_chart_analytic_prediction"
    else:
        final = report["records"][-1]
        checks = {}
        for metric, dof in (
            ("attitude_bias_anees", 9),
            ("physical_heading_anees", 1),
            ("attitude_bias_boundary_anees", 12),
        ):
            bounds = [
                chi2_quantile(dof * args.seeds, q) / args.seeds for q in (0.025, 0.975)
            ]
            checks[metric] = {
                "value": final[metric],
                "bounds": bounds,
                "pass": bounds[0] <= final[metric] <= bounds[1],
            }
        report["ablation_checks"] = checks
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report.get("records", [report])[-1]), flush=True)
    if not args.benchmark and not all(c["pass"] for c in checks.values()):
        raise SystemExit(2)


if __name__ == "__main__":
    main()

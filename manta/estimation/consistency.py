"""Filter consistency — is the EKF's reported covariance honest?

Observability tells you whether a state *can* be estimated; consistency
tells you whether the filter's *uncertainty* is truthful. A filter can be
fully observable and still lie: if it's overconfident (covariance too
small for its actual error) it will diverge and ignore measurements; if
underconfident it throws away information. Neither shows up as an error —
the estimate just quietly drifts while `P` looks fine. (This is the class
of bug that the measurement-timing error was: the estimate was wrong while
the covariance stayed tight.)

The standard test is **NEES** — Normalized Estimation Error Squared. With
the manifold error `e = x_true ⊟ x_est` (tangent) and covariance `P`,

    NEES = eᵀ P⁻¹ e .

For a consistent filter `E[NEES] = n` (the tangent dimension) and `NEES`
is χ²ₙ-distributed. Running a Monte-Carlo ensemble (truth jittered by the
model's process noise, measurements by their R, the initial estimate drawn
from `P₀`) and averaging gives **ANEES**; if it lands outside the χ² 95%
band the filter is inconsistent — too high ⇒ overconfident, too low ⇒
conservative.

This complements `observability`: that finds states you *can't* see; this
finds covariances you *shouldn't trust*. Use both.

Usage::

    from manta.estimation import nees
    rep = nees(world, dt=0.01, steps=400, control={"t.throttle": 14.7})
    print(rep.summary())

Consistency is only meaningful on observable states (an unobservable
direction has unbounded variance and dominates `P⁻¹` ambiguously) — pair
this with an `observability` check, and excite the trajectory via
`control`.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Any, Callable

import numpy as np

# manta imports are deferred into `nees()` to avoid an import cycle
# (estimation → consistency → codegen.numpy → estimation).


# --------------------------------------------------------------------------
# χ² quantiles without scipy (Acklam inverse-normal + Wilson–Hilferty)
# --------------------------------------------------------------------------

def _norm_ppf(p: float) -> float:
    """Inverse standard-normal CDF (Acklam's rational approximation)."""
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = sqrt(-2 * np.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > phigh:
        q = sqrt(-2 * np.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5]) * q / \
           (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


def _chi2_quantile(k: float, p: float) -> float:
    """χ²ₖ quantile via the Wilson–Hilferty cube approximation (accurate for
    the moderate-to-large k this tool produces)."""
    z = _norm_ppf(p)
    return k * (1 - 2 / (9 * k) + z * sqrt(2 / (9 * k))) ** 3


@dataclass
class NEESReport:
    """Result of :func:`nees`.

    Attributes:
        dof        — expected NEES per sample (= tangent dimension).
        anees      — average NEES over the Monte-Carlo ensemble.
        lower/upper — χ² 95%-band on ANEES (sized by the `runs` independent
                      trajectories, which bounds within-run time correlation).
        consistent — True iff `lower ≤ anees ≤ upper`.
        runs, samples — ensemble size and total NEES samples averaged.
    """

    dof: int
    anees: float
    lower: float
    upper: float
    runs: int
    samples: int

    @property
    def consistent(self) -> bool:
        return self.lower <= self.anees <= self.upper

    @property
    def verdict(self) -> str:
        if self.consistent:
            return "consistent"
        return "overconfident" if self.anees > self.upper else "conservative"

    def summary(self) -> str:
        mark = "✓" if self.consistent else "⚠"
        return (f"{mark} filter is {self.verdict}: "
                f"ANEES = {self.anees:.2f} vs dof = {self.dof} "
                f"(95% band [{self.lower:.2f}, {self.upper:.2f}], "
                f"{self.runs} runs × {self.samples // self.runs} samples)")

    def __str__(self) -> str:  # pragma: no cover - thin
        return self.summary()


def _controls_at(control, t) -> dict[str, Any]:
    if control is None:
        return {}
    return control(t) if callable(control) else dict(control)


def nees(world, *, dt: float, steps: int,
         control: Callable[[float], dict] | dict | None = None,
         sensors: list[str] | None = None,
         P0: np.ndarray | None = None, Q: np.ndarray | None = None,
         observable_basis: np.ndarray | None = None,
         runs: int = 20, seed: int = 0, warmup: int | None = None,
         alpha: float = 0.05) -> NEESReport:
    """Monte-Carlo NEES consistency check for `EKF(world)`.

    Each run jitters the truth with the model's process noise (a
    `NoiseDriver`) and the measurements with their R, draws the initial
    estimate from `P0`, runs `sim`↔`ekf` over `steps`, and records NEES at
    each post-`warmup` step. Returns an :class:`NEESReport`.

    Args:
        world    — the model.
        dt, steps — filter step and run length.
        control  — per-tick inputs: `{name: value}` (constant) or a
                   `t -> {name: value}` callable. Excite the trajectory so
                   the observable states are actually exercised.
        sensors  — restrict to these sensor outputs (full names / suffixes);
                   `None` ⇒ all registered. They must declare noise (nonzero
                   R), else the update is singular.
        P0       — initial estimate covariance (tangent). Default `0.1·I`.
        Q        — process-noise override for the predict (else the model's
                   auto-assembled Q). Useful to probe consistency vs. an
                   assumed Q, or to validate the check responds to scaling.
        observable_basis — `(n, r)` orthonormal basis of the observable
                   subspace (from `observability(...).basis` or
                   `observability_trajectory`). When given, NEES is measured
                   only in that subspace (dof = r) — isolating "is my noise
                   modeling right where I *can* estimate?" from overconfidence
                   on unobservable directions.
        runs, seed — ensemble size and base RNG seed.
        warmup   — steps to skip before recording (default `steps // 5`).
        alpha    — significance for the band (default 0.05 ⇒ 95%).
    """
    import casadi as ca
    from ..codegen.numpy import NoiseDriver, TargetNumpy
    from ..ir._names import resolve_suffix
    from ..signal import wire
    from ..sim import Sim
    from .ekf import EKF

    sim_ir, ekf_ir = Sim(world), EKF(world)
    spec = ekf_ir.spec
    n = spec.tangent_dim
    if warmup is None:
        warmup = steps // 5
    if P0 is None:
        P0 = np.eye(n) * 0.1
    L0 = np.linalg.cholesky(P0)

    xa, xb = ca.MX.sym("xa", spec.ambient_dim), ca.MX.sym("xb", spec.ambient_dim)
    boxminus = ca.Function("bm", [xa, xb], [spec.boxminus_sym(xa, xb)])

    # Sensor selection resolves like everywhere else — unknown/ambiguous
    # names raise instead of silently dropping a typo.
    fulls = list(ekf_ir.sys.sensors)
    if sensors is None:
        names = fulls
    else:
        chosen = {resolve_suffix(sn, fulls, label="sensor", who="nees")
                  for sn in sensors}
        names = [f for f in fulls if f in chosen]

    def truth_vec(world_rt) -> np.ndarray:
        return spec.pack_any(world_rt.state)

    nees_samples: list[float] = []
    for r in range(runs):
        rng = np.random.default_rng(seed + r)
        sim = TargetNumpy(sim_ir)
        sim.attach_driver(NoiseDriver(seed=seed + 1000 + r))
        ekf = TargetNumpy(ekf_ir)
        # Truth starts at the world's initial state; the estimate is drawn
        # from N(truth, P0) so the initial NEES is itself χ²-distributed.
        x_truth0 = truth_vec(sim)
        x_est0 = spec.boxplus_num(x_truth0, L0 @ rng.standard_normal(n))
        ekf.reset(state=x_est0, P=P0)
        ekf.Q = Q
        for full in names:
            wire(sim.out(full), ekf.meas(full))

        for i in range(steps):
            t = i * dt
            for nm, v in _controls_at(control, t).items():
                sim.command(nm).set(v)
                ekf.command(nm).set(v)
            sim.step(dt)
            ekf.step(dt)
            if i >= warmup:
                e = np.asarray(boxminus(truth_vec(sim), ekf.x)).reshape(-1)
                P = ekf.P
                if observable_basis is not None:
                    e = observable_basis.T @ e
                    P = observable_basis.T @ P @ observable_basis
                try:
                    nees_samples.append(float(e @ np.linalg.solve(P, e)))
                except np.linalg.LinAlgError:
                    nees_samples.append(float(e @ np.linalg.pinv(P) @ e))

    dof = n if observable_basis is None else observable_basis.shape[1]
    anees = float(np.mean(nees_samples))
    # Band sized by the number of independent trajectories (runs), which
    # accounts for within-run time correlation: runs·ANEES ~ χ²_{runs·dof}.
    k = runs * dof
    lower = _chi2_quantile(k, alpha / 2) / runs
    upper = _chi2_quantile(k, 1 - alpha / 2) / runs
    return NEESReport(dof=dof, anees=anees, lower=lower, upper=upper,
                      runs=runs, samples=len(nees_samples))

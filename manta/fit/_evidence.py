"""Typed held-out fit evidence.

The doctrine's artifact channel: a fitted model's held-out residual bias and
its time-correlated process covariance are *evidence* that a model-aided
estimator consumes explicitly — never an implicit zero. This module owns
that channel end to end:

* `FitEvidence` — the frozen, validated, canonically hashable artifact:
  per-axis held-out residual bias (with the held-out window definition and
  sample count), the fitted per-axis process-noise model (white σ, Gauss–
  Markov τ/σ², or random-walk σ) with its fit statistics, and the
  acceptance decision together with the numeric criteria that produced it.
  `accepted` is *derived* from `FitAcceptanceCriteria` — `FitEvidence`
  refuses construction with any other value.
* `held_out_evidence()` — the pipeline that produces it from a fitted model
  and an untouched held-out window set: mean-prediction residuals, bias,
  an autocorrelation fit of `ρ(l) = a·exp(−l·dt/τ)` with the white-noise
  fallback recorded (never silent) when the fitted τ is below the sample
  interval, and the acceptance checks.
* `hold_out()` — the deterministic training / held-out split.

`ModelForce` builds its error model from a `FitEvidence` and the model-
aided `INS` refuses a `ModelForce` that carries none or whose acceptance
is false.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from ..ir._names import resolve_suffix
from ..ir.module import PortRef, Role
from ._common import Window, pack_u_trace, pack_x0, resolve_traces

# The process-noise vocabulary: the `Noise.kind` strings of `WhiteNoise`,
# `GaussMarkovNoise`, and `RandomWalkNoise` (manta.parts._declarations).
NOISE_KINDS = ("white", "gauss_markov", "random_walk")

_AXES3 = ("x", "y", "z")


def _finite(value, *, name: str, minimum: float | None = None,
            strict: bool = False) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise TypeError(f"{name} must be a finite number, got {value!r}") \
            from None
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite, got {value!r}")
    if minimum is not None:
        ok = number > minimum if strict else number >= minimum
        if not ok:
            rel = ">" if strict else ">="
            raise ValueError(f"{name} must be {rel} {minimum}, got {value!r}")
    return number


def _count(value, *, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer, got {value!r}")
    if int(value) < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value!r}")
    return int(value)


@dataclass(frozen=True)
class ProcessNoiseModel:
    """One axis' fitted process-noise model.

    Args:
        kind  — ``"white"`` (per-sample σ), ``"gauss_markov"`` (stationary σ
                with correlation time ``tau`` seconds), or
                ``"random_walk"`` (σ/√Hz drift density).
        sigma — 1-σ magnitude in the residual's units (≥ 0).
        tau   — correlation time in seconds; required (finite, > 0) for
                ``"gauss_markov"`` and forbidden otherwise.
    """

    kind: str
    sigma: float
    tau: float | None = None

    def __post_init__(self) -> None:
        if self.kind not in NOISE_KINDS:
            raise ValueError(
                f"ProcessNoiseModel.kind must be one of {NOISE_KINDS}, got "
                f"{self.kind!r}")
        object.__setattr__(self, "sigma", _finite(
            self.sigma, name="ProcessNoiseModel.sigma", minimum=0.0))
        if self.kind == "gauss_markov":
            if self.tau is None:
                raise ValueError(
                    "ProcessNoiseModel(kind='gauss_markov') requires tau")
            object.__setattr__(self, "tau", _finite(
                self.tau, name="ProcessNoiseModel.tau", minimum=0.0,
                strict=True))
        elif self.tau is not None:
            raise ValueError(
                f"ProcessNoiseModel(kind={self.kind!r}) takes no tau")


@dataclass(frozen=True)
class HeldOutWindow:
    """Definition of the held-out (acceptance) set the evidence was
    computed on: how many windows, how many samples, at which step, and
    the content digest of every window (the identity a consumer can check
    against its training set)."""

    window_count: int
    sample_count: int
    dt: float
    window_digests: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "window_count", _count(
            self.window_count, name="HeldOutWindow.window_count", minimum=1))
        object.__setattr__(self, "sample_count", _count(
            self.sample_count, name="HeldOutWindow.sample_count", minimum=1))
        object.__setattr__(self, "dt", _finite(
            self.dt, name="HeldOutWindow.dt", minimum=0.0, strict=True))
        digests = tuple(str(d) for d in self.window_digests)
        if len(digests) != self.window_count:
            raise ValueError(
                "HeldOutWindow.window_digests must name every held-out "
                f"window: {len(digests)} digest(s) for {self.window_count} "
                "window(s)")
        if len(set(digests)) != len(digests):
            raise ValueError("HeldOutWindow.window_digests contains a "
                             "duplicate window")
        object.__setattr__(self, "window_digests", digests)

    @property
    def duration_s(self) -> float:
        return self.sample_count * self.dt


@dataclass(frozen=True)
class AxisFitEvidence:
    """Held-out evidence for one residual axis.

    ``residual_bias`` is the held-out mean residual ``mean(z − h)`` with
    its standard error (effective sample count corrected for lag-one
    autocorrelation); ``residual_rms`` the raw root-mean-square residual.
    The autocorrelation fit ``ρ(l) ≈ a·exp(−l·dt/τ)`` over ``lag_count``
    lags yields ``fitted_tau`` and ``fitted_correlated_fraction`` (``a``).
    ``correlation_chi2`` is the fit's significance statistic
    ``n·(SS_white − SS_fit)`` — the reduction in squared autocorrelation
    misfit the two-parameter model buys over the white model, which is
    χ²(2)-distributed for white residuals — and ``correlation_chi2_limit``
    the declared quantile it had to exceed. The decision that followed is
    explicit: ``noise_model`` is the Gauss–Markov model when the correlated
    component is significant and the fitted τ is at or above the sample
    interval, otherwise the white model with ``white_fallback`` set and
    ``white_fallback_reason`` naming why. ``white_sigma`` is the
    uncorrelated per-sample floor (equal to ``noise_model.sigma`` for a
    white model). For a Gauss–Markov model the white fraction of the
    variance is ``max(1 − a, white_floor_fraction)`` with
    ``white_floor_fraction = 1/√n``: a sample autocorrelation over ``n``
    points scatters by about ``1/√n``, so it cannot resolve a white
    component smaller than that — the floor is the data's resolving power
    (recorded here), never a default, and it keeps the pseudo-measurement
    covariance away from the singular ``R = 0`` a saturated fit would
    imply. ``autocorrelation_rmse`` is the RMS misfit between the
    empirical autocorrelation and the *chosen* model's.
    """

    axis: str
    sample_count: int
    residual_bias: float
    residual_bias_stderr: float
    residual_rms: float
    lag_one_autocorrelation: float
    lag_count: int
    fitted_tau: float | None
    fitted_correlated_fraction: float
    correlation_chi2: float
    correlation_chi2_limit: float
    white_floor_fraction: float
    noise_model: ProcessNoiseModel
    white_sigma: float
    autocorrelation_rmse: float
    white_fallback: bool
    white_fallback_reason: str | None

    def __post_init__(self) -> None:
        if not isinstance(self.axis, str) or not self.axis:
            raise TypeError("AxisFitEvidence.axis must be a non-empty string")
        who = f"AxisFitEvidence({self.axis!r})"
        object.__setattr__(self, "sample_count", _count(
            self.sample_count, name=f"{who}.sample_count", minimum=1))
        for name in ("residual_bias", "residual_bias_stderr", "residual_rms",
                     "lag_one_autocorrelation", "fitted_correlated_fraction",
                     "correlation_chi2", "correlation_chi2_limit",
                     "white_floor_fraction", "white_sigma",
                     "autocorrelation_rmse"):
            object.__setattr__(self, name, _finite(
                getattr(self, name), name=f"{who}.{name}"))
        if not 0.0 <= self.white_floor_fraction <= 1.0:
            raise ValueError(f"{who}.white_floor_fraction must lie in [0, 1]")
        for name in ("residual_bias_stderr", "residual_rms", "white_sigma",
                     "autocorrelation_rmse", "correlation_chi2_limit"):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{who}.{name} must be >= 0")
        object.__setattr__(self, "lag_count", _count(
            self.lag_count, name=f"{who}.lag_count", minimum=1))
        if self.fitted_tau is not None:
            object.__setattr__(self, "fitted_tau", _finite(
                self.fitted_tau, name=f"{who}.fitted_tau", minimum=0.0,
                strict=True))
        if not isinstance(self.noise_model, ProcessNoiseModel):
            raise TypeError(f"{who}.noise_model must be a ProcessNoiseModel")
        if not isinstance(self.white_fallback, bool):
            raise TypeError(f"{who}.white_fallback must be a bool")
        if self.white_fallback:
            if self.noise_model.kind != "white":
                raise ValueError(
                    f"{who}: white_fallback=True requires a white noise_model")
            if not self.white_fallback_reason:
                raise ValueError(
                    f"{who}: white_fallback=True requires a "
                    "white_fallback_reason — the fallback is never silent")
        elif self.white_fallback_reason is not None:
            raise ValueError(
                f"{who}: white_fallback_reason given without white_fallback")
        if (self.noise_model.kind == "white"
                and self.white_sigma != self.noise_model.sigma):
            raise ValueError(
                f"{who}: a white noise_model's sigma must equal white_sigma "
                f"({self.noise_model.sigma!r} != {self.white_sigma!r})")

    @property
    def total_sigma(self) -> float:
        """Stationary 1-σ of the modelled residual: white floor plus the
        correlated component (for a random walk, the floor alone — its
        variance is unbounded)."""
        extra = (self.noise_model.sigma ** 2
                 if self.noise_model.kind == "gauss_markov" else 0.0)
        return math.sqrt(self.white_sigma ** 2 + extra)

    @property
    def bias_ratio(self) -> float:
        """|held-out bias| relative to the modelled residual σ."""
        if self.total_sigma > 0.0:
            return abs(self.residual_bias) / self.total_sigma
        return 0.0 if self.residual_bias == 0.0 else math.inf


@dataclass(frozen=True)
class FitAcceptanceCriteria:
    """The numeric thresholds that decide `FitEvidence.accepted`.

    Every criterion applies per axis; the artifact records each check's
    value, limit, and outcome. Defaults:

    * ``max_bias_ratio = 0.5`` — |held-out bias| ≤ 0.5 × the modelled
      residual σ. A bias larger than that is systematic model error, not
      noise the filter can absorb.
    * ``max_autocorrelation_rmse = 0.15`` — the empirical autocorrelation
      over the fitted lags must match the chosen noise model to within
      0.15 RMS (white noise over N ≥ 200 samples scatters at ≈ 1/√N ≈
      0.07, so this accepts sampling scatter and rejects a mis-modelled
      spectrum).
    * ``min_samples = 200`` — held-out samples per axis; fewer cannot
      support the autocorrelation fit.
    * ``max_residual_rms = None`` — optional absolute ceiling on the raw
      held-out residual RMS in the channel's units (``None`` = no
      ceiling; set it from the vehicle's risk policy).
    """

    max_bias_ratio: float = 0.5
    max_autocorrelation_rmse: float = 0.15
    min_samples: int = 200
    max_residual_rms: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "max_bias_ratio", _finite(
            self.max_bias_ratio, name="FitAcceptanceCriteria.max_bias_ratio",
            minimum=0.0))
        object.__setattr__(self, "max_autocorrelation_rmse", _finite(
            self.max_autocorrelation_rmse,
            name="FitAcceptanceCriteria.max_autocorrelation_rmse",
            minimum=0.0))
        object.__setattr__(self, "min_samples", _count(
            self.min_samples, name="FitAcceptanceCriteria.min_samples",
            minimum=1))
        if self.max_residual_rms is not None:
            object.__setattr__(self, "max_residual_rms", _finite(
                self.max_residual_rms,
                name="FitAcceptanceCriteria.max_residual_rms", minimum=0.0,
                strict=True))


@dataclass(frozen=True)
class AcceptanceCheck:
    """One criterion evaluated on one axis."""

    criterion: str
    axis: str
    value: float
    limit: float
    passed: bool


def _evaluate_checks(axes: Sequence[AxisFitEvidence],
                     criteria: FitAcceptanceCriteria
                     ) -> tuple[AcceptanceCheck, ...]:
    checks: list[AcceptanceCheck] = []
    for ax in axes:
        checks.append(AcceptanceCheck(
            "min_samples", ax.axis, float(ax.sample_count),
            float(criteria.min_samples),
            ax.sample_count >= criteria.min_samples))
        checks.append(AcceptanceCheck(
            "max_bias_ratio", ax.axis, float(ax.bias_ratio),
            criteria.max_bias_ratio,
            ax.bias_ratio <= criteria.max_bias_ratio))
        checks.append(AcceptanceCheck(
            "max_autocorrelation_rmse", ax.axis,
            float(ax.autocorrelation_rmse), criteria.max_autocorrelation_rmse,
            ax.autocorrelation_rmse <= criteria.max_autocorrelation_rmse))
        if criteria.max_residual_rms is not None:
            checks.append(AcceptanceCheck(
                "max_residual_rms", ax.axis, float(ax.residual_rms),
                criteria.max_residual_rms,
                ax.residual_rms <= criteria.max_residual_rms))
    return tuple(checks)


@dataclass(frozen=True)
class FitEvidence:
    """The typed held-out fit-evidence artifact.

    Construct through :meth:`evaluate`; ``checks`` and ``accepted`` are a
    pure function of ``axes`` and ``criteria`` and construction refuses any
    other value. The artifact is a frozen dataclass of scalars, strings and
    tuples, so `ModelArtifact`'s canonical derivation hashing covers it
    field by field.
    """

    channel: str
    held_out: HeldOutWindow
    axes: tuple[AxisFitEvidence, ...]
    criteria: FitAcceptanceCriteria
    checks: tuple[AcceptanceCheck, ...]
    accepted: bool

    def __post_init__(self) -> None:
        if not isinstance(self.channel, str) or not self.channel:
            raise TypeError("FitEvidence.channel must be a non-empty string")
        if not isinstance(self.held_out, HeldOutWindow):
            raise TypeError("FitEvidence.held_out must be a HeldOutWindow")
        axes = tuple(self.axes)
        if not axes or not all(isinstance(a, AxisFitEvidence) for a in axes):
            raise TypeError(
                "FitEvidence.axes must be a non-empty tuple of AxisFitEvidence")
        names = [a.axis for a in axes]
        if len(set(names)) != len(names):
            raise ValueError(f"FitEvidence.axes repeats an axis: {names}")
        object.__setattr__(self, "axes", axes)
        if not isinstance(self.criteria, FitAcceptanceCriteria):
            raise TypeError(
                "FitEvidence.criteria must be a FitAcceptanceCriteria")
        expected = _evaluate_checks(axes, self.criteria)
        if tuple(self.checks) != expected:
            raise ValueError(
                "FitEvidence.checks must be the criteria evaluated on the "
                "axes; construct through FitEvidence.evaluate(...)")
        object.__setattr__(self, "checks", expected)
        derived = all(c.passed for c in expected)
        if not isinstance(self.accepted, bool) or self.accepted != derived:
            raise ValueError(
                "FitEvidence.accepted is derived from the acceptance criteria "
                f"({derived}); it cannot be set by the caller")

    @classmethod
    def evaluate(cls, *, channel: str, held_out: HeldOutWindow,
                 axes: Sequence[AxisFitEvidence],
                 criteria: FitAcceptanceCriteria | None = None
                 ) -> FitEvidence:
        """Build the artifact, deciding ``accepted`` from ``criteria``
        (default :class:`FitAcceptanceCriteria`)."""
        criteria = FitAcceptanceCriteria() if criteria is None else criteria
        axes = tuple(axes)
        checks = _evaluate_checks(axes, criteria)
        return cls(channel=channel, held_out=held_out, axes=axes,
                   criteria=criteria, checks=checks,
                   accepted=all(c.passed for c in checks))

    @property
    def failed_checks(self) -> tuple[AcceptanceCheck, ...]:
        return tuple(c for c in self.checks if not c.passed)

    def axis(self, name: str) -> AxisFitEvidence:
        for ax in self.axes:
            if ax.axis == name:
                return ax
        raise KeyError(
            f"FitEvidence({self.channel!r}) has no axis {name!r}; axes: "
            f"{[a.axis for a in self.axes]}")

    def summary(self) -> str:
        verdict = "ACCEPTED" if self.accepted else "REJECTED"
        header = (f"held-out evidence for {self.channel}: "
                  f"{self.held_out.window_count} window(s), "
                  f"{self.held_out.sample_count} samples at "
                  f"dt={self.held_out.dt:g} s — {verdict}")
        lines = [header]
        for ax in self.axes:
            model = ax.noise_model
            desc = (f"gauss_markov tau={model.tau:.4g} s sigma={model.sigma:.4g}"
                    if model.kind == "gauss_markov"
                    else f"{model.kind} sigma={model.sigma:.4g}")
            fallback = (f" [white fallback: {ax.white_fallback_reason}]"
                        if ax.white_fallback else "")
            lines.append(
                f"  {ax.axis}: bias={ax.residual_bias:+.4g}"
                f"±{ax.residual_bias_stderr:.2g} rms={ax.residual_rms:.4g} "
                f"white={ax.white_sigma:.4g} {desc} "
                f"acf_rmse={ax.autocorrelation_rmse:.3f}{fallback}")
        for c in self.failed_checks:
            lines.append(
                f"  FAILED {c.criterion}[{c.axis}]: {c.value:.4g} vs limit "
                f"{c.limit:.4g}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Held-out split and window identity
# ---------------------------------------------------------------------------

def window_digest(window: Window) -> str:
    """Content identity of a `Window` (sha256 over every trace, `dt`,
    `t0`) — what `HeldOutWindow.window_digests` records and what the
    untouched-acceptance-set check compares."""
    if not isinstance(window, Window):
        raise TypeError(f"window_digest expects a Window, got "
                        f"{type(window).__name__}")
    digest = hashlib.sha256(b"manta-window-v1\0")

    def feed(tag: str, mapping: dict) -> None:
        digest.update(tag.encode())
        for key in sorted(mapping, key=str):
            value = mapping[key]
            if isinstance(value, dict):
                feed(f"{tag}/{key}", value)
                continue
            array = np.asarray(value, dtype=float)
            digest.update(repr((str(key), array.shape)).encode())
            digest.update(np.ascontiguousarray(array).tobytes())

    feed("x0", window.x0)
    feed("u", window.u)
    feed("x", window.x)
    feed("x_scale", window.x_scale)
    feed("z", window.z)
    digest.update(repr((float(window.dt), float(window.t0))).encode())
    return digest.hexdigest()


def hold_out(windows: Sequence[Window], *, fraction: float = 0.3
             ) -> tuple[list[Window], list[Window]]:
    """Deterministic training / held-out split: the last
    ``ceil(fraction · n)`` windows (in the order given) are held out and
    must never enter the fit. Both sides must be non-empty."""
    windows = list(windows)
    fraction = _finite(fraction, name="hold_out fraction", minimum=0.0,
                       strict=True)
    if fraction >= 1.0:
        raise ValueError(f"hold_out fraction must be < 1, got {fraction!r}")
    n_held = math.ceil(fraction * len(windows))
    if n_held < 1 or n_held >= len(windows):
        raise ValueError(
            f"hold_out: {len(windows)} window(s) at fraction {fraction:g} "
            f"leaves {n_held} held out — both sides need at least one window")
    return windows[:-n_held], windows[-n_held:]


# ---------------------------------------------------------------------------
# The evidence pipeline
# ---------------------------------------------------------------------------

def _fit_autocorrelation(rho: np.ndarray) -> tuple[float, float]:
    """Least-squares fit of ``ρ(l) ≈ a·φ^l`` over lags ``l = 1..L``.
    Returns ``(φ, a)`` with ``φ ∈ (0, 1)``, ``a ∈ [0, 1]``. For a given φ
    the optimal ``a`` is closed-form; φ is found by a dense grid and a
    golden-section refinement — deterministic, dependency-free."""
    lags = np.arange(1, rho.size + 1, dtype=float)

    def solve_a(phi: float) -> float:
        p = phi ** lags
        return float(np.clip(np.dot(rho, p) / np.dot(p, p), 0.0, 1.0))

    def cost(phi: float) -> float:
        p = phi ** lags
        return float(np.sum((rho - solve_a(phi) * p) ** 2))

    grid = np.linspace(1e-3, 1.0 - 1e-3, 600)
    costs = np.array([cost(phi) for phi in grid])
    k = int(np.argmin(costs))
    lo = grid[max(k - 1, 0)]
    hi = grid[min(k + 1, grid.size - 1)]
    inv = (math.sqrt(5.0) - 1.0) / 2.0
    c = hi - inv * (hi - lo)
    d = lo + inv * (hi - lo)
    fc, fd = cost(c), cost(d)
    for _ in range(60):
        if fc < fd:
            hi, d, fd = d, c, fc
            c = hi - inv * (hi - lo)
            fc = cost(c)
        else:
            lo, c, fc = c, d, fd
            d = lo + inv * (hi - lo)
            fd = cost(d)
    phi = 0.5 * (lo + hi)
    return phi, solve_a(phi)


def _axis_evidence(axis: str, segments: list[np.ndarray], *, dt: float,
                   lag_count: int, chi2_limit: float) -> AxisFitEvidence:
    """Bias, variance, pooled autocorrelation and the τ/σ² decision for
    one axis from its per-window residual segments."""
    pooled = np.concatenate(segments)
    n = int(pooled.size)
    bias = float(np.mean(pooled))
    rms = float(np.sqrt(np.mean(pooled ** 2)))
    centered = [seg - bias for seg in segments]
    var = float(np.mean(np.concatenate(centered) ** 2))
    if var <= 0.0:
        raise ValueError(
            f"held_out_evidence: axis {axis!r} residual has zero variance; "
            "the held-out data cannot have been produced by a noisy sensor")
    # Pooled autocorrelation: lagged products summed over windows (a lag
    # never straddles a window boundary), normalized by the total energy.
    den = sum(float(np.dot(e, e)) for e in centered)
    rho = np.array([
        sum(float(np.dot(e[:-lag], e[lag:])) for e in centered) / den
        for lag in range(1, lag_count + 1)])
    rho1 = float(rho[0])
    phi, a = _fit_autocorrelation(rho)
    lags = np.arange(1, lag_count + 1, dtype=float)
    fitted_tau = -dt / math.log(phi)
    # Significance of the correlated component: for white residuals each
    # sample autocorrelation is ~N(0, 1/n), so n·ΣΔ(ρ²) over the two fitted
    # parameters is ~χ²(2). Below the declared quantile the fit is sampling
    # scatter and the white model is the honest one.
    ss_white = float(np.sum(rho ** 2))
    ss_fit = float(np.sum((rho - a * phi ** lags) ** 2))
    chi2 = n * max(ss_white - ss_fit, 0.0)

    fallback_reason = None
    if a <= 0.0:
        fallback_reason = ("autocorrelation fit found no positive correlated "
                           "variance (fitted correlated fraction a = 0)")
    elif chi2 < chi2_limit:
        fallback_reason = (
            f"correlated component is not significant: chi2={chi2:.4g} is "
            f"below the declared limit {chi2_limit:.4g} (fitted tau="
            f"{fitted_tau:.4g} s, correlated fraction a={a:.3g})")
    elif fitted_tau < dt:
        fallback_reason = (f"fitted correlation time tau={fitted_tau:.4g} s "
                           f"is below the sample interval dt={dt:.4g} s")
    white_floor = 1.0 / math.sqrt(n)
    if fallback_reason is None:
        white_fraction = max(1.0 - a, white_floor)
        sigma_gm = math.sqrt((1.0 - white_fraction) * var)
        white = math.sqrt(white_fraction * var)
        model = ProcessNoiseModel("gauss_markov", sigma_gm, fitted_tau)
        model_acf = a * phi ** lags
    else:
        white = math.sqrt(var)
        model = ProcessNoiseModel("white", white)
        model_acf = np.zeros_like(lags)
    acf_rmse = float(np.sqrt(np.mean((rho - model_acf) ** 2)))
    # Effective sample count under lag-one correlation (AR(1) variance
    # inflation); a negative ρ₁ is treated as uncorrelated.
    r = max(rho1, 0.0)
    n_eff = n * (1.0 - r) / (1.0 + r)
    stderr = math.sqrt(var / max(n_eff, 1.0))
    return AxisFitEvidence(
        axis=axis, sample_count=n, residual_bias=bias,
        residual_bias_stderr=stderr, residual_rms=rms,
        lag_one_autocorrelation=rho1, lag_count=lag_count,
        fitted_tau=fitted_tau, fitted_correlated_fraction=float(a),
        correlation_chi2=chi2, correlation_chi2_limit=chi2_limit,
        white_floor_fraction=white_floor,
        noise_model=model, white_sigma=white, autocorrelation_rmse=acf_rmse,
        white_fallback=fallback_reason is not None,
        white_fallback_reason=fallback_reason)


def held_out_evidence(model, windows: Sequence[Window], *, sensor: str,
                      criteria: FitAcceptanceCriteria | None = None,
                      lag_count: int = 20,
                      correlation_confidence: float = 0.99,
                      training: Sequence[str] = ()) -> FitEvidence:
    """Compute :class:`FitEvidence` for ``sensor`` on untouched held-out
    windows.

    Args:
        model    — the fitted model: a `World` or `ModelArtifact`. Its
                   mean prediction (noise zeroed) is folded from each
                   window's ``x0`` over the recorded controls.
        windows  — the held-out windows; each needs a ``z`` trace for
                   ``sensor`` longer than ``lag_count`` samples and all
                   must share ``dt``.
        sensor   — the measured output whose residual ``z − h`` is the
                   model error (full name or unique suffix).
        criteria — acceptance thresholds (default
                   :class:`FitAcceptanceCriteria`).
        lag_count — autocorrelation lags used for the τ/σ² fit (default
                   20; cover a few correlation times at the sample rate).
        correlation_confidence — the χ²(2) confidence a Gauss–Markov
                   component must reach over the white model to be kept
                   (default 0.99, a limit of 9.21); below it the axis
                   records the white fallback with this number in its
                   reason.
        training — content digests (`window_digest`) of the windows the
                   fit was solved on; a held-out window among them is
                   refused (the acceptance set must be untouched).
    """
    from ..estimation.consistency import chi2_quantile
    from ..sim import Sim

    if not windows:
        raise ValueError("held_out_evidence: needs at least one held-out "
                         "Window")
    lag_count = _count(lag_count, name="held_out_evidence lag_count",
                       minimum=1)
    confidence = _finite(correlation_confidence,
                         name="held_out_evidence correlation_confidence")
    if not 0.0 < confidence < 1.0:
        raise ValueError(
            "held_out_evidence correlation_confidence must lie in (0, 1), "
            f"got {correlation_confidence!r}")
    chi2_limit = float(chi2_quantile(2, confidence))
    training_set = set(training)
    digests = []
    for index, w in enumerate(windows):
        if not isinstance(w, Window):
            raise TypeError(
                f"held_out_evidence: windows[{index}] is not a Window")
        digest = window_digest(w)
        if digest in training_set:
            raise ValueError(
                f"held_out_evidence: windows[{index}] is a training window "
                "(same content) — the acceptance set must be untouched by "
                "the fit")
        digests.append(digest)
    dts = {float(w.dt) for w in windows}
    if len(dts) != 1:
        raise ValueError(
            f"held_out_evidence: held-out windows must share dt, got "
            f"{sorted(dts)}")
    dt = dts.pop()

    sim = Sim(model)
    module = sim.module()
    spec = module.spec
    ep = module.entry("step")
    meas_names = [pt.name for pt in module.ports_by_role(Role.MEASUREMENT)]
    full = resolve_suffix(sensor, meas_names, label="sensor",
                          who="held_out_evidence")
    dim = module.port(full).size
    dims = {n: module.port(n).size for n in meas_names}
    u_fields = module.port("u").fields
    n_noise = module.port("noise").size
    params = next((pt for pt in module.ports if pt.name == "params"), None)
    out_index = 1 + ep.returns.index(full)
    step = module.functions["step"]

    axis_names = _AXES3 if dim == 3 else tuple(str(i) for i in range(dim))
    segments: list[list[np.ndarray]] = [[] for _ in range(dim)]
    total = 0
    for index, w in enumerate(windows):
        traces, K = resolve_traces(w.z, meas_names, dims,
                                   who="held_out_evidence")
        if full not in traces:
            raise ValueError(
                f"held_out_evidence: windows[{index}] has no z trace for "
                f"{full!r}")
        if K <= lag_count:
            raise ValueError(
                f"held_out_evidence: windows[{index}] has {K} samples; the "
                f"autocorrelation fit over {lag_count} lags needs more than "
                f"{lag_count}")
        x0 = pack_x0(sim.world, spec, w)
        U = pack_u_trace(
            w.u, [f.name for f in u_fields],
            [float(np.asarray(f.default).ravel()[0]) for f in u_fields],
            K, who="held_out_evidence")
        call_args = {
            "x": x0,
            "u": U if U.size else np.zeros((0, K)),
            "noise": np.zeros((n_noise, K)),
            "dt": np.full((1, K), dt),
            "t": np.array([[w.t0 + i * dt for i in range(K)]]),
        }
        if params is not None:
            p = np.concatenate([np.atleast_1d(np.asarray(
                f.default, dtype=float)).ravel() for f in params.fields])
            call_args["params"] = np.tile(p.reshape(-1, 1), (1, K))
        ordered = []
        for a in ep.args:
            key = a.name if isinstance(a, PortRef) else "x"
            if key not in call_args:
                raise KeyError(
                    f"held_out_evidence: step entry takes unknown port arg "
                    f"{key!r}")
            ordered.append(call_args[key])
        outs = step.mapaccum(f"evidence_x{K}", K, [0], [0])(*ordered)
        predicted = np.asarray(outs[out_index], dtype=float).reshape(dim, K)
        residual = traces[full] - predicted.T
        if not np.all(np.isfinite(residual)):
            raise ValueError(
                f"held_out_evidence: windows[{index}] produced a non-finite "
                f"residual for {full!r}")
        for i in range(dim):
            segments[i].append(residual[:, i])
        total += K

    axes = [_axis_evidence(axis_names[i], segments[i], dt=dt,
                           lag_count=lag_count, chi2_limit=chi2_limit)
            for i in range(dim)]
    held = HeldOutWindow(window_count=len(windows), sample_count=total,
                         dt=dt, window_digests=tuple(digests))
    return FitEvidence.evaluate(channel=full, held_out=held, axes=axes,
                                criteria=criteria)


__all__ = [
    "NOISE_KINDS",
    "AcceptanceCheck",
    "AxisFitEvidence",
    "FitAcceptanceCriteria",
    "FitEvidence",
    "HeldOutWindow",
    "ProcessNoiseModel",
    "held_out_evidence",
    "hold_out",
    "window_digest",
]

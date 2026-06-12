"""Shared fitting declarations — `Prior`, `Window`, and the plumbing both
fitters share: the `_FitBlock` decision-vector base, window-trace resolution,
and result-table rendering."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import casadi as ca
import numpy as np

from ..ir.state_spec import flatten_nested
from ..ir._names import resolve_suffix


@dataclass(frozen=True)
class Prior:
    """Gaussian prior on one fitted parameter.

    Args:
        sigma — 1-σ width. Scalar (isotropic across the parameter's
                components) or a per-component sequence. With `log=True`
                it is RELATIVE (log-space): `sigma=0.3` ≈ ±30%.
        mean  — prior mean. `None` (default) → the model's declared
                value.
        log   — fit `log(p)` instead of `p`, elementwise. Strictly-
                positive parameters only (mass, moi, a thrust magnitude
                along one axis). Keeps every component positive with no
                constraint and makes pure scale ambiguities linear.
                `NoiseFit` ignores this flag — noise σ is ALWAYS fit in
                log-space, and `sigma` there is always relative.
    """
    sigma: float | tuple = None
    mean: float | tuple | None = None
    log: bool = False


@dataclass(frozen=True)
class Window:
    """One fitting window: a short recorded rollout.

    Args:
        x0 — nested initial state dict (the `sim.state` shape:
             `{craft: {slot: value}}`). Slots omitted fall back to the
             world's initial state.
        u  — recorded controls: `{input name/suffix: scalar | (K,)}`.
             A scalar is held for the whole window; inputs omitted hold
             their model default.
        z  — recorded sensor readings: `{sensor name/suffix:
             (K, dim) | (K,)}`. Row k is the reading produced by step k
             (the step taken FROM state k). For `Fit`, only sensors
             present here enter the loss; for `NoiseFit`, every chosen
             sensor needs a trace.
        dt — fixed step, seconds.
        t0 — world-clock time of x0.
    """
    x0: dict
    u: dict = field(default_factory=dict)
    z: dict = field(default_factory=dict)
    dt: float = 0.01
    t0: float = 0.0


# ---------------------------------------------------------------------------
# Shared fitter plumbing
# ---------------------------------------------------------------------------

class _FitBlock:
    """One block of the optimizer's decision vector — a contiguous slice
    ``[offset, offset+dim)`` carrying a Gaussian prior N(`prior`, `sigma`²) in
    the DECISION space the solver works in (Fit's parameters, optionally
    log-space; NoiseFit's σ, always log-space). `init` seeds the solve.

    Subclasses (`Fit._Block`, `NoiseFit._Channel`) populate the five shared
    slots below — using ONE naming scheme across both fitters — plus their own
    mapping back to ambient values, labels, and metadata."""

    __slots__ = ("offset", "dim", "init", "prior", "sigma")


def prior_penalty(v: ca.MX, blocks: list, *, weight: float = 1.0) -> ca.MX:
    """The Gaussian-prior term `Σ weight·((v_j − prior_j)/σ_j)²` over the
    finite-σ components of every block (flat-prior components contribute
    nothing). `weight` carries each fitter's convention: 1 for Fit's
    squared-residual loss, ½ for NoiseFit's NLL."""
    term = ca.MX(0.0)
    for b in blocks:
        for j in np.flatnonzero(np.isfinite(b.sigma)):
            d = (v[b.offset + j] - float(b.prior[j])) / float(b.sigma[j])
            term = term + weight * d * d
    return term


def solve_blocks_nlp(name: str, x: ca.MX, loss: ca.MX, blocks: list, *,
                     verbose: bool, ipopt_options: dict | None):
    """Build the fitters' shared IPOPT solver, seed it from the blocks'
    `init`, and solve. Returns `(x_opt, objective, stats)`."""
    opts = {"ipopt.print_level": 5 if verbose else 0,
            "print_time": verbose, "ipopt.sb": "yes"}
    opts.update(ipopt_options or {})
    solver = ca.nlpsol(name, "ipopt", {"x": x, "f": loss}, opts)
    sol = solver(x0=np.concatenate([b.init for b in blocks]))
    return np.asarray(sol["x"]).ravel(), sol["f"], solver.stats()


def solver_converged(stats: dict, *, who: str) -> bool:
    """Did IPOPT actually converge? Warns (`RuntimeWarning`) when it did
    not — the values at a failed solve's final iterate are suspect, but
    they are still returned for inspection."""
    ok = bool(stats.get("success", False))
    if not ok:
        warnings.warn(
            f"{who}: IPOPT did NOT converge (return_status="
            f"{stats.get('return_status', 'unknown')!r}) — the fitted "
            f"values and posterior diagnostics are suspect.",
            RuntimeWarning, stacklevel=3)
    return ok


def convergence_line(converged: bool, stats: dict) -> str:
    """The `summary()` header line carrying the solver's verdict."""
    status = stats.get("return_status", "unknown")
    return (f"converged ({status})" if converged
            else f"⚠ NOT CONVERGED ({status}) — values are suspect")


def laplace_sigma(H: np.ndarray) -> np.ndarray:
    """Per-component posterior σ — `√diag(H⁻¹)` — from an information
    matrix, via `eigh` so indefinite/near-singular directions are honest:
    a component touching a non-positive (or numerically zero) eigenvalue
    direction reports `inf` (unidentified), never a fake tight σ."""
    n = H.shape[0]
    try:
        vals, vecs = np.linalg.eigh(0.5 * (H + H.T))
    except np.linalg.LinAlgError:
        return np.full(n, np.inf)
    smax = float(vals[-1]) if n else 0.0          # eigh sorts ascending
    good = vals > max(smax, 0.0) * 1e-12
    var = ((vecs[:, good] ** 2) @ (1.0 / vals[good]) if good.any()
           else np.zeros(n))
    sigma = np.sqrt(var)
    sigma[np.any(np.abs(vecs[:, ~good]) > 1e-12, axis=1)] = np.inf
    return sigma


def pack_x0(world, spec, w: Window) -> np.ndarray:
    """A window's initial state as the spec's ambient column: `Window.x0`
    slots overlaid on the world's initial state."""
    flat = flatten_nested(world._initial_state_dict())
    flat.update(flatten_nested(w.x0))
    return np.asarray(spec.pack_any(flat), dtype=float).reshape(-1, 1)


def pack_u_trace(u: dict, input_names: list[str], defaults, K: int, *,
                 who: str) -> np.ndarray:
    """A window's control trace as `(n_u, K)`: recorded rows (a scalar is
    held for the whole window, else a length-`K` trace), `defaults`
    elsewhere. Any other trace length raises."""
    if not input_names:
        return np.zeros((0, K))
    U = np.tile(np.asarray(defaults, dtype=float).reshape(-1, 1), (1, K))
    for key, val in u.items():
        full = resolve_suffix(key, input_names, label="input", who=who)
        row = input_names.index(full)
        a = np.asarray(val, dtype=float).ravel()
        if a.size == 1:
            U[row, :] = a[0]
        elif a.size == K:
            U[row, :] = a
        else:
            raise ValueError(
                f"{who}: Window.u[{key!r}] expected scalar or length-{K} "
                f"trace, got {a.size}.")
    return U


def resolve_traces(z: dict, sensor_fulls: list[str], dims: dict, *,
                   who: str):
    """Resolve a window's `{sensor name/suffix: array}` to
    `{full name: (K, dim) ndarray}` plus the common window length K. Each
    trace is reshaped to 2-D, its width checked against `dims[full]`, and all
    must share K. Raises if `z` is empty or any shape disagrees. (Each fitter
    then applies its own subset/required policy over the resolved traces.)"""
    traces: dict[str, np.ndarray] = {}
    K = None
    for key, arr in z.items():
        full = resolve_suffix(key, sensor_fulls, label="sensor", who=who)
        a = np.asarray(arr, dtype=float)
        if a.ndim == 1:
            a = a.reshape(-1, 1)
        dim = dims[full]
        if a.shape[1] != dim:
            raise ValueError(
                f"{who}: z[{key!r}] expected (K, {dim}), got {a.shape}.")
        if K is None:
            K = a.shape[0]
        elif a.shape[0] != K:
            raise ValueError(
                f"{who}: z[{key!r}] trace length {a.shape[0]} != {K}.")
        traces[full] = a
    if K is None:
        raise ValueError(f"{who}: window needs at least one sensor trace.")
    return traces, K


def format_table(rows: list) -> str:
    """Render `rows` (the first is the header) as an aligned text table with a
    dashed separator under the header. Every row is a same-length tuple of
    string cells. Shared by both fitters' `result.summary()`."""
    ncol = len(rows[0])
    widths = [max(len(r[c]) for r in rows) for c in range(ncol)]
    lines = ["  ".join(r[c].ljust(widths[c]) for c in range(ncol))
             for r in rows]
    lines.insert(1, "  ".join("-" * w for w in widths))
    return "\n".join(lines)

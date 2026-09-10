"""Shared fitting declarations — `Prior`, `Tied`, `Free`, `Window` — and
the plumbing both fitters share: the `_FitBlock` decision-vector base,
bounds resolution, window-trace resolution, and result-table rendering."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from itertools import count

import casadi as ca
import numpy as np

from ..ir._names import resolve_suffix
from ..ir.state_spec import flatten_nested

_CALLBACK_IDS = count()

DEFAULT_FILL_POLICY_ID = "manta.model-default-fill.v1"


class _BestIterate(ca.Callback):
    """Retain the lowest finite objective IPOPT has actually accepted."""

    def __init__(self, nx: int, initial_x: np.ndarray,
                 initial_objective: float, progress=None) -> None:
        ca.Callback.__init__(self)
        self.nx = int(nx)
        self.best_x = np.asarray(initial_x, dtype=float).ravel().copy()
        self.best_objective = float(initial_objective)
        self.initial_objective = float(initial_objective)
        self.progress = progress
        self.iteration = 0
        self.construct(f"best_iterate_{next(_CALLBACK_IDS)}")

    def get_n_in(self):
        return ca.nlpsol_n_out()

    def get_n_out(self):
        return 1

    def get_name_in(self, index):
        return ca.nlpsol_out(index)

    def get_name_out(self, index):
        return "ret"

    def get_sparsity_in(self, index):
        name = ca.nlpsol_out(index)
        if name == "f":
            return ca.Sparsity.scalar()
        if name in ("x", "lam_x"):
            return ca.Sparsity.dense(self.nx)
        return ca.Sparsity(0, 0)

    def eval(self, args):
        values = {ca.nlpsol_out(i): args[i] for i in range(len(args))}
        objective = float(values["f"])
        current_x = np.asarray(values["x"], dtype=float).ravel().copy()
        if np.isfinite(objective) and objective < self.best_objective:
            self.best_objective = objective
            self.best_x = current_x.copy()
        stop = False
        if self.progress is not None:
            keep_going = self.progress(
                self.iteration,
                current_x,
                objective,
                self.best_x.copy(),
                self.best_objective,
                self.initial_objective,
            )
            stop = keep_going is not None and not bool(keep_going)
        self.iteration += 1
        return [int(stop)]


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
        lower — hard lower bound, AMBIENT space (a value, not a log).
        upper — hard upper bound, ambient. Scalar or per-component.
                Enforced as IPOPT box constraints — the sanity rails
                that keep a physically absurd optimum off the table
                (a thruster gain that can't exceed the motor's rating,
                a mount that must stay inside the hull). The prior pulls
                softly; bounds are walls. In `NoiseFit` they bound σ
                itself. The declared/starting value must satisfy them.
    """
    sigma: float | tuple = None
    mean: float | tuple | None = None
    log: bool = False
    lower: float | tuple | None = None
    upper: float | tuple | None = None


@dataclass(frozen=True)
class Tied:
    """Structural tie: this promoted parameter is a fixed affine function
    of another fitted parameter (or a `Free` variable), not a decision
    variable of its own — `p = scale · p_source + offset`, in AMBIENT
    space (after any log-reparam of the source).

    This is how symmetry is enforced rather than hoped for: identical
    actuators share one gain, mirrored mounts share one geometry. Fewer
    decision variables ⇒ better-conditioned fits, and every data point
    that touches any tied copy informs the shared source.

    Args:
        source — name/suffix of the fitted parameter (or `Free` name)
                 this one derives from. Must itself be a decision
                 variable — chains of ties are not supported; tie all
                 copies to the same source.
        scale  — `None` (identity — identical copies), a scalar, a
                 per-component sequence (elementwise, e.g. a mirror's
                 sign flips `(-1, 1, 1)`), or a full
                 `(target_dim, source_dim)` matrix (e.g. a scalar arm
                 length mapped to a 3-vector mount position).
        offset — additive constant: `None` (zero), scalar, or
                 length-`target_dim`.

    Examples::

        # four identical motors — one fitted gain
        "t1.force_quad": Prior(sigma=3.0),
        "t2.force_quad": Tied("t1.force_quad"),

        # mirrored mount across the y-z plane
        "t2.mount_offset": Tied("t1.mount_offset", scale=(-1, 1, 1)),

        # scalar arm length -> the four X-frame mount positions
        "arm": Free(0.12, prior=Prior(sigma=0.02, lower=0.0)),
        "t1.mount_offset": Tied("arm", scale=[[1], [1], [0]]),
        "t2.mount_offset": Tied("arm", scale=[[-1], [1], [0]]),
    """
    source: str
    scale: object = None
    offset: object = None


@dataclass(frozen=True)
class Free:
    """Auxiliary decision variable that is NOT a promoted parameter of
    the model — it exists to be the source of `Tied` entries (a shared
    arm length, a common incidence angle). Its key in `Fit(parameters=)`
    is a fresh name, not a part parameter.

    Args:
        init  — starting value (scalar or vector); also the prior mean
                unless the prior says otherwise.
        prior — optional `Prior` (sigma/mean/log/bounds), same semantics
                as for a promoted parameter.
    """
    init: object
    prior: Prior | None = None


@dataclass(frozen=True)
class GaussianTangentPrior:
    """Sparse Gaussian prior over a linear combination of fit tangents.

    ``terms`` maps fitted parameter names to scalar, diagonal-vector, or full
    matrix coefficients. The residual is evaluated in each parameter's
    decision tangent about its ordinary :class:`Prior` mean::

        (sum(A_i @ (v_i - vbar_i)) - mean) / sigma

    This is suitable for local calibration relationships such as two mounts
    sharing a tightly known relative translation or rotation, including SO(3)
    parameters. It retains one latent value per parameter, so redundant graph
    edges cannot create independent, cycle-inconsistent transform estimates.

    ``sigma`` is scalar or per residual component. ``mean`` defaults to zero,
    meaning that the declared relative relationship is the prior mean.
    """

    terms: dict[str, object]
    sigma: float | tuple
    mean: float | tuple = 0.0
    name: str | None = None


@dataclass(frozen=True)
class Window:
    """One fitting window: a short recorded rollout.

    Args:
        x0 — nested initial state dict (the `sim.state` shape:
             `{craft: {slot: value}}`). Slots omitted fall back to the
             world's initial state and are recorded as `FitDefaultFill`
             provenance.
        x0_sigma — optional per-slot tangent-space standard deviations for
             multiple shooting. A named slot must be explicitly present in
             `x0`; its value is the prior mean and the fitter optimizes a
             manifold-aware initial perturbation for that window. Scalars are
             broadcast across the slot tangent dimension. An SO(3) value is
             therefore a three-component rotation-vector sigma, never a
             four-component quaternion sigma.
        u  — recorded controls: `{input name/suffix: scalar | (K,)}`.
             A scalar is held for the whole window; inputs omitted hold
             their model default and are recorded as `FitDefaultFill`
             provenance.
        x  — recorded state trajectories: nested or flat mapping from state
             slot name to `(K, ambient_dim)` values. Row k is the state after
             step k. Only named slots enter `Fit`; quaternion slots are
             compared on their SO(3) tangent manifold, not componentwise.
        x_scale — optional positive physical scale per recorded state slot.
             When present, Fit scores that slot as a trajectory mean-square
             ``sum(error²) / (K · scale²)`` instead of a raw sample sum.
             This is the explicit mixed-unit normalization boundary: callers
             choose meaningful floors/tolerances in each slot's native units.
        z  — recorded sensor readings: `{sensor name/suffix:
             (K, dim) | (K,)}`. Row k is the reading produced by step k
             (the step taken FROM state k). For `Fit`, only sensors
             present here enter the loss; for `NoiseFit`, every chosen
             sensor needs a trace.
        z_mask — optional explicit availability masks for multi-rate
             observations. Each key names a trace in `z` and carries a
             boolean `(K,)` array; only true rows enter the fit. Values at
             false rows are storage placeholders and are never observations.
        dt — fixed step, seconds.
        t0 — world-clock time of x0.

    ``dt`` and ``t0`` are always concrete values and are part of
    :func:`window_digest`; `FitDefaultFill` records only actual model-value
    substitutions for omitted ``x0`` and ``u`` fields.
    """
    x0: dict
    x0_sigma: dict = field(default_factory=dict)
    u: dict = field(default_factory=dict)
    x: dict = field(default_factory=dict)
    x_scale: dict = field(default_factory=dict)
    z: dict = field(default_factory=dict)
    z_mask: dict = field(default_factory=dict)
    dt: float = 0.01
    t0: float = 0.0


@dataclass(frozen=True)
class FitDefaultFill:
    """One model value substituted for missing fit-window data.

    The record is deliberately numeric and shape-explicit so model artifact
    provenance has one deterministic JSON representation. ``dataset_role``
    is one of ``training``, ``selection``, or ``acceptance``; ``source`` is
    ``model_initial_state`` or ``model_control_default``.
    """

    dataset_role: str
    window_digest: str
    source: str
    name: str
    shape: tuple[int, ...]
    values: tuple[float, ...]

    def __post_init__(self) -> None:
        if self.dataset_role not in {"training", "selection", "acceptance"}:
            raise ValueError("FitDefaultFill.dataset_role is invalid")
        if self.source not in {"model_initial_state", "model_control_default"}:
            raise ValueError("FitDefaultFill.source is invalid")
        if not isinstance(self.window_digest, str) or not self.window_digest:
            raise TypeError("FitDefaultFill.window_digest must be non-empty")
        if not isinstance(self.name, str) or not self.name:
            raise TypeError("FitDefaultFill.name must be non-empty")
        shape = tuple(self.shape)
        values = tuple(float(value) for value in self.values)
        if any(
            isinstance(dim, bool) or not isinstance(dim, int) or dim < 0
            for dim in shape
        ):
            raise ValueError("FitDefaultFill.shape is invalid")
        size = int(np.prod(shape, dtype=int)) if shape else 1
        if len(values) != size:
            raise ValueError(
                "FitDefaultFill.values does not match its declared shape"
            )
        if not all(np.isfinite(value) for value in values):
            raise ValueError("FitDefaultFill.values must be finite")
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "values", values)


def default_fills_for_window(
    world,
    spec,
    window: Window,
    *,
    dataset_role: str,
    window_digest: str,
    input_names: list[str],
    input_defaults,
    input_fields=None,
    recorded_inputs: dict | None = None,
) -> tuple[FitDefaultFill, ...]:
    """Describe every fallback used when packing one fit window."""

    def record(source: str, name: str, value) -> FitDefaultFill:
        array = np.asarray(value, dtype=float)
        return FitDefaultFill(
            dataset_role=dataset_role,
            window_digest=window_digest,
            source=source,
            name=name,
            shape=tuple(int(dim) for dim in array.shape),
            values=tuple(float(item) for item in array.ravel()),
        )

    base_state = flatten_nested(world._initial_state_dict())
    supplied_state = flatten_nested(window.x0)
    fills = [
        record("model_initial_state", slot.name, base_state[slot.name])
        for slot in sorted(spec.slots, key=lambda item: item.name)
        if slot.name not in supplied_state
    ]

    supplied_inputs = window.u if recorded_inputs is None else recorded_inputs
    supplied_full = {
        resolve_suffix(key, input_names, label="input", who="fit provenance")
        for key in supplied_inputs
    }
    if input_fields is None:
        fields = tuple(zip(input_names, input_defaults, strict=True))
    else:
        fields = tuple(
            (field.name, field.default) for field in input_fields
        )
    fills.extend(
        record("model_control_default", name, default)
        for name, default in sorted(fields)
        if name not in supplied_full
    )
    return tuple(fills)


# ---------------------------------------------------------------------------
# Shared fitter plumbing
# ---------------------------------------------------------------------------

class _FitBlock:
    """One block of the optimizer's decision vector — a contiguous slice
    ``[offset, offset+dim)`` carrying a Gaussian prior N(`prior`, `sigma`²) in
    the DECISION space the solver works in (Fit's parameters, optionally
    log-space; NoiseFit's σ, always log-space). `init` seeds the solve.

    Subclasses (`Fit._Block`, `NoiseFit._Channel`) populate the shared
    slots below — using ONE naming scheme across both fitters — plus their own
    mapping back to ambient values, labels, and metadata. `lower`/`upper` are
    DECISION-space box bounds (±inf when unbounded), fed to IPOPT as
    `lbx`/`ubx`; use `decision_bounds` to build them from a `Prior`."""

    __slots__ = (
        "dim",
        "init",
        "lower",
        "offset",
        "prior_mean",
        "sigma",
        "upper",
    )


def decision_bounds(prior: Prior | None, dim: int, ambient_init: np.ndarray,
                    *, log: bool, full: str, who: str):
    """A prior's ambient `lower`/`upper` as decision-space `(lo, hi)`
    arrays (±inf where unbounded). Validates shape, ordering, and that the
    starting value sits inside the box; with `log=True` a nonpositive
    lower bound is vacuous (log-space is positive by construction)."""
    lo = np.full(dim, -np.inf)
    hi = np.full(dim, np.inf)
    for name, given, dst in (("lower", getattr(prior, "lower", None), lo),
                             ("upper", getattr(prior, "upper", None), hi)):
        if prior is None or given is None:
            continue
        a = np.atleast_1d(np.asarray(given, dtype=float)).ravel()
        if a.size == 1:
            a = np.full(dim, a[0])
        if a.size != dim:
            raise ValueError(
                f"{who}: Prior for {full!r}: {name} bound must be a scalar "
                f"or length-{dim} sequence, got {given!r}.")
        dst[:] = a
    if np.any(lo >= hi):
        raise ValueError(
            f"{who}: Prior for {full!r}: bounds need lower < upper "
            f"elementwise, got lower={lo}, upper={hi}.")
    if np.any(ambient_init < lo) or np.any(ambient_init > hi):
        raise ValueError(
            f"{who}: Prior for {full!r}: starting value {ambient_init} "
            f"violates its own bounds [{lo}, {hi}].")
    if log:
        if np.any(hi <= 0.0):
            raise ValueError(
                f"{who}: Prior for {full!r}: upper bound must be strictly "
                f"positive with a log-space fit.")
        lo = np.where(lo > 0.0, np.log(np.where(lo > 0.0, lo, 1.0)), -np.inf)
        hi = np.where(np.isfinite(hi), np.log(np.where(hi > 0.0, hi, 1.0)),
                      np.inf)
    return lo, hi


def prior_penalty(v: ca.MX, blocks: list, *, weight: float = 1.0) -> ca.MX:
    """The Gaussian-prior term `Σ weight·((v_j − prior_j)/σ_j)²` over the
    finite-σ components of every block (flat-prior components contribute
    nothing). `weight` carries each fitter's convention: 1 for Fit's
    squared-residual loss, ½ for NoiseFit's NLL."""
    term = ca.MX(0.0)
    for b in blocks:
        for j in np.flatnonzero(np.isfinite(b.sigma)):
            d = (v[b.offset + j] - float(b.prior_mean[j])) / float(b.sigma[j])
            term = term + weight * d * d
    return term


def prior_residuals(v: ca.MX, blocks: list) -> list[ca.MX]:
    """Whitened residual form of :func:`prior_penalty`."""
    result = []
    for block in blocks:
        for index in np.flatnonzero(np.isfinite(block.sigma)):
            result.append(
                (v[block.offset + index] - float(block.prior_mean[index]))
                / float(block.sigma[index])
            )
    return result


def expand_or_none(fn: ca.Function):
    """`fn.expand()`, or None when the graph cannot lower to SX (a
    Linsol-bearing joint-space solve). The fitters route every hot
    Function (the NLP and the posterior diagnostics) through this so the
    expanded/interpreted decision is made — and reported — once."""
    try:
        return fn.expand()
    except RuntimeError:
        return None


def solve_blocks_nlp(name: str, x: ca.MX, loss: ca.MX, blocks: list, *,
                     verbose: bool, ipopt_options: dict | None,
                     initial: np.ndarray | None = None,
                     retain_best: bool = False, progress=None):
    """Build the fitters' shared IPOPT solver, seed it from the blocks'
    `init`, apply their box bounds, and solve. Returns
    `(x_opt, objective, stats, expanded)`.

    The NLP is SX-expanded when the graph allows it — a windowed
    `mapaccum` loss is thousands of scalar ops that IPOPT evaluates
    (with its exact Hessian) at every iteration, and expanding cuts that
    by an order of magnitude or more. Expandability is probed ONCE on
    the loss Function (cheap to fail: the probe raises at the first
    Linsol node, before any solver construction) and the expanded graph
    is reused for the solver — no double construction on either path.
    A graph that cannot expand (a craft whose joint-space solve rides
    the default `Linsol`) warns and takes the MX path, and the returned
    `expanded` flag records which path ran so results can carry it.
    Pass `ipopt_options={"expand": False}` to force the MX path."""
    opts = {"ipopt.print_level": 5 if verbose else 0,
            "print_time": verbose, "ipopt.sb": "yes"}
    opts.update(ipopt_options or {})
    expanded = False
    nlp = {"x": x, "f": loss}
    if opts.pop("expand", True):
        f_sx = expand_or_none(ca.Function(f"{name}_f", [x], [loss]))
        if f_sx is None:
            warnings.warn(
                f"{name}: the loss graph cannot SX-expand (a Linsol "
                f"joint-space solve keeps it MX) — IPOPT will evaluate "
                f"the interpreted MX graph, typically an order of "
                f"magnitude slower per iteration. The fit still "
                f"converges to the same optimum.",
                RuntimeWarning, stacklevel=3)
        else:
            v = ca.SX.sym("v", x.numel())
            nlp = {"x": v, "f": f_sx(v)}
            expanded = True
    x0 = (np.concatenate([b.init for b in blocks]) if initial is None
          else np.asarray(initial, dtype=float).ravel())
    callback = None
    if retain_best or progress is not None:
        if "iteration_callback" in opts:
            raise ValueError(
                "retain_best/progress cannot be combined with "
                "iteration_callback")
        initial_fn = ca.Function(
            f"{name}_initial_objective", [nlp["x"]], [nlp["f"]])
        initial_objective = float(ca.DM(initial_fn(x0)))
        callback = _BestIterate(
            x0.size, x0, initial_objective, progress=progress
        )
        opts["iteration_callback"] = callback
    solver = ca.nlpsol(name, "ipopt", nlp, opts)
    sol = solver(x0=x0,
                 lbx=np.concatenate([b.lower for b in blocks]),
                 ubx=np.concatenate([b.upper for b in blocks]))
    solution_x = np.asarray(sol["x"]).ravel()
    solution_f = float(sol["f"])
    if (retain_best and callback is not None
            and callback.best_objective < solution_f):
        solution_x = callback.best_x
        solution_f = callback.best_objective
    return solution_x, solution_f, solver.stats(), expanded


def solve_blocks_least_squares(
    name: str,
    x: ca.MX,
    residual: ca.MX,
    blocks: list,
    *,
    initial: np.ndarray | None = None,
    progress=None,
    options: dict | None = None,
):
    """Bounded damped Gauss-Newton solve without symbolic second derivatives.

    Windowed plant fits are sums of squared residuals. Building IPOPT's exact
    Hessian differentiates the entire unrolled plant twice and can require
    several times more memory than the first-order graph. This solver instead
    evaluates the residual Jacobian and solves a damped normal equation. It is
    intentionally small and dependency-free; Manta does not acquire SciPy.
    """
    opts = {
        "max_iterations": 200,
        "gradient_tolerance": 1e-5,
        "step_tolerance": 1e-8,
        "objective_tolerance": 1e-10,
        "initial_damping": 1e-3,
        "max_line_search": 16,
    }
    opts.update(options or {})
    max_iterations = int(opts["max_iterations"])
    if max_iterations < 0:
        raise ValueError("Gauss-Newton max_iterations must be non-negative")
    for key in (
        "gradient_tolerance",
        "step_tolerance",
        "objective_tolerance",
        "initial_damping",
    ):
        if not np.isfinite(opts[key]) or float(opts[key]) <= 0.0:
            raise ValueError(f"Gauss-Newton {key} must be positive and finite")

    residual_fn = ca.Function(f"{name}_residual", [x], [residual])
    expanded_fn = expand_or_none(residual_fn)
    expanded = expanded_fn is not None
    if expanded_fn is not None:
        decision = ca.SX.sym("decision", x.numel())
        expression = expanded_fn(decision)
    else:
        decision = x
        expression = residual
    evaluation = ca.Function(
        f"{name}_residual_jacobian",
        [decision],
        [expression, ca.jacobian(expression, decision)],
    )

    value = (
        np.concatenate([block.init for block in blocks])
        if initial is None
        else np.asarray(initial, dtype=float).ravel().copy()
    )
    lower = np.concatenate([block.lower for block in blocks])
    upper = np.concatenate([block.upper for block in blocks])
    if value.shape != lower.shape or np.any(value < lower) or np.any(value > upper):
        raise ValueError("Gauss-Newton initial value violates decision bounds")

    def evaluate(at: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        raw_residual, raw_jacobian = evaluation(at)
        vector = np.asarray(raw_residual, dtype=float).ravel()
        jacobian = np.asarray(raw_jacobian, dtype=float)
        objective = float(vector @ vector)
        if not (
            np.isfinite(objective)
            and np.all(np.isfinite(vector))
            and np.all(np.isfinite(jacobian))
        ):
            raise FloatingPointError("non-finite Gauss-Newton residual/Jacobian")
        return vector, jacobian, objective

    vector, jacobian, objective = evaluate(value)
    initial_objective = objective
    best_value = value.copy()
    best_objective = objective
    damping = float(opts["initial_damping"])
    history = [objective]
    gradient_history = []
    step_history = []
    status = "Maximum_Iterations_Exceeded"
    success = False
    prior_objective = objective

    for iteration in range(max_iterations + 1):
        gradient = jacobian.T @ vector
        diagonal = np.sum(np.square(jacobian), axis=0)
        column_scale = np.sqrt(np.maximum(diagonal, 1e-24))
        scaled_gradient = float(np.max(np.abs(gradient) / column_scale))
        gradient_history.append(scaled_gradient)
        if progress is not None:
            keep_going = progress(
                iteration,
                value.copy(),
                objective,
                best_value.copy(),
                best_objective,
                initial_objective,
            )
            if keep_going is not None and not bool(keep_going):
                status = "User_Requested_Stop"
                break
        if scaled_gradient <= float(opts["gradient_tolerance"]):
            status = "Solve_Succeeded"
            success = True
            break
        if iteration == max_iterations:
            break

        normal = jacobian.T @ jacobian
        regularizer = np.maximum(np.diag(normal), 1e-12)
        try:
            step = np.linalg.solve(
                normal + damping * np.diag(regularizer), -gradient
            )
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(
                normal + damping * np.diag(regularizer),
                -gradient,
                rcond=1e-12,
            )[0]
        scaled_step = float(np.max(np.abs(step) * column_scale))
        step_history.append(scaled_step)

        accepted = False
        alpha = 1.0
        candidate = value
        candidate_data = None
        for _attempt in range(int(opts["max_line_search"])):
            candidate = np.clip(value + alpha * step, lower, upper)
            if np.array_equal(candidate, value):
                alpha *= 0.5
                continue
            try:
                trial = evaluate(candidate)
            except FloatingPointError:
                alpha *= 0.5
                continue
            if trial[2] < objective:
                candidate_data = trial
                accepted = True
                break
            alpha *= 0.5
        if accepted and candidate_data is not None:
            value = candidate
            vector, jacobian, objective = candidate_data
            if objective < best_objective:
                best_value = value.copy()
                best_objective = objective
            relative_change = abs(prior_objective - objective) / max(
                1.0, abs(prior_objective)
            )
            prior_objective = objective
            damping = max(1e-12, damping * 0.3)
            history.append(objective)
            if (
                scaled_step <= float(opts["step_tolerance"])
                and relative_change <= float(opts["objective_tolerance"])
            ):
                status = "Solve_Succeeded"
                success = True
                break
        else:
            damping = min(1e12, damping * 10.0)
            history.append(objective)

    stats = {
        "success": success,
        "return_status": status,
        "iter_count": len(gradient_history) - 1,
        "iterations": {
            "obj": history,
            "inf_du": gradient_history,
            "d_norm": step_history,
        },
        "solver": "damped_gauss_newton",
        "final_scaled_gradient": gradient_history[-1],
        "final_damping": damping,
    }
    return best_value, best_objective, stats, expanded


def solver_converged(stats: dict, *, who: str) -> bool:
    """Did the selected solver converge? Warns (`RuntimeWarning`) when it did
    not — the values at a failed solve's final iterate are suspect, but
    they are still returned for inspection."""
    ok = bool(stats.get("success", False))
    if not ok:
        solver = str(stats.get("solver", "IPOPT"))
        warnings.warn(
            f"{who}: {solver} did NOT converge (return_status="
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
    base_record = flatten_nested(world._initial_state_dict())
    overrides = flatten_nested(w.x0)
    unknown = sorted(set(overrides) - set(base_record))
    if unknown:
        raise ValueError(f"Window.x0 contains unknown model keys {unknown}")
    base_record.update(overrides)
    return np.asarray(spec.pack_projected(base_record),
                      dtype=float).reshape(-1, 1)


def pack_u_trace(u: dict, input_names: list[str], defaults, K: int, *,
                 who: str, input_fields=None) -> np.ndarray:
    """A window's control trace as `(n_u, K)`: recorded rows (a scalar is
    held for the whole window, else a length-`K` trace), `defaults`
    elsewhere. Any other trace length raises."""
    if not input_names:
        return np.zeros((0, K))
    U = np.tile(np.asarray(defaults, dtype=float).reshape(-1, 1), (1, K))
    if input_fields is None:
        fields = [(name, 1) for name in input_names]
    else:
        fields = [(field.name, field.dim) for field in input_fields]
    offsets = {}
    off = 0
    for name, dim in fields:
        offsets[name] = slice(off, off + dim)
        off += dim
    for key, val in u.items():
        full = resolve_suffix(key, input_names, label="input", who=who)
        sl = offsets[full]
        dim = sl.stop - sl.start
        a = np.asarray(val, dtype=float)
        if dim == 1 and a.size == 1:
            U[sl, :] = float(a.reshape(-1)[0])
        elif dim == 1 and a.size == K:
            U[sl, :] = a.reshape(1, K)
        elif a.shape == (dim,):
            U[sl, :] = a.reshape(dim, 1)
        elif a.shape == (K, dim):
            U[sl, :] = a.T
        else:
            if dim == 1:
                raise ValueError(
                    f"{who}: Window.u[{key!r}] expected scalar or "
                    f"length-{K} trace, got {a.shape}.")
            raise ValueError(
                f"{who}: Window.u[{key!r}] expected a constant ({dim},) "
                f"or trace ({K}, {dim}), got {a.shape}.")
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


def resolve_state_traces(x: dict, spec, *, who: str):
    """Resolve selected state trajectories against a world's `StateSpec`.

    The returned arrays remain in each slot's ambient representation; the
    fitter applies that slot's manifold `boxminus` when constructing the
    residual, so SO(3) contributes a three-component rotation error rather
    than a sign-ambiguous four-component quaternion subtraction.
    """
    flat = flatten_nested(x)
    names = [slot.name for slot in spec.slots]
    traces: dict[str, np.ndarray] = {}
    K = None
    for key, arr in flat.items():
        full = resolve_suffix(key, names, label="state slot", who=who)
        slot = spec.slot(full)
        a = np.asarray(arr, dtype=float)
        if a.ndim == 1:
            a = a.reshape(-1, 1)
        if a.ndim != 2 or a.shape[1] != slot.ambient_dim:
            raise ValueError(
                f"{who}: x[{key!r}] expected (K, {slot.ambient_dim}), "
                f"got {a.shape}.")
        if K is None:
            K = a.shape[0]
        elif a.shape[0] != K:
            raise ValueError(
                f"{who}: x[{key!r}] trace length {a.shape[0]} != {K}.")
        traces[full] = a
    return traces, K


def resolve_trace_masks(
    masks: dict,
    traces: dict[str, np.ndarray],
    available_names: list[str],
    K: int,
    *,
    who: str,
) -> dict[str, np.ndarray]:
    """Resolve explicit boolean observation masks for already-bound traces."""
    resolved: dict[str, np.ndarray] = {
        full: np.ones(K, dtype=bool) for full in traces
    }
    seen: set[str] = set()
    for key, value in masks.items():
        full = resolve_suffix(
            key, available_names, label="sensor mask", who=who
        )
        if full not in traces:
            raise ValueError(
                f"{who}: z_mask[{key!r}] names a sensor with no z trace"
            )
        if full in seen:
            raise ValueError(f"{who}: duplicate mask for sensor {full!r}")
        mask = np.asarray(value)
        if mask.dtype != np.dtype(bool) or mask.shape != (K,):
            raise ValueError(
                f"{who}: z_mask[{key!r}] must be a boolean ({K},) array, "
                f"got dtype={mask.dtype} shape={mask.shape}"
            )
        resolved[full] = mask.copy()
        seen.add(full)
    return resolved


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

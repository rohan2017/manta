"""Fit — system identification over a World's promotable Parameters.

`Fit(world, parameters={...})` promotes the named Parameters (thruster
gains, mount transforms, masses — any `Parameter` declared with a
manifold) to a live parameter vector via `Sim(world, parameters=[...])`,
then fits them to recorded data by windowed prediction error:

    L(v) = Σ_windows (Σ_states w_x · ‖predicted(v) ⊟ truth‖²
                         + Σ_sensors w_z · ‖predicted(v) − measured‖²)
         + Σ_params ‖(v − v_prior) / σ_prior‖²                 (MAP)

Each `Window` is a short rollout: a known initial state, the recorded
control trace, and ground-truth state and/or sensor trajectories. State
errors use each slot's manifold (including SO(3) quaternion error). Predictions
come from folding the oracle `step` kernel over the entire window (CasADi
`mapaccum`, noise zeroed → mean prediction); gradients are exact
(symbolic), and IPOPT solves the resulting NLP.

Priors are the regularizer that makes jointly-unobservable parameters
(thrust vs. mass) well-posed: the data constrains the observable
combinations, the prior pins the rest. `Prior(sigma=..., log=True)`
puts the prior in log-space (positive scale parameters: mass, thrust
magnitude) where "±30%" is `sigma=0.3`. After the solve, the
Gauss-Newton posterior `(JᵀJ + Σ₀⁻¹)⁻¹` says which parameters the data
actually informed: a posterior σ ≈ prior σ means that number came from
your prior, not from the flight.

Structure is enforced, not hoped for: `Tied` makes one promoted
parameter a fixed affine function of another (identical actuators,
mirrored mounts), `Free` introduces an auxiliary decision variable that
only exists to source ties (a shared arm length), and `Prior(lower=,
upper=)` puts hard box bounds around any decision variable. The fit
then explores only configurations that are still the declared vehicle —
a quadcopter stays a quadcopter — and every window that excites any
tied copy informs the one shared source.

Usage::

    from manta.fit import Fit, Free, Prior, Tied, Window

    fit = Fit(world, parameters={
        "t_fl.force_quad": Prior(sigma=3.0, upper=(0, 0, 25.0)),
        "t_fr.force_quad": Tied("t_fl.force_quad"),       # identical
        "arm":             Free(0.12, prior=Prior(sigma=0.02, lower=0.0)),
        "t_fl.mount_offset":  Tied("arm", scale=[[1], [1], [0]]),
        "t_fr.mount_offset":  Tied("arm", scale=[[1], [-1], [0]]),
        "body.mass":       Prior(sigma=0.05, log=True),   # ±5%
        "imu.mount_offset":   Prior(sigma=0.02),             # ±2 cm
    })
    result = fit.solve(windows)
    print(result.summary())
    result.apply()          # write fitted values (tied ones derived)
                            # back onto the parts; a fresh Sim(world)
                            # bakes them in.

Initial states: each window needs `x0`. For synthetic-recoverability
runs, capture `sim.state` from the truth sim. For real logs without
ground truth, seed from an estimator's output, or extend this with
free per-window initial states (multiple shooting) — not built yet.

Sensor-noise σ values are NOT fittable here: a mean-prediction L2 loss
has zero gradient in σ. Fit them with `NoiseFit` (`_nll.py`) — the
EKF-innovation-likelihood fitter — after applying this fit's result.
"""

from __future__ import annotations

import copy

import casadi as ca
import numpy as np

from ..ir._names import resolve_suffix
from ..ir.module import PortRef, Role
from ..model import ModelArtifact
from ..sim import Sim
from ._common import (
    Free,
    Prior,
    Tied,
    Window,
    _FitBlock,
    convergence_line,
    decision_bounds,
    expand_or_none,
    format_table,
    laplace_sigma,
    pack_u_trace,
    pack_x0,
    prior_penalty,
    resolve_state_traces,
    resolve_traces,
    solve_blocks_nlp,
    solver_converged,
)
from ._evidence import (
    FitAcceptanceCriteria,
    FitEvidence,
    held_out_evidence,
    window_digest,
)
from ._report import derivation_report

# ---------------------------------------------------------------------------
# Internal: one decision-space block per promoted parameter
# ---------------------------------------------------------------------------

class _Block(_FitBlock):
    """One parameter's slice of the decision vector `v` and its prior."""

    __slots__ = ("full", "declared", "log")

    def __init__(self, full: str, dim: int, offset: int,
                 declared: np.ndarray, prior: Prior | None) -> None:
        self.full = full
        self.dim = dim
        self.offset = offset
        self.declared = declared
        self.log = bool(prior.log) if prior is not None else False

        mean = declared if (prior is None or prior.mean is None) else \
            np.atleast_1d(np.asarray(prior.mean, dtype=float)).ravel()
        if mean.size != dim:
            raise ValueError(
                f"Prior for {full!r}: mean has {mean.size} component(s), "
                f"parameter has {dim}.")
        if self.log:
            # Elementwise log-reparam: every component rides as log(p_j),
            # so a vector-positive parameter (e.g. `moi`) stays positive.
            if np.any(declared <= 0.0) or np.any(mean <= 0.0):
                raise ValueError(
                    f"Prior for {full!r}: log=True needs strictly "
                    f"positive declared values and mean (elementwise).")
            self.init = np.log(declared)
            self.prior_mean = np.log(mean)
        else:
            self.init = declared.copy()
            self.prior_mean = mean.copy()

        if prior is None or prior.sigma is None:
            self.sigma = np.full(dim, np.inf)
        else:
            sig = np.atleast_1d(np.asarray(prior.sigma, dtype=float)).ravel()
            if sig.size == 1:
                sig = np.full(dim, sig[0])
            if sig.size != dim or np.any(sig <= 0.0):
                raise ValueError(
                    f"Prior for {full!r}: sigma must be a positive scalar "
                    f"or length-{dim} sequence, got {prior.sigma!r}.")
            self.sigma = sig

        self.lower, self.upper = decision_bounds(
            prior, dim, declared, log=self.log, full=full, who="Fit")

    def p_of_v(self, v_blk: ca.MX) -> ca.MX:
        """Decision slice → parameter values (ambient)."""
        return ca.exp(v_blk) if self.log else v_blk

    def theta_of_v(self, v_blk: np.ndarray) -> np.ndarray:
        return np.exp(v_blk) if self.log else np.asarray(v_blk)

    def labels(self) -> list[str]:
        if self.dim == 1:
            return [self.full]
        return [f"{self.full}[{i}]" for i in range(self.dim)]


def _tie_map(t: Tied, tgt_dim: int, src_dim: int, *,
             target: str) -> tuple[np.ndarray, np.ndarray]:
    """A `Tied` spec as an explicit affine map `(A, b)`:
    `p_target = A @ p_source + b`, with every shorthand normalized."""
    if t.scale is None:
        if src_dim != tgt_dim:
            raise ValueError(
                f"Fit: Tied {target!r}: no scale given, but source "
                f"{t.source!r} has dim {src_dim} != target dim {tgt_dim}.")
        A = np.eye(tgt_dim)
    else:
        a = np.asarray(t.scale, dtype=float)
        if a.ndim == 0:
            if src_dim != tgt_dim:
                raise ValueError(
                    f"Fit: Tied {target!r}: scalar scale needs source dim "
                    f"== target dim, got {src_dim} != {tgt_dim}.")
            A = float(a) * np.eye(tgt_dim)
        elif a.ndim == 1:
            if not (src_dim == tgt_dim == a.size):
                raise ValueError(
                    f"Fit: Tied {target!r}: per-component scale must have "
                    f"length {tgt_dim} == source dim {src_dim}, got "
                    f"{a.size}.")
            A = np.diag(a)
        elif a.ndim == 2:
            if a.shape != (tgt_dim, src_dim):
                raise ValueError(
                    f"Fit: Tied {target!r}: matrix scale must be "
                    f"({tgt_dim}, {src_dim}), got {a.shape}.")
            A = a
        else:
            raise ValueError(
                f"Fit: Tied {target!r}: scale must be a scalar, vector, "
                f"or matrix, got ndim={a.ndim}.")
    if t.offset is None:
        b = np.zeros(tgt_dim)
    else:
        b = np.atleast_1d(np.asarray(t.offset, dtype=float)).ravel()
        if b.size == 1:
            b = np.full(tgt_dim, b[0])
        if b.size != tgt_dim:
            raise ValueError(
                f"Fit: Tied {target!r}: offset must be a scalar or "
                f"length-{tgt_dim} sequence, got {t.offset!r}.")
    return A, b


# ---------------------------------------------------------------------------
# FitResult
# ---------------------------------------------------------------------------

class FitResult:
    """Fitted values + Gauss-Newton posterior diagnostics.

    Attributes:
        values          — `{name: fitted value}` (float for scalars,
                          ndarray for vectors) — every promoted
                          parameter (tied ones derived through their
                          affine map) plus every `Free` variable.
        labels          — one entry per fitted scalar component of the
                          DECISION vector (tied parameters don't appear;
                          their source does).
        log_scale       — per-component bool; True ⇒ the sigmas below
                          are RELATIVE (log-space).
        prior_sigma     — per-component prior σ (inf = no prior).
        posterior_sigma — per-component Gauss-Newton posterior σ from
                          `(JᵀJ + Σ₀⁻¹)⁻¹`. ≈ prior σ ⇒ the data did
                          not inform this component.
        JtJ             — data-only Gauss-Newton information matrix in
                          decision space; its small eigenvalues are the
                          unidentifiable directions.
        objective       — final loss value.
        stats           — IPOPT return stats.
        converged       — IPOPT's success flag; False ⇒ the values below
                          are the failed solve's final iterate (a
                          `RuntimeWarning` was emitted), not an optimum.
        expanded        — True when the NLP ran SX-expanded; False means
                          the loss graph kept a Linsol node and IPOPT
                          evaluated the (order-of-magnitude slower)
                          interpreted MX graph (a `RuntimeWarning` said
                          so at solve time).
    """

    expanded: bool = True

    def __init__(self, blocks, fields, tie_sources, v_opt, p_opt, JtJ,
                 objective, stats, world, source_artifact_id,
                 source_derivation, *, posterior_computed: bool,
                 initial_objective: float) -> None:
        self._blocks = blocks
        self._fields = fields              # [(full, dim)] in port order
        self._tie_sources = tie_sources    # {tied full: source name}
        self._world = world
        self._source_artifact_id = source_artifact_id
        self._source_derivation = dict(source_derivation)
        self.v = np.asarray(v_opt, dtype=float).ravel()
        self._p = np.asarray(p_opt, dtype=float).ravel()
        self.JtJ = JtJ
        self.objective = float(objective)
        self.stats = stats
        self.posterior_computed = posterior_computed
        self.initial_objective = float(initial_objective)
        iteration_objectives = stats.get("iterations", {}).get("obj", ())
        self.objective_history = tuple(
            float(value) for value in iteration_objectives)
        if (not self.objective_history
                or not np.isclose(self.objective_history[0],
                                  self.initial_objective)):
            self.objective_history = (self.initial_objective,
                                      *self.objective_history)
        if (not self.objective_history
                or not np.isclose(self.objective_history[-1], self.objective)):
            # A limited solve can terminate away from an earlier, better
            # accepted iterate. The fitter restores that retained incumbent.
            self.objective_history = (*self.objective_history, self.objective)
        self.converged = solver_converged(stats, who="Fit")

        # Every promoted parameter's ambient value (tied ones included),
        # sliced off the assembled parameter vector…
        self.values: dict[str, object] = {}
        off = 0
        for full, dim in fields:
            theta = self._p[off:off + dim]
            self.values[full] = float(theta[0]) if dim == 1 else theta
            off += dim
        # …plus the Free variables (decision-only, not in the port).
        promoted = {full for full, _ in fields}
        self.labels: list[str] = []
        self.log_scale: list[bool] = []
        prior_sig = []
        for b in blocks:
            theta = b.theta_of_v(self.v[b.offset:b.offset + b.dim])
            if b.full not in promoted:
                self.values[b.full] = float(theta[0]) if b.dim == 1 else theta
            self.labels += b.labels()
            self.log_scale += [b.log] * b.dim
            prior_sig.append(b.sigma)
        self.prior_sigma = np.concatenate(prior_sig)

        prior_prec = np.where(np.isinf(self.prior_sigma), 0.0,
                              1.0 / np.square(self.prior_sigma))
        # eigh-based: flat-prior components the data never touched come
        # back inf, without poisoning the identified ones.
        self.posterior_sigma = (
            laplace_sigma(JtJ + np.diag(prior_prec))
            if posterior_computed else np.full_like(self.prior_sigma, np.nan))

    def weak_directions(self, k: int = 3):
        """The `k` least-informed directions of the DATA alone: list of
        `(eigenvalue, {label: component})` for the smallest eigenvalues
        of JᵀJ. A near-zero eigenvalue is an unidentifiable parameter
        combination (e.g. the thrust/mass scale)."""
        vals, vecs = np.linalg.eigh(self.JtJ)
        out = []
        for i in range(min(k, len(vals))):
            comp = {lbl: float(vecs[j, i])
                    for j, lbl in enumerate(self.labels)
                    if abs(vecs[j, i]) > 1e-3}
            out.append((float(vals[i]), comp))
        return out

    def apply(self) -> None:
        """Write the fitted values — tied parameters derived through
        their affine maps — back onto the world's Part instances. A
        transform built afterwards (`Sim(world)`, `EKF(world)`, a C++
        deploy) bakes them in as constants."""
        if not self.converged:
            raise RuntimeError(
                "FitResult.apply refuses to write an unconverged solve")
        staged = self._staged_updates(self._world)
        for part, pname, value in staged:
            setattr(part, pname, value)

    def _staged_updates(self, world):
        staged = []
        for full, dim in self._fields:
            try:
                craft_name, part_name, pname = full.split(".", 2)
            except ValueError:
                raise ValueError(
                    f"FitResult.apply: parameter name {full!r} does not "
                    f"fit the `craft.part.param` shape.") from None
            craft = next((c for c in world.crafts
                          if c.name == craft_name), None)
            part = (None if craft is None else
                    next((p for p in craft.parts if p.name == part_name),
                         None))
            if part is None:
                raise KeyError(
                    f"FitResult.apply: no part {craft_name}.{part_name} "
                    f"in this world for fitted parameter {full!r} — was "
                    f"the world rebuilt since the fit?")
            theta = np.atleast_1d(self.values[full])
            if theta.size != dim or not np.all(np.isfinite(theta)):
                raise ValueError(
                    f"FitResult.apply: fitted parameter {full!r} is invalid")
            if not hasattr(part, pname):
                raise KeyError(
                    f"FitResult.apply: part {craft_name}.{part_name} has no "
                    f"parameter {pname!r}")
            staged.append((part, pname,
                           float(theta[0]) if dim == 1 else tuple(theta)))
        return staged

    def fitted_world(self):
        """An editable copy of the authoring world with the fitted values
        (tied parameters derived) written in — what `derive()` freezes and
        `evidence()` predicts with. Refuses an unconverged solve."""
        if not self.converged:
            raise RuntimeError(
                "FitResult.fitted_world refuses an unconverged solve")
        derived = copy.deepcopy(self._world)
        for part, name, value in self._staged_updates(derived):
            setattr(part, name, value)
        return derived

    def evidence(self, held_out: list[Window], *, sensor: str,
                 criteria: FitAcceptanceCriteria | None = None,
                 lag_count: int = 20) -> FitEvidence:
        """Held-out evidence for the fitted model (see `held_out_evidence`).

        ``held_out`` must be untouched by the fit: any window whose content
        matches a training window is refused. The result is what
        `derive(evidence=...)` attaches and what a `ModelForce` consumes.
        """
        return held_out_evidence(
            self.fitted_world(), held_out, sensor=sensor, criteria=criteria,
            lag_count=lag_count, training=self._training_digests)

    def derive(self, *, evidence: FitEvidence | None = None):
        """Return a new validated model revision carrying fit provenance.

        ``evidence`` is the typed held-out artifact from `evidence()`; its
        criteria-derived ``accepted`` decision travels with the revision.
        Omitting it preserves exploratory fitting while making the
        resulting artifact visibly unaccepted — a model-aided estimator
        refuses it.
        """
        from ..sim import Sim
        artifact = Sim(self.fitted_world()).model
        if self._source_derivation:
            artifact = artifact.with_derivations(self._source_derivation)
        report = derivation_report(
            "parameter_fit", self._source_artifact_id, self.objective,
            self.values, evidence)
        return artifact.with_derivation("fit", report)

    def summary(self) -> str:
        """Per-component table: fitted value, prior σ vs posterior σ.
        `post/prior ≈ 1` flags a component the data did not inform —
        its fitted value is your prior talking, not the flight. Tied
        parameters follow, showing their derived values and source."""
        rows = [("parameter", "fitted", "prior σ", "post σ", "post/prior")]
        i = 0
        for b in self._blocks:
            theta = b.theta_of_v(self.v[b.offset:b.offset + b.dim])
            for j, lbl in enumerate(b.labels()):
                pri, post = self.prior_sigma[i], self.posterior_sigma[i]
                ratio = ("—" if not np.isfinite(pri) or not np.isfinite(post)
                         else f"{post / pri:.3f}")
                unit = " (rel)" if b.log else ""
                rows.append((lbl, f"{theta[j]:.6g}",
                             ("inf" if np.isinf(pri)
                              else f"{pri:.3g}{unit}"),
                             ("not computed" if np.isnan(post) else
                              "inf" if np.isinf(post)
                              else f"{post:.3g}{unit}"),
                             ratio))
                i += 1
        # Tied parameters carry no decision variable of their own, so
        # they get ONE row each (the derived vector, not a component per
        # line) naming the source that does.
        for full, dim in self._fields:
            src = self._tie_sources.get(full)
            if src is None:
                continue
            theta = np.atleast_1d(self.values[full])
            val = (f"{theta[0]:.6g}" if dim == 1
                   else "[" + " ".join(f"{v:.4g}" for v in theta) + "]")
            rows.append((full, val, f"← {src}", "", ""))
        return (convergence_line(self.converged, self.stats) + "\n"
                + format_table(rows))

    def __repr__(self) -> str:
        return (f"<FitResult {len(self._blocks)} parameter(s), "
                f"objective={self.objective:.6g}, "
                f"converged={self.converged}>")


# ---------------------------------------------------------------------------
# Fit
# ---------------------------------------------------------------------------

class Fit:
    """MAP parameter fit over recorded windows (see module docstring).

    Args:
        world      — the model. Finalized by the internal `Sim`; the
                     fit never mutates it (until `result.apply()`).
        parameters — `{name: Prior | Tied | Free | None}`. `Prior`/`None`
                     keys resolve against the model's promotable
                     Parameters (`<craft>.<part>.<param>`); `None` =
                     flat prior (only safe for parameters the data
                     fully observes). A `Tied` key is promoted but
                     derives from another entry's decision variable; a
                     `Free` key is a fresh auxiliary name (not a model
                     parameter) that exists to source ties.
    """

    def __init__(self, world, parameters: dict) -> None:
        for k, spec in parameters.items():
            if spec is not None and not isinstance(spec, (Prior, Tied, Free)):
                raise TypeError(
                    f"Fit: parameters[{k!r}] must be Prior, Tied, Free, or "
                    f"None, got {type(spec).__name__}.")
        free_specs = {k: v for k, v in parameters.items()
                      if isinstance(v, Free)}
        promoted_keys = [k for k in parameters if k not in free_specs]
        if not promoted_keys:
            raise ValueError(
                "Fit: no promotable parameters named — Free variables "
                "alone fit nothing.")

        # A ModelArtifact is an immutable executable revision, not an
        # authoring surface.  Fit against an editable copy so derive/apply
        # semantics remain coherent without mutating the source artifact.
        source = world
        self.world = (world.world_copy()
                      if isinstance(world, ModelArtifact) else world)
        self.sim = Sim(source, parameters=promoted_keys)
        self.model_world = self.sim.world
        self.module = self.sim.module()
        self._spec = self.module.spec
        port = self.module.port("params")
        fulls = [f.name for f in port.fields]
        by_full = {resolve_suffix(k, fulls, label="parameter", who="Fit"): v
                   for k, v in parameters.items() if k not in free_specs}
        self._fields = [(f.name, f.dim) for f in port.fields]

        # Decision blocks: one per non-tied promoted parameter (in port
        # order == kernel `p` layout), then one per Free variable.
        self._blocks: list[_Block] = []
        self._block_by_name: dict[str, _Block] = {}
        off = 0
        for f in port.fields:
            if isinstance(by_full.get(f.name), Tied):
                continue
            blk = _Block(f.name, f.dim, off,
                         np.asarray(f.default, dtype=float).ravel(),
                         by_full.get(f.name))
            self._blocks.append(blk)
            self._block_by_name[f.name] = blk
            off += f.dim
        for name, fr in free_specs.items():
            if name in {f.name for f in port.fields}:
                raise ValueError(
                    f"Fit: Free name {name!r} collides with a promoted "
                    f"parameter of the same name.")
            init = np.atleast_1d(np.asarray(fr.init, dtype=float)).ravel()
            blk = _Block(name, init.size, off, init, fr.prior)
            self._blocks.append(blk)
            self._block_by_name[name] = blk
            off += init.size
        self.n_v = off

        # Resolve ties: target field → (source block, A, b), ambient.
        sources = list(self._block_by_name)
        self._ties: dict[str, tuple] = {}
        for f in port.fields:
            spec = by_full.get(f.name)
            if not isinstance(spec, Tied):
                continue
            try:
                src_full = resolve_suffix(spec.source, sources,
                                          label="tie source", who="Fit")
            except KeyError:
                tied_names = [n for n, s in by_full.items()
                              if isinstance(s, Tied)]
                try:
                    resolve_suffix(spec.source, tied_names,
                                   label="tie source", who="Fit")
                except KeyError:
                    raise KeyError(
                        f"Fit: Tied {f.name!r}: unknown source "
                        f"{spec.source!r}. Available: {sorted(sources)}")
                raise ValueError(
                    f"Fit: Tied {f.name!r}: source {spec.source!r} is "
                    f"itself tied — chains are not supported; tie every "
                    f"copy to the same free source.")
            src = self._block_by_name[src_full]
            A, b = _tie_map(spec, f.dim, src.dim, target=f.name)
            self._ties[f.name] = (src, A, b)
        self._stepk_cache: dict[int, ca.Function] = {}

    # ------------------------------------------------------------------

    def _p_of_v(self, v: ca.MX) -> ca.MX:
        """Assemble the kernel's ambient parameter vector (port order)
        from the decision vector: own blocks map through their reparam,
        tied fields through their source's ambient value."""
        def ambient(blk):
            return blk.p_of_v(v[blk.offset:blk.offset + blk.dim])
        cols = []
        for name, _dim in self._fields:
            tie = self._ties.get(name)
            if tie is None:
                cols.append(ambient(self._block_by_name[name]))
            else:
                src, A, b = tie
                cols.append(ca.DM(A) @ ambient(src) + ca.DM(b))
        return ca.vertcat(*cols)

    # ------------------------------------------------------------------

    def solve(self, windows: list[Window], *, weights: dict | None = None,
              state_weights: dict | None = None,
              state_robust_delta: float | None = None,
              initial_values: dict | None = None,
              compute_posterior: bool = True,
              verbose: bool = False,
              ipopt_options: dict | None = None) -> FitResult:
        """Build the windowed prediction-error + prior NLP and solve it.

        Args:
            windows — the recorded data (≥ 1 `Window`).
            weights — optional per-sensor scalar weights on the squared
                      residuals (`{sensor name/suffix: w}`); use
                      `1/σ_meas²` to whiten mixed-unit sensors. Default 1.
            state_weights — optional per-state-slot weights on tangent-space
                      trajectory residuals. Default 1 for every recorded slot.
            state_robust_delta — optional positive pseudo-Huber transition in
                      normalized trajectory-RMS units. It applies to state
                      slots carrying ``Window.x_scale`` and limits the
                      influence of one structurally unrepresentable rollout
                      without introducing a non-differentiable clipping point.
            initial_values — optional warm start for decision parameters in
                      ambient units. Keys use the same exact-or-unique-suffix
                      resolution as parameter declarations. Priors and bounds
                      are unchanged; only IPOPT's starting iterate moves.
            compute_posterior — build the full residual Jacobian used only for
                      identifiability diagnostics. Disable on large production
                      fits to save peak memory; fitted values are unchanged.
            verbose — IPOPT iteration output.
            ipopt_options — extra `nlpsol` options, merged last.
        """
        if not windows:
            raise ValueError("Fit.solve: needs at least one Window.")
        if (state_robust_delta is not None
                and (not np.isfinite(state_robust_delta)
                     or state_robust_delta <= 0.0)):
            raise ValueError("state_robust_delta must be positive and finite")

        v = ca.MX.sym("v", self.n_v, 1)
        p = self._p_of_v(v)

        meas_names = [pt.name for pt in
                      self.module.ports_by_role(Role.MEASUREMENT)]
        w_by_full: dict[str, float] = {}
        for k, val in (weights or {}).items():
            full = resolve_suffix(k, meas_names, label="sensor", who="Fit")
            w_by_full[full] = float(val)
        state_names = [slot.name for slot in self._spec.slots]
        wx_by_full: dict[str, float] = {}
        for k, val in (state_weights or {}).items():
            full = resolve_suffix(k, state_names, label="state slot", who="Fit")
            wx_by_full[full] = float(val)

        loss = ca.MX(0.0)
        residuals: list[ca.MX] = []
        for w in windows:
            loss, residuals = self._add_window(
                w, p, loss, residuals, w_by_full, wx_by_full, meas_names,
                state_names, state_robust_delta)

        # MAP prior term (skipped for flat-prior components).
        loss = loss + prior_penalty(v, self._blocks)
        initial_v = np.concatenate([block.init for block in self._blocks])
        block_names = [block.full for block in self._blocks]
        for key, value in (initial_values or {}).items():
            full = resolve_suffix(
                key, block_names, label="warm-start parameter", who="Fit")
            block = self._block_by_name[full]
            ambient = np.atleast_1d(np.asarray(value, dtype=float)).ravel()
            if ambient.size != block.dim:
                raise ValueError(
                    f"Fit: warm start for {full!r} has {ambient.size} "
                    f"component(s), expected {block.dim}.")
            if block.log:
                if np.any(ambient <= 0.0):
                    raise ValueError(
                        f"Fit: warm start for log-space {full!r} must be "
                        "strictly positive.")
                decision = np.log(ambient)
            else:
                decision = ambient
            if (np.any(decision < block.lower)
                    or np.any(decision > block.upper)):
                raise ValueError(
                    f"Fit: warm start for {full!r} violates its bounds.")
            initial_v[block.offset:block.offset + block.dim] = decision
        initial_objective = float(ca.DM(
            ca.Function("fit_initial_loss", [v], [loss])(initial_v)))

        v_opt, objective, stats, expanded = solve_blocks_nlp(
            "fit", v, loss, self._blocks,
            verbose=verbose, ipopt_options=ipopt_options,
            initial=initial_v, retain_best=True)

        # Data-only Gauss-Newton information JᵀJ is optional. It is useful for
        # laboratory identifiability studies but duplicates the largest graph
        # in a production calibration whose acceptance is held-out replay.
        if compute_posterior:
            r = ca.vertcat(*residuals) if residuals else ca.MX.zeros(0, 1)
            J_fn = ca.Function("J", [v], [ca.jacobian(r, v)])
            J_fn = expand_or_none(J_fn) or J_fn
            J = np.asarray(ca.DM(J_fn(v_opt)))
            JtJ = J.T @ J
        else:
            JtJ = np.zeros((self.n_v, self.n_v))

        p_fn = ca.Function("p", [v], [p])
        p_opt = np.asarray(ca.DM(p_fn(v_opt))).ravel()
        tie_sources = {full: src.full
                       for full, (src, _A, _b) in self._ties.items()}
        res = FitResult(self._blocks, self._fields, tie_sources, v_opt,
                        p_opt, JtJ, objective, stats, self.world,
                        self.sim.model.artifact_id,
                        self.sim.model.derivation,
                        posterior_computed=compute_posterior,
                        initial_objective=initial_objective)
        res.expanded = expanded
        # Identity of the training set: `evidence()` refuses any of these
        # as a held-out window (the acceptance set must be untouched).
        res._training_digests = tuple(window_digest(w) for w in windows)
        return res

    # ------------------------------------------------------------------

    def _stepk(self, K: int) -> ca.Function:
        if K not in self._stepk_cache:
            # accumulate output 0 (x_new) -> input 0 (x); everything else
            # is a per-substep column.
            self._stepk_cache[K] = self.module.functions["step"].mapaccum(
                f"step_x{K}", K, [0], [0])
        return self._stepk_cache[K]

    def _add_window(self, w: Window, p: ca.MX, loss, residuals,
                    w_by_full: dict, wx_by_full: dict,
                    meas_names: list[str], state_names: list[str],
                    state_robust_delta: float | None):
        """Append one window's prediction-error terms to the loss."""
        ep = self.module.entry("step")
        u_fields = self.module.port("u").fields
        n_noise = self.module.port("noise").size

        # Window length from ground-truth state and/or measurement traces.
        dims = {n: self.module.port(n).size for n in meas_names}
        x_resolved, Kx = resolve_state_traces(w.x, self._spec, who="Fit")
        z_resolved, Kz = ({}, None)
        if w.z:
            z_resolved, Kz = resolve_traces(w.z, meas_names, dims, who="Fit")
        if Kx is None and Kz is None:
            raise ValueError(
                "Fit: window needs at least one state or sensor trace.")
        if Kx is not None and Kz is not None and Kx != Kz:
            raise ValueError(
                f"Fit: state trace length {Kx} != sensor trace length {Kz}.")
        K = int(Kx if Kx is not None else Kz)

        # Initial state: window slots over the world's initial state.
        x0 = pack_x0(self.model_world, self._spec, w)

        # Control trace (n_u, K): recorded columns, defaults elsewhere.
        U = pack_u_trace(
            w.u, [f.name for f in u_fields],
            [float(np.asarray(f.default).ravel()[0]) for f in u_fields],
            K, who="Fit")

        call_args = {"x": ca.DM(x0),
                     "u": ca.DM(U) if U.size else ca.DM(0, K),
                     "noise": (ca.DM.zeros(n_noise, K) if n_noise
                               else ca.DM(0, K)),
                     "params": ca.repmat(p, 1, K),
                     "dt": ca.repmat(ca.DM(float(w.dt)), 1, K),
                     "t": ca.DM(np.array([[w.t0 + i * w.dt
                                           for i in range(K)]]))}
        # ep.args: the StateRef maps to x; PortRefs by name. An entry arg
        # this loop doesn't know is a contract break — raise, don't guess.
        ordered = []
        for a in ep.args:
            key = a.name if isinstance(a, PortRef) else "x"
            if key not in call_args:
                raise KeyError(
                    f"Fit: step entry takes unknown port arg {key!r} — "
                    f"expected one of {sorted(call_args)}.")
            ordered.append(call_args[key])
        res = self._stepk(K)(*ordered)
        outs = [res] if not isinstance(res, (list, tuple)) else list(res)

        predicted_x = outs[0]
        scale_by_full: dict[str, float] = {}
        for key, value in w.x_scale.items():
            full = resolve_suffix(
                key, state_names, label="state scale", who="Fit")
            scale = float(value)
            if not np.isfinite(scale) or scale <= 0.0:
                raise ValueError(
                    f"Fit: state scale for {full!r} must be positive and "
                    f"finite, got {value!r}.")
            if full not in x_resolved:
                raise ValueError(
                    f"Fit: state scale supplied for {full!r}, but the window "
                    "has no trajectory for that slot.")
            scale_by_full[full] = scale
        for full, X in x_resolved.items():
            slot = self._spec.slot(full)
            a = slot.ambient_offset
            pred = predicted_x[a:a + slot.ambient_dim, :]
            truth = ca.DM(X.T)
            columns = [slot.manifold.boxminus_sym(pred[:, k], truth[:, k])
                       for k in range(K)]
            resid = ca.horzcat(*columns)
            wgt = wx_by_full.get(full, 1.0)
            raw_squared = ca.sumsqr(resid)
            scale = scale_by_full.get(full)
            if scale is None:
                loss = loss + wgt * raw_squared
                residuals.append(
                    np.sqrt(wgt) * ca.reshape(resid, -1, 1))
                continue

            # This is exactly the square of the normalized trajectory RMS
            # used by downstream validation (Euclidean norm across the slot,
            # mean across time). It deliberately does not divide by tangent
            # dimension: a 3-D vector error is one physical state error.
            denominator = float(K) * scale * scale
            normalized_squared = raw_squared / denominator
            if state_robust_delta is None:
                state_loss = normalized_squared
                robust_factor = 1.0
            else:
                delta2 = float(state_robust_delta) ** 2
                # Pseudo-Huber in normalized RMS: quadratic near zero, linear
                # for a whole trajectory that the chosen model class cannot
                # reproduce. Smoothness keeps exact CasADi derivatives useful.
                state_loss = 2.0 * delta2 * (
                    ca.sqrt(1.0 + normalized_squared / delta2) - 1.0)
                robust_factor = ca.if_else(
                    normalized_squared > 0.0,
                    ca.sqrt(state_loss / normalized_squared), 1.0)
            loss = loss + wgt * state_loss
            residuals.append(
                np.sqrt(wgt / denominator) * robust_factor
                * ca.reshape(resid, -1, 1))

        for full, Z in z_resolved.items():
            idx = 1 + ep.returns.index(full)        # 0 is x_new
            pred = outs[idx]                        # (dim, K)
            resid = pred - ca.DM(Z.T)
            wgt = w_by_full.get(full, 1.0)
            loss = loss + wgt * ca.sumsqr(resid)
            residuals.append(np.sqrt(wgt) * ca.reshape(resid, -1, 1))
        return loss, residuals

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
ground truth, seed from an estimator's output and put tangent-space
uncertainties for the uncertain slots in `x0_sigma`. Those slots become
window-local multiple-shooting variables; SO(3) uses a three-component
rotation-vector perturbation rather than optimizing quaternion components.

Sensor-noise σ values are NOT fittable here: a mean-prediction L2 loss
has zero gradient in σ. Fit them with `NoiseFit` (`_nll.py`) — the
EKF-innovation-likelihood fitter — after applying this fit's result.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Callable
from dataclasses import dataclass
from hashlib import sha256

import casadi as ca
import numpy as np

from ..ir._names import resolve_suffix
from ..ir.module import PortRef, Role
from ..ir.state_spec import flatten_nested
from ..model import ModelArtifact, canonical_derivation_bytes
from ..sim import Sim
from ._common import (
    Free,
    GaussianTangentPrior,
    Prior,
    Tied,
    Window,
    _FitBlock,
    convergence_line,
    decision_bounds,
    default_fills_for_window,
    expand_or_none,
    format_table,
    laplace_sigma,
    pack_u_trace,
    pack_x0,
    prior_penalty,
    prior_residuals,
    resolve_state_traces,
    resolve_trace_masks,
    resolve_traces,
    solve_blocks_least_squares,
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

    __slots__ = ("ambient_dim", "declared", "full", "log", "manifold")

    def __init__(self, full: str, ambient_dim: int, offset: int,
                 declared: np.ndarray, prior: Prior | None, *,
                 manifold=None) -> None:
        self.full = full
        self.ambient_dim = ambient_dim
        self.manifold = manifold
        self.dim = (
            int(manifold.tangent_dim)
            if manifold is not None and manifold.kind == "quat"
            else ambient_dim
        )
        self.offset = offset
        self.declared = declared
        self.log = bool(prior.log) if prior is not None else False

        mean = declared if (prior is None or prior.mean is None) else \
            np.atleast_1d(np.asarray(prior.mean, dtype=float)).ravel()
        if mean.size != ambient_dim:
            raise ValueError(
                f"Prior for {full!r}: mean has {mean.size} component(s), "
                f"parameter has {ambient_dim}.")
        if manifold is not None and manifold.kind == "quat":
            if self.log:
                raise ValueError(f"Prior for {full!r}: SO(3) cannot use log=True")
            if prior is not None and (
                prior.lower is not None or prior.upper is not None
            ):
                raise ValueError(
                    f"Prior for {full!r}: componentwise ambient bounds are "
                    "undefined on SO(3)"
                )
            for label, value in (("declared", declared), ("mean", mean)):
                norm = float(np.linalg.norm(value))
                if not np.isfinite(norm) or not np.isclose(
                    norm, 1.0, atol=1e-9
                ):
                    raise ValueError(
                        f"Prior for {full!r}: {label} quaternion must be unit"
                    )
            # Optimize a rotation-vector perturbation around the prior mean,
            # never four unconstrained quaternion coefficients.
            self.declared = mean.copy()
            self.prior_mean = np.zeros(self.dim)
            self.init = np.asarray(
                ca.DM(
                    manifold.boxminus_sym(ca.DM(declared), ca.DM(mean))
                )
            ).ravel()
        elif self.log:
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
            self.sigma = np.full(self.dim, np.inf)
        else:
            sig = np.atleast_1d(np.asarray(prior.sigma, dtype=float)).ravel()
            if sig.size == 1:
                sig = np.full(self.dim, sig[0])
            if sig.size != self.dim or np.any(sig <= 0.0):
                raise ValueError(
                    f"Prior for {full!r}: sigma must be a positive scalar "
                    f"or length-{self.dim} sequence, got {prior.sigma!r}.")
            self.sigma = sig

        if manifold is not None and manifold.kind == "quat":
            self.lower = np.full(self.dim, -np.inf)
            self.upper = np.full(self.dim, np.inf)
        else:
            self.lower, self.upper = decision_bounds(
                prior, self.dim, declared, log=self.log, full=full, who="Fit")

    def p_of_v(self, v_blk: ca.MX) -> ca.MX:
        """Decision slice → parameter values (ambient)."""
        if self.manifold is not None and self.manifold.kind == "quat":
            return self.manifold.boxplus_sym(ca.DM(self.declared), v_blk)
        return ca.exp(v_blk) if self.log else v_blk

    def theta_of_v(self, v_blk: np.ndarray) -> np.ndarray:
        if self.manifold is not None and self.manifold.kind == "quat":
            return self.manifold.boxplus_num(self.declared, v_blk)
        return np.exp(v_blk) if self.log else np.asarray(v_blk)

    def v_from_theta(self, theta: np.ndarray) -> np.ndarray:
        """Ambient warm start → this block's decision coordinates."""
        if self.manifold is not None and self.manifold.kind == "quat":
            if theta.size != self.ambient_dim:
                raise ValueError(
                    f"warm-start quaternion for {self.full!r} needs "
                    f"{self.ambient_dim} components"
                )
            norm = float(np.linalg.norm(theta))
            if not np.isfinite(norm) or not np.isclose(norm, 1.0, atol=1e-9):
                raise ValueError(
                    f"warm-start quaternion for {self.full!r} must be unit"
                )
            return np.asarray(
                ca.DM(
                    self.manifold.boxminus_sym(
                        ca.DM(theta), ca.DM(self.declared)
                    )
                )
            ).ravel()
        if theta.size != self.ambient_dim:
            raise ValueError(
                f"warm start for {self.full!r} has {theta.size} "
                f"component(s), expected {self.ambient_dim}"
            )
        if self.log:
            if np.any(theta <= 0.0):
                raise ValueError(
                    f"warm start for log-space {self.full!r} must be "
                    "strictly positive"
                )
            return np.log(theta)
        return theta

    def labels(self) -> list[str]:
        if self.dim == 1:
            return [self.full]
        if self.manifold is not None and self.manifold.kind == "quat":
            return [f"{self.full}.delta[{i}]" for i in range(self.dim)]
        return [f"{self.full}[{i}]" for i in range(self.dim)]

    def diagnostic_of_v(self, v_blk: np.ndarray) -> np.ndarray:
        if self.manifold is not None and self.manifold.kind == "quat":
            return np.asarray(v_blk, dtype=float)
        return self.theta_of_v(v_blk)


class _InitialStateBlock(_FitBlock):
    """One window-local tangent perturbation around a recorded x0 prior."""

    __slots__ = ("full", "log", "slot_name", "window_index")

    def __init__(
        self,
        *,
        window_index: int,
        slot_name: str,
        dim: int,
        offset: int,
        sigma: np.ndarray,
    ) -> None:
        self.window_index = int(window_index)
        self.slot_name = str(slot_name)
        self.full = f"window[{window_index}].x0.{slot_name}"
        self.dim = int(dim)
        self.offset = int(offset)
        self.init = np.zeros(dim)
        self.prior_mean = np.zeros(dim)
        self.sigma = np.asarray(sigma, dtype=float).reshape(dim)
        self.lower = np.full(dim, -np.inf)
        self.upper = np.full(dim, np.inf)
        self.log = False

    def theta_of_v(self, v_blk: np.ndarray) -> np.ndarray:
        return np.asarray(v_blk, dtype=float)

    def labels(self) -> list[str]:
        if self.dim == 1:
            return [self.full]
        return [f"{self.full}[{index}]" for index in range(self.dim)]


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

@dataclass(frozen=True)
class FitProgress:
    """One accepted IPOPT iteration exposed at the fitting boundary.

    ``values`` contains ambient model parameter values with structural ties
    already resolved, plus any decision-only :class:`Free` values.  It is the
    retained best iterate, not necessarily IPOPT's current trial point, so a
    caller can safely checkpoint it. Return ``False`` from a progress callback
    to request an orderly early stop; ``None`` or ``True`` continues.
    """

    iteration: int
    objective: float
    best_objective: float
    initial_objective: float
    values: dict[str, object]


@dataclass(frozen=True)
class FitPosteriorProgress:
    """Progress while accumulating residual-block posterior information."""

    completed_blocks: int
    total_blocks: int


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
                 objective, stats, world, source_model_id, source_artifact_id,
                 source_derivation, *, posterior_computed: bool,
                 initial_objective: float,
                 tangent_prior_information: np.ndarray | None = None,
                 latent_blocks=()) -> None:
        self._blocks = blocks
        self._latent_blocks = tuple(latent_blocks)
        self._diagnostic_blocks = (*blocks, *self._latent_blocks)
        self._fields = fields              # [(full, dim)] in port order
        self._tie_sources = tie_sources    # {tied full: source name}
        self._world = world
        self._source_model_id = source_model_id
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
        self.window_initial_state_deltas = {
            (block.window_index, block.slot_name): block.theta_of_v(
                self.v[block.offset:block.offset + block.dim]
            )
            for block in self._latent_blocks
        }
        for block in self._latent_blocks:
            self.labels += block.labels()
            self.log_scale += [False] * block.dim
            prior_sig.append(block.sigma)
        self.prior_sigma = np.concatenate(prior_sig)

        prior_prec = np.where(np.isinf(self.prior_sigma), 0.0,
                              1.0 / np.square(self.prior_sigma))
        self.prior_information = np.diag(prior_prec)
        if tangent_prior_information is not None:
            extra = np.asarray(tangent_prior_information, dtype=float)
            if extra.shape != self.prior_information.shape:
                raise ValueError("tangent prior information shape mismatch")
            self.prior_information += extra
        # eigh-based: flat-prior components the data never touched come
        # back inf, without poisoning the identified ones.
        self.posterior_sigma = (
            laplace_sigma(JtJ + self.prior_information)
            if posterior_computed else np.full_like(self.prior_sigma, np.nan))
        self.parameter_component_count = sum(block.dim for block in blocks)
        self.parameter_labels = tuple(
            self.labels[:self.parameter_component_count]
        )
        self.parameter_posterior_ratio = tuple(
            float(post / prior)
            if np.isfinite(prior) and prior > 0.0 else (
                0.0 if np.isfinite(post) else float("inf")
            )
            for prior, post in zip(
                self.prior_sigma[:self.parameter_component_count],
                self.posterior_sigma[:self.parameter_component_count],
                strict=True,
            )
        )
        active_bounds = []
        for block in blocks:
            decision = self.v[block.offset:block.offset + block.dim]
            for index, (value, lower, upper) in enumerate(zip(
                decision, block.lower, block.upper, strict=True
            )):
                scale = max(1.0, abs(float(value)))
                if (
                    np.isfinite(lower) and value - lower <= 1e-6 * scale
                    or np.isfinite(upper) and upper - value <= 1e-6 * scale
                ):
                    active_bounds.append(block.labels()[index])
        self.active_parameter_bounds = tuple(active_bounds)
        self._profile_id = sha256(
            b"manta-parameter-fit-profile-v1\0" + canonical_derivation_bytes({
                "source_model_id": self._source_model_id,
                "source_artifact_id": self._source_artifact_id,
                "objective": self.objective,
                "values": self.values,
            })).hexdigest()

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

    def linear_contrast_posterior_sigma(
        self, coefficients: dict[str, float]
    ) -> float:
        """Posterior sigma of a local linear parameter combination.

        ``coefficients`` names decision-space component labels from
        ``parameter_labels``. The covariance is marginalized over window-local
        initial states, so this can distinguish a well-constrained relative
        quantity (for example one mount offset minus another) from two weak
        absolute parameters. A contrast touching an information-null
        direction reports infinity rather than false confidence.

        This is a local tangent-space diagnostic. SO(3) labels therefore use
        the ``.delta[i]`` components shown by :meth:`summary`.
        """
        if not self.posterior_computed:
            raise RuntimeError(
                "linear contrast requires compute_posterior=True"
            )
        if not coefficients:
            raise ValueError("linear contrast needs at least one coefficient")
        indices = {label: index for index, label in enumerate(self.labels)}
        unknown = set(coefficients) - set(self.parameter_labels)
        if unknown:
            raise KeyError(
                f"linear contrast has unknown parameter labels {sorted(unknown)}"
            )
        contrast = np.zeros(len(self.labels), dtype=float)
        for label, coefficient in coefficients.items():
            value = float(coefficient)
            if not np.isfinite(value):
                raise ValueError("linear contrast coefficients must be finite")
            contrast[indices[label]] = value

        information = self.JtJ + self.prior_information
        try:
            values, vectors = np.linalg.eigh(
                0.5 * (information + information.T)
            )
        except np.linalg.LinAlgError:
            return float("inf")
        largest = float(values[-1]) if len(values) else 0.0
        identified = values > max(largest, 0.0) * 1e-12
        projections = vectors.T @ contrast
        if np.any(np.abs(projections[~identified]) > 1e-12):
            return float("inf")
        variance = float(np.sum(np.square(projections[identified]) / values[identified]))
        return math.sqrt(max(variance, 0.0))

    def parameter_posterior_covariance(
        self, labels: list[str] | tuple[str, ...]
    ) -> np.ndarray:
        """Marginal posterior covariance for selected parameter tangents.

        The returned block is taken from the inverse joint information over
        all fitted parameters and window-local initial states, so nuisance
        variables are marginalized rather than held fixed.  A requested
        coordinate touching an information-null direction is refused instead
        of receiving a spuriously finite pseudoinverse covariance.
        """
        if not self.posterior_computed:
            raise RuntimeError(
                "parameter covariance requires compute_posterior=True"
            )
        selected = tuple(labels)
        if not selected or len(selected) != len(set(selected)):
            raise ValueError(
                "parameter covariance needs unique selected labels"
            )
        index_by_label = {
            label: index for index, label in enumerate(self.labels)
        }
        unknown = set(selected) - set(self.parameter_labels)
        if unknown:
            raise KeyError(
                f"parameter covariance has unknown parameter labels "
                f"{sorted(unknown)}"
            )
        indices = np.asarray(
            [index_by_label[label] for label in selected], dtype=int
        )
        information = self.JtJ + self.prior_information
        try:
            values, vectors = np.linalg.eigh(
                0.5 * (information + information.T)
            )
        except np.linalg.LinAlgError as exc:
            raise ValueError(
                "posterior information eigendecomposition failed"
            ) from exc
        largest = float(values[-1]) if len(values) else 0.0
        identified = values > max(largest, 0.0) * 1e-12
        if np.any(np.abs(vectors[indices][:, ~identified]) > 1e-12):
            raise ValueError(
                "selected parameter covariance touches an unidentified "
                "posterior direction"
            )
        selected_vectors = vectors[indices][:, identified]
        covariance = (
            (selected_vectors / values[identified]) @ selected_vectors.T
            if np.any(identified)
            else np.zeros((len(indices), len(indices)), dtype=float)
        )
        return np.asarray(
            0.5 * (covariance + covariance.T), dtype=float
        )

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

    def _candidate_artifact(self):
        """The exact pre-evidence fitted artifact the replay evaluates."""
        from ..sim import Sim
        artifact = Sim(self.fitted_world()).model
        if self._source_derivation:
            artifact = artifact.with_derivations(self._source_derivation)
        return artifact

    def evidence(self, held_out: list[Window], *, sensor: str,
                 criteria: FitAcceptanceCriteria | None = None,
                 lag_count: int = 20,
                 selection: list[Window] = (),
                 configuration_id: str | None = None,
                 channel_contract_id: str | None = None) -> FitEvidence:
        """Held-out evidence for the fitted model (see `held_out_evidence`).

        ``held_out`` must be untouched by the fit: any window whose content
        matches a training window is refused. The result is what
        `derive(evidence=...)` attaches and what a `ModelForce` consumes.
        """
        candidate = self._candidate_artifact()
        candidate_sim = Sim(candidate)
        candidate_module = candidate_sim.module()
        candidate_u_fields = candidate_module.port("u").fields
        selection_digests = tuple(window_digest(w) for w in selection)
        selection_default_fills = tuple(
            fill
            for window, digest in zip(selection, selection_digests, strict=True)
            for fill in default_fills_for_window(
                candidate_sim.world,
                candidate_module.spec,
                window,
                dataset_role="selection",
                window_digest=digest,
                input_names=[field.name for field in candidate_u_fields],
                input_defaults=[field.default for field in candidate_u_fields],
                input_fields=candidate_u_fields,
            )
        )
        return held_out_evidence(
            candidate, held_out, sensor=sensor,
            criteria=criteria, lag_count=lag_count,
            training=self._training_digests,
            selection=selection_digests,
            source_model_id=self._source_model_id,
            source_artifact_id=self._source_artifact_id,
            configuration_id=(self._source_model_id
                              if configuration_id is None
                              else configuration_id),
            profile_id=self._profile_id,
            channel_contract_id=channel_contract_id,
            training_default_fills=self._training_default_fills,
            selection_default_fills=selection_default_fills)

    def derive(self, *, evidence: FitEvidence | None = None):
        """Return a new validated model revision carrying fit provenance.

        ``evidence`` is the typed held-out artifact from `evidence()`; its
        criteria-derived ``accepted`` decision travels with the revision.
        Omitting it preserves exploratory fitting while making the
        resulting artifact visibly unaccepted — a model-aided estimator
        refuses it.
        """
        artifact = self._candidate_artifact()
        if evidence is not None:
            if not isinstance(evidence, FitEvidence):
                raise TypeError("FitResult.derive evidence must be a "
                                "FitEvidence")
            binding = evidence.binding
            if binding is None:
                raise ValueError("FitResult.derive refuses unbound evidence; "
                                 "use this result's evidence(...) method")
            expected = {
                "fitted_model_id": artifact.model_id,
                "fitted_artifact_id": artifact.artifact_id,
                "source_model_id": self._source_model_id,
                "source_artifact_id": self._source_artifact_id,
                "profile_id": self._profile_id,
                "training_window_digests": self._training_digests,
            }
            mismatch = [name for name, value in expected.items()
                        if getattr(binding, name) != value]
            evidence_training_fills = tuple(
                fill for fill in evidence.default_fills
                if fill.dataset_role == "training"
            )
            if evidence_training_fills != self._training_default_fills:
                mismatch.append("training_default_fills")
            if mismatch:
                raise ValueError("FitResult.derive evidence was issued for a "
                                 "different fit/model scope: "
                                 f"{', '.join(mismatch)}")
        report = derivation_report(
            "parameter_fit", self._source_artifact_id, self.objective,
            self.values, evidence,
            (self._training_default_fills if evidence is None
             else evidence.default_fills))
        return artifact.with_derivation("fit", report)

    def summary(self) -> str:
        """Per-component table: fitted value, prior σ vs posterior σ.
        `post/prior ≈ 1` flags a component the data did not inform —
        its fitted value is your prior talking, not the flight. Tied
        parameters follow, showing their derived values and source."""
        rows = [("parameter", "fitted", "prior σ", "post σ", "post/prior")]
        i = 0
        for b in self._diagnostic_blocks:
            decision = self.v[b.offset:b.offset + b.dim]
            theta = (
                b.diagnostic_of_v(decision)
                if isinstance(b, _Block)
                else b.theta_of_v(decision)
            )
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

    def __init__(self, world, parameters: dict, *,
                 priors: tuple[GaussianTangentPrior, ...] | list[
                     GaussianTangentPrior] = ()) -> None:
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
        # Fitting owns observation availability through Window.z_mask. It
        # needs every declared measurement expression in the batched step;
        # the runtime acquisition scheduler is deliberately irrelevant here.
        self.module = self.sim.inline_module()
        self._spec = self.module.spec
        port = self.module.port("params")
        fulls = [f.name for f in port.fields]
        by_full = {resolve_suffix(k, fulls, label="parameter", who="Fit"): v
                   for k, v in parameters.items() if k not in free_specs}
        self._fields = [(f.name, f.dim) for f in port.fields]
        manifolds = {
            parameter.full: parameter.owner
            .promotable_parameter_declarations()[parameter.name]
            .manifold
            for parameter in self.sim.sys.param_specs
        }

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
                         by_full.get(f.name), manifold=manifolds[f.name])
            self._blocks.append(blk)
            self._block_by_name[f.name] = blk
            off += blk.dim
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

        self._tangent_priors = self._resolve_tangent_priors(priors)

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
            A, b = _tie_map(
                spec, f.dim, src.ambient_dim, target=f.name
            )
            self._ties[f.name] = (src, A, b)
        self._stepk_cache: dict[int, ca.Function] = {}

    def _resolve_tangent_priors(self, priors):
        """Resolve public sparse factors into decision-space matrices."""
        resolved = []
        names: set[str] = set()
        available = list(self._block_by_name)
        for index, prior in enumerate(priors):
            if not isinstance(prior, GaussianTangentPrior):
                raise TypeError(
                    "Fit: priors entries must be GaussianTangentPrior, got "
                    f"{type(prior).__name__}."
                )
            label = prior.name or f"tangent_prior_{index}"
            if label in names:
                raise ValueError(f"Fit: duplicate tangent prior name {label!r}.")
            names.add(label)
            if not prior.terms:
                raise ValueError(f"Fit: tangent prior {label!r} has no terms.")

            raw = []
            rows = None
            for key, coefficient in prior.terms.items():
                full = resolve_suffix(
                    key, available, label="tangent-prior parameter", who="Fit"
                )
                block = self._block_by_name[full]
                value = np.asarray(coefficient, dtype=float)
                if value.ndim == 0:
                    matrix = np.eye(block.dim) * float(value)
                elif value.ndim == 1:
                    if value.size != block.dim:
                        raise ValueError(
                            f"Fit: tangent prior {label!r} coefficient for "
                            f"{full!r} needs {block.dim} diagonal values."
                        )
                    matrix = np.diag(value)
                elif value.ndim == 2 and value.shape[1] == block.dim:
                    matrix = value
                else:
                    raise ValueError(
                        f"Fit: tangent prior {label!r} coefficient for "
                        f"{full!r} has incompatible shape {value.shape}."
                    )
                if rows is None:
                    rows = matrix.shape[0]
                elif rows != matrix.shape[0]:
                    raise ValueError(
                        f"Fit: tangent prior {label!r} term row counts differ."
                    )
                if not np.all(np.isfinite(matrix)):
                    raise ValueError(
                        f"Fit: tangent prior {label!r} coefficients must be finite."
                    )
                raw.append((block, matrix))

            assert rows is not None
            sigma = np.atleast_1d(np.asarray(prior.sigma, dtype=float)).ravel()
            if sigma.size == 1:
                sigma = np.full(rows, sigma[0])
            mean = np.atleast_1d(np.asarray(prior.mean, dtype=float)).ravel()
            if mean.size == 1:
                mean = np.full(rows, mean[0])
            if (
                sigma.size != rows
                or np.any(~np.isfinite(sigma))
                or np.any(sigma <= 0)
            ):
                raise ValueError(
                    f"Fit: tangent prior {label!r} sigma needs {rows} positive "
                    "finite component(s)."
                )
            if mean.size != rows or np.any(~np.isfinite(mean)):
                raise ValueError(
                    f"Fit: tangent prior {label!r} mean needs {rows} finite "
                    "component(s)."
                )
            resolved.append((label, tuple(raw), sigma, mean))
        return tuple(resolved)

    def _tangent_prior_penalty(self, v: ca.MX) -> ca.MX:
        term = ca.MX(0.0)
        for whitened in self._tangent_prior_residuals(v):
            term = term + ca.dot(whitened, whitened)
        return term

    def _tangent_prior_residuals(self, v: ca.MX) -> list[ca.MX]:
        result = []
        for _label, terms, sigma, mean in self._tangent_priors:
            residual = -ca.DM(mean)
            for block, matrix in terms:
                delta = (
                    v[block.offset:block.offset + block.dim]
                    - ca.DM(block.prior_mean)
                )
                residual = residual + ca.DM(matrix) @ delta
            result.append(residual / ca.DM(sigma))
        return result

    def _tangent_prior_information(self, n_decision: int) -> np.ndarray:
        information = np.zeros((n_decision, n_decision), dtype=float)
        for _label, terms, sigma, _mean in self._tangent_priors:
            jacobian = np.zeros((len(sigma), n_decision), dtype=float)
            for block, matrix in terms:
                jacobian[:, block.offset:block.offset + block.dim] += matrix
            jacobian /= sigma[:, None]
            information += jacobian.T @ jacobian
        return information

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

    def _initial_state_blocks(self, windows: list[Window]):
        """Build solve-local multiple-shooting blocks and slot bindings."""
        blocks: list[_InitialStateBlock] = []
        by_window: list[dict[str, _InitialStateBlock]] = []
        state_names = [slot.name for slot in self._spec.slots]
        offset = self.n_v
        for window_index, window in enumerate(windows):
            means = flatten_nested(window.x0)
            sigmas = flatten_nested(window.x0_sigma)
            selected: dict[str, _InitialStateBlock] = {}
            for key, value in sigmas.items():
                full = resolve_suffix(
                    key,
                    state_names,
                    label="initial-state prior",
                    who="Fit",
                )
                if full in selected:
                    raise ValueError(
                        f"Fit: duplicate initial-state prior for {full!r}"
                    )
                if full not in means:
                    raise ValueError(
                        f"Fit: x0_sigma[{key!r}] requires an explicit x0 "
                        "prior mean for the same state slot"
                    )
                slot = self._spec.slot(full)
                sigma = np.atleast_1d(np.asarray(value, dtype=float)).ravel()
                if sigma.size == 1:
                    sigma = np.full(slot.tangent_dim, sigma[0])
                if (
                    sigma.size != slot.tangent_dim
                    or not np.all(np.isfinite(sigma))
                    or np.any(sigma <= 0.0)
                ):
                    raise ValueError(
                        f"Fit: x0_sigma[{key!r}] must be a positive scalar "
                        f"or {slot.tangent_dim}-component tangent sigma"
                    )
                block = _InitialStateBlock(
                    window_index=window_index,
                    slot_name=full,
                    dim=slot.tangent_dim,
                    offset=offset,
                    sigma=sigma,
                )
                blocks.append(block)
                selected[full] = block
                offset += slot.tangent_dim
            by_window.append(selected)
        return blocks, by_window

    def _window_initial_state(
        self,
        window: Window,
        selected: dict[str, _InitialStateBlock],
        v: ca.MX,
    ) -> ca.MX:
        base = ca.DM(pack_x0(self.model_world, self._spec, window))
        if not selected:
            return base
        delta = ca.MX.zeros(self._spec.tangent_dim, 1)
        for full, block in selected.items():
            slot = self._spec.slot(full)
            delta[
                slot.tangent_offset:slot.tangent_offset + slot.tangent_dim
            ] = v[block.offset:block.offset + block.dim]
        return self._spec.boxplus_sym(base, delta)

    # ------------------------------------------------------------------

    def solve(self, windows: list[Window], *, weights: dict | None = None,
              state_weights: dict | None = None,
              window_weights: list[float] | tuple[float, ...] | None = None,
              state_robust_delta: float | None = None,
              initial_values: dict | None = None,
              compute_posterior: bool = True,
              solver: str = "ipopt",
              least_squares_options: dict | None = None,
              verbose: bool = False,
              progress: Callable[[FitProgress], bool | None] | None = None,
              posterior_progress: Callable[[FitPosteriorProgress], None]
              | None = None,
              ipopt_options: dict | None = None) -> FitResult:
        """Build the windowed prediction-error + prior NLP and solve it.

        Args:
            windows — the recorded data (≥ 1 `Window`).
            weights — optional per-sensor scalar weights on the squared
                      residuals (`{sensor name/suffix: w}`); use
                      `1/σ_meas²` to whiten mixed-unit sensors. Default 1.
            state_weights — optional per-state-slot weights on tangent-space
                      trajectory residuals. Default 1 for every recorded slot.
            window_weights — optional positive scalar per window. This is the
                      dataset-composition boundary for independently normalized
                      real and synthetic groups; default 1 for every window.
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
            solver — ``"ipopt"`` or ``"gauss-newton"``. The latter exploits
                      the residual structure without constructing symbolic
                      second derivatives and is preferred for large unrolled
                      plant fits.
            least_squares_options — damped Gauss-Newton tolerances, iteration
                      limit, and damping options. Used only by that solver.
            verbose — IPOPT iteration output.
            progress — optional callback after every accepted IPOPT iteration.
                      It receives the current and retained-best objectives plus
                      checkpoint-safe ambient parameter values. Return False
                      to stop cleanly at that boundary. This is independent of
                      ``verbose`` and cannot be combined with a raw CasADi
                      ``iteration_callback`` option.
            posterior_progress — optional callback after each residual block
                      contributes to the posterior normal matrix.
            ipopt_options — extra `nlpsol` options, merged last.
        """
        if not windows:
            raise ValueError("Fit.solve: needs at least one Window.")
        if solver not in ("ipopt", "gauss-newton"):
            raise ValueError("Fit.solve: solver must be 'ipopt' or 'gauss-newton'")
        if solver == "ipopt" and least_squares_options:
            raise ValueError("least_squares_options require solver='gauss-newton'")
        if solver == "gauss-newton" and ipopt_options:
            raise ValueError("ipopt_options require solver='ipopt'")
        if (state_robust_delta is not None
                and (not np.isfinite(state_robust_delta)
                     or state_robust_delta <= 0.0)):
            raise ValueError("state_robust_delta must be positive and finite")
        if window_weights is None:
            resolved_window_weights = np.ones(len(windows), dtype=float)
        else:
            resolved_window_weights = np.asarray(
                window_weights, dtype=float
            ).ravel()
            if (
                resolved_window_weights.shape != (len(windows),)
                or not np.all(np.isfinite(resolved_window_weights))
                or np.any(resolved_window_weights <= 0.0)
            ):
                raise ValueError(
                    "window_weights must contain one positive finite value "
                    "per window"
                )

        latent_blocks, latent_by_window = self._initial_state_blocks(windows)
        decision_blocks = [*self._blocks, *latent_blocks]
        n_decision = sum(block.dim for block in decision_blocks)
        v = ca.MX.sym("v", n_decision, 1)
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
        for w, selected, window_weight in zip(
            windows, latent_by_window, resolved_window_weights, strict=True
        ):
            loss, residuals = self._add_window(
                w, p, loss, residuals, w_by_full, wx_by_full, meas_names,
                state_names, state_robust_delta,
                x0=self._window_initial_state(w, selected, v),
                window_weight=float(window_weight))

        # MAP prior term (skipped for flat-prior components).
        loss = loss + prior_penalty(v, decision_blocks)
        loss = loss + self._tangent_prior_penalty(v)
        initial_v = np.concatenate([block.init for block in decision_blocks])
        block_names = [block.full for block in self._blocks]
        for key, value in (initial_values or {}).items():
            full = resolve_suffix(
                key, block_names, label="warm-start parameter", who="Fit")
            block = self._block_by_name[full]
            ambient = np.atleast_1d(np.asarray(value, dtype=float)).ravel()
            decision = block.v_from_theta(ambient)
            if (np.any(decision < block.lower)
                    or np.any(decision > block.upper)):
                raise ValueError(
                    f"Fit: warm start for {full!r} violates its bounds.")
            initial_v[block.offset:block.offset + block.dim] = decision
        initial_objective = float(ca.DM(
            ca.Function("fit_initial_loss", [v], [loss])(initial_v)))

        p_fn = ca.Function("p", [v], [p])
        promoted = {full for full, _dim in self._fields}

        def parameter_values(decision: np.ndarray) -> dict[str, object]:
            packed = np.asarray(ca.DM(p_fn(decision))).ravel()
            result: dict[str, object] = {}
            offset = 0
            for full, dim in self._fields:
                value = packed[offset:offset + dim].copy()
                result[full] = float(value[0]) if dim == 1 else value
                offset += dim
            for block in self._blocks:
                if block.full in promoted:
                    continue
                value = block.theta_of_v(
                    decision[block.offset:block.offset + block.dim]
                ).copy()
                result[block.full] = (
                    float(value[0]) if block.dim == 1 else value
                )
            return result

        def emit_progress(
            iteration: int,
            _current_v: np.ndarray,
            objective: float,
            best_v: np.ndarray,
            best_objective: float,
            callback_initial_objective: float,
        ) -> bool | None:
            if progress is None:
                return None
            return progress(FitProgress(
                iteration=iteration,
                objective=float(objective),
                best_objective=float(best_objective),
                initial_objective=float(callback_initial_objective),
                values=parameter_values(best_v),
            ))

        if solver == "gauss-newton":
            complete_residuals = [
                *residuals,
                *prior_residuals(v, decision_blocks),
                *self._tangent_prior_residuals(v),
            ]
            complete_residual = (
                ca.vertcat(*complete_residuals)
                if complete_residuals
                else ca.MX.zeros(0, 1)
            )
            v_opt, objective, stats, expanded = solve_blocks_least_squares(
                "fit",
                v,
                complete_residual,
                decision_blocks,
                initial=initial_v,
                progress=emit_progress if progress is not None else None,
                options=least_squares_options,
            )
        else:
            v_opt, objective, stats, expanded = solve_blocks_nlp(
                "fit", v, loss, decision_blocks,
                verbose=verbose, ipopt_options=ipopt_options,
                initial=initial_v, retain_best=True,
                progress=emit_progress if progress is not None else None)

        # Data-only Gauss-Newton information JᵀJ is optional. Evaluate one
        # residual block at a time: concatenating every window/channel into a
        # monolithic dense Jacobian caused posterior computation to dominate
        # memory even when only a six-dimensional mount block was requested.
        # The accumulated normal matrix is algebraically identical.
        if compute_posterior:
            JtJ = np.zeros((n_decision, n_decision))
            for index, residual in enumerate(residuals):
                J_fn = ca.Function(
                    f"J_block_{index}", [v], [ca.jacobian(residual, v)]
                )
                J_fn = expand_or_none(J_fn) or J_fn
                J = np.asarray(ca.DM(J_fn(v_opt)))
                JtJ += J.T @ J
                if posterior_progress is not None:
                    posterior_progress(FitPosteriorProgress(
                        completed_blocks=index + 1,
                        total_blocks=len(residuals),
                    ))
        else:
            JtJ = np.zeros((n_decision, n_decision))

        p_opt = np.asarray(ca.DM(p_fn(v_opt))).ravel()
        tie_sources = {full: src.full
                       for full, (src, _A, _b) in self._ties.items()}
        res = FitResult(self._blocks, self._fields, tie_sources, v_opt,
                        p_opt, JtJ, objective, stats, self.world,
                        self.sim.model.model_id,
                        self.sim.model.artifact_id,
                        self.sim.model.derivation,
                        posterior_computed=compute_posterior,
                        initial_objective=initial_objective,
                        tangent_prior_information=(
                            self._tangent_prior_information(n_decision)
                        ),
                        latent_blocks=latent_blocks)
        res.expanded = expanded
        # Identity of the training set: `evidence()` refuses any of these
        # as a held-out window (the acceptance set must be untouched).
        res._training_digests = tuple(window_digest(w) for w in windows)
        u_fields = self.module.port("u").fields
        res._training_default_fills = tuple(sorted((
            fill
            for window, digest in zip(
                windows, res._training_digests, strict=True
            )
            for fill in default_fills_for_window(
                self.model_world,
                self._spec,
                window,
                dataset_role="training",
                window_digest=digest,
                input_names=[field.name for field in u_fields],
                input_defaults=[field.default for field in u_fields],
                input_fields=u_fields,
            )
        ), key=lambda fill: (
            fill.dataset_role, fill.window_digest, fill.source, fill.name
        )))
        return res

    # ------------------------------------------------------------------

    def sensor_residuals(
        self,
        result: FitResult,
        windows: list[Window] | tuple[Window, ...],
    ) -> dict[str, np.ndarray]:
        """Replay fitted mean predictions and return raw sensor residuals.

        This is the conditional residual boundary used by inexpensive sensor
        noise characterization after a mean fit. It reuses the same compiled
        step/mapaccum functions and window conventions as :meth:`solve`;
        callers do not need to reconstruct simulation ordering themselves.
        """
        if not isinstance(result, FitResult):
            raise TypeError("sensor_residuals requires a FitResult")
        if result._source_artifact_id != self.sim.model.artifact_id:
            raise ValueError("FitResult was produced by a different model artifact")
        if not result.converged:
            raise RuntimeError("sensor residuals require a converged mean fit")
        measurement_names = [
            port.name for port in self.module.ports_by_role(Role.MEASUREMENT)
        ]
        dimensions = {
            name: self.module.port(name).size for name in measurement_names
        }
        entry = self.module.entry("step")
        input_fields = self.module.port("u").fields
        noise_size = self.module.port("noise").size
        collected: dict[str, list[np.ndarray]] = {}

        for window_index, window in enumerate(windows):
            if not window.z:
                continue
            observed, steps = resolve_traces(
                window.z,
                measurement_names,
                dimensions,
                who="Fit.sensor_residuals",
            )
            masks = resolve_trace_masks(
                window.z_mask,
                observed,
                measurement_names,
                steps,
                who="Fit.sensor_residuals",
            )
            controls = pack_u_trace(
                window.u,
                [field.name for field in input_fields],
                [float(np.asarray(field.default).ravel()[0]) for field in input_fields],
                steps,
                who="Fit.sensor_residuals",
            )
            initial = np.asarray(pack_x0(self.model_world, self._spec, window))
            tangent = np.zeros(self._spec.tangent_dim, dtype=float)
            for (index, slot_name), delta in result.window_initial_state_deltas.items():
                if index != window_index:
                    continue
                slot = self._spec.slot(slot_name)
                tangent[
                    slot.tangent_offset:slot.tangent_offset + slot.tangent_dim
                ] = np.asarray(delta, dtype=float).ravel()
            if np.any(tangent):
                initial = np.asarray(ca.DM(
                    self._spec.boxplus_sym(ca.DM(initial), ca.DM(tangent))
                )).reshape(-1, 1)
            else:
                initial = initial.reshape(-1, 1)
            arguments = {
                "x": ca.DM(initial),
                "u": ca.DM(controls) if controls.size else ca.DM(0, steps),
                "noise": (
                    ca.DM.zeros(noise_size, steps)
                    if noise_size
                    else ca.DM(0, steps)
                ),
                "params": ca.repmat(ca.DM(result._p), 1, steps),
                "dt": ca.repmat(ca.DM(float(window.dt)), 1, steps),
                "t": ca.DM(np.array([[
                    window.t0 + index * window.dt for index in range(steps)
                ]])),
            }
            ordered = [
                arguments[arg.name if isinstance(arg, PortRef) else "x"]
                for arg in entry.args
            ]
            raw_outputs = self._stepk(steps)(*ordered)
            outputs = (
                list(raw_outputs)
                if isinstance(raw_outputs, (list, tuple))
                else [raw_outputs]
            )
            for full, values in observed.items():
                selected = np.flatnonzero(masks[full])
                if not selected.size:
                    continue
                output_index = 1 + entry.returns.index(full)
                predicted = np.asarray(outputs[output_index], dtype=float).T
                collected.setdefault(full, []).append(
                    predicted[selected] - np.asarray(values, dtype=float)[selected]
                )

        return {
            name: np.concatenate(chunks, axis=0)
            for name, chunks in collected.items()
        }

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
                    state_robust_delta: float | None, *, x0: ca.MX,
                    window_weight: float):
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

        # Control trace (n_u, K): recorded columns, defaults elsewhere.
        U = pack_u_trace(
            w.u, [f.name for f in u_fields],
            [float(np.asarray(f.default).ravel()[0]) for f in u_fields],
            K, who="Fit")

        call_args = {"x": x0,
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
                loss = loss + window_weight * wgt * raw_squared
                residuals.append(
                    np.sqrt(window_weight * wgt)
                    * ca.reshape(resid, -1, 1))
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
            loss = loss + window_weight * wgt * state_loss
            residuals.append(
                np.sqrt(window_weight * wgt / denominator) * robust_factor
                * ca.reshape(resid, -1, 1))

        masks = resolve_trace_masks(
            w.z_mask, z_resolved, meas_names, K, who="Fit"
        )
        for full, Z in z_resolved.items():
            idx = 1 + ep.returns.index(full)        # 0 is x_new
            observed = np.flatnonzero(masks[full]).tolist()
            pred = outs[idx][:, observed]
            resid = pred - ca.DM(Z[observed].T)
            wgt = w_by_full.get(full, 1.0)
            loss = loss + window_weight * wgt * ca.sumsqr(resid)
            residuals.append(
                np.sqrt(window_weight * wgt) * ca.reshape(resid, -1, 1)
            )
        return loss, residuals

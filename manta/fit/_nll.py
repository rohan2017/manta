"""NoiseFit — fit noise σ values by EKF-innovation likelihood.

A mean-prediction L2 loss (`Fit`) has zero gradient in a noise σ: the
predicted mean doesn't depend on it. What σ DOES move is the filter's
uncertainty bookkeeping — so σ is fit by running a Kalman filter over
the recorded data and minimizing the negative log-likelihood of its
innovations:

    NLL = ½ Σ_k [ ν_kᵀ S_k⁻¹ ν_k + log det S_k ],
    ν_k = z_k − h(x̂_k),   S_k = H P H ᵀ + R(σ)

Process-noise σ enters through `Q = L Σ Lᵀ` (the same auto-assembly the
EKF transform uses), measurement σ through `R = L_h Σ L_hᵀ`. Too-small
σ makes S underestimate the scatter (the ν ᵀS⁻¹ν term explodes);
too-large σ pays in `log det S` — the optimum is the σ whose predicted
innovation covariance matches the data's actual scatter.

The filter step (manifold-correct update-then-predict, Joseph form,
matching the data convention that `z[k]` is the reading produced by the
step taken from state k) is one symbolic CasADi kernel folded over each
window with `mapaccum`; σ rides in log-space (always positive, priors
are relative); IPOPT minimizes; the Laplace posterior (inverse Hessian
at the optimum) reports which σ the data pinned down.

Usage::

    nf = NoiseFit(world, noise={
        "imu.gyro_noise":  Prior(sigma=1.0),   # ±e¹ relative — loose
        "imu.accel_noise": Prior(sigma=1.0),
    })
    result = nf.solve(windows)      # the same Window type as Fit
    print(result.summary())
    evidence = result.evidence(held_out_windows, sensor="imu.accel")
    artifact = result.derive(evidence=evidence)
    # Or result.apply() to write <name>_sigma onto an editable World.

Typical workflow: fit dynamics/geometry first (`Fit`), derive a validated
model revision, then fit σ on that revision — innovation statistics are only
meaningful once the mean model is right.
"""

from __future__ import annotations

import copy
import hashlib
from collections.abc import Callable
from dataclasses import dataclass

import casadi as ca
import numpy as np

from ..estimation._assembly import _resolve_estimator
from ..estimation._kalman import (
    joseph_update,
    lin_cov,
    require_active_R,
    symmetrize,
)
from ..ir._linalg import spd_logdet, spd_solve
from ..ir._names import resolve_suffix
from ..ir.state_spec import flatten_nested
from ..linearization import LinearizedSystem
from ..model import ModelArtifact, canonical_derivation_bytes
from ._common import (
    Prior,
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
    resolve_trace_masks,
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


class _Channel(_FitBlock):
    """One fitted noise channel: its slot in the tick's noise vector, its
    log-σ decision slot, and its prior. Always scalar (`dim == 1`), always
    log-space."""

    __slots__ = ("alias", "decl_name", "spec")

    def __init__(self, spec, offset: int, prior: Prior | None) -> None:
        self.spec = spec
        self.offset = offset
        self.dim = 1
        # The user-facing name is the DECLARATION name (`gyro_bias`),
        # not the driver-input name (`gyro_bias_driver`).
        self.decl_name = next(
            (n for n, d in spec.owner.noise_declarations().items()
             if d.driver_input_name(n) == spec.name), None)
        if self.decl_name is None:
            raise KeyError(
                f"NoiseFit: noise-vector slot {spec.full!r} matches no "
                f"Noise declaration on {type(spec.owner).__name__}"
                f"('{getattr(spec.owner, 'name', '?')}') — declarations: "
                f"{sorted(spec.owner.noise_declarations())}.")
        self.alias = (spec.full.removesuffix("_driver"))

        mean = None if prior is None else prior.mean
        if mean is None:
            mean = spec.sigma
        mean = float(mean)
        start = spec.sigma if spec.sigma > 0.0 else mean
        if start <= 0.0:
            raise ValueError(
                f"NoiseFit: channel {self.alias!r} has declared σ=0 and "
                f"no positive Prior mean — there is no positive starting "
                f"point for log-σ. Declare a nonzero σ or give "
                f"Prior(mean=...).")
        # ndarray slots (length 1) — the shared-helper contract.
        self.init = np.array([np.log(start)])
        self.prior_mean = np.array([np.log(mean)]) if mean > 0.0 else self.init
        self.sigma = np.array([np.inf if prior is None or prior.sigma is None
                               else float(prior.sigma)])
        if self.sigma[0] <= 0.0:
            raise ValueError(
                f"NoiseFit: Prior.sigma for {self.alias!r} must be "
                f"positive (relative, log-space).")
        # Prior.lower/upper bound σ itself (ambient); log-space decision.
        self.lower, self.upper = decision_bounds(
            prior, 1, np.array([start]), log=True, full=self.alias,
            who="NoiseFit")


@dataclass(frozen=True)
class NoiseFitProgress:
    """One bounded-memory noise-fit iteration.

    ``values`` are ambient noise standard deviations, while the optimizer
    works internally in log space.  Returning ``False`` from a progress
    callback requests an orderly stop after the current checkpoint.
    """

    iteration: int
    objective: float
    best_objective: float
    initial_objective: float
    values: dict[str, float]


class NoiseFitResult:
    """Fitted σ per channel + Laplace posterior diagnostics.

    `prior_sigma` / `posterior_sigma` are RELATIVE (log-space) widths;
    posterior ≈ prior means the data didn't inform that σ. `converged`
    is IPOPT's success flag — False ⇒ the values are the failed solve's
    final iterate (a `RuntimeWarning` was emitted), not an optimum.
    `expanded` records whether the NLL ran SX-expanded (False = a Linsol
    node kept it on the slower interpreted MX path; a `RuntimeWarning`
    said so at solve time)."""

    expanded: bool = True

    def __init__(self, channels, s_opt, hessian, objective, stats,
                 world, source_model_id, source_artifact_id,
                 source_derivation) -> None:
        self._channels = channels
        self._world = world
        self._source_model_id = source_model_id
        self._source_artifact_id = source_artifact_id
        self._source_derivation = dict(source_derivation)
        self.s = np.asarray(s_opt, dtype=float).ravel()
        self.objective = float(objective)
        self.stats = stats
        self.converged = solver_converged(stats, who="NoiseFit")
        self.values = {c.alias: float(np.exp(self.s[c.offset]))
                       for c in channels}
        self.labels = [c.alias for c in channels]
        self.prior_sigma = np.concatenate([c.sigma for c in channels])
        # eigh-based: a non-PD direction (indefinite/near-singular Laplace
        # Hessian) reports inf — never a fake "perfectly identified" 0.
        self.posterior_sigma = laplace_sigma(hessian)
        self.posterior_ratio = tuple(
            float(post / prior)
            if np.isfinite(prior) and prior > 0.0 else (
                0.0 if np.isfinite(post) else float("inf")
            )
            for prior, post in zip(
                self.prior_sigma, self.posterior_sigma, strict=True
            )
        )
        self.active_bounds = tuple(
            channel.alias
            for channel in channels
            if (
                np.isfinite(channel.lower[0])
                and self.s[channel.offset] - channel.lower[0]
                <= 1e-6 * max(1.0, abs(self.s[channel.offset]))
            ) or (
                np.isfinite(channel.upper[0])
                and channel.upper[0] - self.s[channel.offset]
                <= 1e-6 * max(1.0, abs(self.s[channel.offset]))
            )
        )
        self._profile_id = hashlib.sha256(
            b"manta-noise-fit-profile-v1\0" + canonical_derivation_bytes({
                "source_model_id": self._source_model_id,
                "source_artifact_id": self._source_artifact_id,
                "objective": self.objective,
                "values": self.values,
            })).hexdigest()

    def apply(self) -> None:
        """Write the fitted σ back onto the owning parts
        (`<channel>_sigma` attributes); transforms built afterwards
        (an `EKF(world)`'s auto-Q/R, a `NoiseDriver`d truth sim) use
        them."""
        if not self.converged:
            raise RuntimeError(
                "NoiseFitResult.apply refuses to write an unconverged solve")
        staged = self._staged_updates(self._world)
        for owner, attr, value in staged:
            setattr(owner, attr, value)

    def _staged_updates(self, world):
        staged = []
        for c in self._channels:
            pieces = c.alias.split(".")
            owner = None
            if len(pieces) >= 3:
                craft = next((x for x in world.crafts
                              if x.name == pieces[0]), None)
                owner = (None if craft is None else
                         next((p for p in craft.parts
                               if p.name == pieces[1]), None))
            elif len(pieces) == 2:
                owner = next(
                    (d for field in world.fields
                     for d in field.disturbances if d.name == pieces[0]),
                    None)
            if owner is None:
                raise KeyError(
                    f"NoiseFitResult.apply: no owner for channel {c.alias!r} "
                    f"in the authoring world — was it rebuilt since the fit?")
            value = self.values[c.alias]
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(
                    f"NoiseFitResult.apply: invalid sigma for {c.alias!r}")
            attr = f"{c.decl_name}_sigma"
            if not hasattr(owner, attr):
                raise KeyError(
                    f"NoiseFitResult.apply: owner of {c.alias!r} no longer "
                    f"declares {attr!r}")
            staged.append((owner, attr, value))
        return staged

    def fitted_world(self):
        """An editable copy of the authoring world with the fitted σ values
        written in — what `derive()` freezes and `evidence()` predicts
        with. Refuses an unconverged solve."""
        if not self.converged:
            raise RuntimeError(
                "NoiseFitResult.fitted_world refuses an unconverged solve")
        derived = copy.deepcopy(self._world)
        for owner, attr, value in self._staged_updates(derived):
            setattr(owner, attr, value)
        return derived

    def _candidate_artifact(self):
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
        """Held-out evidence for the fitted model (see `held_out_evidence`);
        windows that entered the fit are refused."""
        from ..sim import Sim

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
        """Return a structurally validated model revision carrying the
        typed held-out evidence (or none — visibly unaccepted)."""
        artifact = self._candidate_artifact()
        if evidence is not None:
            if not isinstance(evidence, FitEvidence):
                raise TypeError("NoiseFitResult.derive evidence must be a "
                                "FitEvidence")
            binding = evidence.binding
            if binding is None:
                raise ValueError("NoiseFitResult.derive refuses unbound "
                                 "evidence; use this result's evidence(...) "
                                 "method")
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
                raise ValueError("NoiseFitResult.derive evidence was issued "
                                 "for a different fit/model scope: "
                                 f"{', '.join(mismatch)}")
        report = derivation_report(
            "noise_fit", self._source_artifact_id, self.objective,
            self.values, evidence,
            (self._training_default_fills if evidence is None
             else evidence.default_fills))
        return artifact.with_derivation("noise_fit", report)

    def summary(self) -> str:
        rows = [("channel", "fitted σ", "prior σ(rel)", "post σ(rel)",
                 "post/prior")]
        for i, c in enumerate(self._channels):
            pri, post = self.prior_sigma[i], self.posterior_sigma[i]
            ratio = ("—" if np.isinf(pri) or np.isinf(post)
                     else f"{post / pri:.3f}")
            rows.append((c.alias, f"{self.values[c.alias]:.6g}",
                         "inf" if np.isinf(pri) else f"{pri:.3g}",
                         "inf" if np.isinf(post) else f"{post:.3g}",
                         ratio))
        return (convergence_line(self.converged, self.stats) + "\n"
                + format_table(rows))

    def __repr__(self) -> str:
        return (f"<NoiseFitResult {len(self._channels)} channel(s), "
                f"objective={self.objective:.6g}, "
                f"converged={self.converged}>")


class NoiseFit:
    """Innovation-NLL fit of noise σ values (see module docstring).

    Args:
        world   — the model (dynamics/geometry at their — ideally
                  already fitted — declared values).
        noise   — `{channel name/suffix: Prior | None}`. Channel names
                  are the declaration names (`drone.imu.gyro_noise`,
                  `drone.imu.gyro_bias`); priors are relative
                  (log-space), `None` = flat.
        sensors — measurement outputs the filter consumes (default: all
                  with traces required in every window).
        estimator — optional model-derived estimator transform. Supplying an
                  INS reuses its strapdown transition, selected sensor set,
                  IMU prediction inputs, and measurement-source mapping.
    """

    def __init__(self, world, noise: dict, *,
                 sensors: list[str] | None = None,
                 estimator=None) -> None:
        source = world
        self.world = (world.world_copy()
                      if isinstance(world, ModelArtifact) else world)
        if estimator is None:
            self.estimator = None
            self.sys = LinearizedSystem(source, sensors=sensors)
        else:
            self.estimator = _resolve_estimator(source, estimator)
            self.sys = self.estimator.sys
            if sensors is not None:
                chosen = {
                    resolve_suffix(name, list(self.sys.sensors),
                                   label="sensor", who="NoiseFit")
                    for name in sensors
                }
                if chosen != set(self.sys.sensors):
                    raise ValueError(
                        "NoiseFit: when estimator= is supplied, select its "
                        "sensor set on the estimator transform")
        self.model_world = self.sys.world
        sys = self.sys

        # Resolve requested channels against the tick's noise vector.
        aliases = []
        for spec in sys.noise_specs:
            aliases.append(spec.full.removesuffix("_driver"))
        chosen: dict[int, Prior | None] = {}
        for key, prior in noise.items():
            alias = resolve_suffix(key, aliases, label="noise channel",
                                   who="NoiseFit")
            chosen[aliases.index(alias)] = prior
        self.channels = [
            _Channel(sys.noise_specs[idx], k, prior)
            for k, (idx, prior) in enumerate(sorted(chosen.items()))]
        if not self.channels:
            raise ValueError(
                "NoiseFit: no noise channels selected — name at least one "
                "channel in noise={...}.")
        self._chan_by_spec = {c.spec.full: c for c in self.channels}
        self.n_s = len(self.channels)

        self._step_fn = self._build_step()
        self._fold_cache: dict[int, ca.Function] = {}
        self._window_nll_cache: dict[int, ca.Function] = {}
        self._validate_R0()

    # ------------------------------------------------------------------

    def _sigma_diag(self, s: ca.MX) -> ca.MX:
        """The tick-noise covariance diagonal Σ(s): fitted channels from
        exp(s), the rest at their declared σ."""
        entries = []
        for spec in self.sys.noise_specs:
            c = self._chan_by_spec.get(spec.full)
            var = (ca.exp(2.0 * s[c.offset]) if c is not None
                   else ca.MX(float(spec.sigma) ** 2))
            entries += [var] * spec.dim
        return ca.diag(ca.vertcat(*entries))

    def _build_step(self) -> ca.Function:
        """One symbolic filter step + NLL increment:

            nll_step(x, Pv, u, z, mask, s, dt, t)
                -> x⁺, Pv⁺, ½·mask·(νᵀS⁻¹ν+logdetS)

        Each sensor update is independently gated by its explicit availability
        mask. Then predict once at the base plant cadence — matching the
        recorded-data convention. P rides flattened so `mapaccum` can
        accumulate it."""
        sys = self.sys
        spec = sys.spec
        tan = spec.tangent_dim
        # Most systems currently expose scalar controls, but estimator
        # transforms may add vector-valued prediction inputs (INS adds the
        # selected IMU's accel and gyro samples).  The symbolic width is the
        # authoritative packed-u dimension; ``input_names`` counts fields.
        n_u = int(sys.u_sym.numel())
        zdim = sum(s_.dim for s_ in sys.sensors.values())
        nsensor = len(sys.sensors)

        x_in = ca.MX.sym("x", spec.ambient_dim, 1)
        Pv = ca.MX.sym("Pv", tan * tan, 1)
        u = ca.MX.sym("u", n_u, 1) if n_u else ca.MX.zeros(0, 1)
        z = ca.MX.sym("z", zdim, 1)
        mask = ca.MX.sym("mask", nsensor, 1)
        s = ca.MX.sym("s", self.n_s, 1)
        dt = ca.MX.sym("dt", 1, 1)
        t = ca.MX.sym("t", 1, 1)

        x = x_in                     # evolving estimate (x_in stays symbolic)
        P = ca.reshape(Pv, tan, tan)
        Sigma = self._sigma_diag(s)
        zero_dt = ca.MX.zeros(1, 1)
        nll = ca.MX(0.0)

        # ---- sequential measurement updates at x (pre-step state) ----
        # The Joseph fold is the shared `joseph_update` kernel (R here is
        # symbolic in σ — see estimation/_kalman.py); ν and S come back
        # so the NLL increment reuses the update's own innovation stats.
        off = 0
        for sensor_index, sm in enumerate(sys.sensors.values()):
            zk = z[off:off + sm.dim]
            off += sm.dim
            sub = ca.vertcat(sys.x_sym, sys.u_sym, sys.dt_sym, sys.t_sym)
            vals = ca.vertcat(x, u, zero_dt, t)
            h = ca.substitute(ca.reshape(sm.h_sym, sm.dim, 1), sub, vals)
            H = ca.substitute(sm.H_sym, sub, vals)
            L_h = (ca.substitute(sm.L_h_sym, sub, vals)
                   if sm.L_h_sym is not None else None)
            R = lin_cov(L_h, Sigma, sm.dim)
            x_updated, P_updated, nu, S = joseph_update(
                x, P, h, H, R, zk, spec
            )
            active = mask[sensor_index] > 0.5
            x = ca.if_else(active, x_updated, x)
            P = ca.if_else(active, P_updated, P)
            increment = 0.5 * (ca.dot(nu, spd_solve(S, nu))
                               + spd_logdet(S))
            nll = nll + ca.if_else(active, increment, 0.0)

        # ---- predict from the updated state ---------------------------
        sub = ca.vertcat(sys.x_sym, sys.u_sym, sys.dt_sym, sys.t_sym)
        vals = ca.vertcat(x, u, dt, t)
        x_new = ca.substitute(sys.x_new, sub, vals)
        F = ca.substitute(sys.F_sym, sub, vals)
        L = ca.substitute(sys.L_sym, sub, vals) if sys.L_sym is not None \
            else None
        Q = lin_cov(L, Sigma, tan)
        P = symmetrize(F @ P @ F.T + Q)

        return ca.Function(
            "nll_step", [x_in, Pv, u, z, mask, s, dt, t],
            [x_new, ca.reshape(P, tan * tan, 1), nll],
            ["x", "Pv", "u", "z", "mask", "s", "dt", "t"],
            ["x_new", "Pv_new", "nll"])

    def _validate_R0(self) -> None:
        """Every chosen sensor needs S > 0 from the first update: with all
        its white channels at σ=0 (and unfitted), R is exactly zero and a
        small P0 makes S singular. Probe each sensor's R at the starting σ
        via the shared `require_active_R` guard (diagonal check — a fitted
        R is diagonal in σ)."""
        s0 = np.concatenate([c.init for c in self.channels]).reshape(-1, 1)
        sys = self.sys
        flat = flatten_nested(self.model_world._initial_state_dict())
        x0 = np.asarray(sys.spec.pack_projected(flat), dtype=float)
        s_sym = ca.MX.sym("s", self.n_s, 1)
        zero_dt = ca.MX.zeros(1, 1)
        for full, sm in sys.sensors.items():
            if sm.L_h_sym is None:
                continue
            L_h = ca.substitute(sm.L_h_sym, sys.dt_sym, zero_dt)
            R = lin_cov(L_h, self._sigma_diag(s_sym), sm.dim)
            R_fn = ca.Function("R", [sys.x_sym, sys.u_sym, sys.t_sym,
                                     s_sym], [R])
            require_active_R(
                R, R_fn, ca.vertcat(sys.x_sym, sys.u_sym, sys.t_sym),
                x0=x0, u_defaults=sys.u_defaults, spec=sys.spec,
                full=full, who="NoiseFit", extra=(s0,), diag=True)

    # ------------------------------------------------------------------

    def _fold(self, K: int) -> ca.Function:
        if K not in self._fold_cache:
            self._fold_cache[K] = self._step_fn.mapaccum(
                f"nll_x{K}", K, [0, 1], [0, 1])
        return self._fold_cache[K]

    def _window_nll(self, K: int) -> ca.Function:
        """A reusable one-window NLL/gradient graph.

        Recorded arrays are function inputs rather than embedded constants.
        Consequently a capture with many windows repeatedly evaluates this
        one fixed-size graph instead of concatenating every EKF rollout into
        one enormous symbolic NLP.
        """
        if K in self._window_nll_cache:
            return self._window_nll_cache[K]
        sys = self.sys
        spec = sys.spec
        tan = spec.tangent_dim
        n_u = int(sys.u_sym.numel())
        zdim = sum(sensor.dim for sensor in sys.sensors.values())
        nsensor = len(sys.sensors)
        s = ca.MX.sym("s", self.n_s, 1)
        x0 = ca.MX.sym("x0", spec.ambient_dim, 1)
        Pv0 = ca.MX.sym("Pv0", tan * tan, 1)
        U = ca.MX.sym("U", n_u, K) if n_u else ca.MX.zeros(0, K)
        Z = ca.MX.sym("Z", zdim, K)
        M = ca.MX.sym("M", nsensor, K)
        dt = ca.MX.sym("dt", 1, 1)
        times = ca.MX.sym("times", 1, K)
        folded = self._fold(K)(
            x0,
            Pv0,
            U,
            Z,
            M,
            ca.repmat(s, 1, K),
            ca.repmat(dt, 1, K),
            times,
        )
        objective = ca.sum2(folded[2])
        function = ca.Function(
            f"noise_window_x{K}",
            [s, x0, Pv0, U, Z, M, dt, times],
            [objective, ca.gradient(objective, s)],
            ["s", "x0", "Pv0", "U", "Z", "M", "dt", "times"],
            ["objective", "gradient"],
        )
        self._window_nll_cache[K] = function
        return function

    def _solve_batched(
        self,
        windows: list[Window],
        *,
        P0: float,
        progress: Callable[[NoiseFitProgress], bool | None] | None,
        options: dict | None,
    ):
        """Minimize the exact summed NLL while keeping graph memory bounded.

        The optimization is a projected inverse-BFGS solve over the usually
        tiny log-sigma vector. Each objective/gradient evaluation streams the
        selected windows through reusable one-window CasADi functions. Peak
        symbolic memory therefore depends on the longest window, not on the
        number of windows in the capture.
        """
        opts = {
            "max_iterations": 80,
            "gradient_tolerance": 1e-5,
            "step_tolerance": 1e-8,
            "objective_tolerance": 1e-10,
            "max_line_search": 24,
        }
        opts.update(options or {})
        max_iterations = int(opts["max_iterations"])
        if max_iterations < 0:
            raise ValueError("batched noise-fit max_iterations must be non-negative")
        for name in ("gradient_tolerance", "step_tolerance", "objective_tolerance"):
            value = float(opts[name])
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"batched noise-fit {name} must be positive and finite")

        tan = self.sys.spec.tangent_dim
        prepared = []
        for window in windows:
            x0, U, Z, M, K = self._window_arrays(window)
            prepared.append((
                self._window_nll(K),
                ca.DM(x0),
                ca.DM((P0 * np.eye(tan)).reshape(-1, 1)),
                ca.DM(U) if U.size else ca.DM(0, K),
                ca.DM(Z),
                ca.DM(M),
                ca.DM(float(window.dt)),
                ca.DM(np.array([[window.t0 + i * window.dt for i in range(K)]])),
            ))

        means = np.concatenate([channel.prior_mean for channel in self.channels])
        sigmas = np.concatenate([channel.sigma for channel in self.channels])
        finite_prior = np.isfinite(sigmas)

        def evaluate(at: np.ndarray) -> tuple[float, np.ndarray]:
            point = np.asarray(at, dtype=float).reshape(-1, 1)
            objective = 0.0
            gradient = np.zeros(self.n_s, dtype=float)
            for function, *arguments in prepared:
                raw_objective, raw_gradient = function(point, *arguments)
                objective += float(raw_objective)
                gradient += np.asarray(raw_gradient, dtype=float).ravel()
            delta = np.asarray(at, dtype=float) - means
            if finite_prior.any():
                normalized = delta[finite_prior] / sigmas[finite_prior]
                objective += 0.5 * float(normalized @ normalized)
                gradient[finite_prior] += (
                    delta[finite_prior] / np.square(sigmas[finite_prior])
                )
            if not np.isfinite(objective) or not np.all(np.isfinite(gradient)):
                raise FloatingPointError("non-finite batched noise-fit objective/gradient")
            return objective, gradient

        value = np.concatenate([channel.init for channel in self.channels])
        lower = np.concatenate([channel.lower for channel in self.channels])
        upper = np.concatenate([channel.upper for channel in self.channels])
        objective, gradient = evaluate(value)
        initial_objective = objective
        best_value = value.copy()
        best_objective = objective
        inverse_hessian = np.eye(self.n_s) / max(1.0, np.linalg.norm(gradient, ord=np.inf))
        history = [objective]
        gradient_history = []
        step_history = []
        status = "Maximum_Iterations_Exceeded"
        success = False

        def projected_gradient(at: np.ndarray, raw: np.ndarray) -> np.ndarray:
            projected = raw.copy()
            tolerance = 1e-10
            projected[(at <= lower + tolerance) & (raw > 0.0)] = 0.0
            projected[(at >= upper - tolerance) & (raw < 0.0)] = 0.0
            return projected

        for iteration in range(max_iterations + 1):
            projected = projected_gradient(value, gradient)
            gradient_norm = float(np.linalg.norm(projected, ord=np.inf))
            gradient_history.append(gradient_norm)
            if progress is not None:
                keep_going = progress(NoiseFitProgress(
                    iteration=iteration,
                    objective=objective,
                    best_objective=best_objective,
                    initial_objective=initial_objective,
                    values={
                        channel.alias: float(np.exp(best_value[channel.offset]))
                        for channel in self.channels
                    },
                ))
                if keep_going is not None and not bool(keep_going):
                    status = "User_Requested_Stop"
                    break
            if gradient_norm <= float(opts["gradient_tolerance"]):
                status = "Solve_Succeeded"
                success = True
                break
            if iteration == max_iterations:
                break

            direction = -(inverse_hessian @ projected)
            direction[(value <= lower + 1e-10) & (direction < 0.0)] = 0.0
            direction[(value >= upper - 1e-10) & (direction > 0.0)] = 0.0
            if float(projected @ direction) >= 0.0:
                inverse_hessian = np.eye(self.n_s) / max(1.0, gradient_norm)
                direction = -(inverse_hessian @ projected)

            accepted = False
            candidate = value
            candidate_objective = objective
            candidate_gradient = gradient
            for attempt in range(int(opts["max_line_search"])):
                alpha = 0.5**attempt
                trial = np.clip(value + alpha * direction, lower, upper)
                delta = trial - value
                if not np.any(delta):
                    continue
                trial_objective, trial_gradient = evaluate(trial)
                if trial_objective <= objective + 1e-4 * float(gradient @ delta):
                    candidate = trial
                    candidate_objective = trial_objective
                    candidate_gradient = trial_gradient
                    accepted = True
                    break
            if not accepted:
                status = "Search_Direction_Becomes_Too_Small"
                break

            step = candidate - value
            gradient_change = candidate_gradient - gradient
            curvature = float(step @ gradient_change)
            if curvature > 1e-12 * max(
                1.0,
                np.linalg.norm(step) * np.linalg.norm(gradient_change),
            ):
                rho = 1.0 / curvature
                identity = np.eye(self.n_s)
                left = identity - rho * np.outer(step, gradient_change)
                inverse_hessian = (
                    left @ inverse_hessian @ left.T
                    + rho * np.outer(step, step)
                )
            else:
                inverse_hessian = np.eye(self.n_s) / max(
                    1.0, np.linalg.norm(candidate_gradient, ord=np.inf)
                )
            relative_change = abs(objective - candidate_objective) / max(
                1.0, abs(objective)
            )
            step_norm = float(np.linalg.norm(step, ord=np.inf))
            step_history.append(step_norm)
            value = candidate
            objective = candidate_objective
            gradient = candidate_gradient
            history.append(objective)
            if objective < best_objective:
                best_objective = objective
                best_value = value.copy()
            if (
                step_norm <= float(opts["step_tolerance"])
                and relative_change <= float(opts["objective_tolerance"])
            ):
                status = "Solve_Succeeded"
                success = True
                break

        # The fit normally has only a few log-sigma decisions. Central
        # differences of the streamed analytic gradient provide posterior
        # information without ever differentiating the complete capture twice.
        hessian = np.zeros((self.n_s, self.n_s), dtype=float)
        for index in range(self.n_s):
            step = 1e-5 * max(1.0, abs(float(best_value[index])))
            plus = best_value.copy()
            minus = best_value.copy()
            plus[index] += step
            minus[index] -= step
            hessian[:, index] = (evaluate(plus)[1] - evaluate(minus)[1]) / (2.0 * step)
        hessian = 0.5 * (hessian + hessian.T)
        stats = {
            "success": success,
            "return_status": status,
            "iter_count": len(gradient_history) - 1,
            "iterations": {
                "obj": history,
                "inf_du": gradient_history,
                "d_norm": step_history,
            },
            "solver": "batched_inverse_bfgs",
            "final_projected_gradient": gradient_history[-1],
            "window_graph_count": len(self._window_nll_cache),
        }
        return best_value, best_objective, stats, hessian

    def solve(self, windows: list[Window], *, P0: float = 1e-6,
              solver: str = "ipopt",
              verbose: bool = False,
              progress: Callable[[NoiseFitProgress], bool | None] | None = None,
              batched_options: dict | None = None,
              ipopt_options: dict | None = None,
              hessian_diagnostics: str = "finite-difference") -> NoiseFitResult:
        """Minimize the windows' total innovation NLL + prior over log-σ.

        Args:
            windows — recorded data; every chosen sensor needs a trace
                      in every window.
            P0      — initial tangent covariance per window, `P0 · I`.
                      Keep small when `x0` is trusted (synthetic truth);
                      grow it for estimator-seeded initial states.
        """
        if not windows:
            raise ValueError("NoiseFit.solve: needs at least one Window.")
        if solver not in ("ipopt", "batched"):
            raise ValueError("NoiseFit solver must be 'ipopt' or 'batched'")
        if hessian_diagnostics not in ("finite-difference", "exact"):
            raise ValueError(
                "NoiseFit hessian_diagnostics must be 'finite-difference' "
                "or 'exact'"
            )
        sys = self.sys
        spec = sys.spec
        if solver == "batched":
            if hessian_diagnostics == "exact":
                raise ValueError(
                    "batched NoiseFit intentionally avoids exact symbolic Hessians"
                )
            s_opt, objective, stats, hessian = self._solve_batched(
                windows,
                P0=P0,
                progress=progress,
                options=batched_options,
            )
            expanded = False
        else:
            tan = spec.tangent_dim
            s = ca.MX.sym("s", self.n_s, 1)
            total = ca.MX(0.0)
            for w in windows:
                x0, U, Z, M, K = self._window_arrays(w)
                fold = self._fold(K)
                res = fold(ca.DM(x0),
                           ca.DM((P0 * np.eye(tan)).reshape(-1, 1)),
                           ca.DM(U) if U.size else ca.DM(0, K),
                           ca.DM(Z),
                           ca.DM(M),
                           ca.repmat(s, 1, K),
                           ca.repmat(ca.DM(float(w.dt)), 1, K),
                           ca.DM(np.array([[w.t0 + i * w.dt
                                            for i in range(K)]])))
                total = total + ca.sum2(res[2])

            # ½‖(s − s̄)/σ‖² prior (skipped for flat-prior channels).
            total = total + prior_penalty(s, self.channels, weight=0.5)

            def emit_progress(
                iteration,
                _current,
                current_objective,
                best,
                best_objective,
                initial_objective,
            ):
                if progress is None:
                    return None
                return progress(NoiseFitProgress(
                    iteration=iteration,
                    objective=float(current_objective),
                    best_objective=float(best_objective),
                    initial_objective=float(initial_objective),
                    values={
                        channel.alias: float(np.exp(best[channel.offset]))
                        for channel in self.channels
                    },
                ))

            s_opt, objective, stats, expanded = solve_blocks_nlp(
                "noise_fit", s, total, self.channels,
                verbose=verbose, ipopt_options=ipopt_options,
                retain_best=True, progress=emit_progress)

        if solver == "ipopt" and hessian_diagnostics == "exact":
            # Exact symbolic second derivatives can be useful for small bench
            # problems, but duplicate a large folded EKF graph in production.
            H_fn = ca.Function("H", [s], [ca.hessian(total, s)[0]])
            H_fn = expand_or_none(H_fn) or H_fn
            hessian = np.asarray(ca.DM(H_fn(s_opt)))
        elif solver == "ipopt":
            # Noise fits usually have only a handful of log-sigma decisions.
            # Central differences of the analytic first derivative avoid
            # constructing the folded filter's symbolic second derivative.
            gradient_fn = ca.Function("noise_fit_gradient", [s], [ca.gradient(total, s)])
            gradient_fn = expand_or_none(gradient_fn) or gradient_fn
            hessian = np.zeros((self.n_s, self.n_s), dtype=float)
            for index in range(self.n_s):
                step = 1e-5 * max(1.0, abs(float(s_opt[index])))
                plus = np.asarray(s_opt, dtype=float).copy()
                minus = np.asarray(s_opt, dtype=float).copy()
                plus[index] += step
                minus[index] -= step
                g_plus = np.asarray(ca.DM(gradient_fn(plus))).ravel()
                g_minus = np.asarray(ca.DM(gradient_fn(minus))).ravel()
                hessian[:, index] = (g_plus - g_minus) / (2.0 * step)
            hessian = 0.5 * (hessian + hessian.T)

        res = NoiseFitResult(self.channels, s_opt, hessian, objective,
                             stats, self.world, self.sys.model.model_id,
                             self.sys.model.artifact_id,
                             self.sys.model.derivation)
        res.expanded = expanded
        res._training_digests = tuple(window_digest(w) for w in windows)
        prediction = (() if self.estimator is None else tuple(
            self.estimator.module().metadata.get("prediction_inputs", ())
        ))
        input_fields = getattr(sys, "input_fields", None)
        res._training_default_fills = tuple(sorted((
            fill
            for window, digest in zip(
                windows, res._training_digests, strict=True
            )
            for fill in default_fills_for_window(
                self.model_world,
                spec,
                window,
                dataset_role="training",
                window_digest=digest,
                input_names=sys.input_names,
                input_defaults=sys.u_defaults,
                input_fields=input_fields,
                recorded_inputs={
                    **window.u,
                    **{name: 0.0 for name in prediction},
                },
            )
        ), key=lambda fill: (
            fill.dataset_role, fill.window_digest, fill.source, fill.name
        )))
        return res

    # ------------------------------------------------------------------

    def _window_arrays(self, w: Window):
        """Pack one window: x0, U, Z and per-sensor availability M."""
        sys = self.sys
        sensor_fulls = list(sys.sensors)
        sources = ({} if self.estimator is None else
                   dict(self.estimator.module().metadata.get(
                       "measurement_sources", {})))
        prediction = (() if self.estimator is None else
                      tuple(self.estimator.module().metadata.get(
                          "prediction_inputs", ())))
        required = list(dict.fromkeys(
            [sources.get(full, full) for full in sensor_fulls] + list(prediction)))
        field_dims = {field.name: field.dim
                      for field in getattr(sys, "input_fields", ())}
        dims = {}
        for name in required:
            if name in field_dims:
                dims[name] = field_dims[name]
            else:
                dims[name] = sys.sensors[name].dim
        traces, K = resolve_traces(w.z, required, dims, who="NoiseFit")
        missing = set(required) - set(traces)
        if missing:
            raise ValueError(
                f"NoiseFit: window is missing trace(s) for "
                f"{sorted(missing)} — every measurement source and "
                f"prediction input needs a trace "
                f"(restrict with sensors=[...]).")
        trace_masks = resolve_trace_masks(
            w.z_mask, traces, required, K, who="NoiseFit"
        )
        for name in prediction:
            if not np.all(trace_masks[name]):
                raise ValueError(
                    f"NoiseFit: prediction input {name!r} cannot be masked; "
                    "the process transition needs one value at every base "
                    "step"
                )
        M = np.vstack([
            trace_masks[sources.get(full, full)] for full in sensor_fulls
        ]).astype(float)
        z_rows = []
        for row, full in enumerate(sensor_fulls):
            values = traces[sources.get(full, full)].T.copy()
            # A storage placeholder may be NaN. Do not let an inactive
            # branch poison CasADi's algebra even though its update is gated.
            values[:, M[row] == 0.0] = 0.0
            z_rows.append(values)
        Z = np.vstack(z_rows)
        x0 = pack_x0(self.model_world, sys.spec, w)
        recorded_inputs = dict(w.u)
        recorded_inputs.update({name: traces[name] for name in prediction})
        U = pack_u_trace(
            recorded_inputs, sys.input_names, sys.u_defaults, K,
            who="NoiseFit", input_fields=getattr(sys, "input_fields", None))
        return x0, U, Z, M, K

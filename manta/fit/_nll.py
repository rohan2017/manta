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
    result.apply()                  # writes <name>_sigma onto the parts

Typical workflow: fit dynamics/geometry first (`Fit`), `apply()` it,
then fit σ on the updated model — innovation statistics are only
meaningful once the mean model is right.
"""

from __future__ import annotations

import warnings

import casadi as ca
import numpy as np

from ..estimation._kalman import joseph_update
from ..ir._linalg import spd_logdet, spd_solve
from ..ir.state_spec import flatten_nested
from ..linearization import LinearizedSystem, resolve_suffix
from ._common import (
    Prior, Window, _FitBlock, convergence_line, format_table, laplace_sigma,
    pack_u_trace, pack_x0, prior_penalty, resolve_traces, solve_blocks_nlp,
    solver_converged,
)


class _Channel(_FitBlock):
    """One fitted noise channel: its slot in the tick's noise vector, its
    log-σ decision slot, and its prior. Always scalar (`dim == 1`), always
    log-space."""

    __slots__ = ("spec", "decl_name", "alias")

    def __init__(self, spec, offset: int, prior: Prior | None) -> None:
        self.spec = spec
        self.offset = offset
        self.dim = 1
        # The user-facing name is the DECLARATION name (`gyro_bias`),
        # not the driver-input name (`gyro_bias_driver`).
        self.decl_name = next(
            n for n, d in spec.owner.noise_declarations().items()
            if d.driver_input_name(n) == spec.name)
        self.alias = (spec.full[:-len("_driver")]
                      if spec.full.endswith("_driver") else spec.full)

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
        self.prior = np.array([np.log(mean)]) if mean > 0.0 else self.init
        self.sigma = np.array([np.inf if prior is None or prior.sigma is None
                               else float(prior.sigma)])
        if self.sigma[0] <= 0.0:
            raise ValueError(
                f"NoiseFit: Prior.sigma for {self.alias!r} must be "
                f"positive (relative, log-space).")


class NoiseFitResult:
    """Fitted σ per channel + Laplace posterior diagnostics.

    `prior_sigma` / `posterior_sigma` are RELATIVE (log-space) widths;
    posterior ≈ prior means the data didn't inform that σ. `converged`
    is IPOPT's success flag — False ⇒ the values are the failed solve's
    final iterate (a `RuntimeWarning` was emitted), not an optimum."""

    def __init__(self, channels, s_opt, hessian, objective, stats,
                 world) -> None:
        self._channels = channels
        self._world = world
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

    def apply(self) -> None:
        """Write the fitted σ back onto the owning parts
        (`<channel>_sigma` attributes); transforms built afterwards
        (an `EKF(world)`'s auto-Q/R, a `NoiseDriver`d truth sim) use
        them."""
        if not self.converged:
            warnings.warn(
                "NoiseFitResult.apply: the solve did NOT converge — "
                "writing the failed solve's final iterate onto the parts.",
                RuntimeWarning, stacklevel=2)
        for c in self._channels:
            setattr(c.spec.owner, f"{c.decl_name}_sigma",
                    self.values[c.alias])

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
    """

    def __init__(self, world, noise: dict, *,
                 sensors: list[str] | None = None) -> None:
        self.world = world
        self.sys = LinearizedSystem(world, sensors=sensors)
        sys = self.sys

        # Resolve requested channels against the tick's noise vector.
        aliases = []
        for spec in sys.noise_specs:
            aliases.append(spec.full[:-len("_driver")]
                           if spec.full.endswith("_driver") else spec.full)
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

            nll_step(x, Pv, u, z, s, dt, t) -> x⁺, Pv⁺, ½(νᵀS⁻¹ν+logdetS)

        Update with z (the reading produced by the step FROM x), then
        predict — matching the recorded-data convention. P rides
        flattened so `mapaccum` can accumulate it."""
        sys = self.sys
        spec = sys.spec
        tan = spec.tangent_dim
        n_u = len(sys.input_names)
        zdim = sum(s_.dim for s_ in sys.sensors.values())

        x_in = ca.MX.sym("x", spec.ambient_dim, 1)
        Pv = ca.MX.sym("Pv", tan * tan, 1)
        u = ca.MX.sym("u", n_u, 1) if n_u else ca.MX.zeros(0, 1)
        z = ca.MX.sym("z", zdim, 1)
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
        for full, sm in sys.sensors.items():
            zk = z[off:off + sm.dim]
            off += sm.dim
            sub = ca.vertcat(sys.x_sym, sys.u_sym, sys.dt_sym, sys.t_sym)
            vals = ca.vertcat(x, u, zero_dt, t)
            h = ca.substitute(ca.reshape(sm.h_sym, sm.dim, 1), sub, vals)
            H = ca.substitute(sm.H_sym, sub, vals)
            if sm.L_h_sym is not None:
                L_h = ca.substitute(sm.L_h_sym, sub, vals)
                R = L_h @ Sigma @ L_h.T
            else:
                R = ca.MX.zeros(sm.dim, sm.dim)
            x, P, nu, S = joseph_update(x, P, h, H, R, zk, spec)
            nll = nll + 0.5 * (ca.dot(nu, spd_solve(S, nu))
                               + spd_logdet(S))

        # ---- predict from the updated state ---------------------------
        sub = ca.vertcat(sys.x_sym, sys.u_sym, sys.dt_sym, sys.t_sym)
        vals = ca.vertcat(x, u, dt, t)
        x_new = ca.substitute(sys.x_new, sub, vals)
        F = ca.substitute(sys.F_sym, sub, vals)
        if sys.L_sym is not None:
            L = ca.substitute(sys.L_sym, sub, vals)
            Q = L @ Sigma @ L.T
        else:
            Q = ca.MX.zeros(tan, tan)
        P = F @ P @ F.T + Q
        P = 0.5 * (P + P.T)

        return ca.Function(
            "nll_step", [x_in, Pv, u, z, s, dt, t],
            [x_new, ca.reshape(P, tan * tan, 1), nll],
            ["x", "Pv", "u", "z", "s", "dt", "t"],
            ["x_new", "Pv_new", "nll"])

    def _validate_R0(self) -> None:
        """Every chosen sensor needs S > 0 from the first update: with
        all its white channels at σ=0 (and unfitted), R is exactly zero
        and a small P0 makes S singular. Probe at the starting σ — and,
        as in `EKF`, at a perturbed point too when R is state-dependent
        (a sampled check, not a proof)."""
        s0 = np.concatenate([c.init for c in self.channels]).reshape(-1, 1)
        # Direct numeric probe: evaluate each sensor's R at the start.
        sys = self.sys
        flat = flatten_nested(self.world._initial_state_dict())
        x0 = np.asarray(sys.spec.pack_any(flat), dtype=float)
        s_sym = ca.MX.sym("s", self.n_s, 1)
        for full, sm in sys.sensors.items():
            if sm.L_h_sym is None:
                continue
            R = ca.substitute(
                sm.L_h_sym, sys.dt_sym, ca.MX.zeros(1, 1)) \
                @ self._sigma_diag(s_sym) @ ca.substitute(
                    sm.L_h_sym, sys.dt_sym, ca.MX.zeros(1, 1)).T
            R_fn = ca.Function("R", [sys.x_sym, sys.u_sym, sys.t_sym,
                                     s_sym], [R])
            probes = [(x0, sys.u_defaults, 0.0)]
            if ca.depends_on(R, ca.vertcat(sys.x_sym, sys.u_sym,
                                           sys.t_sym)):
                x1 = sys.spec.boxplus_num(
                    x0.reshape(-1), 1e-3 * np.ones(sys.spec.tangent_dim))
                probes.append((x1, sys.u_defaults + 1e-3, 1.0))
            if all(not np.any(np.abs(np.diag(np.asarray(
                    ca.DM(R_fn(*p, s0))))) > 0.0) for p in probes):
                raise ValueError(
                    f"NoiseFit: sensor {full!r} has zero measurement "
                    f"noise at the starting σ — its first update is "
                    f"singular. Declare/fit a nonzero white-noise σ on "
                    f"it, or exclude it via sensors=[...].")

    # ------------------------------------------------------------------

    def _fold(self, K: int) -> ca.Function:
        if K not in self._fold_cache:
            self._fold_cache[K] = self._step_fn.mapaccum(
                f"nll_x{K}", K, [0, 1], [0, 1])
        return self._fold_cache[K]

    def solve(self, windows: list[Window], *, P0: float = 1e-6,
              verbose: bool = False,
              ipopt_options: dict | None = None) -> NoiseFitResult:
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
        sys = self.sys
        spec = sys.spec
        tan = spec.tangent_dim
        s = ca.MX.sym("s", self.n_s, 1)

        total = ca.MX(0.0)
        for w in windows:
            x0, U, Z, K = self._window_arrays(w)
            fold = self._fold(K)
            res = fold(ca.DM(x0),
                       ca.DM((P0 * np.eye(tan)).reshape(-1, 1)),
                       ca.DM(U) if U.size else ca.DM(0, K),
                       ca.DM(Z),
                       ca.repmat(s, 1, K),
                       ca.repmat(ca.DM(float(w.dt)), 1, K),
                       ca.DM(np.array([[w.t0 + i * w.dt
                                        for i in range(K)]])))
            total = total + ca.sum2(res[2])

        # ½‖(s − s̄)/σ‖² prior (skipped for flat-prior channels).
        total = total + prior_penalty(s, self.channels, weight=0.5)

        s_opt, objective, stats = solve_blocks_nlp(
            "noise_fit", s, total, self.channels,
            verbose=verbose, ipopt_options=ipopt_options)

        H_fn = ca.Function("H", [s], [ca.hessian(total, s)[0]])
        hessian = np.asarray(ca.DM(H_fn(s_opt)))

        return NoiseFitResult(self.channels, s_opt, hessian, objective,
                              stats, self.world)

    # ------------------------------------------------------------------

    def _window_arrays(self, w: Window):
        """Pack one window: x0 (ambient), U (n_u,K), Z (Σdims,K)."""
        sys = self.sys
        sensor_fulls = list(sys.sensors)
        dims = {full: sm.dim for full, sm in sys.sensors.items()}
        traces, K = resolve_traces(w.z, sensor_fulls, dims, who="NoiseFit")
        missing = set(sensor_fulls) - set(traces)
        if missing:
            raise ValueError(
                f"NoiseFit: window is missing trace(s) for "
                f"{sorted(missing)} — every chosen sensor needs a trace "
                f"(restrict with sensors=[...]).")
        Z = np.vstack([traces[f].T for f in sensor_fulls])
        x0 = pack_x0(self.world, sys.spec, w)
        U = pack_u_trace(w.u, sys.input_names, sys.u_defaults, K,
                         who="NoiseFit")
        return x0, U, Z, K

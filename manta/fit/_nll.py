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

import casadi as ca
import numpy as np

from ..ir._linalg import spd_logdet, spd_solve
from ..ir.state_spec import flatten_nested
from ..linearization import LinearizedSystem, resolve_suffix
from ._common import Prior, Window


class _Channel:
    """One fitted noise channel: its slice of the tick's noise vector,
    its log-σ decision slot, and its prior."""

    __slots__ = ("spec", "decl_name", "alias", "s_index",
                 "s_init", "s_prior", "s_sigma")

    def __init__(self, spec, s_index: int, prior: Prior | None) -> None:
        self.spec = spec
        self.s_index = s_index
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
        self.s_init = np.log(start)
        self.s_prior = np.log(mean) if mean > 0.0 else self.s_init
        self.s_sigma = (np.inf if prior is None or prior.sigma is None
                        else float(prior.sigma))
        if self.s_sigma <= 0.0:
            raise ValueError(
                f"NoiseFit: Prior.sigma for {self.alias!r} must be "
                f"positive (relative, log-space).")


class NoiseFitResult:
    """Fitted σ per channel + Laplace posterior diagnostics.

    `prior_sigma` / `posterior_sigma` are RELATIVE (log-space) widths;
    posterior ≈ prior means the data didn't inform that σ."""

    def __init__(self, channels, s_opt, hessian, objective, stats,
                 world) -> None:
        self._channels = channels
        self._world = world
        self.s = np.asarray(s_opt, dtype=float).ravel()
        self.objective = float(objective)
        self.stats = stats
        self.values = {c.alias: float(np.exp(self.s[c.s_index]))
                       for c in channels}
        self.labels = [c.alias for c in channels]
        self.prior_sigma = np.array([c.s_sigma for c in channels])
        try:
            cov = np.linalg.inv(hessian)
            self.posterior_sigma = np.sqrt(np.maximum(np.diag(cov), 0.0))
        except np.linalg.LinAlgError:
            self.posterior_sigma = np.full(len(channels), np.inf)

    def apply(self) -> None:
        """Write the fitted σ back onto the owning parts
        (`<channel>_sigma` attributes); transforms built afterwards
        (an `EKF(world)`'s auto-Q/R, a `NoiseDriver`d truth sim) use
        them."""
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
        widths = [max(len(r[c]) for r in rows) for c in range(5)]
        lines = ["  ".join(r[c].ljust(widths[c]) for c in range(5))
                 for r in rows]
        lines.insert(1, "  ".join("-" * w for w in widths))
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (f"<NoiseFitResult {len(self._channels)} channel(s), "
                f"objective={self.objective:.6g}>")


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
            var = (ca.exp(2.0 * s[c.s_index]) if c is not None
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
        eye = ca.MX.eye(tan)
        nll = ca.MX(0.0)

        # ---- sequential measurement updates at x (pre-step state) ----
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
            nu = zk - h
            S = H @ P @ H.T + R
            nll = nll + 0.5 * (ca.dot(nu, spd_solve(S, nu))
                               + spd_logdet(S, sm.dim))
            K = spd_solve(S, (P @ H.T).T).T
            x = spec.boxplus_sym(x, K @ nu)
            IKH = eye - K @ H
            P = IKH @ P @ IKH.T + K @ R @ K.T

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
        and a small P0 makes S singular. Probe at the starting σ."""
        s0 = np.array([c.s_init for c in self.channels]).reshape(-1, 1)
        # Direct numeric probe: evaluate each sensor's R at the start.
        sys = self.sys
        flat = flatten_nested(self.world._initial_state_dict())
        x0 = np.asarray(sys.spec.pack_any(flat), dtype=float)
        s_sym = ca.MX.sym("s", max(self.n_s, 1), 1)
        for full, sm in sys.sensors.items():
            if sm.L_h_sym is None:
                continue
            R = ca.substitute(
                sm.L_h_sym, sys.dt_sym, ca.MX.zeros(1, 1)) \
                @ self._sigma_diag(s_sym) @ ca.substitute(
                    sm.L_h_sym, sys.dt_sym, ca.MX.zeros(1, 1)).T
            R_fn = ca.Function("R", [sys.x_sym, sys.u_sym, sys.t_sym,
                                     s_sym], [R])
            R0 = np.asarray(ca.DM(R_fn(
                x0, sys.u_defaults, 0.0,
                s0 if self.n_s else np.zeros((1, 1)))))
            if not np.any(np.abs(np.diag(R0)) > 0.0):
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

        for c in self.channels:                    # ½‖(s − s̄)/σ‖² prior
            if np.isfinite(c.s_sigma):
                d = (s[c.s_index] - float(c.s_prior)) / float(c.s_sigma)
                total = total + 0.5 * d * d

        s0 = np.array([c.s_init for c in self.channels])
        opts = {"ipopt.print_level": 5 if verbose else 0,
                "print_time": verbose, "ipopt.sb": "yes"}
        opts.update(ipopt_options or {})
        solver = ca.nlpsol("noise_fit", "ipopt", {"x": s, "f": total}, opts)
        sol = solver(x0=s0)
        s_opt = np.asarray(sol["x"]).ravel()

        H_fn = ca.Function("H", [s], [ca.hessian(total, s)[0]])
        hessian = np.asarray(ca.DM(H_fn(s_opt)))

        return NoiseFitResult(self.channels, s_opt, hessian, sol["f"],
                              solver.stats(), self.world)

    # ------------------------------------------------------------------

    def _window_arrays(self, w: Window):
        """Pack one window: x0 (ambient), U (n_u,K), Z (Σdims,K)."""
        sys = self.sys
        sensor_fulls = list(sys.sensors)
        K = None
        traces = {}
        for key, arr in w.z.items():
            full = resolve_suffix(key, sensor_fulls, label="sensor",
                                  who="NoiseFit")
            a = np.asarray(arr, dtype=float)
            if a.ndim == 1:
                a = a.reshape(-1, 1)
            if K is None:
                K = a.shape[0]
            traces[full] = a
        missing = set(sensor_fulls) - set(traces)
        if missing:
            raise ValueError(
                f"NoiseFit: window is missing trace(s) for "
                f"{sorted(missing)} — every chosen sensor needs a trace "
                f"(restrict with sensors=[...]).")
        for full, a in traces.items():
            dim = sys.sensors[full].dim
            if a.shape != (K, dim):
                raise ValueError(
                    f"NoiseFit: z[{full!r}] expected ({K}, {dim}), got "
                    f"{a.shape}.")
        Z = np.vstack([traces[f].T for f in sensor_fulls])

        flat = flatten_nested(self.world._initial_state_dict())
        flat.update(flatten_nested(w.x0))
        x0 = np.asarray(sys.spec.pack_any(flat),
                        dtype=float).reshape(-1, 1)

        U = np.tile(sys.u_defaults.reshape(-1, 1), (1, K)) \
            if sys.input_names else np.zeros((0, K))
        for key, val in w.u.items():
            full = resolve_suffix(key, sys.input_names, label="input",
                                  who="NoiseFit")
            row = sys.input_names.index(full)
            a = np.asarray(val, dtype=float).ravel()
            U[row, :] = a[0] if a.size == 1 else a
        return x0, U, Z, K

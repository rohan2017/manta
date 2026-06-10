"""Fit — system identification over a World's promotable Parameters.

`Fit(world, parameters={...})` promotes the named Parameters (thruster
gains, mount transforms, masses — any `Parameter` declared with a
manifold) to a live parameter vector via `Sim(world, parameters=[...])`,
then fits them to recorded data by windowed prediction error:

    L(v) = Σ_windows Σ_sensors w_s · ‖predicted(v) − measured‖²
         + Σ_params ‖(v − v_prior) / σ_prior‖²                 (MAP)

Each `Window` is a short rollout: a known initial state, the recorded
control trace, and the recorded sensor traces. The predicted readings
come from folding the oracle `step` kernel over the window (CasADi
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

Usage::

    from manta.fit import Fit, Prior, Window

    fit = Fit(world, parameters={
        "t_fl.force_quad": Prior(sigma=3.0),
        "body.mass":       Prior(sigma=0.05, log=True),   # ±5%
        "imu.transform":   Prior(sigma=0.02),             # ±2 cm
    })
    result = fit.solve(windows)
    print(result.summary())
    result.apply()          # write fitted values back onto the parts;
                            # a fresh Sim(world) bakes them in.

Initial states: each window needs `x0`. For synthetic-recoverability
runs, capture `sim.state` from the truth sim. For real logs without
ground truth, seed from an estimator's output, or extend this with
free per-window initial states (multiple shooting) — not built yet.

Sensor-noise σ values are NOT fittable here: a mean-prediction L2 loss
has zero gradient in σ. Fit them with `NoiseFit` (`_nll.py`) — the
EKF-innovation-likelihood fitter — after applying this fit's result.
"""

from __future__ import annotations

import casadi as ca
import numpy as np

from ..ir.module import Role
from ..ir.state_spec import flatten_nested
from ..linearization import resolve_suffix
from ..sim import Sim
from ._common import Prior, Window


# ---------------------------------------------------------------------------
# Internal: one decision-space block per promoted parameter
# ---------------------------------------------------------------------------

class _Block:
    """One parameter's slice of the decision vector `v` and its prior."""

    __slots__ = ("full", "dim", "offset", "declared", "log",
                 "v_init", "v_prior", "v_sigma")

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
            if dim != 1:
                raise ValueError(
                    f"Prior for {full!r}: log=True needs a scalar "
                    f"parameter (got dim={dim}).")
            if declared[0] <= 0.0 or mean[0] <= 0.0:
                raise ValueError(
                    f"Prior for {full!r}: log=True needs a strictly "
                    f"positive declared value and mean.")
            self.v_init = np.log(declared)
            self.v_prior = np.log(mean)
        else:
            self.v_init = declared.copy()
            self.v_prior = mean.copy()

        if prior is None or prior.sigma is None:
            self.v_sigma = np.full(dim, np.inf)
        else:
            sig = np.atleast_1d(np.asarray(prior.sigma, dtype=float)).ravel()
            if sig.size == 1:
                sig = np.full(dim, sig[0])
            if sig.size != dim or np.any(sig <= 0.0):
                raise ValueError(
                    f"Prior for {full!r}: sigma must be a positive scalar "
                    f"or length-{dim} sequence, got {prior.sigma!r}.")
            self.v_sigma = sig

    def p_of_v(self, v_blk: ca.MX) -> ca.MX:
        """Decision slice → parameter values (ambient)."""
        return ca.exp(v_blk) if self.log else v_blk

    def theta_of_v(self, v_blk: np.ndarray) -> np.ndarray:
        return np.exp(v_blk) if self.log else np.asarray(v_blk)

    def labels(self) -> list[str]:
        if self.dim == 1:
            return [self.full]
        return [f"{self.full}[{i}]" for i in range(self.dim)]


# ---------------------------------------------------------------------------
# FitResult
# ---------------------------------------------------------------------------

class FitResult:
    """Fitted values + Gauss-Newton posterior diagnostics.

    Attributes:
        values          — `{full param name: fitted value}` (float for
                          scalars, ndarray for vectors).
        labels          — one entry per fitted scalar component.
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
    """

    def __init__(self, blocks, v_opt, JtJ, objective, stats, world) -> None:
        self._blocks = blocks
        self._world = world
        self.v = np.asarray(v_opt, dtype=float).ravel()
        self.JtJ = JtJ
        self.objective = float(objective)
        self.stats = stats

        self.values: dict[str, object] = {}
        self.labels: list[str] = []
        self.log_scale: list[bool] = []
        prior_sig, post_var = [], None
        for b in blocks:
            theta = b.theta_of_v(self.v[b.offset:b.offset + b.dim])
            self.values[b.full] = float(theta[0]) if b.dim == 1 else theta
            self.labels += b.labels()
            self.log_scale += [b.log] * b.dim
            prior_sig.append(b.v_sigma)
        self.prior_sigma = np.concatenate(prior_sig)

        prior_prec = np.where(np.isinf(self.prior_sigma), 0.0,
                              1.0 / np.square(self.prior_sigma))
        H = JtJ + np.diag(prior_prec)
        try:
            self.posterior_sigma = np.sqrt(np.diag(np.linalg.inv(H)))
        except np.linalg.LinAlgError:
            # Singular H: flat-prior components the data never touched.
            self.posterior_sigma = np.full(len(self.labels), np.inf)

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
        """Write the fitted values back onto the world's Part instances.
        A transform built afterwards (`Sim(world)`, `EKF(world)`, a C++
        deploy) bakes them in as constants."""
        for b in self._blocks:
            craft_name, part_name, pname = b.full.split(".", 2)
            craft = next(c for c in self._world.crafts
                         if c.name == craft_name)
            part = next(p for p in craft.parts if p.name == part_name)
            theta = b.theta_of_v(self.v[b.offset:b.offset + b.dim])
            setattr(part, pname,
                    float(theta[0]) if b.dim == 1 else tuple(theta))

    def summary(self) -> str:
        """Per-component table: fitted value, prior σ vs posterior σ.
        `post/prior ≈ 1` flags a component the data did not inform —
        its fitted value is your prior talking, not the flight."""
        rows = [("parameter", "fitted", "prior σ", "post σ", "post/prior")]
        i = 0
        for b in self._blocks:
            theta = b.theta_of_v(self.v[b.offset:b.offset + b.dim])
            for j, lbl in enumerate(b.labels()):
                pri, post = self.prior_sigma[i], self.posterior_sigma[i]
                ratio = ("—" if np.isinf(pri) or np.isinf(post)
                         else f"{post / pri:.3f}")
                unit = " (rel)" if b.log else ""
                rows.append((lbl, f"{theta[j]:.6g}",
                             ("inf" if np.isinf(pri)
                              else f"{pri:.3g}{unit}"),
                             ("inf" if np.isinf(post)
                              else f"{post:.3g}{unit}"),
                             ratio))
                i += 1
        widths = [max(len(r[c]) for r in rows) for c in range(5)]
        lines = ["  ".join(r[c].ljust(widths[c]) for c in range(5))
                 for r in rows]
        lines.insert(1, "  ".join("-" * w for w in widths))
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (f"<FitResult {len(self._blocks)} parameter(s), "
                f"objective={self.objective:.6g}>")


# ---------------------------------------------------------------------------
# Fit
# ---------------------------------------------------------------------------

class Fit:
    """MAP parameter fit over recorded windows (see module docstring).

    Args:
        world      — the model. Finalized by the internal `Sim`; the
                     fit never mutates it (until `result.apply()`).
        parameters — `{param name/suffix: Prior | None}`. Names resolve
                     against the model's promotable Parameters
                     (`<craft>.<part>.<param>`); `None` = flat prior
                     (only safe for parameters the data fully observes).
    """

    def __init__(self, world, parameters: dict) -> None:
        self.world = world
        self.sim = Sim(world, parameters=list(parameters))
        self.module = self.sim.module()
        self._spec = self.module.spec
        port = self.module.port("params")
        fulls = [f.name for f in port.fields]
        by_full = {resolve_suffix(k, fulls, label="parameter", who="Fit"): v
                   for k, v in parameters.items()}

        self._blocks: list[_Block] = []
        off = 0
        for f in port.fields:           # port order == kernel `p` layout
            self._blocks.append(_Block(
                f.name, f.dim, off,
                np.asarray(f.default, dtype=float).ravel(),
                by_full.get(f.name)))
            off += f.dim
        self.n_v = off
        self._stepk_cache: dict[int, ca.Function] = {}

    # ------------------------------------------------------------------

    def solve(self, windows: list[Window], *, weights: dict | None = None,
              verbose: bool = False,
              ipopt_options: dict | None = None) -> FitResult:
        """Build the windowed prediction-error + prior NLP and solve it.

        Args:
            windows — the recorded data (≥ 1 `Window`).
            weights — optional per-sensor scalar weights on the squared
                      residuals (`{sensor name/suffix: w}`); use
                      `1/σ_meas²` to whiten mixed-unit sensors. Default 1.
            verbose — IPOPT iteration output.
            ipopt_options — extra `nlpsol` options, merged last.
        """
        if not windows:
            raise ValueError("Fit.solve: needs at least one Window.")

        v = ca.MX.sym("v", self.n_v, 1)
        p = ca.vertcat(*[b.p_of_v(v[b.offset:b.offset + b.dim])
                         for b in self._blocks])

        meas_names = [pt.name for pt in
                      self.module.ports_by_role(Role.MEASUREMENT)]
        w_by_full: dict[str, float] = {}
        for k, val in (weights or {}).items():
            full = resolve_suffix(k, meas_names, label="sensor", who="Fit")
            w_by_full[full] = float(val)

        loss = ca.MX(0.0)
        residuals: list[ca.MX] = []
        for w in windows:
            loss, residuals = self._add_window(
                w, p, loss, residuals, w_by_full, meas_names)

        # MAP prior term (skipped for flat-prior components).
        for b in self._blocks:
            finite = np.isfinite(b.v_sigma)
            if not finite.any():
                continue
            for j in np.flatnonzero(finite):
                d = (v[b.offset + j] - float(b.v_prior[j])) \
                    / float(b.v_sigma[j])
                loss = loss + d * d

        v0 = np.concatenate([b.v_init for b in self._blocks])
        opts = {"ipopt.print_level": 5 if verbose else 0,
                "print_time": verbose, "ipopt.sb": "yes"}
        opts.update(ipopt_options or {})
        solver = ca.nlpsol("fit", "ipopt", {"x": v, "f": loss}, opts)
        sol = solver(x0=v0)
        v_opt = np.asarray(sol["x"]).ravel()

        # Data-only Gauss-Newton information JᵀJ at the solution.
        r = ca.vertcat(*residuals) if residuals else ca.MX.zeros(0, 1)
        J_fn = ca.Function("J", [v], [ca.jacobian(r, v)])
        J = np.asarray(ca.DM(J_fn(v_opt)))
        JtJ = J.T @ J

        return FitResult(self._blocks, v_opt, JtJ, sol["f"],
                         solver.stats(), self.world)

    # ------------------------------------------------------------------

    def _stepk(self, K: int) -> ca.Function:
        if K not in self._stepk_cache:
            # accumulate output 0 (x_new) -> input 0 (x); everything else
            # is a per-substep column.
            self._stepk_cache[K] = self.module.functions["step"].mapaccum(
                f"step_x{K}", K, [0], [0])
        return self._stepk_cache[K]

    def _add_window(self, w: Window, p: ca.MX, loss, residuals,
                    w_by_full: dict, meas_names: list[str]):
        """Append one window's prediction-error terms to the loss."""
        ep = self.module.entry("step")
        u_fields = self.module.port("u").fields
        n_noise = self.module.port("noise").size

        # Window length from the measurement traces.
        z_resolved: dict[str, np.ndarray] = {}
        K = None
        for k, arr in w.z.items():
            full = resolve_suffix(k, meas_names, label="sensor", who="Fit")
            dim = self.module.port(full).size
            a = np.asarray(arr, dtype=float)
            if a.ndim == 1:
                a = a.reshape(-1, 1)
            if a.shape[1] != dim:
                raise ValueError(
                    f"Window.z[{k!r}]: expected (K, {dim}), got {a.shape}.")
            if K is None:
                K = a.shape[0]
            elif a.shape[0] != K:
                raise ValueError(
                    f"Window.z[{k!r}]: trace length {a.shape[0]} != {K}.")
            z_resolved[full] = a
        if not z_resolved:
            raise ValueError("Window.z: needs at least one sensor trace.")

        # Initial state: window slots over the world's initial state.
        flat = flatten_nested(self.world._initial_state_dict())
        flat.update(flatten_nested(w.x0))
        x0 = np.asarray(self._spec.pack_any(flat), dtype=float).reshape(-1, 1)

        # Control trace (n_u, K): recorded columns, defaults elsewhere.
        u_names = [f.name for f in u_fields]
        U = np.tile(np.array([[float(np.asarray(f.default).ravel()[0])]
                              for f in u_fields]), (1, K)) \
            if u_fields else np.zeros((0, K))
        for k, val in w.u.items():
            full = resolve_suffix(k, u_names, label="input", who="Fit")
            row = u_names.index(full)
            a = np.asarray(val, dtype=float).ravel()
            if a.size == 1:
                U[row, :] = a[0]
            elif a.size == K:
                U[row, :] = a
            else:
                raise ValueError(
                    f"Window.u[{k!r}]: expected scalar or length-{K} "
                    f"trace, got {a.size}.")

        call_args = {"x": ca.DM(x0),
                     "u": ca.DM(U) if U.size else ca.DM(0, K),
                     "noise": (ca.DM.zeros(n_noise, K) if n_noise
                               else ca.DM(0, K)),
                     "params": ca.repmat(p, 1, K),
                     "dt": ca.repmat(ca.DM(float(w.dt)), 1, K),
                     "t": ca.DM(np.array([[w.t0 + i * w.dt
                                           for i in range(K)]]))}
        # ep.args: StateRef("x") then PortRefs by name — gather in order.
        ordered = []
        for a in ep.args:
            key = getattr(a, "name", "x")
            ordered.append(call_args[key if key in call_args else "x"])
        res = self._stepk(K)(*ordered)
        outs = [res] if not isinstance(res, (list, tuple)) else list(res)

        for full, Z in z_resolved.items():
            idx = 1 + ep.returns.index(full)        # 0 is x_new
            pred = outs[idx]                        # (dim, K)
            resid = pred - ca.DM(Z.T)
            wgt = w_by_full.get(full, 1.0)
            loss = loss + wgt * ca.sumsqr(resid)
            residuals.append(np.sqrt(wgt) * ca.reshape(resid, -1, 1))
        return loss, residuals

"""Linearization — manifold-aware Jacobians of a compiled world tick.

`Linearization` is the shared model-analysis transform that sits between
the forward dynamics IR (the world tick) and every model-based
estimator/controller. Given a compiled tick + a `StateSpec` (full or a
subset) + the input/noise routing, it produces the symbolic Jacobian
artifacts those algorithms need:

  * predict   f(x, u, dt, t)            — next ambient state, noise zeroed
  * F = ∂δ_out/∂δ_in |₀                 — tangent state-transition
  * B = ∂δ_out/∂u |₀                    — control Jacobian   [if control=]
  * L = ∂δ_out/∂noise |₀                — process-noise gain        [if noise]
  * per output: h, H = ∂h/∂δ |₀, L_h = ∂h/∂noise |₀

The Jacobians are emitted as symbolic `ca.Function`s of `(x, u, dt, t)` —
NOT matrices frozen at one operating point. Each consumer evaluates them
where it needs to: the EKF at the runtime estimate (F re-evaluated every
step), an infinite-horizon LQR at an equilibrium (one-shot), an iLQR
along a trajectory (many points). That is what lets the EKF read like
plain Kalman equations and lets LQR/iLQR reuse the very same engine.

It is a *pure IR transform*: it consumes the compiled CasADi function and
precomputed routing metadata, never Python `World`/`Craft`/`Part`
objects. The caller (e.g. `EKF`) does the model introspection — walking
the world to discover which tick inputs are live Inputs vs. Noise
channels, which outputs are sensors — and hands the result here. That
keeps this layer backend-agnostic and reusable across the numpy EKF, the
C++ extractor, and future controllers.

The manifold awareness lives entirely in `spec.boxplus_sym` /
`boxminus_sym`: the standard error-state pattern

    δ_out = boxminus(f(boxplus(x, δ_in)), f(x))
    F     = ∂δ_out/∂δ_in  at δ_in = 0

composes plain CasADi expressions, so SO(3) (and any future manifold)
linearizes correctly with no special-case code here.
"""

from __future__ import annotations

from dataclasses import dataclass

import casadi as ca
import numpy as np


def slot_of_tan(spec) -> list[str]:
    """Map each tangent index to the name of the slot it belongs to.

    Slot granularity (rather than scalar) keeps composite manifolds like
    an SO(3) orientation atomic for dependency-closure / partitioning.
    """
    m: list[str] = [""] * spec.tangent_dim
    for s in spec.slots:
        for i in range(s.tangent_offset, s.tangent_offset + s.tangent_dim):
            m[i] = s.name
    return m


@dataclass
class OutputLinearization:
    """Linearization of one tick output (a candidate measurement).

    `observed_cols` are the tangent-state columns with structural
    nonzeros in H — the slots this output actually observes. `h_sym`
    (noise-zeroed h(x)) and `H_sym` (∂h/∂δ|₀) are the raw symbolic
    expressions, always populated — a consumer that wants its own naming
    or densification (e.g. the C++ extractor) builds functions from
    these. The `ca.Function`s are convenience wrappers, `None` when the
    parent `Linearization` was built structure-only
    (`build_functions=False`)."""
    full:          str
    dim:           int
    observed_cols: np.ndarray
    h_sym:         ca.MX
    H_sym:         ca.MX
    L_h_sym:       ca.MX | None        # ∂h/∂noise|₀ (None if no noise)
    h_fn:          ca.Function | None
    H_fn:          ca.Function | None
    L_h_fn:        ca.Function | None
    h_noisy_sym:   ca.MX | None = None  # h with noise kept LIVE (=h_sym at n=0)


class Linearization:
    """Manifold-aware linearization of a compiled world tick.

    Args:
        cf            — the compiled tick `casadi.Function` (flat-prefixed
                        `<owner>.<sub>` inputs, plus `dt`/`t`).
        spec          — the `StateSpec` to linearize over (full or subset).
        frozen        — `{tick_input_name: value}` baked in as constants
                        rather than treated as live variables (untracked
                        state slots + excluded inputs). F/L/h/H are born
                        reduced to the kept dims, no post-hoc deletion.
        input_names   — ordered live Input names → the `u` vector.
        noise_specs   — the Noise channels feeding the tick (drives
                        `L`/`Σ`). Each is a `NoiseChannel`-like object
                        with `.full` / `.dim` / `.sigma`
                        (`manta.tick_signature`).
        outputs       — tick-output full names to linearize as `h`/`H`.
        build_functions — when False, only the structural artifacts
                        (`F_sym`, per-output `observed_cols`,
                        `observed_slots`) are produced; the `ca.Function`s
                        and `blocks` are skipped. Used for the cheap
                        full-spec pass that seeds state subsetting.
    """

    def __init__(self, cf, spec, *,
                 frozen: dict,
                 input_names: list[str],
                 noise_specs: list[dict],
                 outputs: list[str],
                 control: bool = False,
                 build_functions: bool = True) -> None:
        self.spec          = spec
        self._cf           = cf
        self._input_names  = input_names
        self._noise_specs  = noise_specs
        self.n_noise       = sum(s.dim for s in noise_specs)

        n_ambient = spec.ambient_dim
        n_tangent = spec.tangent_dim
        n_u       = len(input_names)

        x_sym  = ca.MX.sym("x",  n_ambient, 1)
        u_sym  = ca.MX.sym("u", n_u, 1) if n_u > 0 else ca.MX.zeros(0, 1)
        dt_sym = ca.MX.sym("dt", 1, 1)
        t_sym  = ca.MX.sym("t",  1, 1)
        n_sym  = (ca.MX.sym("noise", self.n_noise, 1) if self.n_noise > 0
                  else ca.MX.zeros(0, 1))
        zero_n = ca.MX.zeros(self.n_noise, 1)
        self.x_sym, self.u_sym, self.n_sym = x_sym, u_sym, n_sym
        self.dt_sym, self.t_sym = dt_sym, t_sym

        args, argn = [x_sym, u_sym, dt_sym, t_sym], ["x", "u", "dt", "t"]

        # --- predict + F ----------------------------------------------------
        outputs_n = self._tick_outputs(x_sym, frozen, u_sym, dt_sym, t_sym,
                                       n_sym)
        x_new_n = self._gather_state(outputs_n)
        x_new_0 = ca.substitute(x_new_n, n_sym, zero_n)

        delta_in = ca.MX.sym("delta_in", n_tangent, 1)
        x_pert   = spec.boxplus_sym(x_sym, delta_in)
        outputs_pert = self._tick_outputs(x_pert, frozen, u_sym, dt_sym,
                                          t_sym, zero_n)
        x_pert_new = self._gather_state(outputs_pert)
        delta_out  = spec.boxminus_sym(x_pert_new, x_new_0)
        self.x_new = x_new_0          # predict expr (noise-zeroed)
        # Noise-LIVE next state (noise kept symbolic). A driven Sim runtime
        # evaluates this with sampled noise; the noiseless predict is just
        # this at noise=0 (they coincide when the model has no noise).
        self.x_new_noisy = x_new_n
        self.F_sym = ca.substitute(
            ca.jacobian(delta_out, delta_in),
            delta_in, ca.MX.zeros(n_tangent, 1))

        # --- control Jacobian B = ∂f/∂u (tangent × n_u) -------------------
        # Same error-state recipe as F, but perturbing the (Euclidean)
        # input u instead of the state: δ(x_new) for a δu nudge, in tangent
        # coords. Opt-in (LQR/iLQR/MPC need it; the EKF + codegen don't).
        self.B_sym = None
        if control and n_u > 0:
            delta_u = ca.MX.sym("delta_u", n_u, 1)
            outputs_pu = self._tick_outputs(
                x_sym, frozen, u_sym + delta_u, dt_sym, t_sym, zero_n)
            x_pu_new = self._gather_state(outputs_pu)
            delta_out_u = spec.boxminus_sym(x_pu_new, x_new_0)
            self.B_sym = ca.substitute(
                ca.jacobian(delta_out_u, delta_u),
                delta_u, ca.MX.zeros(n_u, 1))

        self.predict_fn = None
        self.F_fn       = None
        self.B_fn       = None
        self.L_fn       = None
        self.L_sym      = None       # ∂δ_out/∂noise|₀ expr (None if no noise)
        self.Sigma      = None
        self.blocks: list = []
        self._L_pattern = None

        # L_sym (noise gain) + Σ — symbolic, always built when there is
        # noise (the Kalman recursion in `kalman_functions` needs the
        # expressions, not just the convenience ca.Functions).
        if self.n_noise > 0:
            delta_out_n = spec.boxminus_sym(x_new_n, x_new_0)
            self.L_sym = ca.substitute(
                ca.jacobian(delta_out_n, n_sym), n_sym, zero_n)
            self._L_pattern = np.array(ca.DM(self.L_sym.sparsity()))
            sigmas_sq: list[float] = []
            for ns in noise_specs:
                sigmas_sq.extend([ns.sigma ** 2] * ns.dim)
            self.Sigma = np.diag(sigmas_sq)

        if build_functions:
            self.predict_fn = ca.Function(
                "predict", args, [x_new_0], argn, ["x_new"])
            self.F_fn = ca.Function("F", args, [self.F_sym], argn, ["F"])
            if self.B_sym is not None:
                self.B_fn = ca.Function("B", args, [self.B_sym], argn, ["B"])
            if self.L_sym is not None:
                self.L_fn = ca.Function("L", args, [self.L_sym], argn, ["L"])

        # --- per-output h / H / L_h ----------------------------------------
        self.outputs: dict[str, OutputLinearization] = {}
        self._h_supports: list[np.ndarray] = []
        for full in outputs:
            if full not in outputs_n:
                raise RuntimeError(
                    f"Linearization: tick is missing output {full!r}.")
            h_dim       = int(outputs_n[full].numel())
            h_n_flat    = ca.reshape(outputs_n[full], h_dim, 1)
            h_pert_flat = ca.reshape(outputs_pert[full], h_dim, 1)
            H_sym = ca.substitute(
                ca.jacobian(h_pert_flat, delta_in),
                delta_in, ca.MX.zeros(n_tangent, 1))
            cols = np.flatnonzero(
                np.array(ca.DM(H_sym.sparsity())).any(axis=0))
            self._h_supports.append(cols)
            h_sym = ca.substitute(h_n_flat, n_sym, zero_n)   # noise-zeroed h(x)

            # L_h (∂h/∂noise) — symbolic, built whenever there is noise.
            L_h_sym = None
            if self.n_noise > 0:
                L_h_sym = ca.substitute(
                    ca.jacobian(h_n_flat, n_sym), n_sym, zero_n)

            h_fn = H_fn = L_h_fn = None
            if build_functions:
                safe = full.replace(".", "_")
                h_fn = ca.Function(f"h_{safe}", args, [h_sym], argn, ["h"])
                H_fn = ca.Function(f"H_{safe}", args, [H_sym], argn, ["H"])
                if L_h_sym is not None:
                    L_h_fn = ca.Function(
                        f"Lh_{safe}", args, [L_h_sym], argn, ["L_h"])
            self.outputs[full] = OutputLinearization(
                full=full, dim=h_dim, observed_cols=cols,
                h_sym=h_sym, H_sym=H_sym, L_h_sym=L_h_sym,
                h_fn=h_fn, H_fn=H_fn, L_h_fn=L_h_fn,
                h_noisy_sym=h_n_flat)      # noise-live reading (=h_sym at n=0)

        if build_functions:
            self.blocks = self._compute_blocks(n_tangent)

    # ------------------------------------------------------------------
    # Structural queries (available without build_functions)
    # ------------------------------------------------------------------

    @property
    def observed_slots(self) -> set[str]:
        """Slot names with a structural nonzero in any output's H — the
        seed set for dependency-closure-based state subsetting."""
        sot = slot_of_tan(self.spec)
        out: set[str] = set()
        for cols in self._h_supports:
            for j in cols:
                out.add(sot[int(j)])
        return out

    def dependency_closure(self, seed_slots) -> set[str]:
        """Expand `seed_slots` to the set closed under the dynamics `F`.

        Backward reachability on the structural dependency graph:
        `F_sym[i, j] != 0` (structurally) means slot-row i's next value
        depends on slot-col j, so keeping i requires keeping j. Iterate from
        `seed_slots` to a fixpoint over the slot-level graph (slot
        granularity keeps an SO(3) orientation atomic). Both the EKF
        (`track=`) and LQR use this, so a subset is always self-consistent —
        a tracked slot's dynamics never silently depend on a frozen one.
        """
        sot = slot_of_tan(self.spec)
        pattern = np.array(ca.DM(self.F_sym.sparsity()))
        rows, cols = np.nonzero(pattern)
        deps: dict[str, set[str]] = {}
        for i, j in zip(rows.tolist(), cols.tolist()):
            a, b = sot[i], sot[j]
            if a != b:
                deps.setdefault(a, set()).add(b)
        kept = set(seed_slots)
        frontier = list(seed_slots)
        while frontier:
            s = frontier.pop()
            for dep in deps.get(s, ()):
                if dep not in kept:
                    kept.add(dep)
                    frontier.append(dep)
        return kept

    def kalman_functions(self):
        """Build the full symbolic EKF recursion as `ca.Function`s — the
        logic that otherwise lives hand-written per backend.

        Returns `(predict_fn, process_noise_fn, updates)`:

          * `predict_fn(x, P, Q, u, dt, t) -> (x_new, P_new)`
            with `P_new = F P Fᵀ + Q`.
          * `process_noise_fn(x, u, dt, t) -> Q` = `L Σ Lᵀ`, or `None` when
            the model declares no process noise (caller supplies Q).
          * `updates[full](x, P, z, u, t) -> (x_new, P_new)` — the Joseph
            update for one sensor: `R = L_h Σ L_hᵀ`, gain via the
            codegen-safe `ca.solve(S, ·, "ldl")` (SPD `S`), correction via
            `spec.boxplus_sym` (so SO(3) etc. stay manifold-correct).

        Everything is one fused graph per function, so CasADi CSE shares
        `f`/`F` and the backends — numpy or emitted C — just evaluate it.
        """
        spec = self.spec
        n_tan = spec.tangent_dim
        x, u = self.x_sym, self.u_sym
        dt, t = self.dt_sym, self.t_sym
        P = ca.MX.sym("P", n_tan, n_tan)
        Q = ca.MX.sym("Q", n_tan, n_tan)
        Sig = ca.DM(self.Sigma) if self.Sigma is not None else None

        # predict:  x' = f(x,u,dt,t);  P' = F P Fᵀ + Q
        P_pred = self.F_sym @ P @ self.F_sym.T + Q
        P_pred = 0.5 * (P_pred + P_pred.T)
        predict_fn = ca.Function(
            "ekf_predict", [x, P, Q, u, dt, t], [self.x_new, P_pred],
            ["x", "P", "Q", "u", "dt", "t"], ["x_new", "P_new"])

        process_noise_fn = None
        if self.L_sym is not None and Sig is not None:
            Qexpr = self.L_sym @ Sig @ self.L_sym.T
            process_noise_fn = ca.Function(
                "ekf_process_noise", [x, u, dt, t], [Qexpr],
                ["x", "u", "dt", "t"], ["Q"])

        # per-sensor Joseph update (h/H/L_h evaluated at dt=0, as a
        # measurement is dt-independent — matches the numpy reference).
        eye = ca.MX.eye(n_tan)
        zero_dt = ca.MX.zeros(1, 1)
        updates: dict[str, ca.Function] = {}
        for full, o in self.outputs.items():
            d = o.dim
            z = ca.MX.sym("z", d)
            h = ca.substitute(o.h_sym, dt, zero_dt)
            H = ca.substitute(o.H_sym, dt, zero_dt)
            if o.L_h_sym is not None and Sig is not None:
                L_h = ca.substitute(o.L_h_sym, dt, zero_dt)
                R = L_h @ Sig @ L_h.T
            else:
                R = ca.MX.zeros(d, d)
            S = H @ P @ H.T + R
            K = ca.solve(S, (P @ H.T).T, "ldl").T      # P Hᵀ S⁻¹  (S SPD)
            x_upd = spec.boxplus_sym(x, K @ (z - h))
            IKH = eye - K @ H
            P_upd = IKH @ P @ IKH.T + K @ R @ K.T
            P_upd = 0.5 * (P_upd + P_upd.T)
            safe = full.replace(".", "_")
            updates[full] = ca.Function(
                f"ekf_update_{safe}", [x, P, z, u, t], [x_upd, P_upd],
                ["x", "P", "z", "u", "t"], ["x_new", "P_new"])
        return predict_fn, process_noise_fn, updates

    def _compute_blocks(self, n_tangent: int) -> list:
        """Partition the tangent state into independent subsystems.

        Two tangent indices land in the same block if they're coupled by
        the dynamics (`F`), share a process-noise channel (`L` column), or
        are jointly observed by one output (`H` support). Under this
        partition the covariance stays block-diagonal forever (nothing
        creates a cross-block term), so a predict can propagate each block
        independently — Σ O(n_b³) instead of O(n³). Returns a list of
        sorted tangent-index arrays, ordered by first index.
        """
        parent = list(range(n_tangent))

        def find(a: int) -> int:
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        rows, cols = np.nonzero(np.array(ca.DM(self.F_sym.sparsity())))
        for i, j in zip(rows.tolist(), cols.tolist()):
            union(i, j)
        if self._L_pattern is not None:
            for k in range(self._L_pattern.shape[1]):
                rws = np.flatnonzero(self._L_pattern[:, k])
                for r in rws[1:]:
                    union(int(rws[0]), int(r))
        for cols_touched in self._h_supports:
            for c in cols_touched[1:]:
                union(int(cols_touched[0]), int(c))

        groups: dict[int, list[int]] = {}
        for i in range(n_tangent):
            groups.setdefault(find(i), []).append(i)
        blocks = [np.array(sorted(v)) for v in groups.values()]
        blocks.sort(key=lambda b: int(b[0]))
        return blocks

    # ------------------------------------------------------------------
    # Tick routing (world-agnostic — drives every Jacobian above)
    # ------------------------------------------------------------------

    def _tick_outputs(self, x_sym, frozen, u_sym, dt_sym, t_sym,
                      n_sym) -> dict:
        """Evaluate the compiled tick on flat symbolic inputs; return a
        dict mapping every output name to its MX expression.

        State slots in `spec` are sliced from `x_sym`; slots / inputs in
        `frozen` are injected as baked constants; live inputs come from
        `u_sym`; Noise channels from `n_sym`.
        """
        cf = self._cf
        in_names  = [cf.name_in(i)  for i in range(cf.n_in())]
        out_names = [cf.name_out(i) for i in range(cf.n_out())]
        u_index   = {name: i for i, name in enumerate(self._input_names)}
        noise_offsets: dict[str, tuple[int, int]] = {}
        off = 0
        for ns in self._noise_specs:
            noise_offsets[ns.full] = (off, ns.dim)
            off += ns.dim

        sliced: list = []
        for name in in_names:
            if name == "dt":
                sliced.append(dt_sym)
            elif name == "t":
                sliced.append(t_sym)
            elif name in self.spec:
                slot = self.spec.slot(name)
                sliced.append(x_sym[slot.ambient_offset : slot.ambient_offset + slot.ambient_dim])
            elif name in frozen:
                val = np.atleast_1d(
                    np.asarray(frozen[name], dtype=float)).reshape(-1, 1)
                sliced.append(ca.DM(val))
            elif name in u_index:
                sliced.append(u_sym[u_index[name]])
            elif name in noise_offsets:
                start, dim = noise_offsets[name]
                sliced.append(n_sym[start : start + dim])
            else:
                raise RuntimeError(
                    f"Linearization: tick input {name!r} not handled.")

        result = cf(*sliced)
        if len(out_names) == 1:
            return {out_names[0]: result}
        return {n: result[i] for i, n in enumerate(out_names)}

    def _gather_state(self, outputs_by_name: dict) -> ca.MX:
        """Concatenate state-slot outputs in spec order → ambient vector."""
        chunks = []
        for slot in self.spec.slots:
            r = outputs_by_name[slot.name]
            if r.shape != (slot.ambient_dim, 1):
                r = ca.reshape(r, slot.ambient_dim, 1)
            chunks.append(r)
        return ca.vertcat(*chunks)

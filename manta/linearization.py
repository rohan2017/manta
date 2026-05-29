"""Linearization — manifold-aware Jacobians of a compiled world tick.

`Linearization` is the shared model-analysis transform that sits between
the forward dynamics IR (the world tick) and every model-based
estimator/controller. Given a compiled tick + a `StateSpec` (full or a
subset) + the input/noise routing, it produces the symbolic Jacobian
artifacts those algorithms need:

  * predict   f(x, u, dt, t)            — next ambient state, noise zeroed
  * F = ∂δ_out/∂δ_in |₀                 — tangent state-transition
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
    h_fn:          ca.Function | None
    H_fn:          ca.Function | None
    L_h_fn:        ca.Function | None


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
        noise_specs   — list of `{"full", "dim", "sigma", ...}` for the
                        Noise channels feeding the tick (drives `L`/`Σ`).
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
                 build_functions: bool = True) -> None:
        self.spec          = spec
        self._cf           = cf
        self._input_names  = input_names
        self._noise_specs  = noise_specs
        self.n_noise       = sum(s["dim"] for s in noise_specs)

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
        self.F_sym = ca.substitute(
            ca.jacobian(delta_out, delta_in),
            delta_in, ca.MX.zeros(n_tangent, 1))

        self.predict_fn = None
        self.F_fn       = None
        self.L_fn       = None
        self.Sigma      = None
        self.blocks: list = []
        self._L_pattern = None

        if build_functions:
            self.predict_fn = ca.Function(
                "predict", args, [x_new_0], argn, ["x_new"])
            self.F_fn = ca.Function("F", args, [self.F_sym], argn, ["F"])
            if self.n_noise > 0:
                delta_out_n = spec.boxminus_sym(x_new_n, x_new_0)
                L_sym = ca.substitute(
                    ca.jacobian(delta_out_n, n_sym), n_sym, zero_n)
                self._L_pattern = np.array(ca.DM(L_sym.sparsity()))
                self.L_fn = ca.Function("L", args, [L_sym], argn, ["L"])
                sigmas_sq: list[float] = []
                for ns in noise_specs:
                    sigmas_sq.extend([ns["sigma"] ** 2] * ns["dim"])
                self.Sigma = np.diag(sigmas_sq)

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

            h_fn = H_fn = L_h_fn = None
            if build_functions:
                safe = full.replace(".", "_")
                h_fn = ca.Function(f"h_{safe}", args, [h_sym], argn, ["h"])
                H_fn = ca.Function(f"H_{safe}", args, [H_sym], argn, ["H"])
                if self.n_noise > 0:
                    L_h_sym = ca.substitute(
                        ca.jacobian(h_n_flat, n_sym), n_sym, zero_n)
                    L_h_fn = ca.Function(
                        f"Lh_{safe}", args, [L_h_sym], argn, ["L_h"])
            self.outputs[full] = OutputLinearization(
                full=full, dim=h_dim, observed_cols=cols,
                h_sym=h_sym, H_sym=H_sym,
                h_fn=h_fn, H_fn=H_fn, L_h_fn=L_h_fn)

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
            noise_offsets[ns["full"]] = (off, ns["dim"])
            off += ns["dim"]

        sliced: list = []
        for name in in_names:
            if name == "dt":
                sliced.append(dt_sym)
            elif name == "t":
                sliced.append(t_sym)
            elif name in self.spec:
                slot = self.spec.slot(name)
                sliced.append(x_sym[slot.offset : slot.offset + slot.dim])
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
            if r.shape != (slot.dim, 1):
                r = ca.reshape(r, slot.dim, 1)
            chunks.append(r)
        return ca.vertcat(*chunks)

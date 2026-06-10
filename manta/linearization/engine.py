"""TickLinearizer — the manifold-aware differentiation engine.

One class, one job: evaluate the compiled world tick on symbolic inputs
and differentiate it over a (sub)spec. The recipe is the standard
manifold error-state one, built on `spec.boxplus_sym/boxminus_sym` only:

    δ_out = boxminus(f(boxplus(x, δ_in), …), f(x, …)),  F = ∂δ_out/∂δ_in |₀

so SO(3) (and any future manifold) linearizes correctly with no
special-case code.

Two entry points:

  * `structure(spec, frozen, outputs)` — the CHEAP pass: only the
    structural artifacts (F's sparsity pattern, each sensor's observed
    tangent columns). `LinearizedSystem` runs it over the FULL spec to
    decide the tracked subset — the subset isn't known yet, so this
    cannot be folded into the real pass below.
  * `differentiate(spec, frozen, outputs, control=…)` — the full pass
    over the chosen (sub)spec: forward expressions, F/B/L Jacobians,
    per-sensor measurement models, point-evaluation Functions, and the
    block partition.
"""

from __future__ import annotations

from dataclasses import dataclass

import casadi as ca
import numpy as np

from ..ir.module import entry_ident
from .partition import partition_blocks


@dataclass
class SensorModel:
    """One chosen output's linearized measurement model, by name.

    Symbolic expressions are in the system's `(x_sym, u_sym, n_sym, dt_sym,
    t_sym)` symbols; `h_sym` is noise-zeroed, `h_noisy_sym` keeps the noise
    live (they coincide for a noiseless sensor). `h_fn`/`H_fn` are
    `(x,u,dt,t)` convenience functions for point evaluation/analysis."""
    full:          str
    dim:           int
    h_sym:         ca.MX
    h_noisy_sym:   ca.MX
    H_sym:         ca.MX
    L_h_sym:       ca.MX | None
    observed_cols: np.ndarray
    h_fn:          ca.Function | None
    H_fn:          ca.Function | None


class TickLinearizer:
    """Differentiation engine over one compiled tick.

    Args:
        cf            — the tick's `ca.Function` (named I/O).
        input_names   — the LIVE control inputs, in order (frozen inputs
                        are baked through `frozen` at evaluation).
        noise_specs   — the tick's noise channels (name, dim, σ), in the
                        flat noise-vector order.
        discretization — "exact" | "euler"; how F discretizes (see
                        `differentiate`).
        param_specs   — the tick's promoted-parameter channels (name, dim,
                        declared value), in the flat `p`-vector order.
                        Empty unless the system was built with
                        `parameters=[...]`; when present, every expression
                        and convenience Function gains a `p` argument.
    """

    def __init__(self, cf, input_names, noise_specs,
                 discretization: str, param_specs=()) -> None:
        self.cf = cf
        self.input_names = list(input_names)
        self.noise_specs = list(noise_specs)
        self.n_noise = sum(s.dim for s in self.noise_specs)
        self.param_specs = list(param_specs)
        self.n_param = sum(p.dim for p in self.param_specs)
        self.discretization = discretization

    # ---- entry points ---------------------------------------------------

    def structure(self, spec, frozen, outputs) -> dict:
        """Structural pass: `{"F_pattern", "sensors"}` only — F's slot
        dependency pattern and each sensor's observed tangent columns.
        Always uses the exact (call-node) recipe, so closure/freeze
        decisions stay independent of the discretization choice."""
        return self._linearize(spec, frozen, outputs, control=False,
                               build_functions=False)

    def differentiate(self, spec, frozen, outputs, *, control: bool) -> dict:
        """Full pass over `spec`: forward expressions, error-state
        Jacobians, per-output measurement models, point-evaluation
        Functions, and the independent-block partition."""
        return self._linearize(spec, frozen, outputs, control=control,
                               build_functions=True)

    # ---- the recipe -------------------------------------------------------

    def _linearize(self, spec, frozen, outputs, *, control: bool,
                   build_functions: bool) -> dict:
        n_tangent = spec.tangent_dim
        n_u = len(self.input_names)

        x = ca.MX.sym("x", spec.ambient_dim, 1)
        u = ca.MX.sym("u", n_u, 1) if n_u > 0 else ca.MX.zeros(0, 1)
        dt = ca.MX.sym("dt", 1, 1)
        t = ca.MX.sym("t", 1, 1)
        n = (ca.MX.sym("noise", self.n_noise, 1) if self.n_noise > 0
             else ca.MX.zeros(0, 1))
        zero_n = ca.MX.zeros(self.n_noise, 1)
        p = (ca.MX.sym("p", self.n_param, 1) if self.n_param > 0
             else ca.MX.zeros(0, 1))
        # The `p` argument exists only for parameter-promoted systems, so
        # the convenience-Function signature stays (x, u, dt, t) for every
        # existing consumer (filters, regulators, deploy kernels).
        if self.n_param > 0:
            args, argn = [x, u, p, dt, t], ["x", "u", "p", "dt", "t"]
        else:
            args, argn = [x, u, dt, t], ["x", "u", "dt", "t"]

        # The euler discretization needs the tick INLINED into open MX:
        # the dt→0 fold can then constant-propagate through the discrete
        # integrator and prune it from the differentiated graph. A call
        # node would stay a black box (the fold would stop at its
        # argument) and jacobian-of-jacobian would emit forward-over-
        # forward helpers of the FULL tick — strictly worse. The
        # structural (build_functions=False) pass always stays on the
        # exact call-node recipe: same cost class, and the closure /
        # freeze decisions stay independent of the discretization choice.
        euler = build_functions and self.discretization == "euler"

        # forward (noise live + zeroed)
        outs_n = self._tick_outputs(spec, x, frozen, u, dt, t, n, p_sym=p)
        x_new_n = self._gather_state(spec, outs_n)
        x_new_0 = ca.substitute(x_new_n, n, zero_n)

        # F = ∂δ'/∂δ at δ=0 (error-state recipe)
        delta_in = ca.MX.sym("delta_in", n_tangent, 1)
        outs_pert = self._tick_outputs(spec, spec.boxplus_sym(x, delta_in),
                                       frozen, u, dt, t, zero_n,
                                       p_sym=p, inline=euler)
        x_pert_new = self._gather_state(spec, outs_pert)
        if euler:
            # F = I + dt·M, M = ∂²δ_out/∂dt∂δ at (δ=0, dt=0): the
            # continuous-time linearization, Euler-discretized — exact
            # to O(dt²) including the manifold transport terms (the
            # boxminus base point must be inlined too, or its dt-
            # dependence drags the full tick back in as a helper).
            outs_base = self._tick_outputs(spec, x, frozen, u, dt, t,
                                           zero_n, p_sym=p, inline=True)
            delta_out = spec.boxminus_sym(
                x_pert_new, self._gather_state(spec, outs_base))
            ddt0 = ca.substitute(ca.jacobian(delta_out, dt),
                                 dt, ca.MX.zeros(1, 1))
            M = ca.substitute(ca.jacobian(ddt0, delta_in),
                              delta_in, ca.MX.zeros(n_tangent, 1))
            F_sym = ca.cse(ca.MX.eye(n_tangent) + dt * M)
        else:
            delta_out = spec.boxminus_sym(x_pert_new, x_new_0)
            F_sym = ca.substitute(ca.jacobian(delta_out, delta_in),
                                  delta_in, ca.MX.zeros(n_tangent, 1))
        F_pattern = np.array(ca.DM(F_sym.sparsity()))

        # B = ∂δ'/∂u (regulators only)
        B_sym = None
        if control and n_u > 0:
            delta_u = ca.MX.sym("delta_u", n_u, 1)
            outs_pu = self._tick_outputs(spec, x, frozen, u + delta_u,
                                         dt, t, zero_n, p_sym=p)
            delta_out_u = spec.boxminus_sym(
                self._gather_state(spec, outs_pu), x_new_0)
            B_sym = ca.substitute(ca.jacobian(delta_out_u, delta_u),
                                  delta_u, ca.MX.zeros(n_u, 1))

        # L = ∂δ'/∂noise, Σ
        L_sym, L_pattern, Sigma = None, None, None
        if self.n_noise > 0:
            delta_out_n = spec.boxminus_sym(x_new_n, x_new_0)
            L_sym = ca.substitute(ca.jacobian(delta_out_n, n), n, zero_n)
            L_pattern = np.array(ca.DM(L_sym.sparsity()))
            sig2: list[float] = []
            for ns in self.noise_specs:
                sig2.extend([ns.sigma ** 2] * ns.dim)
            Sigma = np.diag(sig2)

        # per-output measurement models
        sensors: dict[str, SensorModel] = {}
        h_supports: list[np.ndarray] = []
        for full in outputs:
            if full not in outs_n:
                raise RuntimeError(
                    f"TickLinearizer: tick is missing output {full!r}.")
            dim = int(outs_n[full].numel())
            h_noisy = ca.reshape(outs_n[full], dim, 1)
            h_pert = ca.reshape(outs_pert[full], dim, 1)
            H_sym = ca.substitute(ca.jacobian(h_pert, delta_in),
                                  delta_in, ca.MX.zeros(n_tangent, 1))
            if euler:
                # Inlined h_pert → flat self-contained H kernel; cse
                # dedups the kinematic chains the inlining duplicated.
                H_sym = ca.cse(H_sym)
            cols = np.flatnonzero(
                np.array(ca.DM(H_sym.sparsity())).any(axis=0))
            h_supports.append(cols)
            h_sym = ca.substitute(h_noisy, n, zero_n)
            L_h_sym = (ca.substitute(ca.jacobian(h_noisy, n), n, zero_n)
                       if self.n_noise > 0 else None)
            h_fn = H_fn = None
            if build_functions:
                safe = entry_ident(full)
                h_fn = ca.Function(f"h_{safe}", args, [h_sym], argn, ["h"])
                H_fn = ca.Function(f"H_{safe}", args, [H_sym], argn, ["H"])
            sensors[full] = SensorModel(
                full=full, dim=dim,
                h_sym=h_sym, h_noisy_sym=h_noisy, H_sym=H_sym,
                L_h_sym=L_h_sym, observed_cols=cols, h_fn=h_fn, H_fn=H_fn)

        predict_fn = F_fn = B_fn = L_fn = None
        blocks: list = []
        if build_functions:
            predict_fn = ca.Function("predict", args, [x_new_0],
                                     argn, ["x_new"])
            F_fn = ca.Function("F", args, [F_sym], argn, ["F"])
            if B_sym is not None:
                B_fn = ca.Function("B", args, [B_sym], argn, ["B"])
            if L_sym is not None:
                L_fn = ca.Function("L", args, [L_sym], argn, ["L"])
            blocks = partition_blocks(n_tangent, F_pattern, L_pattern,
                                      h_supports)

        return {"x": x, "u": u, "n": n, "p": p, "dt": dt, "t": t,
                "x_new": x_new_0, "x_new_noisy": x_new_n,
                "F_sym": F_sym, "F_pattern": F_pattern, "B_sym": B_sym,
                "L_sym": L_sym, "Sigma": Sigma, "sensors": sensors,
                "predict_fn": predict_fn, "F_fn": F_fn, "B_fn": B_fn,
                "L_fn": L_fn, "blocks": blocks}

    # ---- tick evaluation --------------------------------------------------

    def _tick_outputs(self, spec, x_sym, frozen, u_sym, dt_sym, t_sym,
                      n_sym, *, p_sym=None, inline: bool = False) -> dict:
        """Evaluate the compiled tick on flat symbolic inputs → {output
        name: MX}. Spec slots slice from `x_sym`; frozen entries bake as
        constants; live inputs from `u_sym`; noise channels from `n_sym`;
        promoted parameters from `p_sym`. With `inline=True` the tick's
        MX graph is spliced in open (no call node) so downstream
        substitutions can constant-fold through it — see the
        euler-discretization note in `_linearize`."""
        cf = self.cf
        in_names = [cf.name_in(i) for i in range(cf.n_in())]
        out_names = [cf.name_out(i) for i in range(cf.n_out())]
        u_index = {name: i for i, name in enumerate(self.input_names)}
        noise_off: dict[str, tuple[int, int]] = {}
        off = 0
        for ns in self.noise_specs:
            noise_off[ns.full] = (off, ns.dim)
            off += ns.dim
        param_off: dict[str, tuple[int, int]] = {}
        off = 0
        for ps in self.param_specs:
            param_off[ps.full] = (off, ps.dim)
            off += ps.dim

        sliced: list = []
        for name in in_names:
            if name == "dt":
                sliced.append(dt_sym)
            elif name == "t":
                sliced.append(t_sym)
            elif name in spec:
                s = spec.slot(name)
                sliced.append(
                    x_sym[s.ambient_offset:s.ambient_offset + s.ambient_dim])
            elif name in frozen:
                val = np.atleast_1d(
                    np.asarray(frozen[name], dtype=float)).reshape(-1, 1)
                sliced.append(ca.DM(val))
            elif name in u_index:
                sliced.append(u_sym[u_index[name]])
            elif name in noise_off:
                start, dim = noise_off[name]
                sliced.append(n_sym[start:start + dim])
            elif name in param_off:
                start, dim = param_off[name]
                sliced.append(p_sym[start:start + dim])
            else:
                raise RuntimeError(
                    f"TickLinearizer: tick input {name!r} not handled.")

        if inline:
            result = cf.call(sliced, True, False)   # always_inline
            return {nm: result[i] for i, nm in enumerate(out_names)}
        result = cf(*sliced)
        if len(out_names) == 1:
            return {out_names[0]: result}
        return {nm: result[i] for i, nm in enumerate(out_names)}

    @staticmethod
    def _gather_state(spec, outputs_by_name: dict) -> ca.MX:
        """Concatenate state-slot outputs in spec order → ambient vector."""
        chunks = []
        for slot in spec.slots:
            r = outputs_by_name[slot.name]
            if r.shape != (slot.ambient_dim, 1):
                r = ca.reshape(r, slot.ambient_dim, 1)
            chunks.append(r)
        return ca.vertcat(*chunks)

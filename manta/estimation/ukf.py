"""UKF — the error-state (manifold-aware) Unscented Kalman Filter.

The unscented twin of `manta.estimation.ekf`. It carries the identical
state — `x` (manifold) and `P` (tangent covariance) — and the identical
additive noise (`Q = L Σ Lᵀ` baked from the model, `R = L_h Σ L_hᵀ` per
sensor), and it emits the *same-shape* typed `Module`, so it lowers to
every backend (numpy / C++ / JAX / wasm) through exactly the EKF's path
with no new backend code.

What differs is the math between `x,P` in and `x,P` out: where the EKF
pushes the covariance through a linearization (`F P Fᵀ`, `H P Hᵀ`), the UKF
samples the *nonlinear* `f`/`h` at 2n+1 sigma points and re-summarizes —
no Jacobians:

    predict     {δᵢ}      = ±γ · cholᵢ(P), δ₀ = 0       (tangent sigma set)
                {Xᵢ}      = x ⊞ δᵢ                        (retract to manifold)
                {Yᵢ}      = f(Xᵢ, u, dt, t)               (push through f)
                x'        = mean({Yᵢ})                    (manifold mean)
                P'        = Σ w_cᵢ (Yᵢ ⊟ x')(…)ᵀ + Q
    update_s    {Zᵢ}      = h(Xᵢ, u)                      (push through h)
                ẑ, S, C   = unscented moments + R
                K         = C S⁻¹                          (ldl solve — S SPD)
                x  ⟵ x ⊞ K (z − ẑ)                        (manifold boxplus)
                P  ⟵ P − K S Kᵀ

The sigma spread, mean, and moments are written symbolically (the scalar-
unrolled `chol_lower` from `manta.ir._linalg` keeps it C-codegen- and
SX-expand-clean) and baked once into a fused `ca.Function` per entry, then
exposed as a typed `Module`:

    state    x  (manifold, the tracked StateSpec)   P  (tangent covariance)
    entries  predict(x,P, u,dt,t)            — auto process noise
             predict_with_Q(x,P, Q,u,dt,t)   — explicit-Q override
             update_<sensor>(x,P, z,u,t)     — one per chosen sensor

Lower the Module to a backend to run::

    from manta import UKF, TargetNumpy

    ukf = TargetNumpy(UKF(world, track={"chaser": POSE | TWIST},
                          sensors=["chaser.imu.gyro", "chaser.gps.position"],
                          inputs=["chaser.thruster.throttle"]))
    for t in ts:
        ukf.update("chaser.imu.gyro", gyro_z, t=t)   # fold, then...
        ukf.predict(dt, t=t)                          # ...predict (you order)

The runtime surface is identical to the EKF's (`NumpyFilter`): you own the
update-then-predict loop. `track`/`sensors`/`inputs` carve the state and
I/O exactly as in the EKF — see `manta.linearization`. Q/R auto-assembly
comes from the model's Noise channels.

Tuning is the standard scaled unscented transform (`alpha`, `beta`,
`kappa`); the small-`alpha` default keeps the spread tight, so on a
near-linear model the UKF and EKF agree closely. `mean_iters` controls the
manifold-mean retraction in the predict (1 suffices for the default
spread).
"""

from __future__ import annotations

import casadi as ca
import numpy as np

from ..ir.module import (
    EntryPoint, Hosting, Module, Port, PortField, PortRef, Role, StateField,
    StateLayout, StateRef, entry_ident,
)
from ..ir.state_spec import StateSpec, flatten_nested
from ..ir._names import resolve_suffix
from ..linearization import LinearizedSystem
from ._kalman import (
    lin_cov, require_active_R, sigma_deltas, unscented_weights, ut_predict,
    ut_update,
)


class UKF:
    """Error-state UKF over a `World` — symbolic sigma-point recursion +
    typed Module. Drop-in alternative to `EKF` with the same constructor,
    runtime surface, and emitted Module shape."""

    def __init__(self, world, *,
                 track: dict | None = None,
                 sensors: list[str] | None = None,
                 inputs: list[str] | None = None,
                 alpha: float = 1e-3,
                 beta: float = 2.0,
                 kappa: float = 0.0,
                 mean_iters: int = 1) -> None:
        """Args:
            track:   `{craft_name: SlotSet}` lower bound of what to estimate
                     (closed under the dynamics; the rest freezes). `None`
                     keeps the full state.
            sensors: measurement full-names (or unambiguous suffixes).
                     `None` keeps every output (of tracked crafts).
            inputs:  known control inputs; `None` keeps all, excluded ones
                     freeze at their default.
            alpha:   sigma-point spread (0 < α ≤ 1). Small (default 1e-3)
                     keeps the points near the mean — the canonical scaled
                     UT, which matches the EKF in the linear limit.
            beta:    prior-knowledge term (2.0 is optimal for a Gaussian).
            kappa:   secondary scaling (0.0 by default).
            mean_iters: retraction steps for the predict's manifold mean
                     (1 is plenty for the default spread; raise it for a
                     wide spread on a strongly-curved manifold).

        Unlike the EKF there is no `discretization` knob: the UKF pushes
        sigma points through the exact nonlinear discrete tick `f`, so the
        Euler/exact distinction (which only shapes the EKF's linearized F)
        does not arise.
        """
        sys = LinearizedSystem(world, track=track, sensors=sensors,
                               inputs=inputs, track_mode="closure")
        self.sys = sys
        self.world = world
        self.crafts = sys.crafts
        self.spec: StateSpec = sys.spec
        self.alpha, self.beta, self.kappa = alpha, beta, kappa
        self.mean_iters = mean_iters

        # ---- the unscented recursion, symbolically, once ----------------
        spec, n_tan = sys.spec, sys.spec.tangent_dim
        x, u = sys.x_sym, sys.u_sym
        dt, t = sys.dt_sym, sys.t_sym
        P = ca.MX.sym("P", n_tan, n_tan)
        Q = ca.MX.sym("Q", n_tan, n_tan)

        _, w_m, w_c, gamma = unscented_weights(n_tan, alpha, beta, kappa)

        # Prior sigma set, shared by predict and (regenerated identically by)
        # each update: tangent offsets → retract onto the manifold.
        deltas = sigma_deltas(P, gamma, n_tan)
        sigma_pts = [spec.boxplus_sym(x, d) for d in deltas]

        # predict: push each sigma point through the nonlinear tick (inline
        # via substitution so the kernel stays one expandable scalar graph),
        # then the unscented mean/cov + auto process noise Q = L Σ Lᵀ.
        Q_auto = lin_cov(sys.L_sym,
                         ca.DM(sys.Sigma) if sys.L_sym is not None else None,
                         n_tan)
        propagated = [ca.substitute(sys.x_new, x, Xi) for Xi in sigma_pts]
        x_pred, P_pred = ut_predict(deltas, propagated, Q_auto,
                                    w_m, w_c, spec, mean_iters)
        x_pred_q, P_pred_q = ut_predict(deltas, propagated, Q,
                                        w_m, w_c, spec, mean_iters)
        predict_fn = ca.Function(
            "ukf_predict", [x, P, u, dt, t], [x_pred, P_pred],
            ["x", "P", "u", "dt", "t"], ["x_new", "P_new"])
        predict_q_fn = ca.Function(
            "ukf_predict_with_Q", [x, P, Q, u, dt, t], [x_pred_q, P_pred_q],
            ["x", "P", "Q", "u", "dt", "t"], ["x_new", "P_new"])

        # per-sensor unscented update. A measurement is dt-independent, so dt
        # is eliminated (substituted to 0) — the kernel honestly takes only
        # (x, P, z, u, t).
        init_flat = flatten_nested(world._initial_state_dict())
        x0 = spec.pack_any(init_flat)
        zero_dt = ca.MX.zeros(1, 1)
        updates: dict[str, ca.Function] = {}
        for full, s in sys.sensors.items():
            z = ca.MX.sym("z", s.dim)
            h = ca.substitute(s.h_sym, dt, zero_dt)
            L_h = (ca.substitute(s.L_h_sym, dt, zero_dt)
                   if s.L_h_sym is not None and sys.Sigma is not None
                   else None)
            R = lin_cov(L_h, ca.DM(sys.Sigma) if L_h is not None else None,
                        s.dim)
            # Refuse a σ=0 sensor (R ≡ 0 → singular innovation S).
            R_fn = ca.Function("R0", [x, u, t], [R])
            require_active_R(R, R_fn, ca.vertcat(x, u, t),
                             x0=x0, u_defaults=sys.u_defaults, spec=spec,
                             full=full, who="UKF")
            measured = [ca.substitute(h, x, Xi) for Xi in sigma_pts]
            x_upd, P_upd, _, _ = ut_update(x, P, deltas, measured, R, z,
                                           w_m, w_c, spec)
            updates[full] = ca.Function(
                f"ukf_update_{entry_ident(full)}",
                [x, P, z, u, t], [x_upd, P_upd],
                ["x", "P", "z", "u", "t"], ["x_new", "P_new"])

        # ---- the typed Module (identical shape to EKF's) ----------------
        fields = (
            StateField("x", "manifold", (spec.ambient_dim,),
                       init=x0, manifold=spec),
            StateField("P", "matrix", (n_tan, n_tan),
                       init=np.eye(n_tan) * 1e-2),
        )
        ports = [
            Port("u", Role.CONTROL, (len(sys.input_names),), fields=tuple(
                PortField(n, 1, float(sys.input_defaults[n]),
                          rate=sys.sample_rates.get(n))
                for n in sys.input_names)),
            Port("dt", Role.TIMESTEP),
            Port("t", Role.TIME),
            Port("Q", Role.MATRIX, (n_tan, n_tan)),
        ]
        functions = {"predict": predict_fn, "predict_with_Q": predict_q_fn}
        entries = [
            EntryPoint("predict", "predict",
                       (StateRef("x"), StateRef("P"), PortRef("u"),
                        PortRef("dt"), PortRef("t")),
                       writes=("x", "P")),
            EntryPoint("predict_with_Q", "predict_with_Q",
                       (StateRef("x"), StateRef("P"), PortRef("Q"),
                        PortRef("u"), PortRef("dt"), PortRef("t")),
                       writes=("x", "P")),
        ]
        for full, s in sys.sensors.items():
            ident = entry_ident(full)
            ports.append(Port(full, Role.MEASUREMENT, (s.dim,),
                              rate=sys.sample_rates.get(full)))
            functions[f"update_{ident}"] = updates[full]
            entries.append(EntryPoint(
                f"update_{ident}", f"update_{ident}",
                (StateRef("x"), StateRef("P"), PortRef(full),
                 PortRef("u"), PortRef("t")),
                writes=("x", "P")))

        self._module = Module(
            name=f"{world.name}_ukf", state=StateLayout(fields),
            ports=tuple(ports), functions=functions,
            entry_points=tuple(entries), hosting=Hosting.HELD)

    def module(self) -> Module:
        """The typed `Module` IR a backend lowers."""
        return self._module

    # ------------------------------------------------------------------
    # Analysis surface
    # ------------------------------------------------------------------

    @property
    def n_blocks(self) -> int:
        """Independent tangent subsystems (structurally decoupled crafts)."""
        return len(self.sys.blocks)

    def _build_u(self, u: dict[str, float] | None) -> np.ndarray:
        """Resolve `u` to a flat input vector (full or suffix names)."""
        names = self.sys.input_names
        out = self.sys.u_defaults.copy()
        if u:
            index = {n: i for i, n in enumerate(names)}
            for k, v in u.items():
                full = resolve_suffix(k, names, label="input", who="UKF")
                out[index[full]] = float(v)
        return out

    def observability(self, **kwargs):
        """Local observability of the chosen sensor set at an operating
        point (the linearized analysis shared with the EKF — see
        `manta.estimation.observability`)."""
        from .observability import observability
        return observability(self, **kwargs)

    def sigma_horizon(self, **kwargs):
        """Per-slot σ attainable after a horizon — the linearized covariance
        recursion run open-loop (see `manta.estimation.observability`)."""
        from .observability import sigma_horizon
        return sigma_horizon(self, **kwargs)

    def __repr__(self) -> str:
        return (f"<UKF tangent={self.spec.tangent_dim} "
                f"sensors={list(self.sys.sensors)} n_blocks={self.n_blocks}>")

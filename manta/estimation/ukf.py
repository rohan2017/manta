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
                P  ⟵ Σ w_cᵢ (δᵢ − K dzᵢ)(…)ᵀ + K R Kᵀ    (sigma-point Joseph)

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
`kappa`). The auto default (`alpha=None`) pins the sigma spread at
`√3·σ` independent of state size — see the constructor docstring for why
both α=1 (manifold wrap at large n) and the textbook small-α (a −10⁶
central weight and a finite-difference spread) are traps at
robotics-sized tangent dims. `mean_iters` controls the manifold-mean
retraction in the predict; `jitter` regularizes the sigma factorization
of the iterated covariance.
"""

from __future__ import annotations

import math
import warnings

import casadi as ca

from ..ir._linalg import spd_solve
from ..ir.module import entry_ident
from ..linearization import LinearizedSystem
from ._assembly import (
    _FilterBase,
    _q_auto,
    emit_filter_module,
    initial_ambient,
    prepared_sensors,
    resolve_gates,
)
from ._kalman import sigma_deltas, unscented_weights, ut_predict, ut_update


class UKF(_FilterBase):
    """Error-state UKF over a `World` — symbolic sigma-point recursion +
    typed Module. Drop-in alternative to `EKF` with the same constructor,
    runtime surface, and emitted Module shape. The analysis surface
    (`module`, `n_blocks`, `observability`, `sigma_horizon`) is the shared
    `_FilterBase` tail."""

    def __init__(self, world, *,
                 track: dict | None = None,
                 sensors: list[str] | None = None,
                 inputs: list[str] | None = None,
                 alpha: float | None = None,
                 beta: float = 2.0,
                 kappa: float = 0.0,
                 mean_iters: int = 1,
                 jitter: float = 1e-12,
                 gates: float | dict[str, float] | None = None) -> None:
        """Args:
            track:   `{craft_name: SlotSet}` lower bound of what to estimate
                     (closed under the dynamics; the rest freezes). `None`
                     keeps the full state.
            sensors: measurement full-names (or unambiguous suffixes).
                     `None` keeps every output (of tracked crafts).
            inputs:  known control inputs; `None` keeps all, excluded ones
                     freeze at their default.
            alpha:   sigma-point spread (0 < α ≤ 1). `None` (default)
                     resolves to `min(1, √(3/n))` for tangent dim `n`, which
                     pins the spread at `γ = √(n+λ) = √3` — i.e. sigma
                     points at ~1.7σ — *independent of state size*. The two
                     failure modes this dodges are both real: α=1 spreads
                     points at `√n`·σ, which at robotics-sized `n` wraps
                     rotation offsets past π and corrupts the re-summarized
                     covariance; the textbook "small α" (1e-3) drives the
                     central covariance weight to ≈ −n/α² (−10⁶ at n=13,
                     the classic indefinite-P failure) while shrinking the
                     spread until the transform is a noisy finite
                     difference of the EKF.
            beta:    prior-knowledge term (2.0 is optimal for a Gaussian).
            kappa:   secondary scaling (0.0 by default).
            mean_iters: retraction steps for the predict's manifold mean
                     (1 suffices for the default spread; raise it for a
                     wide spread on a strongly-curved manifold).
            jitter:  variance floor (added as `jitter·I`, and as a pivot
                     floor) when factoring P into sigma points. P is an
                     iterated covariance, so roundoff can leave it
                     marginally indefinite; the jitter turns the NaN cliff
                     into a regularized factor. 0 disables.
            gates:   optional normalized-innovation-squared threshold: one
                     positive scalar for every sensor, or a mapping from
                     sensor name/suffix to threshold. Rejection preserves
                     the prior state and covariance.

        An *explicit* tuning that produces a negative central covariance
        weight (`w_c[0] < 0`) is accepted but warns: the covariance sums
        lose their PSD guarantee and rely on the jitter backstop. (The
        auto default accepts a mildly negative `w_c[0]` at large `n` —
        the price of the bounded spread — which the sigma-point Joseph
        update form and the jitter absorb.)

        Unlike the EKF there is no `discretization` knob: the UKF pushes
        sigma points through the exact nonlinear discrete tick `f`, so the
        Euler/exact distinction (which only shapes the EKF's linearized F)
        does not arise.
        """
        sys = LinearizedSystem(world, track=track, sensors=sensors,
                               inputs=inputs, track_mode="closure")
        self._bind_system(world, sys)
        resolved_gates = resolve_gates(sys, gates, who="UKF")

        n_tan = sys.spec.tangent_dim
        explicit_alpha = alpha is not None
        if alpha is None:
            alpha = min(1.0, math.sqrt(3.0 / n_tan))
        self.alpha, self.beta, self.kappa = alpha, beta, kappa
        self.mean_iters = mean_iters
        self.jitter = jitter

        # ---- the unscented recursion, symbolically, once ----------------
        spec = sys.spec
        x, u = sys.x_sym, sys.u_sym
        dt, t = sys.dt_sym, sys.t_sym
        P = ca.MX.sym("P", n_tan, n_tan)
        Q = ca.MX.sym("Q", n_tan, n_tan)

        w_m, w_c, gamma = unscented_weights(n_tan, alpha, beta, kappa)
        if explicit_alpha and w_c[0] < 0.0:
            warnings.warn(
                f"UKF: alpha={alpha}, beta={beta}, kappa={kappa} give a "
                f"negative central covariance weight (w_c[0] = {w_c[0]:.3g}"
                f" at tangent dim {n_tan}) — the covariance updates lose "
                f"their PSD-by-construction guarantee and rely on the "
                f"jitter backstop. alpha=None (auto) bounds the spread at "
                f"√3·σ with a near-minimal negative weight.",
                RuntimeWarning, stacklevel=2)

        # Prior sigma set, shared by predict and (regenerated identically by)
        # each update: tangent offsets → retract onto the manifold. The
        # jitter regularizes the factorization of an iterated covariance.
        deltas = sigma_deltas(P, gamma, n_tan, jitter=jitter)
        sigma_pts = [spec.boxplus_sym(x, d) for d in deltas]

        # predict: push each sigma point through the nonlinear tick (inline
        # via substitution so the kernel stays one expandable scalar graph),
        # then the unscented mean/cov + auto process noise Q = L Σ Lᵀ.
        # `ca.cse` shrinks the graph: the 2n+1 substituted tick copies share
        # nearly all their subexpressions (same reason the linearization
        # engine cse's its inlined H kernels).
        Q_auto = _q_auto(sys)
        propagated = [ca.substitute(sys.x_new, x, Xi) for Xi in sigma_pts]
        x_pred, P_pred = ut_predict(propagated, Q_auto,
                                    w_m, w_c, spec, mean_iters)
        x_pred_q, P_pred_q = ut_predict(propagated, Q,
                                        w_m, w_c, spec, mean_iters)
        x_pred, P_pred = ca.cse([x_pred, P_pred])
        x_pred_q, P_pred_q = ca.cse([x_pred_q, P_pred_q])
        predict_fn = ca.Function(
            "ukf_predict", [x, P, u, dt, t], [x_pred, P_pred],
            ["x", "P", "u", "dt", "t"], ["x_new", "P_new"])
        predict_q_fn = ca.Function(
            "ukf_predict_with_Q", [x, P, Q, u, dt, t], [x_pred_q, P_pred_q],
            ["x", "P", "Q", "u", "dt", "t"], ["x_new", "P_new"])

        # per-sensor unscented update. `prepared_sensors` eliminates dt and
        # refuses σ=0 sensors — the kernel honestly takes only
        # (x, P, z, u, t).
        x0 = initial_ambient(sys.world, spec)
        updates: dict[str, ca.Function] = {}
        diagnostic_updates: dict[str, ca.Function] = {}
        override_updates: dict[str, ca.Function] = {}
        for ps in prepared_sensors(sys, spec, x0=x0, who="UKF"):
            measured = [ca.substitute(ps.h, x, Xi) for Xi in sigma_pts]

            threshold = resolved_gates[ps.full]

            def expressions(
                R, measured=measured, ps=ps, threshold=threshold
            ):
                x_candidate, P_candidate, nu, S = ut_update(
                    x, P, deltas, measured, R, ps.z, w_m, w_c, spec)
                nis = ca.dot(nu, spd_solve(S, nu))
                accepted = (ca.MX.ones(1, 1) if threshold is None
                            else nis <= threshold)
                return (ca.if_else(accepted, x_candidate, x),
                        ca.if_else(accepted, P_candidate, P),
                        nu, S, nis, accepted)

            x_upd, P_upd, nu, S, nis, accepted = expressions(ps.R)
            x_upd, P_upd, nu, S, nis, accepted = ca.cse(
                [x_upd, P_upd, nu, S, nis, accepted])
            updates[ps.full] = ca.Function(
                f"ukf_update_{entry_ident(ps.full)}",
                [x, P, ps.z, u, t], [x_upd, P_upd],
                ["x", "P", "z", "u", "t"], ["x_new", "P_new"])
            diagnostic_updates[ps.full] = ca.Function(
                f"ukf_update_diagnostic_{entry_ident(ps.full)}",
                [x, P, ps.z, u, t],
                [x_upd, P_upd, nu, S, nis, accepted],
                ["x", "P", "z", "u", "t"],
                ["x_new", "P_new", "innovation", "innovation_covariance",
                 "nis", "accepted"])
            R_override = ca.MX.sym(f"R_{entry_ident(ps.full)}", ps.dim,
                                   ps.dim)
            xo, Po, nuo, So, niso, acceptedo = expressions(R_override)
            xo, Po, nuo, So, niso, acceptedo = ca.cse(
                [xo, Po, nuo, So, niso, acceptedo])
            override_updates[ps.full] = ca.Function(
                f"ukf_update_with_R_{entry_ident(ps.full)}",
                [x, P, ps.z, R_override, u, t],
                [xo, Po, nuo, So, niso, acceptedo],
                ["x", "P", "z", "R", "u", "t"],
                ["x_new", "P_new", "innovation", "innovation_covariance",
                 "nis", "accepted"])

        self._module = emit_filter_module(
            sys, spec, name=f"{world.name}_ukf", x0=x0,
            predict_fn=predict_fn, predict_q_fn=predict_q_fn,
            updates=updates, diagnostic_updates=diagnostic_updates,
            override_updates=override_updates, gates=resolved_gates)

    # module() / n_blocks / observability() / sigma_horizon() / __repr__
    # are the shared `_FilterBase` analysis tail (estimation/_assembly.py).

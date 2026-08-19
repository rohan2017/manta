"""EKF — the error-state (manifold-aware) Extended Kalman Filter.

This file IS the filter. `EKF(world, …)` builds a `LinearizedSystem` (which
owns all the tick/subset/sensor plumbing) and then writes the Kalman
recursion symbolically, right here, over the system's artifacts:

    predict     x' = f(x, u, dt, t)
                P' = F P Fᵀ + Q            (Q = L Σ Lᵀ baked, or supplied)
    update_s    S  = H P Hᵀ + R            (R = L_h Σ L_hᵀ, at dt = 0)
                K  = P Hᵀ S⁻¹              (ldl solve — S is SPD)
                x  ⟵ x ⊞ K (z − h)         (manifold-correct boxplus)
                P  ⟵ (I−KH) P (I−KH)ᵀ + K R Kᵀ        (Joseph form)

Each line is baked once into a fused `ca.Function` — kernels are *honest*
(a measurement kernel never takes `dt`; the auto-`Q` predict has `L Σ Lᵀ`
inside), and `EKF.module()` exposes them as a typed `Module`:

    state    x  (manifold, the tracked StateSpec)   P  (tangent covariance)
    entries  predict(x,P, u,dt,t)            — auto process noise
             predict_with_Q(x,P, Q,u,dt,t)   — explicit-Q override
             update_<sensor>(x,P, z,u,t)     — one per chosen sensor

Lower the Module to a backend to run::

    from manta import EKF, TargetNumpy

    ekf = TargetNumpy(EKF(world, track={"chaser": POSE | TWIST},
                          sensors=["chaser.imu.gyro", "chaser.gps.position"],
                          inputs=["chaser.thruster.throttle"]))
    for t in ts:
        ekf.update("chaser.imu.gyro", gyro_z, t=t)   # fold, then...
        ekf.predict(dt, t=t)                          # ...predict (you order)

`track` is a *lower bound*: the system expands it (and whatever the chosen
sensors observe) to a set closed under the dynamics and freezes the rest —
see `manta.linearization`. Q/R auto-assembly comes from the model's
Noise channels; `observability()` analyzes the chosen sensor set at an
operating point.
"""

from __future__ import annotations

import casadi as ca

from ..ir._linalg import spd_solve
from ..ir.module import entry_ident
from ..ir.state_spec import StateSpec
from ..linearization import LinearizedSystem
from ._assembly import (
    _FilterBase,
    _q_auto,
    emit_filter_module,
    initial_ambient,
    prepared_sensors,
    resolve_gates,
)
from ._kalman import joseph_update, symmetrize


class EKF(_FilterBase):
    """Error-state EKF over a `World` — symbolic recursion + typed Module.
    The analysis surface (`module`, `n_blocks`, `observability`,
    `sigma_horizon`) is the shared `_FilterBase` tail."""

    def __init__(self, world, *,
                 track: dict | None = None,
                 sensors: list[str] | None = None,
                 inputs: list[str] | None = None,
                 discretization: str = "exact",
                 gates: float | dict[str, float] | None = None) -> None:
        """Args:
            track:   `{craft_name: SlotSet}` lower bound of what to estimate
                     (closed under the dynamics; the rest freezes). `None`
                     keeps the full state.
            sensors: measurement full-names (or unambiguous suffixes).
                     `None` keeps every output (of tracked crafts).
            inputs:  known control inputs; `None` keeps all, excluded ones
                     freeze at their default.
            discretization: how F discretizes the dynamics — "exact"
                     (default; jacobian of the full discrete tick) or
                     "euler" (F = I + dt·∂ẋ/∂δ; O(dt²) from exact, much
                     smaller generated deploy code). See LinearizedSystem.
            gates:   optional normalized-innovation-squared threshold: one
                     positive scalar for every sensor, or a mapping from
                     sensor name/suffix to threshold. Rejected updates leave
                     both state and covariance unchanged while still
                     returning their innovation diagnostics.
        """
        sys = LinearizedSystem(world, track=track, sensors=sensors,
                               inputs=inputs, track_mode="closure",
                               discretization=discretization)
        self._bind_system(world, sys)
        resolved_gates = resolve_gates(sys, gates, who="EKF")

        # ---- the Kalman recursion, symbolically, once -------------------
        spec, n_tan = sys.spec, sys.spec.tangent_dim
        x, u = sys.x_sym, sys.u_sym
        dt, t = sys.dt_sym, sys.t_sym
        P = ca.MX.sym("P", n_tan, n_tan)
        Q = ca.MX.sym("Q", n_tan, n_tan)
        F = sys.F_sym

        # predict: auto process noise Q = L Σ Lᵀ baked into the kernel
        # (zero when the model declares none) + an explicit-Q override.
        Q_auto = _q_auto(sys)
        predict_fn = ca.Function(
            "ekf_predict", [x, P, u, dt, t],
            [sys.x_new, symmetrize(F @ P @ F.T + Q_auto)],
            ["x", "P", "u", "dt", "t"], ["x_new", "P_new"])
        predict_q_fn = ca.Function(
            "ekf_predict_with_Q", [x, P, Q, u, dt, t],
            [sys.x_new, symmetrize(F @ P @ F.T + Q)],
            ["x", "P", "Q", "u", "dt", "t"], ["x_new", "P_new"])

        # per-sensor Joseph update (the shared `joseph_update` kernel —
        # see estimation/_kalman.py). `prepared_sensors` eliminates dt and
        # refuses σ=0 sensors — the kernel honestly takes only
        # (x, P, z, u, t).
        x0 = initial_ambient(sys.world, spec)
        zero_dt = ca.MX.zeros(1, 1)
        updates: dict[str, ca.Function] = {}
        diagnostic_updates: dict[str, ca.Function] = {}
        override_updates: dict[str, ca.Function] = {}
        for ps in prepared_sensors(sys, spec, x0=x0, who="EKF"):
            H = ca.substitute(sys.sensors[ps.full].H_sym, dt, zero_dt)
            threshold = resolved_gates[ps.full]

            def expressions(R, ps=ps, H=H, threshold=threshold):
                x_candidate, P_candidate, nu, S = joseph_update(
                    x, P, ps.h, H, R, ps.z, spec)
                nis = ca.dot(nu, spd_solve(S, nu))
                accepted = (ca.MX.ones(1, 1) if threshold is None
                            else nis <= threshold)
                x_result = ca.if_else(accepted, x_candidate, x)
                P_result = ca.if_else(accepted, P_candidate, P)
                return x_result, P_result, nu, S, nis, accepted

            x_upd, P_upd, nu, S, nis, accepted = expressions(ps.R)
            updates[ps.full] = ca.Function(
                f"ekf_update_{entry_ident(ps.full)}",
                [x, P, ps.z, u, t], [x_upd, P_upd],
                ["x", "P", "z", "u", "t"], ["x_new", "P_new"])
            diagnostic_updates[ps.full] = ca.Function(
                f"ekf_update_diagnostic_{entry_ident(ps.full)}",
                [x, P, ps.z, u, t],
                [x_upd, P_upd, nu, S, nis, accepted],
                ["x", "P", "z", "u", "t"],
                ["x_new", "P_new", "innovation", "innovation_covariance",
                 "nis", "accepted"])
            R_override = ca.MX.sym(f"R_{entry_ident(ps.full)}", ps.dim,
                                   ps.dim)
            xo, Po, nuo, So, niso, acceptedo = expressions(R_override)
            override_updates[ps.full] = ca.Function(
                f"ekf_update_with_R_{entry_ident(ps.full)}",
                [x, P, ps.z, R_override, u, t],
                [xo, Po, nuo, So, niso, acceptedo],
                ["x", "P", "z", "R", "u", "t"],
                ["x_new", "P_new", "innovation", "innovation_covariance",
                 "nis", "accepted"])

        self._module = emit_filter_module(
            sys, spec, name=f"{world.name}_ekf", x0=x0,
            predict_fn=predict_fn, predict_q_fn=predict_q_fn,
            updates=updates, diagnostic_updates=diagnostic_updates,
            override_updates=override_updates, gates=resolved_gates)

    # module() / n_blocks / observability() / sigma_horizon() / __repr__
    # are the shared `_FilterBase` analysis tail (estimation/_assembly.py).


# ---------------------------------------------------------------------------
# Low-level h_sym helpers (custom runtime measurements / tests)
# ---------------------------------------------------------------------------

def measurement_slot(spec: StateSpec, name: str):
    """An h_sym callable that reads slot `name` directly from x."""
    slot = spec.slot(name)
    def h_sym(x):
        return x[slot.ambient_offset : slot.ambient_offset + slot.ambient_dim]
    return h_sym


def measurement_component(spec: StateSpec, name: str, component: int):
    """An h_sym callable for a single component of a slot."""
    slot = spec.slot(name)
    if not (0 <= component < slot.ambient_dim):
        raise IndexError(
            f"measurement_component: slot {name!r} has dim "
            f"{slot.ambient_dim}, component {component} out of range")
    def h_sym(x):
        return x[slot.ambient_offset + component]
    return h_sym

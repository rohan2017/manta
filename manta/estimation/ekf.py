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
        ekf.feed("chaser.imu.gyro", gyro_z, t=t)
        ekf.step(dt, t=t)            # fold fresh measurements, then predict

`track` is a *lower bound*: the system expands it (and whatever the chosen
sensors observe) to a set closed under the dynamics and freezes the rest —
see `manta.linearization`. Q/R auto-assembly comes from the model's
Noise channels; `observability()` analyzes the chosen sensor set at an
operating point.
"""

from __future__ import annotations

import casadi as ca
import numpy as np

from ..ir.module import (
    EntryPoint, Hosting, Module, Port, PortField, PortRef, Role, StateField,
    StateLayout, StateRef, entry_ident,
)
from ..ir.state_spec import StateSpec, flatten_nested
from ..linearization import LinearizedSystem, resolve_suffix
from ._kalman import joseph_update


class EKF:
    """Error-state EKF over a `World` — symbolic recursion + typed Module."""

    def __init__(self, world, *,
                 track: dict | None = None,
                 sensors: list[str] | None = None,
                 inputs: list[str] | None = None,
                 discretization: str = "exact") -> None:
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
        """
        sys = LinearizedSystem(world, track=track, sensors=sensors,
                               inputs=inputs, track_mode="closure",
                               discretization=discretization)
        self.sys = sys
        self.world = world
        self.crafts = sys.crafts
        self.spec: StateSpec = sys.spec
        self._blocks = sys.blocks
        self._F_fn = sys.F_fn                      # for observability
        self._sensors = {full: {"full": full, "dim": s.dim,
                                "h_fn": s.h_fn, "H_fn": s.H_fn}
                         for full, s in sys.sensors.items()}

        # ---- the Kalman recursion, symbolically, once -------------------
        spec, n_tan = sys.spec, sys.spec.tangent_dim
        x, u = sys.x_sym, sys.u_sym
        dt, t = sys.dt_sym, sys.t_sym
        P = ca.MX.sym("P", n_tan, n_tan)
        Q = ca.MX.sym("Q", n_tan, n_tan)
        F = sys.F_sym

        def _sym(M):                              # keep P symmetric
            return 0.5 * (M + M.T)

        # predict: auto process noise Q = L Σ Lᵀ baked into the kernel
        # (zero when the model declares none) + an explicit-Q override.
        if sys.L_sym is not None:
            Q_auto = sys.L_sym @ ca.DM(sys.Sigma) @ sys.L_sym.T
        else:
            Q_auto = ca.MX.zeros(n_tan, n_tan)
        predict_fn = ca.Function(
            "ekf_predict", [x, P, u, dt, t],
            [sys.x_new, _sym(F @ P @ F.T + Q_auto)],
            ["x", "P", "u", "dt", "t"], ["x_new", "P_new"])
        predict_q_fn = ca.Function(
            "ekf_predict_with_Q", [x, P, Q, u, dt, t],
            [sys.x_new, _sym(F @ P @ F.T + Q)],
            ["x", "P", "Q", "u", "dt", "t"], ["x_new", "P_new"])

        # per-sensor Joseph update (the shared `joseph_update` kernel —
        # see estimation/_kalman.py). A measurement is dt-independent, so
        # dt is eliminated here (substituted to 0) — the kernel honestly
        # takes only (x, P, z, u, t).
        init_flat = flatten_nested(world._initial_state_dict())
        x0 = spec.pack_any(init_flat)
        zero_dt = ca.MX.zeros(1, 1)
        updates: dict[str, ca.Function] = {}
        for full, s in sys.sensors.items():
            z = ca.MX.sym("z", s.dim)
            h = ca.substitute(s.h_sym, dt, zero_dt)
            H = ca.substitute(s.H_sym, dt, zero_dt)
            if s.L_h_sym is not None and sys.Sigma is not None:
                L_h = ca.substitute(s.L_h_sym, dt, zero_dt)
                R = L_h @ ca.DM(sys.Sigma) @ L_h.T
            else:
                R = ca.MX.zeros(s.dim, s.dim)
            # A σ=0 sensor bakes R = 0: the first update collapses its
            # covariance block to exactly zero, the second meets a singular
            # S and NaNs silently. Refuse at construction. R can be
            # state-dependent (L_h varying with x), so probe a perturbed
            # point too — this is a sampled check, not a proof; an R that
            # vanishes only away from both probes still slips through.
            R_fn = ca.Function("R0", [x, u, t], [R])
            probes = [(x0, sys.u_defaults, 0.0)]
            if ca.depends_on(R, ca.vertcat(x, u, t)):
                x1 = spec.boxplus_num(
                    np.asarray(x0, dtype=float).reshape(-1),
                    1e-3 * np.ones(n_tan))
                probes.append((x1, sys.u_defaults + 1e-3, 1.0))
            if all(not np.any(np.abs(np.asarray(R_fn(*p))) > 0.0)
                   for p in probes):
                raise ValueError(
                    f"EKF: sensor {full!r} has no active noise channel — its "
                    f"baked R is zero and the update is singular (the second "
                    f"fold NaNs). Declare a nonzero noise σ on the sensor, or "
                    f"exclude it via sensors=[...].")
            x_upd, P_upd, _, _ = joseph_update(x, P, h, H, R, z, spec)
            updates[full] = ca.Function(
                f"ekf_update_{entry_ident(full)}",
                [x, P, z, u, t], [x_upd, P_upd],
                ["x", "P", "z", "u", "t"], ["x_new", "P_new"])

        # ---- the typed Module -------------------------------------------
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
            name=f"{world.name}_ekf", state=StateLayout(fields),
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
        """Independent tangent subsystems (block-diagonal predict)."""
        return len(self._blocks)

    def _build_u(self, u: dict[str, float] | None) -> np.ndarray:
        """Resolve `u` to a flat input vector (full or suffix names)."""
        names = self.sys.input_names
        out = self.sys.u_defaults.copy()
        if u:
            index = {n: i for i, n in enumerate(names)}
            for k, v in u.items():
                full = resolve_suffix(k, names, label="input", who="EKF")
                out[index[full]] = float(v)
        return out

    def observability(self, **kwargs):
        """Local observability of the chosen sensor set at an operating
        point (see `manta.estimation.observability`)."""
        from .observability import observability
        return observability(self, **kwargs)

    def sigma_horizon(self, **kwargs):
        """Per-slot σ attainable after a horizon — the covariance
        recursion run open-loop, resolving the weak/slow observability the
        rank test can't (see `manta.estimation.observability`)."""
        from .observability import sigma_horizon
        return sigma_horizon(self, **kwargs)

    def __repr__(self) -> str:
        return (f"<EKF tangent={self.spec.tangent_dim} "
                f"sensors={list(self._sensors)} n_blocks={self.n_blocks}>")


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

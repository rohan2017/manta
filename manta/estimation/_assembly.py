"""Shared EKF/UKF plumbing — everything identical between the twin filters.

The two filters differ only in the math between `(x, P)` in and `(x, P)`
out (Jacobian push vs sigma-point sample). Everything around that math —
the per-sensor measurement preparation (dt elimination, `R = L_h Σ L_hᵀ`
assembly, the σ=0 refusal), the typed-Module emission, and the shared
analysis surface (`_FilterBase`) — is contract, not math, and lives here
exactly once so a Module-shape change can never land in one filter and
silently skip the other. The math kernels stay in `_kalman.py`; the
analysis tools (`observability` / `sigma_horizon` / `nees`) resolve their
filter argument through the helpers at the bottom of this file.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import casadi as ca
import numpy as np

from ..ir.module import (
    EntryPoint, Hosting, Module, Port, PortField, PortRef, Role, StateField,
    StateLayout, StateRef, entry_ident,
)
from ..ir.state_spec import StateSpec, flatten_nested
from ..ir._names import resolve_suffix
from ._kalman import lin_cov, require_active_R


@dataclass(frozen=True)
class PreparedSensor:
    """One sensor's measurement artifacts, ready for an update kernel:
    `h` and `R` with dt eliminated (a measurement is dt-independent), `z`
    a fresh measurement symbol, and the σ=0 refusal already applied."""
    full: str
    dim: int
    z: ca.MX
    h: ca.MX
    R: ca.MX


def initial_ambient(world, spec: StateSpec) -> np.ndarray:
    """The world's initial state packed into the tracked spec's ambient
    vector — the filter's `x` init and the R-probe operating point."""
    return spec.pack_any(flatten_nested(world._initial_state_dict()))


def sensor_R_expr(sys, sm) -> ca.MX:
    """One sensor's measurement covariance `R = L_h Σ L_hᵀ` with dt
    eliminated (a measurement is dt-independent — the filters' convention),
    or a zero `dim×dim` when the model declares no noise on it. The single
    R assembly shared by `prepared_sensors` (both filters' baked update
    kernels) and `sigma_horizon`'s per-sensor recursion."""
    L_h = (ca.substitute(sm.L_h_sym, sys.dt_sym, ca.MX.zeros(1, 1))
           if sm.L_h_sym is not None and sys.Sigma is not None
           else None)
    return lin_cov(L_h, ca.DM(sys.Sigma) if L_h is not None else None,
                   sm.dim)


def prepared_sensors(sys, spec: StateSpec, *, x0: np.ndarray,
                     who: str) -> list[PreparedSensor]:
    """Prepare every chosen sensor of a `LinearizedSystem` for update-kernel
    construction. Refuses σ=0 sensors (`require_active_R`) so both filters
    reject a singular R identically, at construction."""
    x, u, dt, t = sys.x_sym, sys.u_sym, sys.dt_sym, sys.t_sym
    zero_dt = ca.MX.zeros(1, 1)
    out = []
    for full, s in sys.sensors.items():
        z = ca.MX.sym("z", s.dim)
        h = ca.substitute(s.h_sym, dt, zero_dt)
        R = sensor_R_expr(sys, s)
        R_fn = ca.Function("R0", [x, u, t], [R])
        require_active_R(R, R_fn, ca.vertcat(x, u, t),
                         x0=x0, u_defaults=sys.u_defaults, spec=spec,
                         full=full, who=who)
        out.append(PreparedSensor(full=full, dim=s.dim, z=z, h=h, R=R))
    return out


def emit_filter_module(sys, spec: StateSpec, *, name: str, x0: np.ndarray,
                       predict_fn: ca.Function, predict_q_fn: ca.Function,
                       updates: dict[str, ca.Function]) -> Module:
    """The typed filter Module both twins emit — identical shape by
    construction:

        state    x  (manifold)              P  (tangent covariance)
        entries  predict(x,P, u,dt,t)              — auto process noise
                 predict_with_Q(x,P, Q,u,dt,t)     — explicit-Q override
                 update_<sensor>(x,P, z,u,t)       — one per chosen sensor
    """
    n_tan = spec.tangent_dim
    fields = (
        StateField("x", "manifold", (spec.ambient_dim,),
                   init=x0, spec=spec),
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
    return Module(name=name, state=StateLayout(fields),
                  ports=tuple(ports), functions=functions,
                  entry_points=tuple(entries), hosting=Hosting.HELD)


def _q_auto(sys) -> ca.MX:
    """The model's auto process noise `Q = L Σ Lᵀ` over the tracked tangent
    space — zero when the model declares no noise. Both filters bake this
    into their auto-Q predict kernel."""
    return lin_cov(sys.L_sym,
                   ca.DM(sys.Sigma) if sys.L_sym is not None else None,
                   sys.spec.tangent_dim)


class _FilterBase:
    """The twin filters' shared non-math surface. Each subclass's ctor
    does its own recursion math, then everything after — the system/spec
    attribute block (`_bind_system`), the Module accessor, and the
    analysis tail — is identical by inheritance, so it can never drift
    between the twins. The analysis is *linearized* either way (it reads
    the shared `LinearizedSystem`), which is exactly why it is shared."""

    def _bind_system(self, world, sys) -> None:
        """The ctor attribute block both filters open with."""
        self.sys = sys
        self.world = world
        self.crafts = sys.crafts
        self.spec: StateSpec = sys.spec

    def module(self) -> Module:
        """The typed `Module` IR a backend lowers."""
        return self._module

    @property
    def n_blocks(self) -> int:
        """Independent tangent subsystems (structurally decoupled crafts) —
        a structural diagnostic; see `linearization.partition_blocks`."""
        return len(self.sys.blocks)

    def observability(self, **kwargs):
        """Local observability of the chosen sensor set at an operating
        point (see `manta.estimation.observability`)."""
        from .observability import observability
        return observability(self, **kwargs)

    def sigma_horizon(self, **kwargs):
        """Per-slot σ attainable after a horizon — the linearized
        covariance recursion run open-loop, resolving the weak/slow
        observability the rank test can't (see
        `manta.estimation.observability`)."""
        from .observability import sigma_horizon
        return sigma_horizon(self, **kwargs)

    def __repr__(self) -> str:
        return (f"<{type(self).__name__} tangent={self.spec.tangent_dim} "
                f"sensors={list(self.sys.sensors)} n_blocks={self.n_blocks}>")


# ---------------------------------------------------------------------------
# Analysis-layer resolution — how observability / sigma_horizon / nees turn
# their arguments into the filter IR, a sensor subset, and per-tick controls.
# ---------------------------------------------------------------------------

def _resolve_ir(ekf):
    """Require a filter transform carrying the IR (`EKF`/`UKF`); the runtime
    view doesn't carry it. The analysis is the *linearized* observability of
    the model's sensor set, so it applies to either filter unchanged."""
    if isinstance(ekf, _FilterBase):
        return ekf
    raise TypeError(
        f"observability: expected the EKF/UKF transform (e.g. "
        f"EKF(world, ...)), got {type(ekf).__name__}")


def _resolve_estimator(world, estimator):
    """The `estimator=` convention of the world-level analyses
    (`observability_trajectory`, `nees`): an `EKF`/`UKF` *instance* over
    `world`, or a class/callable applied to it; `None` defaults to `EKF`."""
    if estimator is None:
        from .ekf import EKF          # deferred — ekf.py imports this module
        estimator = EKF
    return _resolve_ir(estimator if not callable(estimator)
                       else estimator(world))


def resolve_sensor_set(ekf, sensors, *, who: str) -> list[str]:
    """The chosen sensor full-names. `sensors=None` keeps every registered
    sensor; otherwise each entry resolves like everywhere else (full name or
    unique `.<suffix>`) — an unknown or ambiguous name raises instead of
    silently dropping a typo. Shared by observability and `nees`."""
    fulls = list(ekf.sys.sensors)
    if sensors is None:
        return fulls
    chosen = {resolve_suffix(s, fulls, label="sensor", who=who)
              for s in sensors}
    return [f for f in fulls if f in chosen]


def _controls_at(control, t) -> dict[str, Any]:
    """The analyses' shared `control` convention at time `t`: `None` ⇒ no
    overrides, a dict is constant, a callable is sampled at `t`."""
    if control is None:
        return {}
    return control(t) if callable(control) else dict(control)

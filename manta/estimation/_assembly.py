"""Shared EKF/UKF plumbing — everything identical between the twin filters.

The two filters differ only in the math between `(x, P)` in and `(x, P)`
out (Jacobian push vs sigma-point sample). Everything around that math —
the per-sensor measurement preparation (dt elimination, `R = L_h Σ L_hᵀ`
assembly, the σ=0 refusal), the typed-Module emission, and the input-vector
resolution — is contract, not math, and lives here exactly once so a
Module-shape change can never land in one filter and silently skip the
other. The math kernels stay in `_kalman.py`.
"""

from __future__ import annotations

from dataclasses import dataclass

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
        L_h = (ca.substitute(s.L_h_sym, dt, zero_dt)
               if s.L_h_sym is not None and sys.Sigma is not None
               else None)
        R = lin_cov(L_h, ca.DM(sys.Sigma) if L_h is not None else None,
                    s.dim)
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
    return Module(name=name, state=StateLayout(fields),
                  ports=tuple(ports), functions=functions,
                  entry_points=tuple(entries), hosting=Hosting.HELD)


def resolve_u(sys, u: dict[str, float] | None, *, who: str) -> np.ndarray:
    """Resolve a `{name: value}` input dict (full or unambiguous suffix
    names) to the system's flat input vector, defaults elsewhere."""
    names = sys.input_names
    out = sys.u_defaults.copy()
    if u:
        index = {n: i for i, n in enumerate(names)}
        for k, v in u.items():
            full = resolve_suffix(k, names, label="input", who=who)
            out[index[full]] = float(v)
    return out

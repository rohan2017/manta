"""Shared EKF/UKF/INS plumbing for the common filter contract.

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

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import casadi as ca
import numpy as np

from ..ir._names import resolve_suffix
from ..ir.module import (
    EntryPoint,
    Hosting,
    Module,
    ModuleKind,
    Port,
    PortField,
    PortRef,
    Role,
    StateField,
    StateLayout,
    StateRef,
    entry_ident,
)
from ..ir.state_spec import StateSpec, flatten_nested
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


def resolve_gates(sys, gates, *, who: str) -> dict[str, float | None]:
    """Resolve construction-time NIS gates to full sensor names.

    ``None`` disables gating, one positive scalar applies to every sensor,
    and a mapping may use full or unambiguous suffix names. Unmentioned
    sensors remain ungated. The threshold is deliberately part of the
    generated estimator artifact: deployed and numpy filters cannot silently
    disagree because of runtime-only policy.
    """
    out: dict[str, float | None] = {name: None for name in sys.sensors}
    if gates is None:
        return out
    if isinstance(gates, (int, float)):
        items = [(name, gates) for name in sys.sensors]
    elif isinstance(gates, Mapping):
        items = []
        for name, value in gates.items():
            full = resolve_suffix(name, list(sys.sensors), label="sensor",
                                  who=who)
            items.append((full, value))
    else:
        raise TypeError(f"{who}: gates must be None, a scalar, or a mapping")
    for full, value in items:
        threshold = float(value)
        if not np.isfinite(threshold) or threshold <= 0.0:
            raise ValueError(
                f"{who}: NIS gate for {full!r} must be finite and > 0, "
                f"got {value!r}")
        out[full] = threshold
    return out


def initial_ambient(world, spec: StateSpec) -> np.ndarray:
    """The world's initial state packed into the tracked spec's ambient
    vector — the filter's `x` init and the R-probe operating point."""
    return spec.pack_projected(flatten_nested(world._initial_state_dict()))


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
                       updates: dict[str, ca.Function],
                       diagnostic_updates: dict[str, ca.Function],
                       override_updates: dict[str, ca.Function],
                       gates: dict[str, float | None],
                       metadata_extra: Mapping[str, Any] | None = None) -> Module:
    """The typed filter Module both twins emit — identical shape by
    construction:

        state    x  (manifold)              P  (tangent covariance)
        entries  predict(x,P, u,dt,t)              — auto process noise
                 predict_with_Q(x,P, Q,u,dt,t)     — explicit-Q override
                 update_<sensor>(x,P, z,u,t)       — compatible state fold
                 update_diagnostic_<sensor>(...)  — fold + innovation/NIS
                 update_with_R_<sensor>(...,R,...) — per-sample covariance
    """
    n_tan = spec.tangent_dim
    fields = (
        StateField("x", "manifold", (spec.ambient_dim,),
                   init=x0, spec=spec),
        StateField("P", "matrix", (n_tan, n_tan),
                   init=np.eye(n_tan) * 1e-2),
    )
    input_fields = getattr(sys, "input_fields", None)
    if input_fields is None:
        input_fields = tuple(
            PortField(n, 1, float(sys.input_defaults[n]),
                      rate=sys.sample_rates.get(n))
            for n in sys.input_names)
    ports = [
        Port("u", Role.CONTROL, (sum(f.dim for f in input_fields),),
             fields=tuple(input_fields)),
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
        r_name = f"R_{ident}"
        innovation = f"innovation_{ident}"
        innovation_cov = f"innovation_covariance_{ident}"
        nis = f"nis_{ident}"
        accepted = f"accepted_{ident}"
        ports.extend((
            Port(r_name, Role.MATRIX, (s.dim, s.dim)),
            Port(innovation, Role.DIAGNOSTIC, (s.dim,)),
            Port(innovation_cov, Role.DIAGNOSTIC, (s.dim, s.dim)),
            Port(nis, Role.DIAGNOSTIC, (1,)),
            Port(accepted, Role.DIAGNOSTIC, (1,)),
        ))
        functions[f"update_{ident}"] = updates[full]
        functions[f"update_diagnostic_{ident}"] = diagnostic_updates[full]
        functions[f"update_with_R_{ident}"] = override_updates[full]
        entries.append(EntryPoint(
            f"update_{ident}", f"update_{ident}",
            (StateRef("x"), StateRef("P"), PortRef(full),
             PortRef("u"), PortRef("t")),
            writes=("x", "P")))
        returns = (innovation, innovation_cov, nis, accepted)
        entries.append(EntryPoint(
            f"update_diagnostic_{ident}", f"update_diagnostic_{ident}",
            (StateRef("x"), StateRef("P"), PortRef(full),
             PortRef("u"), PortRef("t")),
            writes=("x", "P"), returns=returns))
        entries.append(EntryPoint(
            f"update_with_R_{ident}", f"update_with_R_{ident}",
            (StateRef("x"), StateRef("P"), PortRef(full), PortRef(r_name),
             PortRef("u"), PortRef("t")),
            writes=("x", "P"), returns=returns))
    # Keep deploy/runtime metadata out of naming conventions. Backends and
    # consumers can inspect this immutable map directly.
    metadata = {"nis_gates": MappingProxyType(dict(gates))}
    if metadata_extra:
        overlap = set(metadata) & set(metadata_extra)
        if overlap:
            raise ValueError(
                f"emit_filter_module: duplicate metadata keys {sorted(overlap)}")
        metadata.update(metadata_extra)
    return Module(name=name, state=StateLayout(fields),
                  ports=tuple(ports), functions=functions,
                  entry_points=tuple(entries), kind=ModuleKind.FILTER,
                  hosting=Hosting.HELD,
                  metadata=metadata)


def _q_auto(sys) -> ca.MX:
    """The model's auto process noise `Q = L Σ Lᵀ` over the tracked tangent
    space — zero when the model declares no noise. Both filters bake this
    into their auto-Q predict kernel."""
    return lin_cov(sys.L_sym,
                   ca.DM(sys.Sigma) if sys.L_sym is not None else None,
                   sys.spec.tangent_dim)


class _FilterBase:
    """The filters' shared non-math surface. Each subclass's ctor
    does its own recursion math, then everything after — the system/spec
    attribute block (`_bind_system`), the Module accessor, and the
    analysis tail — is identical by inheritance, so it can never drift
    between the twins. The analysis is *linearized* either way (it reads
    the shared `LinearizedSystem`), which is exactly why it is shared."""

    _module: Module

    def _bind_system(self, world, sys) -> None:
        """The ctor attribute block both filters open with."""
        self.sys = sys
        self.authoring_world = world
        self.world = sys.world
        self.model = sys.model
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
    """Require a filter transform carrying the IR (`EKF`/`UKF`/`INS`); the runtime
    view doesn't carry it. The analysis is the *linearized* observability of
    the model's sensor set, so it applies to either filter unchanged."""
    if isinstance(ekf, _FilterBase):
        return ekf
    raise TypeError(
        f"observability: expected an EKF/UKF/INS transform (e.g. "
        f"EKF(world, ...)), got {type(ekf).__name__}")


def _resolve_estimator(world, estimator):
    """The `estimator=` convention of the world-level analyses
    (`observability_trajectory`, `nees`): an `EKF`/`UKF` *instance* over
    `world`, or a class/callable applied to it; `None` defaults to `EKF`."""
    if estimator is None:
        from .ekf import EKF  # deferred — ekf.py imports this module
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


def estimator_inputs(estimator, controls, *, reading=None,
                     dt: float | None = None) -> dict[str, Any]:
    """Merge physical controls with any sensor samples driving prediction.

    EKF/UKF declare no ``prediction_inputs`` and pass through unchanged. INS
    declares its selected IMU outputs in Module metadata; trajectory/NEES
    tools supply those readings from the truth simulator through this one
    transform-neutral adapter.
    """
    out = dict(controls or {})
    metadata = estimator.module().metadata
    names = metadata.get("prediction_inputs", ())
    packet_map = dict(metadata.get("preintegration_input_map", {}))
    if packet_map and reading is not None:
        if dt is None:
            raise ValueError(
                f"{type(estimator).__name__}: preintegrated truth adapter "
                "needs dt")
        from .imu_preintegrator import _single_sample_packet
        accel_name = packet_map["end_accel"]
        gyro_name = packet_map["end_gyro"]
        accel = out[accel_name] if accel_name in out else reading(accel_name)
        gyro = out[gyro_name] if gyro_name in out else reading(gyro_name)
        packet = _single_sample_packet(
            accel=accel, gyro=gyro, dt=dt,
            accel_noise_sigma=float(estimator.sys.imu.accel_noise_sigma),
            gyro_noise_sigma=float(estimator.sys.imu.gyro_noise_sigma))
        for short, full in packet_map.items():
            out.setdefault(full, packet[short])
        return out
    if names and reading is None:
        missing = [name for name in names if name not in out]
        if missing:
            raise ValueError(
                f"{type(estimator).__name__}: prediction needs readings "
                f"{missing}")
    for name in names:
        if name not in out:
            value = reading(name)
            if value is None:
                raise ValueError(f"prediction input {name!r} has no reading")
            out[name] = value
    return out

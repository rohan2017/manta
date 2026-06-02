"""Observability analysis — does your sensor set actually constrain your
state?

A faithful EKF (correct `f`, correct Jacobians) is still only as good as
the *observability* of the model it filters: whether the measurements,
through the dynamics, pin down every state direction. Observability is a
property of *(dynamics + sensor set + operating point)* — **not** of the
model alone — so "make the model correct and the EKF follows" silently
fails on unobservable modes. They don't error; they drift, and the filter
reports tight covariance while doing it. (The canonical example: heading
is unobservable from GPS + DVL + gyro — only its *rate* is measured — so a
submarine's yaw estimate wanders until a compass is added.)

manta is well placed to catch this automatically, because the symbolic
state-transition `F` and per-sensor measurement Jacobians `H` already
exist on the `EKF` IR (via the shared `Linearization`). This module builds
the discrete observability matrix at an operating point

    O = [H; H·F; H·F²; …; H·F^(n-1)]            (n = tangent dimension)

and reports its rank. A rank deficiency means an unobservable subspace;
its null space, projected onto the state slots, tells you *which* states
(e.g. ``sub.orientation``) you can't see — turning a silent drift into a
setup-time warning.

Usage::

    from manta.estimation import observability
    rep = observability(EKF(world))
    print(rep.summary())
    if not rep.observable:
        ...   # add a sensor, or accept the drift on those states

This is *local* observability at the supplied operating point (the
standard linear test, evaluated along the dynamics for ``n`` steps). Check
a few representative points if your system is strongly nonlinear.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class ObservabilityReport:
    """Result of :func:`observability`.

    Attributes:
        tangent_dim   — total tangent (error-state) dimension `n`.
        rank          — rank of the observability matrix (≤ n).
        observable    — True iff `rank == tangent_dim`.
        sensors       — sensor output names folded into the analysis.
        unobservable  — list of `(slot_name, strength)` for slots with a
                        component in the unobservable subspace; `strength`
                        is the Frobenius norm of the null-space basis
                        restricted to that slot (0 ⇒ fully observable).
        singular_values — singular values of the observability matrix.
    """

    tangent_dim: int
    rank: int
    sensors: list[str]
    unobservable: list[tuple[str, float]] = field(default_factory=list)
    singular_values: np.ndarray | None = None

    @property
    def observable(self) -> bool:
        return self.rank == self.tangent_dim

    def summary(self) -> str:
        head = ("✓ fully observable" if self.observable
                else "⚠ NOT fully observable")
        lines = [f"{head}: rank {self.rank}/{self.tangent_dim}  "
                 f"(sensors: {', '.join(self.sensors) or 'none'})"]
        if not self.observable:
            lines.append("  unobservable state directions:")
            for name, strength in self.unobservable:
                lines.append(f"    - {name:28s} (strength {strength:.2f})")
            lines.append("  → add a sensor that constrains these, or accept "
                         "that they will drift.")
        return "\n".join(lines)

    def __str__(self) -> str:  # pragma: no cover - thin
        return self.summary()


def _resolve_ir(ekf):
    """Accept either an `EKF` IR or a `NumpyEKF` runtime (unwrap `._ekf`)."""
    from .ekf import EKF
    if isinstance(ekf, EKF):
        return ekf
    inner = getattr(ekf, "_ekf", None)
    if isinstance(inner, EKF):
        return inner
    raise TypeError(
        f"observability: expected an EKF IR or NumpyEKF, got "
        f"{type(ekf).__name__}")


def _operating_point(ekf, state) -> np.ndarray:
    """Pack `state` (nested or flat, merged over the world's initial
    state) into the EKF spec's ambient vector — same convention as
    `NumpyEKF.reset`."""
    spec = ekf.spec
    flat: dict[str, Any] = {}
    for owner, slots in ekf.world._initial_state_dict().items():
        for k, v in slots.items():
            flat[f"{owner}.{k}"] = v
    if state is not None:
        for k, v in state.items():
            if isinstance(v, dict):
                for slot, val in v.items():
                    flat[f"{k}.{slot}"] = val
            else:
                flat[k] = v
    return spec.pack({k: v for k, v in flat.items() if k in spec})


def _select_sensors(ekf, sensors):
    """Pick the `(H_fn, full_name)` pairs to analyze. `sensors=None` uses
    every registered sensor; otherwise each entry matches a sensor whose
    full name equals it or ends with `.<entry>`."""
    out = []
    for key, spec_o in ekf._sensors.items():
        full = spec_o["full"]
        if sensors is None or full in sensors or any(
                full == s or full.endswith("." + s) for s in sensors):
            out.append((spec_o["H_fn"], full))
    return out


def observability(ekf, *, state=None, inputs=None, sensors=None,
                  dt: float = 0.02, t: float = 0.0,
                  rtol: float = 1e-6) -> ObservabilityReport:
    """Local observability of an `EKF` at an operating point.

    Args:
        ekf      — an `EKF` IR (`EKF(world)`) or a `NumpyEKF`.
        state    — operating-point state (nested `{owner: {slot: v}}` or
                   flat), merged over the world's initial state. Defaults
                   to the world's initial state.
        inputs   — operating-point control inputs `{name: value}`.
        sensors  — restrict the analysis to these sensor outputs (full
                   names or unambiguous suffixes); `None` ⇒ all registered.
                   Use this to ask "what would I lose without the compass?"
        dt, t    — the step / clock the Jacobians are evaluated at.
        rtol     — relative singular-value threshold for the rank test.

    Returns:
        :class:`ObservabilityReport`.
    """
    ir = _resolve_ir(ekf)
    spec = ir.spec
    n = spec.tangent_dim
    x = _operating_point(ir, state)
    u = ir._build_u(inputs)

    F = np.asarray(ir._F_fn(x, u, dt, t), dtype=float).reshape(n, n)
    pairs = _select_sensors(ir, sensors)
    names = [full for _, full in pairs]
    if not pairs:
        # No measurements ⇒ nothing observable.
        return ObservabilityReport(tangent_dim=n, rank=0, sensors=[],
                                   unobservable=[(s.name, 1.0)
                                                 for s in spec.slots],
                                   singular_values=np.zeros(1))

    H = np.vstack([np.asarray(h(x, u, dt, t), dtype=float).reshape(-1, n)
                   for h, _ in pairs])

    # Discrete observability matrix O = [H; H F; …; H F^(n-1)].
    blocks, Fp = [], np.eye(n)
    for _ in range(n):
        blocks.append(H @ Fp)
        Fp = Fp @ F
    O = np.vstack(blocks)

    U, sv, Vt = np.linalg.svd(O)
    smax = sv[0] if sv.size else 0.0
    rank = int(np.sum(sv > rtol * smax)) if smax > 0 else 0

    # Null space of O = unobservable subspace (rows of Vt past the rank).
    null_basis = Vt[rank:] if rank < n else np.zeros((0, n))
    unobservable: list[tuple[str, float]] = []
    for s in spec.slots:
        lo, hi = s.tangent_offset, s.tangent_offset + s.tangent_dim
        strength = (float(np.linalg.norm(null_basis[:, lo:hi]))
                    if null_basis.size else 0.0)
        if strength > 1e-6:
            unobservable.append((s.name, strength))

    return ObservabilityReport(
        tangent_dim=n, rank=rank, sensors=names,
        unobservable=unobservable, singular_values=sv)

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
submarine's yaw estimate wanders until a compass is added. Caveat: that
holds on a non-rotating model. On a spinning planet the Earth rate makes
heading *weakly* observable through the dynamics — gyrocompassing — at
singular values far below this rank test's threshold; `sigma_horizon`
below is the tool that resolves such slow channels.)

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

The rank test is binary and short-horizon (`n` steps ≈ a fraction of a
second), so it cannot grade *weak* observability — information that
arrives through slow dynamics over many seconds. For that, use
:func:`sigma_horizon`: it runs the filter's own covariance recursion
(no data needed — F, H, Q, R are already baked) along the nominal
trajectory and reports how tight each state *can* get after `horizon`
seconds. A slot this rank test calls unobservable but whose σ shrinks
there is weakly observable, with the convergence time read off directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from ..ir.module import entry_ident
from ..ir._names import resolve_suffix
from ._kalman import joseph_update_np, lin_cov, symmetrize


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
        basis         — orthonormal basis of the **observable** subspace,
                        shape `(tangent_dim, rank)`. Feed to `nees(...,
                        observable_basis=...)` to check consistency only
                        where the state is actually observable.
    """

    tangent_dim: int
    rank: int
    sensors: list[str]
    unobservable: list[tuple[str, float]] = field(default_factory=list)
    singular_values: np.ndarray | None = None
    basis: np.ndarray | None = None

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
    """Require the `EKF` transform (the runtime doesn't carry the IR)."""
    from .ekf import EKF
    if isinstance(ekf, EKF):
        return ekf
    raise TypeError(
        f"observability: expected the EKF transform (EKF(world, ...)), got "
        f"{type(ekf).__name__}")


def _operating_point(ekf, state) -> np.ndarray:
    """Pack `state` (nested or flat, merged over the world's initial
    state) into the EKF spec's ambient vector — same convention as
    the runtime's `reset`."""
    return ekf.spec.pack_any(state, base=ekf.world._initial_state_dict())


def _select_sensors(ekf, sensors, *, who: str):
    """Pick the `(H_fn, full_name)` pairs to analyze. `sensors=None` uses
    every registered sensor; otherwise each entry resolves like everywhere
    else (full name or unique `.<suffix>`) — an unknown or ambiguous name
    raises instead of silently dropping a typo."""
    fulls = list(ekf.sys.sensors)
    if sensors is None:
        chosen = set(fulls)
    else:
        chosen = {resolve_suffix(s, fulls, label="sensor", who=who)
                  for s in sensors}
    return [(ekf.sys.sensors[f].H_fn, f) for f in fulls if f in chosen]


def observability(ekf, *, state=None, inputs=None, sensors=None,
                  dt: float = 0.02, t: float = 0.0,
                  rtol: float = 1e-6) -> ObservabilityReport:
    """Local observability of an `EKF` at an operating point.

    Args:
        ekf      — the `EKF` transform (`EKF(world, ...)`).
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

    pairs = _select_sensors(ir, sensors, who="observability")
    names = [full for _, full in pairs]
    if not pairs:
        return ObservabilityReport(n, 0, [], [(s.name, 1.0) for s in spec.slots],
                                   np.zeros(1), np.zeros((n, 0)))
    return _report_from_O(_local_O(ir, x, u, dt, t, pairs, n), spec, names, rtol)


def _local_O(ir, x, u, dt, t, pairs, n) -> np.ndarray:
    """Local discrete observability matrix at one point: `[H; H·F; …;
    H·F^(n-1)]`. `F^p` stays bounded (F ≈ I + A·dt), so this is well
    conditioned — unlike `H·F^k` propagated from a trajectory start."""
    F = np.asarray(ir.sys.F_fn(x, u, dt, t), dtype=float).reshape(n, n)
    H = np.vstack([np.asarray(h(x, u, dt, t), dtype=float).reshape(-1, n)
                   for h, _ in pairs])
    blocks, Fp = [], np.eye(n)
    for _ in range(n):
        blocks.append(H @ Fp)
        Fp = Fp @ F
    return np.vstack(blocks)


def _report_from_O(O, spec, names, rtol) -> ObservabilityReport:
    """SVD of an observability matrix → rank, observable basis, and the
    unobservable slots."""
    n = spec.tangent_dim
    _, sv, Vt = np.linalg.svd(O)
    smax = sv[0] if sv.size else 0.0
    rank = int(np.sum(sv > rtol * smax)) if smax > 0 else 0

    basis = Vt[:rank].T.copy() if rank else np.zeros((n, 0))     # observable
    null_basis = Vt[rank:] if rank < n else np.zeros((0, n))     # unobservable
    unobservable: list[tuple[str, float]] = []
    for s in spec.slots:
        lo, hi = s.tangent_offset, s.tangent_offset + s.tangent_dim
        strength = (float(np.linalg.norm(null_basis[:, lo:hi]))
                    if null_basis.size else 0.0)
        if strength > 1e-6:
            unobservable.append((s.name, strength))

    return ObservabilityReport(
        tangent_dim=n, rank=rank, sensors=names,
        unobservable=unobservable, singular_values=sv, basis=basis)


def observability_trajectory(world, *, dt: float, steps: int,
                             control: "Callable | dict | None" = None,
                             sensors: list[str] | None = None,
                             samples: int = 30,
                             rtol: float = 1e-6) -> ObservabilityReport:
    """Observability accumulated **over a trajectory**, not at one point.

    Local `observability()` is evaluated at a single operating point, where
    a state can look unobservable that a maneuver would reveal (absolute
    heading from GPS+DVL is the classic case: unobservable at rest,
    observable while turning). This rolls out the *deterministic* nominal
    trajectory under `control`, evaluates the local observability matrix at
    ~`samples` points along it, and reports the rank / unobservable slots /
    observable basis of their **union** — i.e. a direction counts as
    observable if it is locally observable at *any* configuration the
    trajectory visits. (Union of bounded local matrices, so it stays well
    conditioned — unlike propagating one matrix through the whole run,
    where the integrators blow the conditioning up.)

    Args mirror `nees`/`observability`: `control` is `{name: value}` or a
    `t -> {name: value}` callable; `sensors` restricts the suite; `samples`
    caps how many trajectory points are linearized.
    """
    from ..codegen.numpy import TargetNumpy
    from ..sim import Sim
    from .ekf import EKF

    sim_ir, ekf_ir = Sim(world), EKF(world)
    spec = ekf_ir.spec
    n = spec.tangent_dim
    sim = TargetNumpy(sim_ir)

    pairs = _select_sensors(ekf_ir, sensors, who="observability_trajectory")
    names = [full for _, full in pairs]
    if not pairs:
        return ObservabilityReport(n, 0, [], [(s.name, 1.0) for s in spec.slots],
                                   np.zeros(1), np.zeros((n, 0)))

    def truth_vec():
        return spec.pack_any(sim.state)

    every = max(1, steps // max(1, samples))
    blocks = []
    for i in range(steps):
        t = i * dt
        u_dict = control(t) if callable(control) else (control or {})
        for nm, v in u_dict.items():
            sim.command(nm).set(v)
        if i % every == 0:
            blocks.append(_local_O(ekf_ir, truth_vec(),
                                   ekf_ir._build_u(u_dict or None), dt, t, pairs, n))
        sim.step(dt)
    return _report_from_O(np.vstack(blocks), spec, names, rtol)


# ---------------------------------------------------------------------------
# Covariance-horizon analysis — how tight can each state get in T seconds?
# ---------------------------------------------------------------------------

@dataclass
class SigmaHorizonReport:
    """Result of :func:`sigma_horizon`.

    Attributes:
        horizon, dt — the analyzed window and step.
        sensors     — sensor output names folded into the recursion.
        times       — recorded sample times, shape `(k,)`.
        sigmas      — `{slot_name: (k,) array}` of the slot's
                      worst-direction σ (√ of the largest eigenvalue of its
                      tangent covariance block) at each recorded time.
        P_final     — full tangent covariance at `horizon`.
    """

    horizon: float
    dt: float
    sensors: list[str]
    times: np.ndarray
    sigmas: dict[str, np.ndarray]
    P_final: np.ndarray

    def sigma0(self, slot: str) -> float:
        return float(self.sigmas[slot][0])

    def sigmaT(self, slot: str) -> float:
        return float(self.sigmas[slot][-1])

    def summary(self) -> str:
        lines = [f"σ-horizon {self.horizon:g} s @ dt {self.dt:g}  "
                 f"(sensors: {', '.join(self.sensors) or 'none'})",
                 f"  {'slot':28s} {'σ₀':>9}   {'σ(T)':>9}  {'ratio':>7}"]
        for name, sig in self.sigmas.items():
            s0, sT = float(sig[0]), float(sig[-1])
            ratio = sT / s0 if s0 > 0 else float("inf")
            if ratio < 0.5:
                verdict = "converges"
            elif ratio < 0.95:
                verdict = "converging (slow)"
            elif ratio <= 1.05:
                verdict = "static — unconstrained over this horizon"
            else:
                verdict = "growing — process noise, unconstrained"
            lines.append(f"  {name:28s} {s0:>9.3g} → {sT:>9.3g}  "
                         f"{ratio:>7.3f}  {verdict}")
        lines.append(
            "  (a slot that converges here but fails the observability() "
            "rank test is weakly observable — constrained through slow "
            "dynamics, e.g. gyrocompassing)")
        return "\n".join(lines)

    def __str__(self) -> str:  # pragma: no cover - thin
        return self.summary()


def sigma_horizon(ekf, *, horizon: float, dt: float = 0.02,
                  state=None, control: "Callable | dict | None" = None,
                  sensors: list[str] | None = None,
                  P0: "np.ndarray | float | None" = None,
                  Q: np.ndarray | None = None,
                  t0: float = 0.0, samples: int = 60,
                  record: int = 200) -> SigmaHorizonReport:
    """Per-slot σ attainable after `horizon` seconds — the covariance
    recursion run open-loop (no data), i.e. the linear-Gaussian CRLB along
    the nominal trajectory.

    Where :func:`observability` asks the binary "is this direction in the
    observable subspace?" over an `n`-step window, this asks the graded
    question the rank test cannot: *how tight does each state get, and how
    fast?* It propagates the nominal state with the filter's own predict,
    evaluates F, Q, H, R along it (all already baked on the EKF), and runs

        P ← F P Fᵀ + Q;   per due sensor: P ← Joseph(P, H, R)

    recording each slot's worst-direction σ. Slow channels the rank test
    misses — heading from the Earth rate (gyrocompassing), drag-coupled
    biases — show up as σ trajectories with their convergence time
    readable directly.

    Args:
        ekf      — the `EKF` transform (`EKF(world, ...)`).
        horizon  — analysis window (s).
        dt       — filter step (s).
        state    — starting operating point (nested or flat), merged over
                   the world's initial state.
        control  — `{name: value}` or `t -> {name: value}` nominal inputs.
        sensors  — restrict to these outputs (full names / suffixes);
                   `None` ⇒ all registered on the EKF.
        P0       — initial tangent covariance: full matrix, diagonal
                   vector, or scalar·I. Default `0.1·I` (as `nees`). Use a
                   generous prior on the states in question — σ can only
                   show convergence the prior leaves room for.
        Q        — process-noise override (else the model's auto `L Σ Lᵀ`).
        t0       — start time (matters on a time-dependent world, e.g. a
                   rotating planet).
        samples  — how many points along the trajectory F/Q/H/R are
                   re-evaluated at (held constant in between).
        record   — max number of recorded σ samples.

    Returns:
        :class:`SigmaHorizonReport`.
    """
    import casadi as ca

    ir = _resolve_ir(ekf)
    sys = ir.sys
    spec = ir.spec
    n = spec.tangent_dim
    steps = max(1, int(round(horizon / dt)))

    # --- P0: scalar | diag vector | full matrix --------------------------
    if P0 is None:
        P = np.eye(n) * 0.1
    else:
        P0 = np.asarray(P0, dtype=float)
        if P0.ndim == 0:
            P = np.eye(n) * float(P0)
        elif P0.ndim == 1:
            P = np.diag(P0)
        else:
            P = P0.copy()
    if P.shape != (n, n):
        raise ValueError(f"sigma_horizon: P0 must be scalar, ({n},) or "
                         f"({n},{n}); got shape {np.shape(P0)}")

    # --- per-sensor H/R/rate ---------------------------------------------
    x_s, u_s, dt_s, t_s = sys.x_sym, sys.u_sym, sys.dt_sym, sys.t_sym
    zero_dt = ca.MX.zeros(1, 1)
    chosen = []                                  # (full, H_fn, R_fn, period)
    for H_fn, full in _select_sensors(ir, sensors, who="sigma_horizon"):
        s = sys.sensors[full]
        L_h = (ca.substitute(s.L_h_sym, dt_s, zero_dt)
               if s.L_h_sym is not None and sys.Sigma is not None else None)
        R_expr = lin_cov(L_h, ca.DM(sys.Sigma) if L_h is not None else None,
                         s.dim)
        R_fn = ca.Function(f"R_{entry_ident(full)}",
                           [x_s, u_s, t_s], [R_expr])
        rate = sys.sample_rates.get(full)
        period = max(1, int(round(1.0 / (rate * dt)))) if rate else 1
        chosen.append((full, s.dim, H_fn, R_fn, period))
    names = [full for full, *_ in chosen]

    # --- Q: model auto L Σ Lᵀ unless overridden ---------------------------
    Q_fn = None
    if Q is not None:
        Q_const = np.asarray(Q, dtype=float)
    elif sys.L_sym is not None:
        Q_fn = ca.Function("Q_auto", [x_s, u_s, dt_s, t_s],
                           [lin_cov(sys.L_sym, ca.DM(sys.Sigma), n)])
        Q_const = None
    else:
        Q_const = np.zeros((n, n))

    # --- the recursion -----------------------------------------------------
    x = _operating_point(ir, state)
    refresh = max(1, steps // max(1, samples))
    rec_every = max(1, steps // max(1, record))
    slots = [(s.name, s.tangent_offset, s.tangent_offset + s.tangent_dim)
             for s in spec.slots]

    times: list[float] = []
    sigmas: dict[str, list[float]] = {name: [] for name, _, _ in slots}

    def record_point(t: float) -> None:
        times.append(t)
        for name, lo, hi in slots:
            block = P[lo:hi, lo:hi]
            sigmas[name].append(float(np.sqrt(max(
                np.max(np.linalg.eigvalsh(0.5 * (block + block.T))), 0.0))))

    record_point(t0)
    F = Qk = None
    HRs: list = []
    for i in range(steps):
        t = t0 + i * dt
        u_dict = control(t) if callable(control) else (control or None)
        u_vec = ir._build_u(u_dict)
        if i % refresh == 0:                     # re-linearize along the way
            F = np.asarray(sys.F_fn(x, u_vec, dt, t), float).reshape(n, n)
            Qk = (Q_const if Q_fn is None else
                  np.asarray(Q_fn(x, u_vec, dt, t), float).reshape(n, n))
            HRs = []
            for full, dim, H_fn, R_fn, period in chosen:
                H = np.asarray(H_fn(x, u_vec, 0.0, t), float).reshape(dim, n)
                R = np.asarray(R_fn(x, u_vec, t), float).reshape(dim, dim)
                if not np.any(np.abs(R) > 0):
                    raise ValueError(
                        f"sigma_horizon: sensor {full!r} has zero R — the "
                        f"update is singular. Declare a noise σ on it or "
                        f"exclude it via sensors=[...].")
                HRs.append((H, R, period))
        x = np.asarray(sys.predict_fn(x, u_vec, dt, t), float).reshape(-1)
        P = F @ P @ F.T + Qk
        for H, R, period in HRs:
            if i % period:
                continue
            _, P, _, _ = joseph_update_np(P, H, R)   # covariance-only fold
        P = symmetrize(P)
        if (i + 1) % rec_every == 0 or i == steps - 1:
            record_point(t0 + (i + 1) * dt)

    return SigmaHorizonReport(
        horizon=horizon, dt=dt, sensors=names, times=np.asarray(times),
        sigmas={k: np.asarray(v) for k, v in sigmas.items()}, P_final=P)

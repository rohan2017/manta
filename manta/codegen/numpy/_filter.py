"""NumpyFilter — the HELD predict/update filter view."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import casadi as ca
import numpy as np

from ..._validation import require_finite, require_positive
from ...estimation._kalman import joseph_update_np
from ...ir._names import resolve_suffix
from ...ir.module import entry_ident
from ._runtime import NumpyRuntime


@dataclass(frozen=True)
class FilterCheckpoint:
    """Complete restart point for a filter runtime.

    Arrays are owned snapshots rather than views into the live runtime.
    ``time`` is the filter's logical model time, not a wall clock.
    """

    x: np.ndarray
    P: np.ndarray
    time: float


@dataclass(frozen=True)
class UpdateResult:
    """Diagnostics and disposition of one measurement fold."""

    sensor: str
    innovation: np.ndarray
    innovation_covariance: np.ndarray
    nis: float
    accepted: bool
    gate: float | None
    covariance_overridden: bool


class NumpyFilter(NumpyRuntime):
    """A predict/update filter over a held `x`/`P`, with baked per-sensor
    update kernels — the same surface every backend emits.

    You own the loop, identically in numpy and C++: fold each fresh
    measurement at the pre-predict state, then predict.

        for nm in sensors:
            if gate[nm].due(t):
                ekf.update(nm, sim.reading(nm), u=u)   # update-then-...
        ekf.predict(dt, u=u)                           # ...-predict

    The update-then-predict order is yours to keep: a reading sampled at the
    interval start belongs against the current (pre-predict) state.

    Clock: same convention as `NumpySim` — the runtime tracks `t`,
    `predict(dt)` advances it, and an explicit `t=` overrides it for that
    call. (The kernels stay pure; this is caller-side bookkeeping the two
    runtimes must agree on: a filter that silently pinned `t=0` while the
    sim advanced left every time-dependent world linearized at t=0.)
    """

    def __init__(self, module) -> None:
        super().__init__(module)
        self._Q: np.ndarray | None = None        # default process noise
        self._custom_h_cache: dict = {}          # h_sym -> (h_fn, H_fn)

    # ---- estimate access -------------------------------------------------

    @property
    def x(self) -> np.ndarray:
        return self._state["x"].copy()

    @property
    def P(self) -> np.ndarray:
        return self._state["P"].copy()

    @property
    def estimate_vector(self) -> np.ndarray:
        return self._state["x"]

    @property
    def time(self) -> float:
        """Current logical filter time (advanced only by ``predict``)."""
        return self._t

    def state_dict(self) -> dict[str, dict[str, Any]]:
        """Current estimate nested by owner."""
        return self._spec.to_nested(self._state["x"])

    def reset(self, state: dict | None = None, *,
              P: np.ndarray | None = None) -> None:
        """Reset state, covariance, and clock from the Module defaults.

        ``state`` is merged over the declared initial state and ``P`` may
        replace the declared initial covariance. To move only the nominal
        state while deliberately preserving covariance, use
        :meth:`set_state_keep_covariance`.
        """
        self._t = 0.0
        x_field = self.module.state.field("x")
        self._state["x"] = self._spec.pack_any(
            state, base=x_field.init) if state is not None else np.asarray(
                x_field.init, dtype=float).reshape(-1).copy()
        pf = self.module.state.field("P")
        self._state["P"] = np.asarray(
            pf.init, dtype=float).reshape(pf.shape).copy()
        if P is not None:
            P = np.asarray(P, dtype=float)
            expected = (self._spec.tangent_dim, self._spec.tangent_dim)
            if P.shape != expected:
                raise ValueError(
                    f"reset: P shape {P.shape} doesn't match tangent dim "
                    f"{expected}")
            self._state["P"] = P.copy()

    def checkpoint(self) -> FilterCheckpoint:
        """Capture nominal state, covariance, and logical time atomically."""
        return FilterCheckpoint(self._state["x"].copy(),
                                self._state["P"].copy(), float(self._t))

    def restore(self, checkpoint: FilterCheckpoint) -> None:
        """Restore a checkpoint after strict shape/finite validation.

        Restore never partially mutates the live filter: all values are
        validated and copied before any runtime field changes.
        """
        if not isinstance(checkpoint, FilterCheckpoint):
            raise TypeError("restore: expected FilterCheckpoint")
        x = np.asarray(checkpoint.x, dtype=float)
        P = np.asarray(checkpoint.P, dtype=float)
        expected_x = (self._spec.ambient_dim,)
        expected_P = (self._spec.tangent_dim, self._spec.tangent_dim)
        if x.shape != expected_x:
            raise ValueError(
                f"restore: x shape {x.shape} doesn't match {expected_x}")
        if P.shape != expected_P:
            raise ValueError(
                f"restore: P shape {P.shape} doesn't match {expected_P}")
        t = float(checkpoint.time)
        if not np.all(np.isfinite(x)) or not np.all(np.isfinite(P)) \
                or not np.isfinite(t):
            raise ValueError("restore: checkpoint contains non-finite values")
        if not np.allclose(P, P.T, rtol=1e-10, atol=1e-12):
            raise ValueError("restore: P must be symmetric")
        if np.linalg.eigvalsh(P).min() < -1e-12:
            raise ValueError("restore: P must be positive semidefinite")
        new_x, new_P = x.copy(), P.copy()
        self._state["x"], self._state["P"], self._t = new_x, new_P, t

    def set_state_keep_covariance(self, state: dict) -> None:
        """Replace the nominal state while preserving covariance and clock.

        This is intentionally separate from :meth:`reset`: retaining a
        covariance after moving its linearization point is an advanced,
        explicit operation.
        """
        self._state["x"] = self._spec.pack_any(
            state, base=self.module.state.field("x").init)

    @property
    def Q(self):
        """Default process noise for `predict` (overridden per-call by
        `predict(dt, Q=...)`; `None` uses the model's baked `L Σ Lᵀ`)."""
        return self._Q

    @Q.setter
    def Q(self, value) -> None:
        self._Q = value

    # ---- predict / update (the uniform kernel surface) -------------------

    def predict(self, dt: float, *, t: float | None = None,
                u: dict[str, Any] | None = None,
                Q: np.ndarray | None = None) -> None:
        """Advance the estimate by `dt`. Process noise: an explicit `Q`, else
        `self.Q`, else the model's baked `L Σ Lᵀ`. `u` is `{input: value}`
        (unset inputs fall to the Module's declared defaults); pass the same
        held `u` truth ran on. `t=None` uses (and advances) the runtime's
        clock, matching `NumpySim.step`; an explicit `t` overrides and
        resynchronizes it."""
        dt = require_positive(dt, name=f"{type(self).__name__}.predict dt")
        t0 = self._t if t is None else float(require_finite(
            t, name=f"{type(self).__name__}.predict t"))
        self._predict_kernel(dt, t0, self.build_u(u),
                             Q if Q is not None else self._Q)
        self._t = t0 + dt

    def update(self, target, z=None, R=None, *, t: float | None = None,
               u: dict[str, Any] | None = None) -> UpdateResult:
        """Fold one measurement at the current state.

        * `update("gps.position", z)` — by sensor name (full or suffix),
          through the baked covariance and gate.
        * `update("gps.position", z, R=sample_R)` — typed per-sample
          covariance override through the deployable Module entry point.
        * `update(h_sym, z, R=R)` — a caller-supplied `h(x)` callable +
          measurement covariance (custom measurements; numpy-only).

        `t=None` reads the runtime's clock (which `predict` advances); a
        measurement is dt-independent, so `update` never advances it.
        """
        if callable(target):
            if z is None or R is None:
                raise TypeError("update(h_sym, z, R=...): z and R required")
            return self._update_custom(target, z, R)
        return self._fold_sensor(
            self._resolve_sensor(target), z, self.build_u(u), R=R,
            t=self._t if t is None else float(require_finite(
                t, name=f"{type(self).__name__}.update t")))

    # ---- kernels ---------------------------------------------------------

    def _predict_kernel(self, dt: float, t: float, u_vec: np.ndarray,
                        Q: np.ndarray | None) -> None:
        if Q is None:
            self._run(self.module.entry("predict"),
                      {"u": u_vec, "dt": dt, "t": t})
        else:
            self._run(self.module.entry("predict_with_Q"),
                      {"Q": np.asarray(Q, dtype=float), "u": u_vec,
                       "dt": dt, "t": t})

    def _resolve_sensor(self, name: str) -> str:
        return resolve_suffix(name, [p.name for p in self._meas_ports_ir],
                              label="sensor", who=type(self).__name__)

    def _fold_sensor(self, full: str, z, u_vec: np.ndarray, *,
                     R=None, t: float = 0.0) -> UpdateResult:
        """Fold a sensor and return the generated kernel's diagnostics."""
        port = self.module.port(full)
        z_arr = np.atleast_1d(np.asarray(z, dtype=float)).reshape(-1)
        if z_arr.size != port.size:
            raise ValueError(
                f"update {full}: expected z of size {port.size}, got "
                f"{z_arr.size}.")
        ident = entry_ident(full)
        values = {full: z_arr, "u": u_vec, "t": t}
        if R is None:
            method = f"update_diagnostic_{ident}"
        else:
            R = self._validate_measurement_covariance(R, port.size, full)
            method = f"update_with_R_{ident}"
            values[f"R_{ident}"] = R
        result = self._run(self.module.entry(method), values)
        gate = self.module.metadata.get("nis_gates", {}).get(full)
        return UpdateResult(
            sensor=full,
            innovation=np.asarray(result[f"innovation_{ident}"], dtype=float)
            .reshape(-1).copy(),
            innovation_covariance=np.asarray(
                result[f"innovation_covariance_{ident}"], dtype=float).copy(),
            nis=float(np.asarray(result[f"nis_{ident}"]).reshape(-1)[0]),
            accepted=bool(
                np.asarray(result[f"accepted_{ident}"]).reshape(-1)[0]),
            gate=gate,
            covariance_overridden=R is not None,
        )

    @staticmethod
    def _validate_measurement_covariance(R, dim: int,
                                         full: str) -> np.ndarray:
        R = np.asarray(R, dtype=float)
        if R.shape != (dim, dim):
            raise ValueError(
                f"update {full}: R shape {R.shape} doesn't match "
                f"{(dim, dim)}")
        if not np.all(np.isfinite(R)):
            raise ValueError(f"update {full}: R contains non-finite values")
        if not np.allclose(R, R.T, rtol=1e-10, atol=1e-12):
            raise ValueError(f"update {full}: R must be symmetric")
        try:
            np.linalg.cholesky(R)
        except np.linalg.LinAlgError as exc:
            raise ValueError(
                f"update {full}: R must be positive definite") from exc
        return R.copy()

    def _custom_h_fns(self, h_sym: Callable):
        """The `(h, H)` `ca.Function` pair for a caller-supplied `h(x)`,
        built once per callable and memoized — the symbolic construction
        (trace + jacobian + two Functions) is far too expensive to redo
        on every in-loop update."""
        try:
            cached = self._custom_h_cache.get(h_sym)
        except TypeError:                        # unhashable callable
            cached = None
        if cached is not None:
            return cached
        spec = self._spec
        x_sym = ca.MX.sym("x", spec.ambient_dim, 1)
        h_mx = h_sym(x_sym)
        h_mx = ca.reshape(h_mx, h_mx.numel(), 1)
        delta = ca.MX.sym("delta", spec.tangent_dim, 1)
        h_pert = h_sym(spec.boxplus_sym(x_sym, delta))
        h_pert = ca.reshape(h_pert, h_pert.numel(), 1)
        H_sym = ca.substitute(ca.jacobian(h_pert, delta), delta,
                              ca.MX.zeros(spec.tangent_dim, 1))
        fns = (ca.Function("h", [x_sym], [h_mx]),
               ca.Function("H", [x_sym], [H_sym]))
        try:
            self._custom_h_cache[h_sym] = fns
        except TypeError:
            pass
        return fns

    def _update_custom(self, h_sym: Callable, z, R) -> UpdateResult:
        """Joseph update for a caller-supplied `h(x)` — built on the spec's
        manifold ops; the one genuinely runtime-defined measurement."""
        spec = self._spec
        z = np.asarray(z, dtype=float).reshape(-1)
        h_fn, H_fn = self._custom_h_fns(h_sym)
        x_now = self._state["x"]
        h_x = np.asarray(h_fn(x_now)).reshape(-1)
        H = np.asarray(H_fn(x_now))
        if h_x.size != z.size:
            raise ValueError(
                f"update: h(x) size {h_x.size} doesn't match z size {z.size}")
        R = self._validate_measurement_covariance(R, z.size, "custom")
        x_new, P_new, innovation, S = joseph_update_np(
            self._state["P"], H, R,
            x=x_now, z=z, h=h_x, boxplus=spec.boxplus_num)
        self._state["x"] = x_new
        self._state["P"] = P_new
        nis = float(innovation @ np.linalg.solve(S, innovation))
        return UpdateResult("custom", innovation.copy(), S.copy(), nis,
                            True, None, True)

"""NumpyFilter — the HELD predict/update filter view."""

from __future__ import annotations

from typing import Any, Callable

import casadi as ca
import numpy as np

from ...bus import MeasurementBus
from ...ir.module import entry_ident
from ...linearization import resolve_suffix
from ._runtime import NumpyRuntime


class NumpyFilter(NumpyRuntime):
    """A predict/update filter. Held `x`/`P`; baked per-sensor updates by
    NAME; the measurement bus (`feed` + `step`, wired `meas`/`command`/
    `estimate` ports) on top."""

    def __init__(self, module) -> None:
        super().__init__(module)
        self._bus = MeasurementBus(
            self,
            sensors={p.name: {"dim": p.size, "key": p.name}
                     for p in self._meas_ports_ir},
            input_names=self._input_names(),
            sample_rates={f.name: f.rate for f in self._u_fields()
                          if f.rate is not None},
            estimate_dim=self._spec.ambient_dim,
            estimate_layout={s.name: (s.ambient_offset, s.ambient_dim)
                             for s in self._spec.slots})

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

    def state_dict(self) -> dict[str, dict[str, Any]]:
        """Current estimate nested by owner."""
        return self._spec.to_nested(self._state["x"])

    def reset(self, state: dict | None = None, *,
              P: np.ndarray | None = None) -> None:
        """Reset held state. `state` is a nested/flat dict merged over
        the Module's declared initial values, or a flat ambient vector
        taken verbatim; `P` resets the covariance."""
        x_field = self.module.state.field("x")
        if state is not None:
            self._state["x"] = self._spec.pack_any(state, base=x_field.init)
        elif P is None:
            self._state["x"] = np.asarray(
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
        self._bus.publish_estimate()

    # ---- predict / update ------------------------------------------------

    def predict(self, dt: float, *, t: float = 0.0,
                u: dict[str, Any] | None = None,
                Q: np.ndarray | None = None) -> None:
        """Advance the estimate by `dt`. Process noise: the model's baked
        `L Σ Lᵀ` unless an explicit `Q` overrides it."""
        u_vec = self.build_u(u)
        if Q is None:
            self._run(self.module.entry("predict"),
                      {"u": u_vec, "dt": dt, "t": t})
        else:
            self._run(self.module.entry("predict_with_Q"),
                      {"Q": np.asarray(Q, dtype=float), "u": u_vec,
                       "dt": dt, "t": t})

    def update(self, target, z=None, R=None, *, t: float = 0.0,
               u: dict[str, Any] | None = None) -> None:
        """Fold one measurement.

        * `update("gps.position", z)` — by sensor name (full or suffix),
          through the baked Joseph-update kernel.
        * `update(h_sym, z, R=R)` — a caller-supplied `h(x)` callable +
          measurement covariance (custom measurements; numpy-only).
        """
        if callable(target):
            if z is None or R is None:
                raise TypeError("update(h_sym, z, R=...): z and R required")
            return self._update_low_level(target, z, R)
        self.fold(self._resolve_sensor(target),
                  z, self.build_u(u if u is not None else
                                  (self.inputs if self.inputs else None)),
                  t=t)

    def _resolve_sensor(self, name: str) -> str:
        return resolve_suffix(name, [p.name for p in self._meas_ports_ir],
                              label="sensor", who=type(self).__name__)

    def fold(self, full: str, z, u_vec: np.ndarray, *,
             t: float = 0.0) -> None:
        """Fold one measurement (full sensor name) — the bus's hook."""
        port = self.module.port(full)
        z_arr = np.atleast_1d(np.asarray(z, dtype=float)).reshape(-1)
        if z_arr.size != port.size:
            raise ValueError(
                f"update {full}: expected z of size {port.size}, got "
                f"{z_arr.size}.")
        self._run(self.module.entry(f"update_{entry_ident(full)}"),
                  {full: z_arr, "u": u_vec, "t": t})

    def _update_low_level(self, h_sym: Callable, z, R) -> None:
        """Joseph update for a caller-supplied `h(x)` — built on the spec's
        manifold ops; the one genuinely runtime-defined measurement."""
        spec = self._spec
        z = np.asarray(z, dtype=float).reshape(-1)
        R = np.asarray(R, dtype=float)
        if R.shape != (z.size, z.size):
            raise ValueError(
                f"update: R shape {R.shape} doesn't match z size {z.size}")
        x_sym = ca.MX.sym("x", spec.ambient_dim, 1)
        h_mx = h_sym(x_sym)
        h_mx = ca.reshape(h_mx, h_mx.numel(), 1)
        delta = ca.MX.sym("delta", spec.tangent_dim, 1)
        h_pert = h_sym(spec.boxplus_sym(x_sym, delta))
        h_pert = ca.reshape(h_pert, h_pert.numel(), 1)
        H_sym = ca.substitute(ca.jacobian(h_pert, delta), delta,
                              ca.MX.zeros(spec.tangent_dim, 1))
        x_now = self._state["x"]
        h_x = np.asarray(ca.Function("h", [x_sym], [h_mx])(x_now)).reshape(-1)
        H = np.asarray(ca.Function("H", [x_sym], [H_sym])(x_now))
        if h_x.size != z.size:
            raise ValueError(
                f"update: h(x) size {h_x.size} doesn't match z size {z.size}")
        P = self._state["P"]
        S = H @ P @ H.T + R
        K = np.linalg.solve(S.T, (P @ H.T).T).T
        self._state["x"] = spec.boxplus_num(x_now, K @ (z - h_x))
        IKH = np.eye(spec.tangent_dim) - K @ H
        P = IKH @ P @ IKH.T + K @ R @ K.T
        self._state["P"] = 0.5 * (P + P.T)

    # ---- the measurement bus ----------------------------------------------

    def step(self, dt: float, *, t: float | None = None,
             Q: np.ndarray | None = None) -> None:
        """Fold fresh measurements (interval start), then predict by `dt`."""
        return self._bus.step(dt, t=t, Q=Q)

    def feed(self, name: str, z, *, t: float | None = None) -> None:
        self._bus.feed(name, z, t=t)

    def meas(self, name: str):
        return self._bus.meas(name)

    def command(self, name: str):
        return self._bus.command(name)

    @property
    def estimate(self):
        """Producer port: the estimate as a flat ambient vector."""
        return self._bus.estimate

    @property
    def inputs(self) -> dict:
        return self._bus.inputs

    @inputs.setter
    def inputs(self, value: dict) -> None:
        self._bus.inputs = value

    @property
    def Q(self):
        return self._bus.Q

    @Q.setter
    def Q(self, value) -> None:
        self._bus.Q = value

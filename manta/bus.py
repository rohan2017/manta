"""MeasurementBus — the backend-agnostic measurement + command bus.

A stateful filter run online needs more than its baked kernels: a mailbox
for sensor readings that arrive at their own rates, zero-order-hold of the
latest control command, staleness rejection, and a published state estimate
other blocks can wire to. None of that is filter math — it's the same
event-plumbing for any predict/update filter on any backend — so the target
architecture lifts it OUT of the runtime into this one layer over `Signal`
ports (`manta.signal`). It drives a filter through a tiny *math-only*
interface (`FilterRuntime` below): build a control vector, fold one sensor,
predict, expose the estimate. A numpy filter satisfies it today; a torch one
would tomorrow, and inherit the bus unchanged.

The order is **update-then-predict**: a reading published over the step
`[t, t+dt]` was sampled at its *start* `t`, so it is folded against the
current (pre-predict) state, and only then is the state propagated over the
interval. Folding it after a full predict would meet it with the `t+dt`
state and bias rate-derived states by O(dt).
"""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np

from .linearized_system import resolve_suffix
from .signal import Signal


class FilterRuntime(Protocol):
    """The math-only surface the bus drives. A backend's filter runtime
    (e.g. `NumpyEKF`) implements it; the bus owns everything else."""

    def build_u(self, u: dict | None) -> np.ndarray:
        """Resolve a `{name: value}` command dict to the flat control vector
        in the filter's input order (defaults for anything unset)."""

    def fold(self, sensor_key, z, u_vec: np.ndarray) -> None:
        """Fold one measurement `z` (for the sensor registered under
        `sensor_key`) into the state at the current estimate."""

    def predict(self, dt: float, *, t: float, u: dict | None,
                Q: np.ndarray | None) -> None:
        """Advance the nominal state + covariance over `[t, t+dt]`."""

    @property
    def estimate_vector(self) -> np.ndarray:
        """The current estimate as a flat ambient vector (for publishing)."""


class MeasurementBus:
    """Measurement + command ports, mailbox, and the update-then-predict
    `step` for a stateful filter — all backend-agnostic, over `Signal`s.

    `sensors` is `{full_name: {"dim": int, "key": Any}}` (the routing key is
    opaque — whatever `runtime.fold` expects). `input_names` / `sample_rates`
    describe the control ports. `estimate_dim` / `estimate_layout` size the
    published estimate port.
    """

    def __init__(self, runtime: FilterRuntime, *,
                 sensors: dict[str, dict[str, Any]],
                 input_names: list[str],
                 sample_rates: dict[str, float] | None = None,
                 estimate_dim: int,
                 estimate_layout: dict[str, tuple[int, int]] | None = None,
                 ) -> None:
        self._rt = runtime
        self._input_names = list(input_names)
        self._sample_rates = sample_rates or {}
        self._estimate_dim = estimate_dim
        self._estimate_layout = estimate_layout

        # One mailbox per registered sensor, in registration order (= the
        # order step() folds them). feed() drops a reading; step() drains it.
        self._meas: dict[str, dict[str, Any]] = {}
        for full, s in sensors.items():
            self._meas[full] = {"value": None, "fresh": False, "t": None,
                                "dim": s["dim"], "key": s["key"]}

        # Latched (zero-order-hold) control commands, read by step()'s predict.
        self.inputs: dict[str, Any] = {}
        # Default process noise used by step() when none is passed.
        self.Q: np.ndarray | None = None
        # Filter clock — advances by dt each step (or to the supplied t).
        self._t = 0.0
        # Signal ports (lazy). `_seen_meas` tracks the last upstream version
        # folded in, so a wired measurement is consumed once per fresh sample.
        self._meas_ports: dict[str, Signal] = {}
        self._cmd_ports: dict[str, Signal] = {}
        self._seen_meas: dict[str, int] = {}
        self._est_port: Signal | None = None

    # ---- Ports ---------------------------------------------------------

    def meas(self, name: str) -> Signal:
        """Consumer port for a sensor measurement (`"craft.part.output"` or
        an unambiguous suffix). Wire a sim output into it, or `set()` it
        yourself; `step()` folds it in once per fresh sample."""
        full = resolve_suffix(name, self._meas, label="sensor",
                              who="MeasurementBus.feed")
        if full not in self._meas_ports:
            self._meas_ports[full] = Signal(full, dim=self._meas[full]["dim"])
        return self._meas_ports[full]

    def command(self, name: str) -> Signal:
        """Consumer port for a known control input (drives the predict).
        Latched / zero-order-hold, gated at the part's declared intake rate
        (`ctx.hold`) so predict sees the same held command as truth."""
        full = resolve_suffix(name, self._input_names, label="input",
                              who="MeasurementBus.command")
        if full not in self._cmd_ports:
            self._cmd_ports[full] = Signal(
                full, latched=True, rate=self._sample_rates.get(full))
        return self._cmd_ports[full]

    @property
    def estimate(self) -> Signal:
        """Producer port carrying the current estimate as the flat ambient
        vector, refreshed after every `step()`/reset. Its `layout` lets a
        consumer build a fixed name-keyed gather (codegen-honest)."""
        if self._est_port is None:
            self._est_port = Signal("estimate", dim=self._estimate_dim)
            self._est_port.layout = self._estimate_layout
            self._est_port.set(np.asarray(self._rt.estimate_vector).copy())
        return self._est_port

    def publish_estimate(self) -> None:
        """Refresh the estimate port from the runtime (after step/reset)."""
        if self._est_port is not None:
            self._est_port.set(np.asarray(self._rt.estimate_vector).copy())

    # ---- Feed + step ---------------------------------------------------

    def feed(self, name: str, z, *, t: float | None = None) -> None:
        """Drop a measurement into the mailbox and mark it fresh.

        `name` is a registered sensor's full name or an unambiguous suffix;
        `t` is an optional sample timestamp used for staleness rejection in
        `step()`. The reading is consumed by the next `step()`."""
        full = resolve_suffix(name, self._meas, label="sensor",
                              who="MeasurementBus.feed")
        m = self._meas[full]
        z_arr = np.atleast_1d(np.asarray(z, dtype=float)).reshape(-1)
        if z_arr.size != m["dim"]:
            raise ValueError(
                f"MeasurementBus.feed: {name}: expected z of size {m['dim']}, "
                f"got {z_arr.size}.")
        m["value"], m["fresh"], m["t"] = z_arr, True, t

    def step(self, dt: float, *, t: float | None = None,
             Q: np.ndarray | None = None) -> None:
        """Fold the interval-start measurements, then predict by `dt`.

        Pulls wired ports first: latched `command` ports refresh `inputs`
        (ZOH), and each `meas` port whose upstream advanced is dropped into
        the mailbox. Folds every fresh, non-stale reading at the current
        (pre-predict) state, then predicts over `[t, t+dt]`. `Q` overrides
        the process noise (else `self.Q`, else the model default)."""
        t_start = t if t is not None else self._t

        # Pull wired command ports (ZOH, rate-gated) into the latched dict.
        for full, port in self._cmd_ports.items():
            v = port.latched_value(t_start)
            if v is not None:
                self.inputs[full] = float(v) if np.ndim(v) == 0 else v
        # Pull wired measurement ports: fold once per fresh sample.
        for full, port in self._meas_ports.items():
            ver = port.cur_version
            if ver > self._seen_meas.get(full, 0):
                self._seen_meas[full] = ver
                self.feed(full, port.read(), t=port.cur_t)

        u_dict = self.inputs if self.inputs else None
        u_vec = self._rt.build_u(u_dict)
        # Update first: fold interval-start readings at the pre-predict state.
        for full, m in self._meas.items():
            if not m["fresh"]:
                continue
            m["fresh"] = False
            if m["t"] is not None and m["t"] < t_start - 1e-9:
                continue                       # stale: predates current state
            self._rt.fold(m["key"], m["value"], u_vec)
        # Then predict over [t_start, t_start + dt].
        self._rt.predict(dt, t=t_start, u=u_dict,
                         Q=Q if Q is not None else self.Q)
        self._t = t_start + dt
        self.publish_estimate()

"""MeasurementBus — the backend-agnostic measurement + command bus.

A stateful filter run online needs more than its baked kernels: a mailbox
for sensor readings that arrive at their own rates, zero-order-hold of the
latest control command, staleness rejection, and a published state estimate
other blocks can wire to. None of that is filter math — it's the same
event-plumbing for any predict/update filter on any backend — so the target
architecture lifts it OUT of the runtime into this one layer over `Signal`
ports (`manta.signal`). It drives a filter through a tiny *math-only*
interface (`FilterRuntime` below): build a control vector, fold one sensor,
predict, expose the estimate. `NumpyRuntime` satisfies it today; a torch
runtime would tomorrow, and inherit the bus unchanged.

The order is **update-then-predict**: a reading published over the step
`[t, t+dt]` was sampled at its *start* `t`, so it is folded against the
current (pre-predict) state, and only then is the state propagated over the
interval. Folding it after a full predict would meet it with the `t+dt`
state and bias rate-derived states by O(dt).
"""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np

from .linearization import resolve_suffix
from .signal import Signal


class PortSet:
    """Lazy named `Signal` ports + the pull/publish disciplines the block
    runtimes share (sim, recurrence, …). Consumers are latched (zero-order-
    hold); producers publish rate-gated. The runtime keeps its own
    compute/step orchestration and just calls `pull()` then `publish()` —
    the mechanical port bookkeeping (which the design wants to live once,
    over `Signal`) is here, not duplicated per façade."""

    def __init__(self) -> None:
        self.consumers: dict[str, Signal] = {}
        self.producers: dict[str, Signal] = {}

    def consumer(self, name: str, *, dim: int | None = None,
                 rate: float | None = None) -> Signal:
        if name not in self.consumers:
            self.consumers[name] = Signal(name, dim=dim, latched=True, rate=rate)
        return self.consumers[name]

    def producer(self, name: str, *, dim: int | None = None,
                 rate: float | None = None) -> Signal:
        if name not in self.producers:
            self.producers[name] = Signal(name, dim=dim, rate=rate)
        return self.producers[name]

    def pull(self, t: float) -> dict[str, Any]:
        """Latched value of every consumer port at time `t` (ZOH); omits
        ports that have never received a value."""
        out: dict[str, Any] = {}
        for name, port in self.consumers.items():
            v = port.latched_value(t)
            if v is not None:
                out[name] = v
        return out

    def publish(self, values: dict[str, Any], t: float) -> None:
        """Set each producer whose name is in `values`, rate-gated by `due`
        (sample-and-hold in between firings)."""
        for name, port in self.producers.items():
            if name in values and port.due(t):
                port.set(np.atleast_1d(np.asarray(values[name])).ravel(), t=t)


class FilterRuntime(Protocol):
    """The math-only surface the bus drives. A backend's filter runtime
    (e.g. `NumpyRuntime`) implements it; the bus owns everything else."""

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

        # ONE Signal per registered sensor, in registration order (= the
        # order step() folds them). It is both the mailbox (`feed()` sets
        # it) and the wireable port (`meas()` returns it); the Signal's
        # version counter is the single freshness mechanism — step() folds
        # once per version advance. `_keys` routes a fold to the runtime.
        self._meas: dict[str, Signal] = {}
        self._keys: dict[str, Any] = {}
        for full, s in sensors.items():
            self._meas[full] = Signal(full, dim=s["dim"])
            self._keys[full] = s["key"]
        self._seen_meas: dict[str, int] = {}

        # Latched (zero-order-hold) control commands, read by step()'s predict.
        self.inputs: dict[str, Any] = {}
        # Default process noise used by step() when none is passed.
        self.Q: np.ndarray | None = None
        # Filter clock — advances by dt each step (or to the supplied t).
        self._t = 0.0
        self._cmd_ports: dict[str, Signal] = {}
        self._est_port: Signal | None = None

    # ---- Ports ---------------------------------------------------------

    def meas(self, name: str) -> Signal:
        """The sensor's measurement Signal (`"craft.part.output"` or an
        unambiguous suffix). Wire a sim output into it, or `set()` it
        yourself; `step()` folds it once per fresh sample (version)."""
        full = resolve_suffix(name, self._meas, label="sensor",
                              who="MeasurementBus.meas")
        return self._meas[full]

    def command(self, name: str) -> Signal:
        """Consumer port for a known control input (drives the predict).
        Latched / zero-order-hold, gated at the part's declared intake rate
        (`PartUpdate.rates`) so predict sees the same held command as truth."""
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
        """Drop a measurement onto the sensor's Signal.

        `name` is a registered sensor's full name or an unambiguous suffix;
        `t` is an optional sample timestamp used for staleness rejection in
        `step()`. The reading is consumed by the next `step()`."""
        full = resolve_suffix(name, self._meas, label="sensor",
                              who="MeasurementBus.feed")
        sig = self._meas[full]
        z_arr = np.atleast_1d(np.asarray(z, dtype=float)).reshape(-1)
        if z_arr.size != sig.dim:
            raise ValueError(
                f"MeasurementBus.feed: {name}: expected z of size {sig.dim}, "
                f"got {z_arr.size}.")
        sig.set(z_arr, t=t)

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

        u_dict = self.inputs if self.inputs else None
        u_vec = self._rt.build_u(u_dict)
        # Update first: fold interval-start readings at the pre-predict
        # state — once per version advance (fed directly or via a wire).
        for full, sig in self._meas.items():
            ver = sig.cur_version
            if ver <= self._seen_meas.get(full, 0):
                continue
            self._seen_meas[full] = ver
            zt = sig.cur_t
            if zt is not None and zt < t_start - 1e-9:
                continue                       # stale: predates current state
            z = np.atleast_1d(np.asarray(sig.read(), dtype=float)).reshape(-1)
            self._rt.fold(self._keys[full], z, u_vec)
        # Then predict over [t_start, t_start + dt].
        self._rt.predict(dt, t=t_start, u=u_dict,
                         Q=Q if Q is not None else self.Q)
        self._t = t_start + dt
        self.publish_estimate()

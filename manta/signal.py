"""Signal — a typed value channel with freshness + wiring.

A `Signal` is a small mailbox: a named slot holding a value, a freshness
flag, a timestamp, and a monotonic version counter. `wire(producer,
consumer)` connects two Signals so the consumer pulls the producer's
value on read. The version counter is how a consumer tells that the
producer has emitted something *new* (multi-rate freshness) without the
producer having to clear a flag that other consumers might still need —
so one producer can fan out to many consumers cleanly.

Two disciplines share the one struct:

  * **measurement** — event-like. A consumer folds it in only when the
    producer's `version` advances (a fresh sample arrived). This is what
    lets the EKF `step()` independently of the sim and still update only
    when a new reading shows up.
  * **command** — latched / zero-order-hold. The value persists and is
    read every tick regardless of freshness.

The shape is deliberately codegen-trivial — `{value buffer, bool fresh,
double t, int version, Signal* source}`. Pull-through is a pointer chase
(or a static copy pass in pointer-free targets); it never touches the
compiled-kernel ABIs. The runtimes expose Signals as named *ports*
(`sim.out(...)`, `ekf.meas(...)`, `lqr.command(...)`, `ekf.estimate`),
and `wire()` is the only thing that connects them.
"""

from __future__ import annotations

from typing import Any


class Signal:
    """A named value channel: value + freshness + version + optional wire.

    A Signal with `source is None` is its own producer — writes via
    `set()` bump its version. A wired Signal (`source` set by `wire()`)
    mirrors its upstream: `read()` / `version` / `t` all reflect the
    source. `dim` is an optional shape hint (None ⇒ structured / opaque,
    e.g. a nested state-estimate dict) used only for wire-time
    compatibility checks.
    """

    __slots__ = ("name", "dim", "value", "fresh", "t", "version",
                 "source", "latched", "rate", "_last_fire_t", "_held")

    def __init__(self, name: str, dim: int | None = None, *,
                 latched: bool = False, rate: float | None = None) -> None:
        self.name    = name
        self.dim     = dim
        self.value:   Any = None
        self.fresh   = False
        self.t:       float | None = None
        self.version = 0          # bumps on every set()
        self.source:  "Signal | None" = None
        self.latched = latched
        # Rate gating (Hz). None ⇒ every tick. The part declares this via
        # `ctx.sample()` / `ctx.hold()`; the runtime stamps it onto the
        # port. `_last_fire_t` / `_held` are the gate's bookkeeping.
        self.rate    = rate
        self._last_fire_t: float | None = None
        self._held:  Any = None

    # ---- Producer side ------------------------------------------------

    def set(self, value: Any, t: float | None = None) -> None:
        """Write a value, stamp it, and mark a new version. Called on the
        producing end (or on an unwired port the user feeds directly)."""
        self.value   = value
        self.t       = t
        self.fresh   = True
        self.version += 1

    # ---- Consumer side (pull through the wire) ------------------------

    @property
    def _eff(self) -> "Signal":
        """The effective producer: the wired source, else self."""
        return self.source if self.source is not None else self

    def read(self) -> Any:
        """Current value, pulled from the wired source if any."""
        return self._eff.value

    @property
    def cur_version(self) -> int:
        return self._eff.version

    @property
    def cur_t(self) -> float | None:
        return self._eff.t

    # ---- Rate gating --------------------------------------------------

    def _fires(self, t: float) -> bool:
        """Whether a rate boundary is crossed at time `t` (and record it).
        `rate is None` fires every call; otherwise once per 1/rate window.
        Shared by the producer gate (`due`) and the consumer gate
        (`latched_value`) — a port is one or the other, so the single
        `_last_fire_t` is unambiguous."""
        if self.rate is None:
            return True
        period = 1.0 / self.rate
        if (self._last_fire_t is None
                or t - self._last_fire_t >= period - 1e-9):
            self._last_fire_t = t
            return True
        return False

    def due(self, t: float) -> bool:
        """Producer gate: True when this port should publish a new sample
        at time `t` (sensor sample-and-hold). Between firings the producer
        holds — it neither writes a value nor bumps its version."""
        return self._fires(t)

    def latched_value(self, t: float) -> Any:
        """Consumer gate (zero-order-hold): latch a fresh value from the
        wired source at each rate boundary and hold it in between. With
        `rate is None` this is a live pull-through every tick."""
        if self.rate is None:
            return self.read()
        if self._fires(t):
            self._held = self.read()
        return self._held

    def __repr__(self) -> str:
        wired = "" if self.source is None else f" <-{self.source.name}"
        kind  = "cmd" if self.latched else "sig"
        return f"<{kind} {self.name} v{self.cur_version}{wired}>"


def wire(producer: Signal, consumer: Signal) -> Signal:
    """Connect `producer` → `consumer`: the consumer now pulls the
    producer's value (and version/timestamp) on read.

    Raises on a dim mismatch (when both declare one) or on re-wiring a
    consumer that is already attached to a different producer. Returns
    the consumer.
    """
    if not isinstance(producer, Signal) or not isinstance(consumer, Signal):
        raise TypeError("wire(producer, consumer): both must be Signals")
    if (producer.dim is not None and consumer.dim is not None
            and producer.dim != consumer.dim):
        raise ValueError(
            f"wire: dim mismatch {producer.name}(dim={producer.dim}) -> "
            f"{consumer.name}(dim={consumer.dim})")
    if consumer.source is not None and consumer.source is not producer:
        raise ValueError(
            f"wire: {consumer.name!r} is already wired to "
            f"{consumer.source.name!r}; cannot re-wire to {producer.name!r}")
    consumer.source = producer
    return consumer

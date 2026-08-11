"""Rate-gating tools for the driving loop — `RateGate` + `CommandLatch`.

manta's runtimes are pure: a kernel is a function of its inputs, holding no
clock. Sensor sample rates and actuator intake rates are *loop-level* state
(they remember the last window boundary), so — like the loop itself — they
belong to **you**, the same way in every backend. These two small helpers
carry that state; wire them into your loop:

    gps = sim.rate_gate("craft.gps.position")     # once, before the loop
    latch = sim.command_latch()
    ...
    u = latch.latch(lqr.control(ekf.state_dict()), t)   # ZOH the commands
    sim.step(dt, u=u)
    if gps.due(t):                                       # fold at the GPS rate
        ekf.update("craft.gps.position", sim.reading("craft.gps.position"), u=u)
    ekf.predict(dt, u=u)

(`sim.rate_gate` / `sim.command_latch` just build these pre-configured from
the model's declared rates; a C++ user writes the equivalent two lines.)
"""

from __future__ import annotations

from typing import Any

from ._validation import require_finite, require_positive


class RateGate:
    """Window gate: `due(t)` fires once per `1/rate` window (and records
    it); `rate is None` fires every call. Holds only the last fire time —
    the one bit of state a sample-and-hold window needs."""

    __slots__ = ("rate", "_last")

    def __init__(self, rate: float | None) -> None:
        self.rate = (None if rate is None else
                     require_positive(rate, name="RateGate.rate"))
        self._last: float | None = None

    def due(self, t: float) -> bool:
        t = float(require_finite(float(t), name="RateGate time"))
        if self.rate is None:
            return True
        if self._last is None or t - self._last >= 1.0 / self.rate - 1e-9:
            self._last = t
            return True
        return False


class CommandLatch:
    """Zero-order-hold for rate-gated control inputs. `latch(inputs, t)`
    returns `inputs` with every gated entry replaced by the value latched
    at its last window boundary — mid-window writes are ignored until the
    next boundary. Apply it once per step and pass the held dict to BOTH
    `sim.step` and `ekf.predict`, so the filter predicts on the same held
    command truth acted on. Ungated inputs pass through untouched (and an
    empty gate set is a no-op that returns the dict unchanged)."""

    def __init__(self, rates: dict[str, float]) -> None:
        self._gates = {n: RateGate(r) for n, r in rates.items()}
        self._held: dict[str, Any] = {}

    def latch(self, inputs: dict[str, Any], t: float) -> dict[str, Any]:
        if not self._gates:
            return inputs
        out = dict(inputs)
        for n, gate in self._gates.items():
            if n in inputs and gate.due(t):
                self._held[n] = inputs[n]
            if n in self._held:
                out[n] = self._held[n]
        return out

"""NumpyRecurrence — the HELD stateful-dataflow view."""

from __future__ import annotations

from typing import Any

import numpy as np

from ._runtime import NumpyRuntime, pack_fields, unpack_fields


class NumpyRecurrence(NumpyRuntime):
    """A stateful dataflow block (PID, Madgwick, …): `step(dt, **inputs)`
    advances the held state and computes the readouts."""

    def __init__(self, module) -> None:
        super().__init__(module)
        self._y = np.zeros(self._y_port.size)

    @property
    def state(self) -> dict[str, Any]:
        """Held state, `{slot: value}` by name."""
        return self._spec.unpack(self._state["x"])

    def reset(self) -> None:
        """Reset the held state to the Module's declared initial values."""
        x_field = self.module.state.field("x")
        self._state["x"] = np.asarray(
            x_field.init, dtype=float).reshape(-1).copy()
        self._y = np.zeros(self._y_port.size)

    def step(self, dt: float, *, t: float | None = None,
             **inputs) -> dict[str, Any]:
        tt = self._t if t is None else t
        u = pack_fields(self._u_fields(), inputs, required=True, who="step")
        ret = self._run(self.module.entry("step"),
                        {"u": u, "dt": dt, "t": tt})
        self._y = ret[self._y_port.name]
        self._t = tt + dt
        return self.readouts()

    def readouts(self) -> dict[str, Any]:
        """Last-computed readouts by output-field name (scalars unwrapped)."""
        return unpack_fields(self._y_port.fields, self._y)

    # ---- ports -----------------------------------------------------------

    def input(self, name: str):
        """Consumer port for a recurrence input (latched / ZOH)."""
        names = self._input_names()
        if name not in names:
            raise KeyError(f"unknown input port {name!r}. Available: {names}")
        f = next(f for f in self._u_fields() if f.name == name)
        return self._ports.consumer(name, dim=f.dim)

    def output(self, name: str):
        """Producer port for a recurrence readout."""
        names = [f.name for f in self._y_port.fields]
        if name not in names:
            raise KeyError(f"unknown output port {name!r}. Available: {names}")
        f = next(f for f in self._y_port.fields if f.name == name)
        return self._ports.producer(name, dim=f.dim)

    def compute(self, dt: float, *, t: float | None = None) -> dict[str, Any]:
        """Pull wired input ports, `step(dt)`, publish readouts (stamped
        at start-of-step)."""
        tt = self._t if t is None else t
        out = self.step(dt, t=tt, **self._ports.pull(tt))
        self._ports.publish(out, t=tt)
        return out

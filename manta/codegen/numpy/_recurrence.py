"""NumpyRecurrence — the HELD stateful-dataflow view."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..._validation import require_finite, require_positive
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
        self._t = 0.0

    def step(self, dt: float, *, t: float | None = None,
             **inputs) -> dict[str, Any]:
        dt = require_positive(dt, name=f"{type(self).__name__}.step dt")
        tt = self._t if t is None else float(require_finite(
            t, name=f"{type(self).__name__}.step t"))
        next_t = float(require_finite(
            tt + dt, name=f"{type(self).__name__}.step resulting time"))
        u = pack_fields(self._u_fields(), inputs, required=True, who="step")
        ret = self._run(self.module.entry("step"),
                        {"u": u, "dt": dt, "t": tt})
        next_y = ret[self._y_port.name].copy()
        self._y = next_y
        self._t = next_t
        return self.readouts()

    def readouts(self) -> dict[str, Any]:
        """Last-computed readouts by output-field name (scalars unwrapped)."""
        return unpack_fields(self._y_port.fields, self._y)

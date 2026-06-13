"""NumpyRegulator — the THREADED stateless control-law view."""

from __future__ import annotations

from typing import Any

import numpy as np

from ...ir.module import Role
from ._runtime import NumpyRuntime, unpack_fields


class NumpyRegulator(NumpyRuntime):
    """A stateless control law: map a state estimate to commands via
    `control(estimate) -> {input: value}`.

    Holds the live reference point `x_ref` (seeded from the Module's
    built operating point); `retarget()` moves it at runtime."""

    def __init__(self, module) -> None:
        super().__init__(module)
        st = module.ports_by_role(Role.STATE)
        self._ref_port = st[1] if len(st) > 1 else None
        self._x_ref = (np.asarray(self._ref_port.init, dtype=float)
                       .reshape(-1).copy() if self._ref_port is not None
                       else np.asarray(self._x_port.init,
                                       dtype=float).reshape(-1).copy())

    def retarget(self, state: dict) -> None:
        """Move the reference point the law regulates to (nested or flat
        dict, merged over the CURRENT reference). The gain K is NOT
        re-solved: exact wherever the dynamics are invariant along the
        moved direction (e.g. translating a hover setpoint); build a new
        LQR for a genuinely different operating point (new A/B or trim).
        """
        if self._ref_port is None:
            raise AttributeError(
                f"{type(self).__name__}: module {self.module.name!r} has "
                "no reference port — its control law is not retargetable.")
        self._x_ref = self._ref_port.manifold.pack_any(state,
                                                       base=self._x_ref)

    @property
    def x_ref(self) -> np.ndarray:
        """The live reference point (flat ambient vector)."""
        return self._x_ref.copy()

    def u(self, x_flat) -> np.ndarray:
        """Control vector for a flat ambient state (full-spec layout)."""
        vals = {"x": np.asarray(x_flat, dtype=float)}
        if self._ref_port is not None:
            vals[self._ref_port.name] = self._x_ref
        return self.call("control", vals)["u"]

    def control(self, state: dict) -> dict[str, Any]:
        """Map a state estimate (nested or flat dict) → `{input: value}`,
        merged over the live reference point (unsupplied slots sit at
        the reference, i.e. zero error)."""
        x = self._x_port.manifold.pack_any(state, base=self._x_ref)
        return unpack_fields(self._u_fields(), self.u(x))

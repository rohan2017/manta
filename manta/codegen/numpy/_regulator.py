"""NumpyRegulator — the THREADED stateless control-law view."""

from __future__ import annotations

from typing import Any

import numpy as np

from ...ir.module import Role
from ._runtime import NumpyRuntime, unpack_fields


class NumpyRegulator(NumpyRuntime):
    """A stateless control law: map a state estimate to commands via
    `control(estimate) -> {input: value}`.

    Holds the live operating point the law flies about — the reference
    `x_ref` plus every MATRIX-role coefficient the Module declares (an
    LQR's gain `K` and feed-forward `u_ff`), each seeded from its
    declared default. `retarget()` moves the reference alone;
    `reprogram()` installs a whole re-solved operating point."""

    def __init__(self, module) -> None:
        super().__init__(module)
        st = module.ports_by_role(Role.STATE)
        self._ref_port = st[1] if len(st) > 1 else None
        self._x_ref = (np.asarray(self._ref_port.init, dtype=float)
                       .reshape(-1).copy() if self._ref_port is not None
                       else np.asarray(self._x_port.init,
                                       dtype=float).reshape(-1).copy())
        # Law coefficients supplied as data (defaults = the built solve).
        self._coeffs = {p.name: np.asarray(p.init, dtype=float).copy()
                        for p in module.ports_by_role(Role.MATRIX)
                        if p.init is not None}

    def retarget(self, state: dict) -> None:
        """Move the reference point the law regulates to (nested or flat
        dict, merged over the CURRENT reference). The gain is NOT
        re-solved: exact wherever the dynamics are invariant along the
        moved direction (e.g. translating a hover setpoint), but NOT for
        a heading change — there the world-frame feedback rotates with
        the reference. Use `reprogram(lqr.resolve_at(...))` for those.
        """
        if self._ref_port is None:
            raise AttributeError(
                f"{type(self).__name__}: module {self.module.name!r} has "
                "no reference port — its control law is not retargetable.")
        self._x_ref = self._ref_port.manifold.pack_any(state,
                                                       base=self._x_ref)

    def reprogram(self, solution) -> None:
        """Install a re-solved operating point — gain, feed-forward, and
        reference at once (a gain is only valid about the point it was
        solved at, so they move together).

        `solution` is anything carrying `K`, `u_ff` and `x_ref` — an
        `LQRSolution` from `LQR.resolve_at`, or a plain object/namespace
        rebuilt from JSON on the far side of a retarget service.
        """
        missing = [n for n in ("K", "u_ff", "x_ref")
                   if getattr(solution, n, None) is None]
        if missing:
            raise TypeError(
                f"{type(self).__name__}.reprogram: solution is missing "
                f"{missing}; expected an LQRSolution-shaped value "
                f"(K, u_ff, x_ref).")
        for name in ("K", "u_ff"):
            if name not in self._coeffs:
                raise AttributeError(
                    f"{type(self).__name__}: module {self.module.name!r} "
                    f"declares no {name!r} port — its control law is not "
                    f"reprogrammable.")
            want = self._coeffs[name].shape
            arr = np.asarray(getattr(solution, name),
                             dtype=float).reshape(want)
            self._coeffs[name] = arr
        self.retarget(np.asarray(solution.x_ref, dtype=float).reshape(-1))

    @property
    def x_ref(self) -> np.ndarray:
        """The live reference point (flat ambient vector)."""
        return self._x_ref.copy()

    @property
    def gain(self) -> np.ndarray:
        """The live feedback gain `K` (n_u × regulated tangent)."""
        return self._coeffs["K"].copy()

    @property
    def u_ff(self) -> np.ndarray:
        """The live feed-forward command (n_u)."""
        return self._coeffs["u_ff"].reshape(-1).copy()

    def u(self, x_flat) -> np.ndarray:
        """Control vector for a flat ambient state (full-spec layout)."""
        vals = {"x": np.asarray(x_flat, dtype=float)}
        if self._ref_port is not None:
            vals[self._ref_port.name] = self._x_ref
        vals.update(self._coeffs)
        return self.call("control", vals)["u"]

    def control(self, state: dict) -> dict[str, Any]:
        """Map a state estimate (nested or flat dict) → `{input: value}`,
        merged over the live reference point (unsupplied slots sit at
        the reference, i.e. zero error)."""
        x = self._x_port.manifold.pack_any(state, base=self._x_ref)
        return unpack_fields(self._u_fields(), self.u(x))

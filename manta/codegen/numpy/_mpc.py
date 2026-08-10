"""NumpyMpc — the HELD receding-horizon controller view."""

from __future__ import annotations

from typing import Any

import numpy as np

from ._runtime import NumpyRuntime, unpack_fields


class NumpyMpc(NumpyRuntime):
    """A receding-horizon controller: the warm plan is HELD state, one
    `tick(x, goal)` per controller period.

    Each tick runs the Module's fixed-work solve from the held plan,
    applies the receding-horizon shift (folded into the kernel), and
    returns the first control as `{input name: value}`. The plan
    carries between calls exactly like a filter's covariance — the
    view adds no controller logic, only the typed surface:

        mpc = TargetNumpy(MPC(world, ...))     # or compile=True
        u = mpc.tick(x, goal)                  # dict for the actuators
        mpc.J                                  # last plan cost (QoS)

    Plan birth: the kernel bootstraps from the zero plan on hulls that
    can act from rest; for underactuated hulls or basin-sensitive
    transits, seed with `reset_plan(mpc.plan(x, goal).U)` — the
    transform's offline solver (line-searched, multi-start, optional
    full-DDP) at mission load. `reset_plan()` (no argument) re-zeros
    it — e.g. on a goal change large enough that the old plan is the
    wrong basin.
    """

    def __init__(self, module) -> None:
        super().__init__(module)
        self._plan_field = module.state.field("plan")
        self.J: float | None = None      #: last tick's plan cost

    def _enable_compile(self) -> "NumpyMpc":
        """The tick is loop-structured: large MX node count but compact
        generated C (one emitted body per kernel + loops), so the
        engine's node-count cost gate does not apply — compile
        unconditionally. First build ~10-30 s, disk-cached by source
        hash; measured ~12x per tick."""
        from ._compile import _compiled_functions
        self._functions = _compiled_functions(
            dict(self.module.functions), max_instr=2 ** 31)
        return self

    # ---- the controller surface ----------------------------------------

    def tick(self, x, goal) -> dict[str, Any]:
        """One controller period: solve from the held plan, return the
        first control as `{input name: value}`, hold the shifted rest.

        `x` is the current state — packed ambient vector, nested dict
        (`{craft: {slot: v}}`) or flat dict (`{"craft.slot": v}`);
        `goal` is the world-frame target position (3-vector).
        """
        spec = self._x_port.manifold
        if not isinstance(x, np.ndarray):
            x = spec.pack_any(x, base=np.asarray(self._x_port.init,
                                                 dtype=float))
        out = self.call("tick", values={
            "x": np.asarray(x, dtype=float).reshape(-1),
            "goal": np.asarray(goal, dtype=float).reshape(-1)})
        self.J = float(np.asarray(out["J"]).reshape(-1)[0])
        return unpack_fields(self._u_port.fields, out["u"])

    # ---- the plan --------------------------------------------------------

    @property
    def plan(self) -> np.ndarray:
        """The held warm plan (nu × N), next tick's starting point."""
        return self._state["plan"]

    def reset_plan(self, U=None) -> None:
        """Replace the held plan — `U` as (nu × N) or (N × nu) (an
        offline `DdpResult.U` transposes in), or None to re-zero."""
        shape = self._plan_field.shape
        if U is None:
            self._state["plan"] = np.zeros(shape)
            return
        U = np.asarray(U, dtype=float)
        if U.shape == shape[::-1]:
            U = U.T
        if U.shape != shape:
            raise ValueError(
                f"reset_plan: expected {shape} (or its transpose), "
                f"got {U.shape}")
        self._state["plan"] = U.copy()

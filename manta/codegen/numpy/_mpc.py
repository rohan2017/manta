"""NumpyMpc — the HELD receding-horizon controller view."""

from __future__ import annotations

from typing import Any

import numpy as np

from ._runtime import NumpyRuntime, unpack_fields


class NumpyMpc(NumpyRuntime):
    """A receding-horizon controller: the warm plan is HELD state, one
    `tick(x, position)` per controller period.

    Each tick runs the Module's fixed-work solve from the held plan,
    applies the receding-horizon shift (folded into the kernel), and
    returns the first control as `{input name: value}`. The plan
    carries between calls exactly like a filter's covariance — the
    view adds no controller logic, only the typed surface:

        mpc = TargetNumpy(MPC(world, ...))     # or compile=True
        u = mpc.tick(x, position)              # dict for the actuators
        mpc.J                                  # last plan cost (QoS)

    Plan birth: the kernel bootstraps from the zero plan on hulls that
    can act from rest; for underactuated hulls or basin-sensitive
    transits, seed with `reset_plan(mpc.plan(x, position).U)` — the
    transform's offline solver (line-searched, multi-start, optional
    full-DDP) at mission load. `reset_plan()` (no argument) re-zeros
    it — e.g. on a target change large enough that the old plan is the
    wrong basin.
    """

    def __init__(self, module) -> None:
        super().__init__(module)
        self._plan_field = module.state.field("plan")
        self.J: float | None = None      #: last tick's plan cost
        # the held objective — targets, weights, terminal quadratic —
        # seeded from the Module's declared defaults (the transform's
        # constructor objective) and updated via set_objective()
        self._target = np.asarray(module.port("target").init,
                                  dtype=float).reshape(-1).copy()
        self._weights = np.asarray(module.port("weights").init,
                                   dtype=float).reshape(-1).copy()
        self._PT = np.asarray(module.port("PT").init,
                              dtype=float).copy()
        self._nu = len(self._u_port.fields)

    # ---- the objective ---------------------------------------------------

    def set_objective(self, *, position=None, velocity=None,
                      body_velocity=None, heading=None, orientation=None,
                      rates=None, w_position=None, w_velocity=None,
                      w_body_velocity=None, w_heading=None,
                      w_attitude=None, w_rates=None, w_effort=None, P=None) -> None:
        """Update the held objective — the control surface for a
        guidance-law switch. Targets: `position`/`velocity` world-frame
        3-vectors, `heading` a 2- or 3-vector direction (normalized
        here; commands the NOSE, not the course — hull anisotropy makes
        them agree in steady flight). Weights: scalars broadcast, or
        per-axis (3,) / per-channel (nu,) vectors; a mask is a zero.
        `P` replaces the terminal quadratic (re-solve via
        `LQR.resolve_at` when running weights change on a
        station-keeping law). Unnamed pieces keep their current
        values; whether an unweighted axis is FREE or HOLD-CURRENT is
        the caller's semantics, applied here by what you pass."""
        if position is not None:
            self._target[0:3] = np.asarray(position, dtype=float).ravel()
        if velocity is not None:
            self._target[3:6] = np.asarray(velocity, dtype=float).ravel()
        if body_velocity is not None:
            self._target[6:9] = np.asarray(body_velocity, dtype=float).ravel()
        if heading is not None:
            d = np.zeros(3)
            h = np.asarray(heading, dtype=float).ravel()
            d[:h.size] = h
            n = np.linalg.norm(d)
            if n < 1e-9:
                raise ValueError("set_objective: heading must be a "
                                 "non-zero direction")
            self._target[9:12] = d / n
        if orientation is not None:
            q = np.asarray(orientation, dtype=float).ravel()
            n = np.linalg.norm(q)
            if q.size != 4 or n < 1e-9:
                raise ValueError("set_objective: orientation must be a non-zero wxyz quaternion")
            self._target[12:16] = q / n
        if rates is not None:
            self._target[16:19] = np.asarray(rates, dtype=float).ravel()
        for lo, hi, val in ((0, 3, w_position), (3, 12, w_velocity),
                            (12, 21, w_body_velocity), (21, 22, w_heading),
                            (22, 25, w_attitude), (25, 28, w_rates),
                            (28, 28 + self._nu, w_effort)):
            if val is not None:
                a = np.asarray(val, dtype=float)
                if hi - lo == 9:
                    a = (np.eye(3) * float(a) if a.ndim == 0 else
                         np.diag(a.ravel()) if a.size == 3 else a.reshape(3, 3))
                    if not np.allclose(a, a.T) or np.linalg.eigvalsh(a).min() < -1e-10:
                        raise ValueError("set_objective: velocity weight must be symmetric positive semidefinite")
                self._weights[lo:hi] = a.ravel()
        if P is not None:
            self._PT = np.asarray(P, dtype=float).reshape(self._PT.shape)

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

    def tick(self, x, position=None) -> dict[str, Any]:
        """One controller period: solve from the held plan, return the
        first control as `{input name: value}`, hold the shifted rest.

        `x` is the current state — packed ambient vector, nested dict
        (`{craft: {slot: v}}`) or flat dict (`{"craft.slot": v}`).
        `position` is the position-target shorthand (a world 3-vector,
        persisted into the held objective); the full command family —
        velocity, heading, per-term weights — is `set_objective`'s.
        """
        if position is not None:
            self._target[0:3] = np.asarray(position,
                                            dtype=float).ravel()
        spec = self._x_port.spec
        if not isinstance(x, np.ndarray):
            x = spec.pack_any(x, base=np.asarray(self._x_port.init,
                                                 dtype=float))
        out = self.call("tick", values={
            "x": np.asarray(x, dtype=float).reshape(-1),
            "target": self._target,
            "weights": self._weights,
            "PT": self._PT})
        self.J = float(np.asarray(out["J"]).reshape(-1)[0])
        return unpack_fields(self._u_port.fields, out["u"])

    # ---- the plan --------------------------------------------------------

    @property
    def plan(self) -> np.ndarray:
        """The held warm plan (nu × N), next tick's starting point."""
        return self._state["plan"]

    def reset_plan(self, U=None) -> None:
        """Replace the held plan — `U` as (nu × N) or (N × nu) (an
        offline `MPCPlan.U`, which is (N × nu), transposes in), or None
        to re-zero."""
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

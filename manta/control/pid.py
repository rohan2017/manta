"""PID — a scalar proportional-integral-derivative controller.

`PID(kp, ki, kd, …)` is a recurrence block (a sibling of `LQR`, lowered
through the same generic Module path). It carries its own state — the
integral accumulator and the previous measurement — and bakes the
control law into one `ca.Function` the runtime evaluates each step:

    error   = setpoint − measurement
    integral ← clamp(integral + error·dt,  ±integral_limit)
    command  = kp·error + ki·integral − kd·d(measurement)/dt   (clamped)

Two textbook robustness choices are built in:

  * **Derivative on measurement**, not on error — so a setpoint step does
    not kick the derivative term. The measurement rate is a backward
    finite difference held in state; a `primed` flag zeros it on the very
    first step (deterministic across backends — no first-call special case
    in the runtime), so the kernel stays pure dataflow.
  * **Integral anti-windup** via an optional symmetric clamp on the
    accumulator (`integral_limit`); an optional `output_limit` saturates
    the command.

Gains are baked at construction (the IR boundary is finite-step, no online
retune — see the IR design notes). Scalar by design: drive each axis with
its own `PID`. A vector PID awaits a parametric Rⁿ manifold (only R3 is
implemented today); per-axis instances are the usual idiom regardless.

Lower it like any block::

    pid = TargetNumpy(PID(kp=2.0, ki=5.0, kd=0.5, integral_limit=2.0))
    cmd = pid.step(dt, setpoint=des, measurement=meas)["command"]
"""

from __future__ import annotations

import casadi as ca

from .._validation import require_finite, require_positive
from ..ir.manifold import ScalarManifold
from ..recurrence import RecurrenceBlock


class PID(RecurrenceBlock):
    """Scalar PID controller as a `recurrence` block.

    Args:
        kp, ki, kd      — proportional / integral / derivative gains.
        integral_limit  — symmetric clamp on the integral accumulator
                          (anti-windup). `None` disables it.
        output_limit    — symmetric clamp on the command. `None` disables.
        name            — codegen basename / default C++ class stem.

    Ports: inputs ``setpoint`` + ``measurement`` (scalars); output
    ``command``. State: ``integral``, ``prev_measurement``, ``primed``.
    """

    def __init__(self, kp: float, ki: float = 0.0, kd: float = 0.0, *,
                 integral_limit: float | None = None,
                 output_limit: float | None = None,
                 name: str = "pid") -> None:
        self.kp = float(require_finite(kp, name="PID.kp"))
        self.ki = float(require_finite(ki, name="PID.ki"))
        self.kd = float(require_finite(kd, name="PID.kd"))
        self.integral_limit = (None if integral_limit is None
                               else require_positive(
                                   integral_limit, name="PID.integral_limit"))
        self.output_limit = (None if output_limit is None
                             else require_positive(
                                 output_limit, name="PID.output_limit"))

        kp_, ki_, kd_ = self.kp, self.ki, self.kd
        i_lim, o_lim = self.integral_limit, self.output_limit

        def rec(x, u, dt, t):
            err   = u["setpoint"] - u["measurement"]
            integ = x["integral"] + err * dt
            if i_lim is not None:
                integ = ca.fmin(ca.fmax(integ, -i_lim), i_lim)
            # Derivative on measurement; zeroed on the first step via
            # `primed`, and at dt == 0 (a paused/stalled driving loop) —
            # the bare division there is 0·inf = NaN, which poisons the
            # command. At dt = 0 the block degrades to P + held-I.
            d_rate = ca.if_else(
                dt > 0,
                (u["measurement"] - x["prev_measurement"]) / ca.fmax(dt, 1e-300),
                0.0)
            d_meas = x["primed"] * d_rate
            cmd = kp_ * err + ki_ * integ - kd_ * d_meas
            if o_lim is not None:
                cmd = ca.fmin(ca.fmax(cmd, -o_lim), o_lim)
            x_next = {
                "integral":         integ,
                "prev_measurement": u["measurement"],
                "primed":           ca.MX.ones(1, 1),
            }
            return x_next, {"command": cmd}

        self._build_recurrence(
            name=name,
            state=[("integral",         ScalarManifold()),
                   ("prev_measurement", ScalarManifold()),
                   ("primed",           ScalarManifold())],
            inputs=[("setpoint", 1), ("measurement", 1)],
            outputs=[("command", 1)],
            x0={"integral": 0.0, "prev_measurement": 0.0, "primed": 0.0},
            recurrence=rec)

    def __repr__(self) -> str:
        lim = "" if self.integral_limit is None else f" i_lim={self.integral_limit}"
        return (f"<PID kp={self.kp} ki={self.ki} kd={self.kd}{lim}>")

"""Madgwick — a gyro + accelerometer attitude (AHRS) filter.

`Madgwick(beta, …)` is a `recurrence` block (a sibling of `PID`, lowered by
the same `lower_evaluator` backend path). It fuses a rate gyro with an
accelerometer into a single orientation quaternion using Madgwick's
gradient-descent complementary filter — the gyro integrates the fast
motion, and a normalized gradient step nudges the estimate so the
quaternion's predicted gravity direction tracks the measured accelerometer
direction, with `beta` setting how hard.

State: one `orientation` quaternion (SO(3) storage; the kernel does
Madgwick's own quaternion integration + renormalization, so the runtime
just stores it). Inputs: `gyro` (rad/s) and `accel` (any units — only its
direction is used). Output: the `orientation` quaternion ``(w, x, y, z)``.

Convention follows Madgwick's reference IMU implementation: gyro and accel
are in the sensor frame, and the estimated gravity in the sensor frame is

    g_pred = (2(q1 q3 − q0 q2), 2(q0 q1 + q2 q3), 1 − 2 q1² − 2 q2²),

which the filter drives toward the normalized accelerometer reading. A
recurrence block is forward-only (never differentiated, unlike the EKF), so
the normalizations here need no smooth-kernel treatment — just an epsilon
guard against a zero-magnitude accelerometer sample.

    ahrs = TargetNumpy(Madgwick(beta=0.1))
    q = ahrs.step(dt, gyro=gyro_xyz, accel=accel_xyz)["orientation"]
"""

from __future__ import annotations

import casadi as ca

from .._validation import require_positive
from ..ir.manifold import SO3Manifold
from ..recurrence import RecurrenceBlock


class Madgwick(RecurrenceBlock):
    """Madgwick gyro+accel AHRS filter as a `recurrence` block.

    Args:
        beta — filter gain (rad/s): the accelerometer correction rate.
               Higher trusts the accelerometer more (faster convergence,
               more noise); lower trusts the gyro. ~0.1 is a typical start.
        name — codegen basename / default C++ class stem.
    """

    def __init__(self, beta: float = 0.1, *, name: str = "madgwick") -> None:
        self.beta = require_positive(beta, name="Madgwick.beta", allow_zero=True)
        b = self.beta

        def rec(x, u, dt, t):
            q0, q1, q2, q3 = x["orientation"][0], x["orientation"][1], \
                x["orientation"][2], x["orientation"][3]
            gx, gy, gz = u["gyro"][0], u["gyro"][1], u["gyro"][2]
            ax, ay, az = u["accel"][0], u["accel"][1], u["accel"][2]

            # Rate of change of quaternion from the gyroscope.
            qd0 = 0.5 * (-q1 * gx - q2 * gy - q3 * gz)
            qd1 = 0.5 * (q0 * gx + q2 * gz - q3 * gy)
            qd2 = 0.5 * (q0 * gy - q1 * gz + q3 * gx)
            qd3 = 0.5 * (q0 * gz + q1 * gy - q2 * gx)

            # Normalize the accelerometer (eps-guarded; direction only).
            a_norm = ca.sqrt(ax * ax + ay * ay + az * az + 1e-18)
            ax, ay, az = ax / a_norm, ay / a_norm, az / a_norm

            # Gradient-descent corrective step (Madgwick reference IMU form).
            _2q0, _2q1, _2q2, _2q3 = 2 * q0, 2 * q1, 2 * q2, 2 * q3
            _4q0, _4q1, _4q2 = 4 * q0, 4 * q1, 4 * q2
            _8q1, _8q2 = 8 * q1, 8 * q2
            q0q0, q1q1, q2q2, q3q3 = q0 * q0, q1 * q1, q2 * q2, q3 * q3

            s0 = _4q0 * q2q2 + _2q2 * ax + _4q0 * q1q1 - _2q1 * ay
            s1 = (_4q1 * q3q3 - _2q3 * ax + 4 * q0q0 * q1 - _2q0 * ay
                  - _4q1 + _8q1 * q1q1 + _8q1 * q2q2 + _4q1 * az)
            s2 = (4 * q0q0 * q2 + _2q0 * ax + _4q2 * q3q3 - _2q3 * ay
                  - _4q2 + _8q2 * q1q1 + _8q2 * q2q2 + _4q2 * az)
            s3 = 4 * q1q1 * q3 - _2q1 * ax + 4 * q2q2 * q3 - _2q2 * ay
            s_norm = ca.sqrt(s0 * s0 + s1 * s1 + s2 * s2 + s3 * s3 + 1e-18)

            # Apply the feedback step and integrate.
            q0n = q0 + (qd0 - b * s0 / s_norm) * dt
            q1n = q1 + (qd1 - b * s1 / s_norm) * dt
            q2n = q2 + (qd2 - b * s2 / s_norm) * dt
            q3n = q3 + (qd3 - b * s3 / s_norm) * dt

            qn = ca.vertcat(q0n, q1n, q2n, q3n)
            qn = qn / ca.sqrt(ca.dot(qn, qn) + 1e-18)   # renormalize
            return {"orientation": qn}, {"orientation": qn}

        self._build_recurrence(
            name=name,
            state=[("orientation", SO3Manifold())],
            inputs=[("gyro", 3), ("accel", 3)],
            outputs=[("orientation", 4)],
            x0={"orientation": [1.0, 0.0, 0.0, 0.0]},
            recurrence=rec)

    def __repr__(self) -> str:
        return f"<Madgwick beta={self.beta}>"

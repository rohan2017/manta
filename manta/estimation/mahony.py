"""Mahony — a gyro + accelerometer attitude (AHRS) filter with bias estimation.

`Mahony(kp, ki, …)` is a `recurrence` block (a sibling of `Madgwick`,
lowered by the same `lower_evaluator` path — zero backend code). It is the
explicit complementary filter on SO(3): the orientation error is the cross
product of the measured accelerometer direction with the estimate's
predicted gravity, and that error drives the gyro through a PI law. The
**integral** term accumulates into an estimated gyro bias — Mahony's
defining feature over Madgwick — so a constant rate-gyro offset is learned
out rather than integrated into drift.

State: an `orientation` quaternion (SO(3)) and a `gyro_bias` 3-vector (the
integral-feedback term, *added* to the gyro — so it is the negative of a
physical additive offset, opposite in sign to the IMU part's `gyro_bias`).
Inputs: `gyro` (rad/s) and `accel` (direction only, eps-guarded). Output:
the `orientation` quaternion; the learned correction is readable from the
state.

Convention matches the reference Mahony IMU implementation (sensor-frame
gyro/accel). Like every recurrence block this is forward-only — the kernel
does the quaternion integration + renormalization directly.

    ahrs = TargetNumpy(Mahony(kp=0.5, ki=0.1))
    q = ahrs.step(dt, gyro=gyro_xyz, accel=accel_xyz)["orientation"]
"""

from __future__ import annotations

import casadi as ca

from ..ir.manifold import R3Manifold, SO3Manifold
from ..recurrence import RecurrenceBlock


class Mahony(RecurrenceBlock):
    """Mahony gyro+accel AHRS filter (with bias estimation) as a `recurrence`
    block.

    Args:
        kp   — proportional gain on the accelerometer error (how hard the
               accel pulls the attitude).
        ki   — integral gain; the rate at which the gyro-bias estimate
               learns. `0` disables bias estimation (pure complementary).
        name — codegen basename / default C++ class stem.
    """

    def __init__(self, kp: float = 0.5, ki: float = 0.0, *,
                 name: str = "mahony") -> None:
        self.kp = float(kp)
        self.ki = float(ki)
        two_kp, two_ki = 2.0 * self.kp, 2.0 * self.ki

        def rec(x, u, dt, t):
            q0, q1, q2, q3 = (x["orientation"][0], x["orientation"][1],
                              x["orientation"][2], x["orientation"][3])
            bx, by, bz = x["gyro_bias"][0], x["gyro_bias"][1], x["gyro_bias"][2]
            gx, gy, gz = u["gyro"][0], u["gyro"][1], u["gyro"][2]
            ax, ay, az = u["accel"][0], u["accel"][1], u["accel"][2]

            # Normalize the accelerometer (eps-guarded; direction only).
            a_norm = ca.sqrt(ax * ax + ay * ay + az * az + 1e-18)
            ax, ay, az = ax / a_norm, ay / a_norm, az / a_norm

            # Estimated direction of gravity (half), and the error = measured
            # × estimated (cross product).
            halfvx = q1 * q3 - q0 * q2
            halfvy = q0 * q1 + q2 * q3
            halfvz = q0 * q0 - 0.5 + q3 * q3
            halfex = ay * halfvz - az * halfvy
            halfey = az * halfvx - ax * halfvz
            halfez = ax * halfvy - ay * halfvx

            # Integral feedback → gyro-bias estimate; corrected rate.
            bx_n = bx + two_ki * halfex * dt
            by_n = by + two_ki * halfey * dt
            bz_n = bz + two_ki * halfez * dt
            gxc = gx + bx_n + two_kp * halfex
            gyc = gy + by_n + two_kp * halfey
            gzc = gz + bz_n + two_kp * halfez

            # First-order quaternion integration (snapshot the old q).
            hx, hy, hz = 0.5 * dt * gxc, 0.5 * dt * gyc, 0.5 * dt * gzc
            q0n = q0 + (-q1 * hx - q2 * hy - q3 * hz)
            q1n = q1 + (q0 * hx + q2 * hz - q3 * hy)
            q2n = q2 + (q0 * hy - q1 * hz + q3 * hx)
            q3n = q3 + (q0 * hz + q1 * hy - q2 * hx)
            qn = ca.vertcat(q0n, q1n, q2n, q3n)
            qn = qn / ca.sqrt(ca.dot(qn, qn) + 1e-18)
            return ({"orientation": qn,
                     "gyro_bias": ca.vertcat(bx_n, by_n, bz_n)},
                    {"orientation": qn})

        self._build_recurrence(
            name=name,
            state=[("orientation", SO3Manifold()),
                   ("gyro_bias", R3Manifold())],
            inputs=[("gyro", 3), ("accel", 3)],
            outputs=[("orientation", 4)],
            x0={"orientation": [1.0, 0.0, 0.0, 0.0],
                "gyro_bias": [0.0, 0.0, 0.0]},
            recurrence=rec)

    def __repr__(self) -> str:
        return f"<Mahony kp={self.kp} ki={self.ki}>"

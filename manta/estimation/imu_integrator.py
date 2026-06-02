"""IMUIntegrator — strapdown inertial dead-reckoning from accel + gyro.

`IMUIntegrator(gravity, …)` is a `recurrence` block (a sibling of the AHRS
filters, lowered by the same `lower_evaluator` path — zero backend code).
It mechanizes the strapdown INS equations: integrate the body-frame gyro
into attitude, rotate the body-frame specific force into the world frame,
add gravity, and integrate that into velocity and position.

    q' = q ⊗ exp(ω·dt)                      (body-frame attitude increment)
    a_world = R(q)·f_body + g
    v' = v + a_world·dt
    p' = p + v·dt + ½ a_world·dt²

State: `position` (R3, world), `velocity` (R3, world), `orientation` (SO(3),
body→world). Inputs: `accel` — the accelerometer's specific force in the
body frame (so at rest, level, it reads +g upward) — and `gyro` (rad/s,
body). Outputs all three states, so a consumer can read pose + twist
directly (the first recurrence block with more than one readout port).

Pure dead-reckoning: with no correction it drifts (that is what the EKF /
AHRS filters are for); this is the deterministic propagation underneath
them, and the deploy-to-robot strapdown front end. The attitude increment
uses the exact SO(3) exponential (forward-only block — no autodiff, so no
smooth-kernel constraint); position/velocity use the constant-acceleration
update over the step.

    ins = TargetNumpy(IMUIntegrator())
    s = ins.step(dt, accel=accel_body, gyro=gyro_body)
    p, v, q = s["position"], s["velocity"], s["orientation"]
"""

from __future__ import annotations

import casadi as ca

from ..ir.manifold import R3Manifold, SO3Manifold, _quat_mul_mx, _so3_exp
from ..recurrence import RecurrenceBlock


class IMUIntegrator(RecurrenceBlock):
    """Strapdown INS dead-reckoner as a `recurrence` block.

    Args:
        gravity — world-frame gravity vector (default ``(0, 0, -9.81)``).
        p0, v0  — initial position / velocity (world). Default zero.
        q0      — initial orientation quaternion ``(w, x, y, z)``, body→world.
                  Default identity.
        name    — codegen basename / default C++ class stem.
    """

    def __init__(self, *, gravity=(0.0, 0.0, -9.81),
                 p0=(0.0, 0.0, 0.0), v0=(0.0, 0.0, 0.0),
                 q0=(1.0, 0.0, 0.0, 0.0), name: str = "imu_integrator") -> None:
        self.gravity = tuple(float(x) for x in gravity)
        g = ca.DM(list(self.gravity))

        def rec(x, u, dt, t):
            q = x["orientation"]
            v = x["velocity"]
            p = x["position"]
            accel = u["accel"]
            gyro = u["gyro"]

            # Specific force, body → world (quaternion sandwich), + gravity.
            q_conj = ca.vertcat(q[0], -q[1], -q[2], -q[3])
            a_body_q = ca.vertcat(0, accel[0], accel[1], accel[2])
            a_world_q = _quat_mul_mx(_quat_mul_mx(q, a_body_q), q_conj)
            a_world = a_world_q[1:4] + g

            v_next = v + a_world * dt
            p_next = p + v * dt + 0.5 * a_world * dt * dt
            # Exact body-frame attitude increment for the step.
            q_next = _quat_mul_mx(q, _so3_exp(gyro * dt))

            nxt = {"position": p_next, "velocity": v_next,
                   "orientation": q_next}
            return nxt, dict(nxt)

        self._build_recurrence(
            name=name,
            state=[("position", R3Manifold()),
                   ("velocity", R3Manifold()),
                   ("orientation", SO3Manifold())],
            inputs=[("accel", 3), ("gyro", 3)],
            outputs=[("position", 3), ("velocity", 3), ("orientation", 4)],
            x0={"position": list(map(float, p0)),
                "velocity": list(map(float, v0)),
                "orientation": list(map(float, q0))},
            recurrence=rec)

    def __repr__(self) -> str:
        return f"<IMUIntegrator gravity={self.gravity}>"

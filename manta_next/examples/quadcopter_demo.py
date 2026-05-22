"""Quadcopter demo — 4× linear Thrusters with yaw-reaction torque.

Mirrors legacy ex2_quadcopter. Each rotor is a 1st-order `Thruster` with:
  * thrust along +z (body) scaled by throttle
  * yaw torque about body z with sign alternating CW/CCW between
    diagonal pairs (the standard quadcopter chirality)

Demonstrates:
  * Hover when all four rotors run at the throttle that exactly cancels
    gravity, with no net yaw torque from the alternating chirality.
  * Yaw control by raising two opposite-chirality rotors and lowering
    the other pair — the four torques no longer cancel → net z torque.

Run::

    .venv/bin/python -m manta_next.examples.quadcopter_demo
"""

import numpy as np

from manta_next import Craft, World
from manta_next.fields import GravityField
from manta_next.parts import IMU, Mass, PositionSensor, Thruster


def _build_quadcopter() -> Craft:
    """Quad in '+' configuration: rotors at (+L,0), (0,+L), (-L,0), (0,-L)
    in body x-y plane. Chirality alternates CW/CCW so opposing rotors
    have the same sense — standard layout."""
    L         = 0.25     # arm length
    mass      = 1.0
    max_thrust = 6.0     # per rotor, N
    k_yaw      = 0.05    # yaw torque coefficient per rotor (N·m / unit thrust)

    c = Craft("quad")
    c.add(Mass("body", mass=mass, moi=(0.01, 0.01, 0.02)))
    # Rotors — alternating CW (+τ_z) and CCW (-τ_z). force vector is
    # along body +z scaled by max_thrust; torque vector is along the
    # same axis scaled by yaw coefficient (signed for chirality).
    F_vec = (0.0, 0.0, max_thrust)
    τ_cw  = (0.0, 0.0, +k_yaw * max_thrust)
    τ_ccw = (0.0, 0.0, -k_yaw * max_thrust)
    c.add(Thruster("rotor_front",
                   force=F_vec, torque=τ_cw,  transform=(+L, 0, 0)))
    c.add(Thruster("rotor_right",
                   force=F_vec, torque=τ_ccw, transform=(0, +L, 0)))
    c.add(Thruster("rotor_back",
                   force=F_vec, torque=τ_cw,  transform=(-L, 0, 0)))
    c.add(Thruster("rotor_left",
                   force=F_vec, torque=τ_ccw, transform=(0, -L, 0)))
    # Sensors.
    c.add(IMU("imu"))
    c.add(PositionSensor("gps"))
    return c


def main() -> None:
    g = (0.0, 0.0, -9.81)
    quad = _build_quadcopter()
    w = World().add_field(GravityField().add_uniform(g))
    w.add_craft(quad, position=(0, 0, 2))
    cw = w.compile()
    state = cw.initial_state()

    # 4 rotors × per-rotor max_thrust must collectively counter m·g.
    # m·g = 9.81 N; total max thrust = 4·6 = 24 N; throttle_hover = 9.81/24
    # ≈ 0.409.
    m = 1.0
    throttle_hover = (m * 9.81) / (4 * 6.0)

    print(f"{'t (s)':>6} {'z (m)':>8} {'yaw (deg)':>10} {'ω_z':>10} "
          f"{'cmd':>6}")
    dt = 0.005
    n  = 800     # 4 s

    yaw_pulse_window = (1.0, 2.0)   # raise diagonal rotors during this window

    for i in range(n):
        t = i * dt
        base = throttle_hover

        # During the pulse window, raise the CW-pair (front+back), lower
        # the CCW-pair (right+left) → net positive yaw torque → spin in +z.
        if yaw_pulse_window[0] < t < yaw_pulse_window[1]:
            cw_boost = +0.05
        else:
            cw_boost = 0.0

        state["quad"]["rotor_front.throttle"] = base + cw_boost
        state["quad"]["rotor_back.throttle"]  = base + cw_boost
        state["quad"]["rotor_right.throttle"] = base - cw_boost
        state["quad"]["rotor_left.throttle"]  = base - cw_boost
        state = cw.step(state, dt=dt)

        if (i + 1) % 100 == 0:
            p = np.array(state["quad"]["position"]).ravel()
            q = np.array(state["quad"]["orientation"]).ravel()
            # yaw = atan2(2(qw·qz + qx·qy), 1 - 2(qy² + qz²))
            yaw = np.degrees(np.arctan2(
                2.0 * (q[0]*q[3] + q[1]*q[2]),
                1.0 - 2.0 * (q[2]**2 + q[3]**2)))
            wz = float(np.array(state["quad"]["angular_velocity"])[2])
            print(f"{t:>6.2f} {p[2]:>8.3f} {yaw:>10.2f} {wz:>10.4f} "
                  f"{cw_boost:>+6.3f}")


if __name__ == "__main__":
    main()

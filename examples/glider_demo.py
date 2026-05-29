"""Glider demo — NACA airfoil gliding under gravity.

A wing with no propulsion released into a still atmosphere. Lift from
forward motion balances gravity; the glider sustains horizontal flight
with a steady descent.

Run::

    .venv/bin/python -m examples.glider_demo
"""

import numpy as np

from manta import Craft, Sim, TargetNumpy, World
from manta.fields import GravityField, FluidField
from manta.parts import Mass, Naca00xx, IMU


def main() -> None:
    rho_air = 1.225
    g = (0.0, 0.0, -9.81)

    glider = Craft("glider")
    # 5 kg glider with a 1.5 m² wing.
    glider.add(Mass("body", mass=5.0, moi=(0.1, 0.5, 0.5)))
    glider.add(Naca00xx("wing",
                         area=1.5, CL_max=1.2,
                         CD_0=0.02, induced_k=0.05,
                         chord_axis=(1.0, 0.0, 0.0),
                         normal_axis=(0.0, 0.0, 1.0)))
    glider.add(IMU("imu"))

    w = World().add_field(GravityField().add_uniform(g)).add_field(FluidField().add_uniform(density=rho_air))
    # Launched horizontal at 15 m/s with the chord level.
    w.add_craft(glider, position=(0.0, 0.0, 100.0),
                velocity=(15.0, 0.0, 0.0))
    cw = TargetNumpy(Sim(w))
    state = cw.initial_state()

    dt = 0.01
    n  = 1500     # 15 s

    print(f"{'t (s)':>6} {'x (m)':>8} {'z (m)':>8} {'vx':>7} {'vz':>7} "
          f"{'|v|':>7} {'gamma (deg)':>11}")
    for i in range(n):
        state = cw.step(state, dt=dt)
        if (i + 1) % 100 == 0:
            t = (i + 1) * dt
            p = np.array(state["glider"]["position"]).ravel()
            v = np.array(state["glider"]["velocity"]).ravel()
            vmag = float(np.linalg.norm(v))
            # Flight path angle (negative = descending).
            gamma = np.degrees(np.arctan2(v[2], v[0]))
            print(f"{t:>6.2f} {p[0]:>8.2f} {p[2]:>8.2f} "
                  f"{v[0]:>7.2f} {v[2]:>+7.2f} {vmag:>7.2f} {gamma:>+11.2f}")


if __name__ == "__main__":
    main()

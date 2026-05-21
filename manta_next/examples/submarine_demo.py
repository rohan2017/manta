"""Submarine demo — Mass + PointBuoy + DragSurface in a seawater FluidField.

Mirrors legacy ex8_submarine:
  * neutrally buoyant body (ρ_water · V = m)
  * forward propulsion via a Thruster at the tail
  * 3-axis hydrodynamic drag (low Cd axial, higher Cd lateral)
  * IMU + DVL + PositionSensor for state observation

Runs a "commanded forward + dive" maneuver: apply forward thrust for 5
seconds, then add a downward thrust component to dive. Prints depth +
forward velocity at intervals.

Run::

    .venv/bin/python -m manta_next.examples.submarine_demo
"""

import numpy as np

from manta_next import Craft, World
from manta_next.parts import (
    DVL, DragSurface, IMU, Mass, PointBuoy, PositionSensor, Thruster,
)


def main() -> None:
    g_world = (0.0, 0.0, -9.81)
    rho_sea = 1025.0

    # Sub geometry: 2 m long, 0.3 m diameter cylinder. Mass 100 kg.
    L     = 2.0
    R     = 0.15
    mass  = 100.0
    V_disp = mass / rho_sea   # neutrally buoyant

    # DragSurface is isotropic in v1 — drag = ½·ρ·A·Cd·|v|·v regardless
    # of orientation. Mounting one with a tuned (A, Cd) for a streamlined
    # sub gives a realistic terminal forward velocity. (Anisotropic drag —
    # different Cd along vs. across — is a v2 feature; the legacy
    # ex8_submarine uses the same single-coefficient model.)
    A_drag  = np.pi * R**2     # axial frontal area
    Cd_hull = 0.4              # streamlined body

    sub = Craft("nautilus")
    sub.add(Mass("hull", mass=mass, moi=(2.0, 8.0, 8.0)))
    sub.add(PointBuoy("buoy", volume=V_disp))
    sub.add(DragSurface("hull_drag", area=A_drag, drag_coefficient=Cd_hull))
    # Stern thruster along +x.
    sub.add(Thruster("propulsion", force=(1.0, 0.0, 0.0),
                     transform=(-L/2, 0.0, 0.0)))
    # Vertical thruster for diving.
    sub.add(Thruster("dive", force=(0.0, 0.0, -1.0)))
    # Sensors.
    sub.add(IMU("imu"))
    sub.add(DVL("dvl"))
    sub.add(PositionSensor("gps"))

    w = (World()
         .add_uniform_gravity(g_world)
         .add_uniform_fluid(density=rho_sea))
    w.add_craft(sub, position=(0.0, 0.0, -10.0))    # 10 m depth
    cw = w.compile()
    state = cw.initial_state()

    dt = 0.01
    n_total = 1500   # 15 s

    print(f"{'t (s)':>6} {'depth (m)':>10} {'vx (m/s)':>10} "
          f"{'vz (m/s)':>10} {'pitch (deg)':>12} {'thrust_x':>10} {'thrust_z':>10}")
    for i in range(n_total):
        t = i * dt
        # Forward thrust ramps up over first 2s to 500 N steady.
        thrust_x = min(500.0, 500.0 * t / 2.0)
        # Dive thrust on after t=5s: 200 N downward (in body frame).
        thrust_z = 200.0 if t > 5.0 else 0.0

        state["nautilus"]["propulsion.throttle"] = thrust_x
        state["nautilus"]["dive.throttle"]       = thrust_z
        state = cw.step(state, dt=dt)

        if (i + 1) % 150 == 0:
            p   = np.array(state["nautilus"]["position"]).ravel()
            v   = np.array(state["nautilus"]["velocity"]).ravel()
            q   = np.array(state["nautilus"]["orientation"]).ravel()
            # extract pitch from quat (rotation about body y).
            pitch = np.degrees(np.arcsin(
                np.clip(2 * (q[0]*q[2] - q[3]*q[1]), -1.0, 1.0)))
            print(f"{t:>6.2f} {p[2]:>10.2f} {v[0]:>10.3f} "
                  f"{v[2]:>10.3f} {pitch:>12.2f} {thrust_x:>10.1f} {thrust_z:>10.1f}")


if __name__ == "__main__":
    main()

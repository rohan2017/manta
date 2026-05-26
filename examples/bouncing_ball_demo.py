"""Bouncing-ball demo — Collider + CollisionField ground plane.

A unit-mass ball dropped from 2 m above a ground plane. The Collider
provides spring + damping contact; energy bleeds out of each bounce
until the ball settles at the compression-equilibrium depth (m·g/k).

Run::

    .venv/bin/python -m examples.bouncing_ball_demo
"""

import numpy as np

from manta import Craft, World, TargetNumpy
from manta.fields import CollisionField, HalfSpace, GravityField
from manta.parts import Collider, Mass


def main() -> None:
    g = 9.81
    k = 5.0e3
    c_damp = 30.0

    cf = CollisionField()
    cf.add(HalfSpace(origin=(0, 0, 0), normal=(0, 0, 1)))
    w = (World()
         .add_field(GravityField().add_uniform((0, 0, -g)))
         .add_field(cf))

    ball = Craft("ball")
    ball.add(Mass("body", mass=1.0, moi=(0.01, 0.01, 0.01)))
    ball.add(Collider("contact", stiffness=k, damping=c_damp))
    w.add_craft(ball, position=(0, 0, 2.0))
    cw = TargetNumpy(w.compile())
    state = cw.initial_state()

    dt = 0.001
    n  = 4000     # 4 s

    print(f"{'t (s)':>6} {'z (m)':>10} {'vz (m/s)':>10} {'KE+PE (J)':>10}")
    z_peaks = []
    last_z = state["ball"]["position"][2]
    going_up = False
    for i in range(n):
        state = cw.step(state, dt=dt)
        z  = float(state["ball"]["position"][2])
        vz = float(state["ball"]["velocity"][2])

        # Detect local max (apex of a bounce).
        if going_up and vz < 0:
            z_peaks.append(z)
        going_up = vz > 0

        if (i + 1) % 250 == 0:
            t  = (i + 1) * dt
            ke = 0.5 * (vz ** 2)
            pe = max(0.0, z) * g
            print(f"{t:>6.2f} {z:>10.4f} {vz:>10.4f} {(ke + pe):>10.4f}")

    print()
    print(f"Bounce apex heights (m): {[f'{p:.3f}' for p in z_peaks[:6]]}")
    eq_depth = 1.0 * g / k
    print(f"Compression equilibrium depth: {eq_depth*1000:.3f} mm "
          f"(theoretical m·g/k)")


if __name__ == "__main__":
    main()

"""Bouncing ball — `Collider` + `CollisionField` contact, visualized.

A unit-mass ball is dropped onto a ground plane. The `Collider` is a
spring + damper: each bounce loses energy until the ball settles at the
static-compression depth ``m·g/k``. Watch the bounces decay in the rerun
viewer; the terminal prints the apex heights and the settling depth.

Run::

    .venv/bin/python -m examples.physics.bouncing_ball
    .venv/bin/python -m examples.physics.bouncing_ball --no-viz   # headless
"""

from __future__ import annotations

import argparse

import numpy as np

from manta import Craft, Sim, TargetNumpy, World
from manta.fields import CollisionField, GravityField, HalfSpace
from manta.parts import Collider, Mass

from .._viz import Viz

RADIUS = 0.1     # ball radius, for drawing only


def build_world(g: float, k: float, c_damp: float):
    cf = CollisionField()
    cf.add(HalfSpace(origin=(0, 0, 0), normal=(0, 0, 1)))
    w = (World()
         .add_field(GravityField().add_uniform((0, 0, -g)))
         .add_field(cf))
    ball = Craft("ball")
    ball.add(Mass("body", mass=1.0, moi=(0.01, 0.01, 0.01)))
    ball.add(Collider("contact", stiffness=k, damping=c_damp))
    w.add_craft(ball, position=(0, 0, 2.0))
    return w


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--no-viz", action="store_true", help="run headless")
    p.add_argument("--duration", type=float, default=4.0)
    args = p.parse_args()

    g, k, c_damp = 9.81, 5.0e3, 30.0
    sim = TargetNumpy(Sim(build_world(g, k, c_damp)))
    state = sim.initial_state()

    viz = Viz("manta/bouncing_ball", spawn=not args.no_viz)
    viz.plane("world/ground", z=0.0, size=3.0, color=(70, 110, 70, 200))

    dt = 0.001
    n = int(args.duration / dt)

    print(f"{'t (s)':>6} {'z (m)':>10} {'vz (m/s)':>10} {'KE+PE (J)':>10}")
    z_peaks, last_vz = [], 0.0
    for i in range(n):
        state = sim.step(state, dt=dt)
        z = float(state["ball"]["position"][2])
        vz = float(state["ball"]["velocity"][2])
        if last_vz > 0 and vz <= 0:          # apex of a bounce
            z_peaks.append(z)
        last_vz = vz

        t = (i + 1) * dt
        viz.t(t)
        viz.point("world/ball", state["ball"]["position"],
                  color=(240, 180, 60), radius=RADIUS)
        viz.trail("world/ball/trail", state["ball"]["position"])

        if (i + 1) % 250 == 0:
            ke, pe = 0.5 * vz ** 2, max(0.0, z) * g
            print(f"{t:>6.2f} {z:>10.4f} {vz:>10.4f} {(ke + pe):>10.4f}")

    print()
    print(f"Bounce apex heights (m): {[f'{p:.3f}' for p in z_peaks[:6]]}")
    print(f"Compression equilibrium depth: {1.0 * g / k * 1000:.3f} mm "
          f"(theoretical m·g/k)")


if __name__ == "__main__":
    main()

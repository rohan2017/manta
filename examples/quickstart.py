"""Quickstart — the smallest useful manta program.

A single ball (one `Mass`) is thrown into uniform gravity and follows a
parabola. We build the model, lower it to the native-Python backend with
`TargetNumpy(Sim(world))`, integrate forward, and report the apex height
the ball reached — then check it against the textbook value
``z0 + vz0² / (2g)``.

If this prints "manta is working" you have a working install. From here:

  * ``examples/physics/``  — bouncing ball, spinning top, Foucault pendulum
                             (with rerun visualization)
  * ``examples/vehicles/`` — quadcopter (EKF + LQR), airplane, submarine,
                             hydrofoil (visualization + keyboard control)

Run::

    .venv/bin/python -m examples.quickstart
"""

import numpy as np

from manta import Craft, Sim, TargetNumpy, World
from manta.fields import GravityField
from manta.parts import Mass


def main() -> None:
    mass = 2.0                       # kg
    g = 9.81                         # m/s²
    v0 = (5.0, 0.0, 12.0)            # initial velocity (m/s): up and forward
    z0 = 0.0                         # launch height (m)

    # Model: one-part craft (just a point mass) in uniform downward gravity.
    ball = Craft("ball")
    ball.add(Mass("body", mass=mass))
    world = World().add_field(GravityField(g=(0.0, 0.0, -g)))
    world.add_craft(ball, position=(0.0, 0.0, z0), velocity=v0)

    # Lower to the native-Python backend and integrate. The runtime holds
    # the state: `sim.state` is the nested dict, `sim.step(dt)` advances it.
    sim = TargetNumpy(Sim(world))

    dt = 0.001
    apex = z0
    t_apex = 0.0
    t = 0.0
    while True:
        sim.step(dt)
        t += dt
        z = float(sim.state["ball"]["position"][2])
        if z > apex:
            apex, t_apex = z, t
        # Stop once the ball has fallen back to (or below) the launch height.
        if z < z0 and t > 2.0 * dt:
            break

    expected = z0 + v0[2] ** 2 / (2.0 * g)
    print("manta is working ✔")
    print(f"  mass {mass} kg, gravity {g} m/s², launch velocity {v0} m/s")
    print(f"  simulated apex : {apex:.3f} m  at t = {t_apex:.3f} s")
    print(f"  textbook apex  : {expected:.3f} m  (z0 + vz0²/2g)")
    print(f"  agreement      : {abs(apex - expected) * 1000:.1f} mm")


if __name__ == "__main__":
    main()

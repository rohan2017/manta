"""Spinning top — gyroscopic precession via a passive `Joint`, visualized.

A rotor (`Mass` on a passive `Joint`) sits on a body. The Joint carries
the gyroscopic-torque-on-body reaction ``-ω_body × L_rotor``, so when a
brief lateral impulse hits the body it **precesses** about the spin axis
instead of toppling — the classic top. Two runs, same rig:

  1. rotor spinning at 100 rad/s → the spin axis sweeps a precession cone;
  2. rotor stationary           → no angular momentum, the body tumbles.

The viewer shows run (1): the body box, the spin axis, and the trail its
tip sweeps. The terminal prints angular-velocity history for both.

Run::

    .venv/bin/python -m examples.physics.spinning_top
    .venv/bin/python -m examples.physics.spinning_top --no-viz
"""

from __future__ import annotations

import argparse

import numpy as np

from manta import Craft, Sim, TargetNumpy, World
from manta.fields import GravityField
from manta.parts import Joint, Mass, Thruster

from .._viz import Viz


def _build_top() -> Craft:
    """Body MOI ~ rotor MOI·rate so the precession period spans many ticks
    (keeps the symplectic integrator comfortable)."""
    c = Craft("top")
    c.add(Mass("body", mass=0.2, moi=(0.005, 0.005, 0.001)))
    rotor = Joint("rotor", mode="passive", axis=(0.0, 0.0, 1.0))
    rotor.add(Mass("rotor_disk", mass=0.05, moi=(0.0001, 0.0001, 0.001)))
    c.add(rotor)
    # Off-axis thruster, pulsed briefly for a lateral body-frame torque.
    c.add(Thruster("kick", force=(0.0, 0.0, 1.0), transform=(0.0, 0.1, 0.0)))
    return c


def _run(label: str, rotor_rate: float, viz: Viz | None, dt: float, n: int):
    w = World().add_field(GravityField().add_uniform((0.0, 0.0, 0.0)))  # g off
    w.add_craft(_build_top())
    sim = TargetNumpy(Sim(w))
    sim.state["top"]["rotor.rate"] = rotor_rate

    if viz is not None:
        viz.box("world/top/body", (0.06, 0.06, 0.012), color=(200, 200, 210))
        viz.box("world/top/rotor", (0.04, 0.04, 0.006), color=(230, 150, 60))

    print(f"\n=== {label}  rotor_rate = {rotor_rate} rad/s ===")
    print(f"{'t (s)':>5} {'ω_x':>10} {'ω_y':>10} {'ω_z':>10} {'rotor':>9}")
    for i in range(n):
        t = i * dt
        sim.state["top"]["kick.throttle"] = 0.5 if 0.1 <= t < 0.2 else 0.0
        sim.step(dt)

        if viz is not None:
            q = np.asarray(sim.state["top"]["orientation"]).ravel()
            viz.t(t)
            viz.pose("world/top", (0, 0, 0), q)
            # Spin axis (body z) drawn in the body frame; its world tip
            # traces the precession cone.
            viz.arrow("world/top/axis", (0, 0, 0), (0, 0, 0.25),
                      color=(90, 170, 255), radius=0.004)
            w_, x_, y_, z_ = q
            zaxis = np.array([2*(x_*z_ + w_*y_), 2*(y_*z_ - w_*x_),
                              1 - 2*(x_*x_ + y_*y_)]) * 0.25
            viz.trail("world/tip", zaxis, color=(90, 170, 255))

        if i in (99, 199, 499, 999, 1999, 2999):
            omega = np.asarray(sim.state["top"]["angular_velocity"]).ravel()
            print(f"{t:>5.2f} {omega[0]:>10.4f} {omega[1]:>10.4f} "
                  f"{omega[2]:>10.4f} {float(sim.state['top']['rotor.rate']):>9.1f}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--no-viz", action="store_true", help="run headless")
    p.add_argument("--duration", type=float, default=3.0)
    args = p.parse_args()

    dt = 0.001
    n = int(args.duration / dt)
    viz = None if args.no_viz else Viz("manta/spinning_top")

    print("Spinning top — lateral impulse during t ∈ [0.1, 0.2] s.")
    print("Spinning rotor → precession (ω_x, ω_y oscillate in quadrature);")
    print("stationary rotor → toppling (ω_x runs away).")
    _run("SPINNING", 100.0, viz, dt, n)        # visualized
    _run("STATIONARY", 0.0, None, dt, n)       # terminal only


if __name__ == "__main__":
    main()

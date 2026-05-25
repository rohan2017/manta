"""Coriolis demo — pseudo-forces emerge from frame transforms, not from
the integrator.

Setup: a ball launched radially outward from the rotation axis of a
fast-spinning "planet." There are no real forces — no gravity, no
thrust, nothing. The integrator runs in WorldFrame (inertial), so in
WorldFrame the ball flies in a straight line.

When we ask "where is the ball relative to the planet?", we transform
the WorldFrame state back through `Planet.world_to_planet(t)`. The
result curves — the classic Coriolis deflection. The framework adds
no fictitious forces; the deflection falls out of the rotation of
the observation frame.

This is exactly the design point: keep Newton-Euler honest in
WorldFrame, let frame transforms do the bookkeeping for everything
else.

Run::

    .venv/bin/python -m manta_next.examples.coriolis_drop_demo
"""

from __future__ import annotations

from manta_next import Craft, Planet, TargetNumpy, World
from manta_next.parts import Mass


def main() -> None:
    # Exaggerated rotation rate so Coriolis is visible over a 2 s run.
    # Earth's ω is ~7.27e-5 rad/s — too small to see deflection without
    # running for hours.
    planet = Planet(rotation_axis=(0.0, 0.0, 1.0), omega=0.5)

    # Ball at PlanetFrame origin (on the rotation axis) with initial
    # PlanetFrame velocity = 2 m/s along +x. Sitting on the rotation
    # axis means p × ω = 0, so there's no tangential velocity to add —
    # WorldFrame velocity equals PlanetFrame velocity.
    c = Craft("ball")
    c.add(Mass("body", mass=1.0))

    w = World()
    w.add_planet(planet)
    w.add_craft(c,
                position=planet.position(0.0, 0.0, 0.0),
                velocity=planet.velocity(2.0, 0.0, 0.0))

    sim   = TargetNumpy(w.compile())
    state = sim.initial_state()

    print(f"planet: {planet}")
    print(f"initial PlanetFrame: p=(0,0,0), v=(2,0,0)")
    print()
    print(f"{'t (s)':>6} {'p_world_x':>11} {'p_world_y':>11} "
          f"{'p_planet_x':>11} {'p_planet_y':>11}")

    dt = 0.001
    t  = 0.0
    print_every = 200   # every 0.2 s
    for step in range(1, 2001):  # 2 s
        state = sim.step(state, dt=dt, t=t)
        t += dt
        if step % print_every == 0:
            p_w = state["ball"]["position"]
            p_p, _ = planet.world_to_planet(p_w, state["ball"]["velocity"], t)
            print(f"{t:6.2f} {p_w[0]:11.4f} {p_w[1]:11.4f} "
                  f"{p_p[0]:11.4f} {p_p[1]:11.4f}")

    print()
    print("Observe: p_world_y stays ~0 (ball flies straight in"
          " WorldFrame, no forces). p_planet_y is large and negative — "
          "Coriolis deflection viewed in the rotating frame.")


if __name__ == "__main__":
    main()

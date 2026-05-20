"""Dual-craft demo — two crafts joined by a spring-damper Tether.

Mirrors legacy ex9_dual_craft. Two AUV-like crafts (Mass + DragSurface
in seawater) are tethered together with a 10 m rest-length spring. One
is given a perpendicular initial velocity; the spring pulls them into
a coupled motion, and drag damps it out.

Run::

    .venv/bin/python -m manta_next.examples.dual_craft_demo
"""

import numpy as np

from manta_next import Craft, World
from manta_next.couplings import Tether
from manta_next.parts import DragSurface, Mass, TetherEndpoint


def _build_buddy(name: str) -> Craft:
    c = Craft(name)
    c.add(Mass("body", mass=10.0, moi=(0.5, 0.5, 0.5)))
    c.add(TetherEndpoint("hook"))
    c.add(DragSurface("drag", area=0.005, drag_coefficient=0.4))
    return c


def main() -> None:
    # Both crafts in the same seawater field, no gravity (sideways view).
    rho_sea = 1025.0
    w = (World()
         .add_uniform_gravity((0.0, 0.0, 0.0))
         .add_uniform_fluid(density=rho_sea))

    alice = _build_buddy("alice")
    bob   = _build_buddy("bob")

    w.add_craft(alice, position=(0, 0, 0),  velocity=(0, 2.0, 0))
    w.add_craft(bob,   position=(10, 0, 0))

    # 10 m rest length tether — they start exactly at rest length. Alice
    # has perpendicular initial velocity → arcing motion. Low drag,
    # low damping → visible oscillation before settling.
    w.add_coupling(Tether(alice, "hook", bob, "hook",
                          stiffness=20.0, damping=0.5, rest_length=10.0))

    cw = w.compile()
    state = cw.initial_state()
    comp = list(cw.components)[0]
    print(f"Connected component '{comp}' carries both crafts.")
    print(f"State keys: {sorted(state[comp].keys())[:4]} …")

    dt = 0.01
    n  = 1200

    print()
    print(f"{'t (s)':>6} {'alice xy':>22} {'bob xy':>22} {'|sep|':>8}")
    for i in range(n):
        state = cw.step(state, dt=dt)
        if (i + 1) % 100 == 0:
            t = (i + 1) * dt
            pa = np.array(state[comp]["alice.position"]).ravel()
            pb = np.array(state[comp]["bob.position"]).ravel()
            sep = np.linalg.norm(pb - pa)
            print(f"{t:>6.2f} "
                  f"({pa[0]:>+6.2f},{pa[1]:>+6.2f},{pa[2]:>+6.2f}) "
                  f"({pb[0]:>+6.2f},{pb[1]:>+6.2f},{pb[2]:>+6.2f}) "
                  f"{sep:>8.3f}")


if __name__ == "__main__":
    main()

"""ex8 — single drone Craft type, instantiated 3 times in the user main.

Each drone is a small free-flying body with one thruster pointing along
+x. Tether endpoints are added by the user main (not by codegen) because
they need per-instance Tether references that vary across instances of
the same craft type — Pattern A multi-craft, glued at the C++ level.

Codegen:

    PYTHONPATH=python/manta_codegen/src \\
        python -m manta_codegen.cli examples/ex8_swarm/craft.py \\
            --workflow library
"""

from manta_codegen import Craft, World
from manta_codegen.parts import PointMass, Thruster


def make_world() -> World:
    c = Craft("ex8")  # no fields — drag-free vacuum
    c.add(PointMass("body", mass=1.0))
    c.add(Thruster("forward",
                   max_thrust=2.0,
                   direction=(1.0, 0.0, 0.0)))

    # Library workflow: user main does multi-instance pub/sub via tethers.
    return World().add_craft(c)

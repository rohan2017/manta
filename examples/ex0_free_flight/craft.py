"""ex0 — trivial 6-thruster craft in zero gravity.

Codegen the equivalent of the hand-written `ex0.cpp` from this descriptor.
Run from the repo root:

    PYTHONPATH=python/manta_codegen/src \
        python -m manta_codegen.cli examples/ex0_free_flight/craft.py --workflow binary

Output lands in `examples/ex0_free_flight/generated/ex0/`.
"""

from manta_codegen import Craft
from manta_codegen.parts import PointMass, Thruster


# Six unit-vector directions: ±x, ±y, ±z. Iterate to keep the Python tidy.
THRUSTER_DIRS: list[tuple[str, tuple[float, float, float]]] = [
    ("tx_p", ( 1.0,  0.0,  0.0)),
    ("tx_n", (-1.0,  0.0,  0.0)),
    ("ty_p", ( 0.0,  1.0,  0.0)),
    ("ty_n", ( 0.0, -1.0,  0.0)),
    ("tz_p", ( 0.0,  0.0,  1.0)),
    ("tz_n", ( 0.0,  0.0, -1.0)),
]


def make_craft() -> Craft:
    c = Craft("ex0")  # no fields registered — zero-gravity
    c.root.add(PointMass("body", mass=1.0))
    for name, direction in THRUSTER_DIRS:
        c.root.add(Thruster(name, max_thrust=5.0, direction=direction))
    return c

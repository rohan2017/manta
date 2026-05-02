"""ex0 — trivial 6-thruster craft in zero gravity.

Codegen the equivalent of the hand-written `ex0.cpp` from this descriptor.
Run from the repo root:

    PYTHONPATH=python/manta_codegen/src \
        python -m manta_codegen.cli examples/ex0_free_flight/craft.py --workflow binary

Output lands in `examples/ex0_free_flight/generated/ex0/`.
"""

from manta_codegen import Craft, World
from manta_codegen.parts import Mass, Thruster


# Six unit-vector directions: ±x, ±y, ±z. Iterate to keep the Python tidy.
THRUSTER_DIRS: list[tuple[str, tuple[float, float, float]]] = [
    ("tx_p", ( 1.0,  0.0,  0.0)),
    ("tx_n", (-1.0,  0.0,  0.0)),
    ("ty_p", ( 0.0,  1.0,  0.0)),
    ("ty_n", ( 0.0, -1.0,  0.0)),
    ("tz_p", ( 0.0,  0.0,  1.0)),
    ("tz_n", ( 0.0,  0.0, -1.0)),
]


def make_world() -> World:
    body      = Mass("body", mass=1.0)
    thrusters = [Thruster(name, max_thrust=5.0, direction=d) for name, d in THRUSTER_DIRS]

    c = Craft("ex0")
    c.add(body)
    for t in thrusters:
        c.add(t)

    state = {
        "t": c.time,
        "p": c.position, "q": c.orientation,
        "v": c.vel_linear, "w": c.vel_angular,
    }
    for t in thrusters:
        state[t.name] = t.throttle
    c.publish(state, "manta/ex0/state")
    for t in thrusters:
        c.subscribe(t.set_throttle, f"manta/ex0/{t.name}/cmd")

    # Zero-gravity free-flight: no fields, no planet. Default initial state.
    return (World()
            .add_craft(c)
            .run(dt=0.001, sim_rate_mult=1.0))

"""ex1 — same 6-thruster craft as ex0, in 1 km circular orbit around Earth.

Inverse-square central gravity, no atmosphere, no rotation.
Sim runs at 200x realtime so a full orbit completes in ~28 wall-seconds.

Generate from the repo root:

    PYTHONPATH=python/manta_codegen/src \
        python -m manta_codegen.cli examples/ex1_orbit/craft.py --workflow binary
"""

import math

from manta_codegen import Craft, World
from manta_codegen.parts  import PointGravityPart, PointMass, Thruster
from manta_codegen.fields import PointGravityField


# Earth.
MU              = 3.986004418e14   # m^3/s^2
EARTH_RADIUS    = 6.371e6           # m
ALTITUDE        = 1.0e3             # 1 km
ORBIT_R         = EARTH_RADIUS + ALTITUDE
V_CIRC          = math.sqrt(MU / ORBIT_R)


THRUSTER_DIRS: list[tuple[str, tuple[float, float, float]]] = [
    ("tx_p", ( 1.0,  0.0,  0.0)),
    ("tx_n", (-1.0,  0.0,  0.0)),
    ("ty_p", ( 0.0,  1.0,  0.0)),
    ("ty_n", ( 0.0, -1.0,  0.0)),
    ("tz_p", ( 0.0,  0.0,  1.0)),
    ("tz_n", ( 0.0,  0.0, -1.0)),
]


def make_world() -> World:
    body      = PointMass("body", mass=1.0)
    thrusters = [Thruster(name, max_thrust=5.0, direction=d) for name, d in THRUSTER_DIRS]
    grav      = PointGravityPart("grav")

    c = Craft("ex1")
    c.add(body)
    for t in thrusters:
        c.add(t)
    c.add(grav)

    # Bundled state topic: craft pose + per-thruster throttles. Wire format
    # matches what the viewer expects (top-level "p","q","v","w" arrays).
    state = {
        "p": c.position, "q": c.orientation,
        "v": c.vel_linear, "w": c.vel_angular,
    }
    for t in thrusters:
        state[t.name] = t.throttle
    c.publish(state, "manta/ex1/state")
    for t in thrusters:
        c.subscribe(t.set_throttle, f"manta/ex1/{t.name}/cmd")

    # Sim setup: gravity field on the world, craft at (R, 0, 0) with
    # tangential velocity for a circular orbit. 200 sim-s per wall-s.
    return (World()
            .add_field(PointGravityField(mu=MU))
            .add_craft(c, pos=(ORBIT_R, 0.0, 0.0), vel=(0.0, V_CIRC, 0.0))
            .run(dt=0.005, sim_rate_mult=200.0))

"""ex1 — same 6-thruster craft as ex0, in 1 km circular orbit around Earth.

Inverse-square central gravity, no atmosphere, no rotation.
Sim runs at 200x realtime so a full orbit completes in ~28 wall-seconds.

Generate from the repo root:

    PYTHONPATH=python/manta_codegen/src \
        python -m manta_codegen.cli examples/ex1_orbit/craft.py --workflow binary
"""

import math

from manta_codegen import Craft
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


def make_craft() -> Craft:
    c = Craft("ex1", fields=[PointGravityField(mu=MU)])
    c.root.add(PointMass("body", mass=1.0))
    for name, direction in THRUSTER_DIRS:
        c.root.add(Thruster(name, max_thrust=5.0, direction=direction))
    c.root.add(PointGravityPart("grav"))

    # 200 sim-s per wall-s, 5 ms tick.
    c.sim_config(dt=0.005, sim_rate_mult=200.0)

    # Initial state: at (R, 0, 0), tangential velocity (0, V_circ, 0) → CCW orbit.
    c.initial_state(
        position   = (ORBIT_R, 0.0, 0.0),
        vel_linear = (0.0, V_CIRC, 0.0),
    )
    return c

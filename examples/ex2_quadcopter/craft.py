"""ex2 — quadcopter X-config with 4 PropThrusters, GravityPart, IMU.

Library workflow: codegen emits the Craft type and telemetry; the user's
main.cpp does the rate PIDs, X-config mixing, and Zenoh wiring.

Generate from the repo root:

    PYTHONPATH=python/manta_codegen/src \
        python -m manta_codegen.cli examples/ex2_quadcopter/craft.py --workflow library
"""

from manta_codegen import Craft, World, tf
from manta_codegen.parts  import GravityPart, IMU, PointMass, PropThruster
from manta_codegen.fields import GravityField


MASS  = 1.0
ARM_L = 0.25     # half-diagonal in m
KT    = 0.02     # counter-torque coefficient
MAX_THRUST_PER_PROP = 2 * (MASS * 9.81) / 4  # 2× hover thrust per prop


def make_world() -> World:
    c = Craft("ex2")

    # Pencil-shaped body inertia: lump everything into the central PointMass.
    c.add(PointMass("body", mass=MASS, moi=(0.01, 0.01, 0.02)))

    # X-config: looking down +z,
    #   front-right (+x,-y) : CCW
    #   front-left  (+x,+y) : CW
    #   back-left   (-x,+y) : CCW
    #   back-right  (-x,-y) : CW
    layout = [
        ("fr", +ARM_L, -ARM_L, False),
        ("fl", +ARM_L, +ARM_L, True),
        ("bl", -ARM_L, +ARM_L, False),
        ("br", -ARM_L, -ARM_L, True),
    ]
    for name, x, y, cw in layout:
        c.add(PropThruster(
            name,
            max_thrust=MAX_THRUST_PER_PROP,
            kt=KT,
            cw=cw,
            transform=tf((x, y, 0)),
        ))

    c.add(GravityPart("grav"))
    c.add(IMU("imu"))

    # Library workflow: user main does pub/sub manually. World still owns the
    # field registration so codegen can emit any required headers/glue.
    return World().add_field(GravityField()).add_craft(c)

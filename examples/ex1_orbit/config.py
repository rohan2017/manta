"""ex1 — 6-thruster manoeuvring craft in a 1 km circular orbit.

Two bipolar thrusters per body axis, each offset from the centerline so
differential thrust produces a torque. Equal throttles on a pair give
pure translation along that axis; opposite throttles give pure rotation
about the perpendicular axis they lever against.

Axis layout (body frame, X forward / Y left / Z up):

    Pair        thrust axis    offset axis    differential ⇒ rotation
    ----        -----------    -----------    -----------------------
    tx_zp/tx_zn  ±X             ±Z             about Y  (pitch)
    ty_xp/ty_xn  ±Y             ±X             about Z  (yaw)
    tz_yp/tz_yn  ±Z             ±Y             about X  (roll)

Inverse-square central gravity, no atmosphere, no rotation.
Sim runs at 200x realtime so a full orbit completes in ~28 wall-seconds.

Generate from the repo root:

    PYTHONPATH=python/manta_codegen/src \
        python -m manta_codegen.cli examples/ex1_orbit/config.py --workflow binary

A keyboard controller for live manoeuvring is in `controller.py`:
WASD = XY translation, QE = Z translation, arrow keys = pitch/yaw, ZX = roll.
"""

import math

from manta_codegen import (Craft, MantaConfig, Target, World, publish,
                           subscribe, tf)
from manta_codegen.parts  import Mass, Thruster
from manta_codegen.fields import GravityField


# Earth.
MU              = 3.986004418e14    # m^3/s^2
EARTH_RADIUS    = 6.371e6           # m
ALTITUDE        = 1.0e3             # 1 km
ORBIT_R         = EARTH_RADIUS + ALTITUDE
V_CIRC          = math.sqrt(MU / ORBIT_R)

# Thruster geometry. Lever arm L sets the moment arm for the differential
# thrust rotation modes; max_thrust scales force and torque proportionally.
LEVER_ARM       = 0.5               # m, offset of each thruster from the body center
MAX_THRUST      = 5.0               # N per thruster

# Body inertia (kg·m²). Picked so a max differential moment of
# 2·L·max_thrust = 5 N·m gives ~25 rad/s² peak angular accel — fast
# enough to be responsive in the demo but not twitchy.
BODY_MOI        = (0.2, 0.2, 0.2)


THRUSTERS: list[tuple[str, tuple[float, float, float], tuple[float, float, float]]] = [
    # name,    direction,         body-frame offset
    ("tx_zp",  ( 1.0, 0.0, 0.0),  (0.0, 0.0,  LEVER_ARM)),   # +X thrust, +Z offset
    ("tx_zn",  ( 1.0, 0.0, 0.0),  (0.0, 0.0, -LEVER_ARM)),   # +X thrust, -Z offset
    ("ty_xp",  ( 0.0, 1.0, 0.0),  ( LEVER_ARM, 0.0, 0.0)),   # +Y thrust, +X offset
    ("ty_xn",  ( 0.0, 1.0, 0.0),  (-LEVER_ARM, 0.0, 0.0)),   # +Y thrust, -X offset
    ("tz_yp",  ( 0.0, 0.0, 1.0),  (0.0,  LEVER_ARM, 0.0)),   # +Z thrust, +Y offset
    ("tz_yn",  ( 0.0, 0.0, 1.0),  (0.0, -LEVER_ARM, 0.0)),   # +Z thrust, -Y offset
]


def make_config() -> MantaConfig:
    body      = Mass("body", mass=1.0, moi=BODY_MOI)
    thrusters = [
        Thruster(name, max_thrust=MAX_THRUST, direction=d, transform=tf(pos))
        for name, d, pos in THRUSTERS
    ]

    c = Craft("ex1")
    c.add(body)
    for t in thrusters:
        c.add(t)

    # Sim setup: gravity field on the world, craft at (R, 0, 0) with
    # tangential velocity for a circular orbit. 200 sim-s per wall-s.
    # `synchronized=True` opts the gravity field into Zenoh disturbance
    # replication — a second binary attached to the same fabric will
    # see the same point-mass disturbance even though it didn't add it.
    grav = GravityField().add_point_mass(mu=MU)
    grav.synchronized = True

    w = (World()
            .add_field(grav)
            .add_craft(c, pos=(ORBIT_R, 0.0, 0.0), vel=(0.0, V_CIRC, 0.0)))

    # Bundled state topic: craft pose + per-thruster throttles. Wire format
    # matches what the viewer expects (top-level "p","q","v","w" arrays).
    state = {
        "t": c.time,
        "p": c.position, "q": c.orientation,
        "v": c.vel_linear, "w": c.vel_angular,
    }
    for t in thrusters:
        state[t.name] = t.throttle
    publish(state, "manta/ex1/state")
    for t in thrusters:
        subscribe(t.set_throttle, f"manta/ex1/{t.name}/cmd")

    return MantaConfig(targets=[
        Target("ex1", drives=[w], dt=0.005, sim_rate_mult=200.0),
    ])

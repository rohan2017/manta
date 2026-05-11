"""ex1 — 6-thruster lunar-orbit manoeuvring craft (Apollo-LM scale).

A 15 t lander with two bipolar thrusters per body axis, each offset 3 m
from the centerline so differential thrust produces a torque. Equal
throttles on a pair give pure translation along that axis; opposite
throttles give pure rotation about the perpendicular axis they lever
against.

Axis layout (body frame, X forward / Y left / Z up):

    Pair        thrust axis    offset axis    differential ⇒ rotation
    ----        -----------    -----------    -----------------------
    tx_zp/tx_zn  ±X             ±Z             about Y  (pitch)
    ty_xp/ty_xn  ±Y             ±X             about Z  (yaw)
    tz_yp/tz_yn  ±Z             ±Y             about X  (roll)

The craft starts in a circular orbit ~100 m above the Moon's mean
surface. Inverse-square lunar gravity, no atmosphere, no rotation. Sim
runs at 200x realtime so a full orbit completes in ~17 wall-seconds.

Generate from the repo root:

    .venv/bin/python -m manta_codegen.cli examples/ex1_orbit/config.py --workflow binary

A keyboard controller for live manoeuvring is in `controller.py`:
WASD = XY translation, QE = Z translation, arrow keys = pitch/yaw, ZX = roll.
"""

import math

from manta_codegen import (Craft, MantaConfig, Target, World, publish,
                           subscribe, tf)
from manta_codegen.parts  import Mass, Thruster
from manta_codegen.fields import GravityField


# Moon.
MU              = 4.9048695e12      # m^3/s^2  (lunar gravitational parameter)
MOON_RADIUS     = 1.7374e6          # m
ALTITUDE        = 100.0             # m — ~100 m above the surface
ORBIT_R         = MOON_RADIUS + ALTITUDE
V_CIRC          = math.sqrt(MU / ORBIT_R)   # ≈ 1680 m/s

# Craft geometry / mass budget. The 15 t mass is Apollo-LM scale.
# LEVER_ARM = 3 m is large enough that the thrusters bracket the body
# visibly in the viewer, even with the Apollo_LM.gltf model at meter
# scale; max_thrust is sized so 6 thrusters at full throttle can hold
# the craft against lunar gravity with margin.
LEVER_ARM       = 3.0               # m
MAX_THRUST      = 5000.0            # N per thruster (Moon hover: 15000·1.62 ≈ 24.3 kN)
CRAFT_MASS      = 15000.0           # kg

# Body inertia (kg·m²). Order-of-magnitude estimate for a 15 t craft of
# ~3 m extent: I ≈ (1/6)·m·L² ≈ 22500 kg·m². At max differential moment
# 2·L·max_thrust = 30 kN·m, peak angular accel ≈ 1.3 rad/s² — responsive
# but not twitchy at human reaction time.
BODY_MOI        = (22500.0, 22500.0, 22500.0)


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
    body      = Mass("body", mass=CRAFT_MASS, moi=BODY_MOI)
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

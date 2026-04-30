"""ex3 — TVC rocket "hopper" with a single gimbaled engine 1m below CoM.

Library workflow: codegen emits the Craft type and telemetry; the user's
main.cpp does the rate PIDs, gimbal mixing, and Zenoh wiring.

Generate from the repo root:

    PYTHONPATH=python/manta_codegen/src \
        python -m manta_codegen.cli examples/ex3_tvc_rocket/craft.py --workflow library
"""

from manta_codegen import Craft, tf
from manta_codegen.parts  import GimbaledThruster, GravityPart, IMU, PointMass
from manta_codegen.fields import GravityField


MASS  = 5.0
G     = 9.81
HOVER_THRUST = MASS * G            # ~49 N
MAX_THRUST   = 1.5 * HOVER_THRUST  # 50% headroom
ENGINE_Z     = -1.0                # 1 m below CoM
MAX_GIMBAL   = 0.15                # ~8.6 deg


def make_craft() -> Craft:
    c = Craft("ex3", fields=[GravityField()])

    # Pencil-shaped rocket inertia: Ix=Iy >> Iz.
    c.root.add(PointMass("body", mass=MASS, moi=(0.8, 0.8, 0.05)))
    c.root.add(GimbaledThruster(
        "engine",
        max_thrust=MAX_THRUST,
        max_angle=MAX_GIMBAL,
        transform=tf((0.0, 0.0, ENGINE_Z)),
    ))
    c.root.add(GravityPart("grav"))
    c.root.add(IMU("imu"))
    return c

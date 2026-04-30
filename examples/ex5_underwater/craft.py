"""ex5 — neutrally-buoyant submarine.

A cylindrical-volume underwater vehicle modeled as a Hull with sample
points distributed along its body, plus an IMU, a DVL, and a single aft
Thruster. Floats at neutral buoyancy in OceanAtmosField below the sea
surface, with GravityField pulling it down. The Hull's runtime
dynamic_cast detects OceanAtmosField and blends water/air density across
the surface — visible if the sub crosses sea_level.

Codegen:

    PYTHONPATH=python/manta_codegen/src \\
        python -m manta_codegen.cli examples/ex5_underwater/craft.py \\
            --workflow binary

Workflow: binary. Topics:
  manta/ex5/state            — pose + IMU + DVL telemetry (publish-only)
  manta/ex5/aft_thrust/cmd   — float[1] throttle [0,1]
"""

from manta_codegen import Craft, tf
from manta_codegen.parts  import DVL, GravityPart, Hull, IMU, PointMass, Thruster
from manta_codegen.fields import GravityField, OceanAtmosField


# --- Vehicle geometry ---
# Cylinder: 1 m long, 0.15 m radius. Volume ≈ π·r²·L ≈ 0.0707 m³.
# At neutral buoyancy in fresh water (ρ=1000), mass = ρ·V ≈ 70.7 kg.
LENGTH = 1.0
RADIUS = 0.15
import math
VOLUME = math.pi * RADIUS * RADIUS * LENGTH      # ≈ 0.0707 m³
MASS   = 1000.0 * VOLUME                          # neutrally buoyant in water
MAX_THRUST = 50.0                                 # N — modest aft thruster

# Sample points distributed along the cylinder axis (x). 8 samples covers
# the body well enough that crossing the surface gives a smooth force ramp.
N_SAMPLES = 8
SAMPLE_POINTS = [
    (-LENGTH/2 + (i + 0.5) * LENGTH / N_SAMPLES, 0.0, 0.0)
    for i in range(N_SAMPLES)
]


def make_craft() -> Craft:
    c = Craft("ex5", fields=[
        GravityField(),                    # default g = (0,0,-9.81)
        OceanAtmosField(sea_level=0.0),    # surface at z=0
    ])

    # Hull: provides buoyancy + holds the bulk mass/MOI of the sub.
    # MOI of a uniform cylinder: I_xx = (1/2)·m·r², I_yy = I_zz = (1/12)·m·(3r²+L²).
    Ixx = 0.5 * MASS * RADIUS * RADIUS
    Iyy = (1/12.0) * MASS * (3 * RADIUS * RADIUS + LENGTH * LENGTH)
    Izz = Iyy

    hull = c.root.add(Hull("hull",
                           volume=VOLUME,
                           sample_points=SAMPLE_POINTS,
                           publish_state=False))
    # Hull doesn't carry mass on its own (it's a buoyancy-only model). We
    # bolt the mass onto the hull part itself by post-construction.

    # Sensors and actuator. The Hull is mass-less (buoyancy-only model);
    # lump the body mass and inertia into a dedicated PointMass.
    c.root.add(PointMass("body", mass=MASS, moi=(Ixx, Iyy, Izz)))

    # Gravity: Hull only reads g for the buoyancy direction; we need a
    # separate GravityPart to apply mg to the body.
    c.root.add(GravityPart("gravity"))

    c.root.add(IMU("imu", accel_sigma=0.05, gyro_sigma=0.005))
    c.root.add(DVL("dvl", velocity_sigma=0.02))

    # Aft thruster on the −x face, pointing +x (along forward axis).
    c.root.add(Thruster("aft_thrust",
                        max_thrust=MAX_THRUST,
                        direction=(1.0, 0.0, 0.0),
                        transform=tf((-LENGTH/2, 0.0, 0.0))))

    # Initial state: 0.5 m below sea surface, at rest.
    c.initial_state(position=(0.0, 0.0, -0.5))
    return c

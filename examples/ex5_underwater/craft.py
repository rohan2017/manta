"""ex5 — neutrally-buoyant submarine.

A cylindrical-volume underwater vehicle modeled as a Hull with sample
points distributed along its body, plus an IMU, a DVL, and a single aft
Thruster. Floats at neutral buoyancy below the sea surface; gravity
pulls it down. Earth provides the ocean+atmosphere fluid disturbance and
the SeaSurface that Hull's smoothstep blend uses to ramp buoyancy across
the air-water interface.

Codegen:

    PYTHONPATH=python/manta_codegen/src \\
        python -m manta_codegen.cli examples/ex5_underwater/craft.py \\
            --workflow binary

Workflow: binary. Topics:
  manta/ex5/state            — pose + IMU + DVL telemetry (publish-only)
  manta/ex5/aft_thrust/cmd   — float[1] throttle [0,1]
"""

from manta_codegen import Craft, World, tf
from manta_codegen.parts   import DVL, IMU, Mass, PointBuoy, Thruster
from manta_codegen.fields  import GravityField
from manta_codegen.planets import Earth


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


def make_world() -> World:
    # Earth registers a FluidField (water below sea level + atmosphere above).
    # The user adds a separate GravityField on the world so Mass auto-applies
    # weight; buoyancy is composed from N PointBuoy parts distributed along
    # the cylinder axis (replaces the old `Hull` part — same physical idea
    # built from atomic primitives).
    c = Craft("ex5")

    # Body inertia of a uniform cylinder.
    Ixx = 0.5 * MASS * RADIUS * RADIUS
    Iyy = (1/12.0) * MASS * (3 * RADIUS * RADIUS + LENGTH * LENGTH)
    Izz = Iyy

    body       = Mass("body", mass=MASS, moi=(Ixx, Iyy, Izz))
    imu        = IMU("imu", accel_sigma=0.05, gyro_sigma=0.005)
    dvl        = DVL("dvl", velocity_sigma=0.02)
    aft_thrust = Thruster("aft_thrust",
                          max_thrust=MAX_THRUST,
                          direction=(1.0, 0.0, 0.0),
                          transform=tf((-LENGTH/2, 0.0, 0.0)))

    c.add(body)
    c.add(imu)
    c.add(dvl)
    c.add(aft_thrust)

    # Distribute volume across N PointBuoys to mimic the old Hull's
    # sample-points buoyancy. Each carries V/N volume; placed at the same
    # offsets so righting moments behave the same.
    V_per = VOLUME / N_SAMPLES
    for i, pt in enumerate(SAMPLE_POINTS):
        c.add(PointBuoy(f"buoy_{i}", volume=V_per, transform=tf(pt)))

    # Bundled state: pose + sensor signals + thrust. Smoke test only reads
    # `state["p"]` and `state["v"]` so the legacy wire shape is preserved
    # for those fields; sensor data uses flat per-signal keys.
    c.publish({
        "t": c.time,
        "p": c.position,         "q": c.orientation,
        "v": c.vel_linear,       "w": c.vel_angular,
        "imu_accel": imu.last_accel,
        "imu_gyro":  imu.last_gyro,
        "dvl_vel":   dvl.last_velocity,
        "throttle":  aft_thrust.throttle,
    }, "manta/ex5/state")
    c.subscribe(aft_thrust.set_throttle, "manta/ex5/aft_thrust/cmd")

    # World setup: gravity field + Earth planet (registers OceanAtmosField
    # and FlatSeaSurface automatically). Craft starts 0.5 m below the sea
    # surface, at rest.
    earth = Earth(sea_level=0.0)
    return (World()
            .add_field(GravityField())
            .add_planet(earth)
            .add_craft(c, on=earth, pos=(0.0, 0.0, -0.5))
            .run(dt=0.001, sim_rate_mult=1.0))

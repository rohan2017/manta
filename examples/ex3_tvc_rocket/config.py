"""ex3 — TVC rocket "hopper" with a 2-axis gimballed engine 1m below CoM.

The deleted `GimbaledThruster` is decomposed into a yaw motor (axis x)
hosting a pitch motor (axis y) hosting a `Thruster1` engine. The
controller in `main.cpp` runs a stiff position PD on each motor to
emulate the old `set_gimbal(pitch, yaw)` semantics.

Library workflow: codegen emits the Craft type and telemetry; the user's
main.cpp does the rate PIDs, gimbal mapping, and Zenoh wiring.

Generate from the repo root:

    PYTHONPATH=python/manta_codegen/src \
        python -m manta_codegen.cli examples/ex3_tvc_rocket/config.py --workflow library
"""

from manta_codegen import Craft, MantaConfig, Target, World, tf
from manta_codegen.parts  import IMU, Mass, Motor, Thruster1
from manta_codegen.fields import GravityField


MASS         = 5.0
G            = 9.81
HOVER_THRUST = MASS * G            # ~49 N
MAX_THRUST   = 1.5 * HOVER_THRUST  # 50% headroom
ENGINE_Z     = -1.0                # 1 m below CoM
GIMBAL_STALL = 100.0               # N·m — stiff enough to track angle commands


def make_config() -> MantaConfig:
    c = Craft("ex3")

    # Pencil-shaped rocket inertia: Ix=Iy >> Iz.
    c.add(Mass("body", mass=MASS, moi=(0.8, 0.8, 0.05)))

    # Two-axis gimbal stack at the engine mount, 1 m below the CoM. Yaw
    # outermost (rotates everything below it including the pitch axis),
    # pitch inside, then the engine itself.
    yaw_motor = Motor("yaw_motor",
                      axis=(1.0, 0.0, 0.0),
                      stall_torque=GIMBAL_STALL,
                      damping=0.0,
                      transform=tf((0.0, 0.0, ENGINE_Z)))
    pitch_motor = Motor("pitch_motor",
                        axis=(0.0, 1.0, 0.0),
                        stall_torque=GIMBAL_STALL,
                        damping=0.0)
    engine = Thruster1("engine",
                       force_coefs=[(0.0, 0.0, MAX_THRUST)],
                       torque_coefs=[(0.0, 0.0, 0.0)])

    c.add(yaw_motor)
    yaw_motor.add(pitch_motor)
    pitch_motor.add(engine)

    c.add(IMU("imu"))

    w = World().add_field(GravityField(g=(0.0, 0.0, -G))).add_craft(c)
    return MantaConfig(targets=[Target("ex3", drives=[w])])

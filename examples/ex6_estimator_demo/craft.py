"""ex6 — Phase-4 EKF demo: sim + estimator side-by-side, in one process.

Same craft definition, two consumers in the same binary:
  1. The sim runs the dynamics, drives noisy IMU + DVL outputs.
  2. An EKF (in the user main) consumes IMU as process input + DVL as
     measurement and produces an estimated 6-DOF pose.

Both truth and estimate are published to Zenoh on separate topics for
overlay in rerun.

Workflow: library — codegen emits Ex6Craft + telemetry; the user main
in ex6.cpp wires the EKF and Zenoh I/O.

Codegen:

    PYTHONPATH=python/manta_codegen/src \\
        python -m manta_codegen.cli examples/ex6_estimator_demo/craft.py \\
            --workflow library
"""

from manta_codegen import Craft, World
from manta_codegen.parts  import DVL, IMU, Mass, Thruster
from manta_codegen.fields import GravityField


def make_world() -> World:
    c = Craft("ex6")

    # Body inertia: lumped point with small diagonal MOI. Mass auto-applies
    # gravity from the registered GravityField each tick.
    c.add(Mass("body", mass=1.0, moi=(0.05, 0.05, 0.05)))

    # Sensors. Modest noise — enough to be realistic, not so much that the
    # EKF can't lock on within a few seconds.
    c.add(IMU("imu", accel_sigma=0.05, gyro_sigma=0.005))
    c.add(DVL("dvl", velocity_sigma=0.02))

    # A diagonal aft thruster — pulse to drive 3-axis motion. Commanded by
    # the user main via Zenoh ('manta/ex6/thrust/cmd').
    c.add(Thruster("thrust", max_thrust=15.0, direction=(1.0, 1.0, 0.5)))

    # Library workflow: user main does Zenoh I/O + EKF wiring.
    return World().add_field(GravityField()).add_craft(c)

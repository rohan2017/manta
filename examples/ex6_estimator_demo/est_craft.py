"""ex6 — estimator-side craft (templated).

Same parts as the sim craft (`craft.py`) but with `scalar_templated=True`,
so codegen emits `Ex6EstCraftT<Scalar>`. The estimator instantiates this
with `double` (value step) and `ceres::Jet<double, 13>` (Jacobian step)
inside `CraftEKF`.

This is Pattern A from estimation_workflow.md: same model on both sides,
matching part names → automatic data-flow glue. The user pipes sim
outputs into the est instance's `set_measurement(...)` hooks each tick,
and `CraftEKF` does the rest.

Codegen:

    PYTHONPATH=python/manta_codegen/src \\
        python -m manta_codegen.cli examples/ex6_estimator_demo/est_craft.py \\
            --workflow library
"""

from manta_codegen import Craft, World
from manta_codegen.parts  import DVL, IMU, Mass, Thruster
from manta_codegen.fields import GravityField


def make_world() -> World:
    c = Craft("ex6_est")
    # Opt in to templated codegen — required for CraftEKF wrapping.
    c.scalar_templated = True

    # Mass auto-applies gravity from the registered GravityField.
    c.add(Mass("body", mass=1.0, moi=(0.05, 0.05, 0.05)))
    c.add(IMU("imu", accel_sigma=0.05, gyro_sigma=0.005))
    c.add(DVL("dvl", velocity_sigma=0.02))
    # The estimator-side thruster has no command path — the EKF doesn't pilot.
    # We still keep the part so inertia and geometric offsets match the sim
    # model.
    c.add(Thruster("thrust", max_thrust=15.0, direction=(1.0, 1.0, 0.5)))

    return World().add_field(GravityField()).add_craft(c)

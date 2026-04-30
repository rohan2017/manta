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

from manta_codegen import Craft
from manta_codegen.parts  import DVL, GravityPart, IMU, PointMass, Thruster
from manta_codegen.fields import GravityField


def make_craft() -> Craft:
    c = Craft("ex6_est", fields=[GravityField()])
    # Opt in to templated codegen — required for CraftEKF wrapping.
    c.scalar_templated = True

    c.root.add(PointMass("body", mass=1.0, moi=(0.05, 0.05, 0.05)))
    c.root.add(GravityPart("gravity"))
    c.root.add(IMU("imu", accel_sigma=0.05, gyro_sigma=0.005))
    c.root.add(DVL("dvl", velocity_sigma=0.02))
    # No subscribe_command on the est-side thruster — the EKF doesn't pilot.
    # We still keep the part so the inertia and any geometric offsets match
    # the sim model.
    c.root.add(Thruster("thrust",
                        max_thrust=15.0,
                        direction=(1.0, 1.0, 0.5),
                        subscribe_command=False))

    return c

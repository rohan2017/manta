"""ex7 — real-data-only estimator craft (Pattern C).

There is no sim Craft in this example. The estimator runs against external
sensor data fed via Zenoh (typical robot deployment). The codegen emits the
Craft as a class template (`scalar_templated=True`) so it plugs into
`manta::estimation::CraftEKF<Ex7EstCraftT, 3>`.

Topology:
    fake_sensors.py        →  publishes  manta/ex7/imu, manta/ex7/dvl
    ex7_real_data_estimator → subscribes, calls set_measurement, runs EKF
                              publishes  manta/ex7/estimate

Codegen:
    PYTHONPATH=python/manta_codegen/src \\
        python -m manta_codegen.cli examples/ex7_real_data_estimator/est_craft.py \\
            --workflow library
"""

from manta_codegen import Craft, World
from manta_codegen.parts import DVL, IMU, Mass


def make_world() -> World:
    c = Craft("ex7_est")
    c.scalar_templated = True

    # Body inertia for the 13-DOF rigid-body state. No GravityPart here:
    # the IMU's reading already accounts for the inertial acceleration
    # (the EKF's process model integrates it directly), so the estimator
    # doesn't separately apply gravity.
    c.add(Mass("body", mass=1.0, moi=(0.05, 0.05, 0.05)))

    c.add(IMU("imu", accel_sigma=0.05, gyro_sigma=0.005))
    c.add(DVL("dvl", velocity_sigma=0.02))

    return World().add_craft(c)

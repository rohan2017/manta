"""ex6 — real-data-only estimator (Pattern C from estimation_workflow.md).

No sim Craft in this binary. The estimator runs against external sensor
data fed via Zenoh — typical robot deployment shape:

    fake_sensors.py        → publishes  manta/ex6/imu, manta/ex6/dvl
    ex6_real_data_estimator → subscribes, runs the codegen-emitted EKF,
                              publishes  manta/ex6/estimate

The Craft is `scalar_templated=True` so it plugs into
`manta::estimation::CraftEKF<Ex6EstCraftT, MeasDim>`. Only the DVL is
treated as a measurement update — the IMU's readings flow into
`set_measurement` for the predict step but are not currently used as an
EKF update (Phase-A codegen scope).

Codegen:
    PYTHONPATH=python/manta_codegen/src \\
        python -m manta_codegen.cli examples/ex6_real_data_estimator/est_craft.py \\
            --workflow binary
"""

from manta_codegen import (Craft, EKF, MantaConfig, Target, World,
                           publish, subscribe)
from manta_codegen.parts import DVL, IMU, Mass


def make_config() -> MantaConfig:
    c = Craft("ex6_est")
    c.scalar_templated = True

    c.add(Mass("body", mass=1.0, moi=(0.05, 0.05, 0.05)))
    imu = IMU("imu", accel_sigma=0.05, gyro_sigma=0.005)
    dvl = DVL("dvl", velocity_sigma=0.02)
    c.add(imu)
    c.add(dvl)

    w = World("ex6_est").add_craft(c)

    ekf = EKF(w, measurements=[dvl], process_noise=1e-6,
              initial_covariance=1.0)

    # Real-time sensor feed. IMU is fed in but does not currently drive an
    # update (Phase-A: only DVL h(x) is implemented). DVL `consume_fresh()`
    # gates the update step inside the generated tick loop.
    subscribe(imu.set_measurement, "manta/ex6/imu")
    subscribe(dvl.set_measurement, "manta/ex6/dvl")

    publish({
        "p": ekf.position,
        "v": ekf.vel_linear,
        "p_stddev": ekf.position_stddev,
        "v_stddev": ekf.vel_linear_stddev,
    }, "manta/ex6/estimate")

    return MantaConfig(targets=[
        Target("ex6_est", drives=[ekf], dt=0.001, sim_rate_mult=1.0),
    ])

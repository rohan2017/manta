"""ex6 — real-data-only estimator (Pattern C from estimation_workflow.md).

No sim Craft in this binary. The estimator runs against external sensor
data fed via Zenoh — typical robot deployment shape:

    fake_sensors.py        → publishes  manta/ex6/imu, manta/ex6/dvl
    ex6_real_data_estimator → subscribes, runs the codegen-emitted EKF,
                              publishes  manta/ex6/estimate

The Craft plugs into `manta::estimation::WorldEKF`; the EKF descriptor
flips the wrapped craft to scalar-templated automatically so the codegen
emits `Ex6EstCraftT<Scalar>`. Both IMU and DVL drive measurement updates;
each sensor's `consume_fresh()` gates its own update inside the
generated tick loop.

Codegen:
    PYTHONPATH=python/manta_codegen/src \\
        python -m manta_codegen.cli examples/ex6_real_data_estimator/config.py \\
            --workflow binary
"""

from manta_codegen import (Craft, EKF, MantaConfig, Target, World,
                           publish, subscribe)
from manta_codegen.parts import DVL, IMU, Mass


def make_config() -> MantaConfig:
    c = Craft("ex6_est")

    c.add(Mass("body", mass=1.0, moi=(0.05, 0.05, 0.05)))
    imu = IMU("imu", accel_sigma=0.05, gyro_sigma=0.005)
    dvl = DVL("dvl", velocity_sigma=0.02)
    c.add(imu)
    c.add(dvl)

    w = World("ex6_est").add_craft(c)

    ekf = EKF(w, measurements=[imu, dvl], process_noise=1e-6,
              initial_covariance=1.0)

    # Real-time sensor feeds — each set_measurement call also flips the
    # part's freshness bit, so the EKF's per-sensor `consume_fresh()` gate
    # picks up the new reading on the next predict tick.
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

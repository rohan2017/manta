"""ukf_smoke — minimal UKF-codegen smoke target.

Exercises the UKF codegen path end-to-end: Mass + IMU + DVL on a
non-templated craft (UKF doesn't require scalar templating), wrapped in
a `manta::estimation::CraftUKFOf<...>`. Mostly here so CI catches
regressions in the UKF emit path.

Codegen:

    PYTHONPATH=python/manta_codegen/src \\
        python -m manta_codegen.cli examples/ukf_smoke/craft.py \\
            --workflow binary
"""

from manta_codegen import (Craft, MantaConfig, Target, UKF, World,
                           publish, subscribe)
from manta_codegen.parts import DVL, IMU, Mass


def make_config() -> MantaConfig:
    c = Craft("ukf_smoke")
    # No scalar_templated=True — UKF works on plain crafts.
    c.add(Mass("body", mass=1.0, moi=(0.05, 0.05, 0.05)))
    imu = IMU("imu", accel_sigma=0.05, gyro_sigma=0.005)
    dvl = DVL("dvl", velocity_sigma=0.02)
    c.add(imu)
    c.add(dvl)

    w = World("ukf_smoke").add_craft(c)

    ukf = UKF(w, measurements=[imu, dvl],
              process_noise=1e-6, initial_covariance=1.0,
              alpha=1e-3, beta=2.0, kappa=0.0)

    subscribe(imu.set_measurement, "manta/ukf_smoke/imu")
    subscribe(dvl.set_measurement, "manta/ukf_smoke/dvl")
    publish({
        "p": ukf.position,
        "v": ukf.vel_linear,
        "p_stddev": ukf.position_stddev,
        "v_stddev": ukf.vel_linear_stddev,
    }, "manta/ukf_smoke/estimate")

    return MantaConfig(targets=[
        Target("ukf_smoke", drives=[ukf], dt=0.001, sim_rate_mult=1.0),
    ])

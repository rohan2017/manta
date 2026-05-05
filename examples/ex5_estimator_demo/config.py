"""ex5 — sim + EKF side-by-side, in one process (Pattern A).

Two crafts in the same Target's drive list:
  * `Ex5Craft` — sim, runs in a manta::World/Scene with full physics.
  * `Ex5EstCraftT<Scalar>` — templated estimator wrapped by WorldEKF.

Cross-world `connect()` pipes the sim's noisy IMU + DVL outputs into the
est's `set_measurement()` hooks each tick, mirrors the throttle so the
EKF's predict step models the same input the sim is applying, and the
codegen-emitted main runs both in lockstep.

Codegen:

    PYTHONPATH=python/manta_codegen/src \\
        python -m manta_codegen.cli examples/ex5_estimator_demo/config.py \\
            --workflow binary
"""

from manta_codegen import (Craft, EKF, MantaConfig, Target, World,
                           connect, publish, subscribe)
from manta_codegen.parts  import DVL, IMU, Mass, Thruster
from manta_codegen.fields import GravityField


def make_config() -> MantaConfig:
    grav = GravityField()

    # ---- Sim craft ----
    sim_c = Craft("ex5")
    sim_c.add(Mass("body", mass=1.0, moi=(0.05, 0.05, 0.05)))
    sim_imu = IMU("imu", accel_sigma=0.05, gyro_sigma=0.005)
    sim_dvl = DVL("dvl", velocity_sigma=0.02)
    sim_thrust = Thruster("thrust", max_thrust=15.0, direction=(1.0, 0.0, 0.0))
    sim_c.add(sim_imu)
    sim_c.add(sim_dvl)
    sim_c.add(sim_thrust)

    sim_w = World("ex5_sim").add_field(grav).add_craft(sim_c)

    # ---- Est craft (templated) ----
    est_c = Craft("ex5_est")
    est_c.add(Mass("body", mass=1.0, moi=(0.05, 0.05, 0.05)))
    est_imu = IMU("imu", accel_sigma=0.05, gyro_sigma=0.005)
    est_dvl = DVL("dvl", velocity_sigma=0.02)
    est_thrust = Thruster("thrust", max_thrust=15.0, direction=(1.0, 0.0, 0.0))
    est_c.add(est_imu)
    est_c.add(est_dvl)
    est_c.add(est_thrust)

    est_w = World("ex5_est").add_field(grav).add_craft(est_c)

    # ex5's sensor suite (IMU + DVL + gyro) is body-frame only: there's
    # no absolute-attitude reference. Both the sim and the EKF start at
    # q = identity (the default in World.add_craft, which the EKF picks
    # up automatically). The default per-block variances tell the EKF
    # we're moderately confident in that initial — small enough that
    # body-frame measurement noise can't pull q far from identity, but
    # not a hard lock. That's sufficient because nothing in the
    # dynamics generates torque, so q should stay near identity for
    # the whole run.
    ekf = EKF(est_w, measurements=[est_imu, est_dvl],
              process_noise=1e-6, initial_covariance=1.0,
              initial_position_var=1e-4,
              initial_attitude_var=1e-4,
              initial_velocity_var=1e-2,
              initial_angular_velocity_var=1e-4)

    # ---- In-process plumbing ----
    # Cross-world: sim sensor outputs feed est sensor measurement inputs
    # so the EKF's predict + update see the same readings the sim produced.
    connect(sim_imu.last_accel, est_imu.set_measurement_accel)
    connect(sim_imu.last_gyro,  est_imu.set_measurement_gyro)
    connect(sim_dvl.last_velocity, est_dvl.set_measurement)
    # Mirror commanded throttle so predict's force model matches the sim.
    connect(sim_thrust.throttle, est_thrust.set_throttle)

    # ---- Zenoh I/O ----
    subscribe(sim_thrust.set_throttle, "manta/ex5/cmd")
    # Truth pose from the sim craft.
    publish({
        "p": sim_c.position,
        "q": sim_c.orientation,
        "v": sim_c.vel_linear,
        "w": sim_c.vel_angular,
    }, "manta/ex5/state")
    publish({
        "p": ekf.position,
        "v": ekf.vel_linear,
        "p_stddev": ekf.position_stddev,
        "v_stddev": ekf.vel_linear_stddev,
    }, "manta/ex5/estimate")

    return MantaConfig(targets=[
        Target("ex5", drives=[sim_w, ekf], dt=0.001, sim_rate_mult=1.0),
    ])

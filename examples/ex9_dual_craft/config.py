"""ex9 — Dual-craft EKF (Pattern C, real-data inputs).

Validates the multi-craft architecture end-to-end:

  * Two free-flying drones in the SAME World, each with Mass + IMU + DVL.
  * One `WorldEKF<NumCrafts=2, MeasDim=18>` estimates the joint 26-DOF
    state — concat of both crafts' [p (3) | q (4) | v (3) | ω (3)].
  * Per-craft slice accessors (`ekf.position(0)`, `ekf.position(1)`)
    expose each craft's belief without flattening through the wrapper.
  * Per-sensor measurement updates run with craft_idx routing — sensor
    on craft 0 reads state segment [0, 13); sensor on craft 1 reads
    segment [13, 26). State-offset arithmetic is codegen-baked into the
    measurement functor.

No physical inter-craft coupling in this example (drones are
independent), so the cross-craft Jacobian blocks are zero. The
templated-World architecture is what makes that case correct
*and* would automatically pick up coupling (tether, fluid, contact)
without further codegen work — the physics is written once and the
Jet world propagates through it.

Topics:
    manta/ex9/imu/0      ← IMU 0 readings   [ax,ay,az, gx,gy,gz]   (50 Hz)
    manta/ex9/imu/1      ← IMU 1 readings                         (50 Hz)
    manta/ex9/dvl/0      ← DVL 0 reading    [vx,vy,vz]            (10 Hz)
    manta/ex9/dvl/1      ← DVL 1 reading                          (10 Hz)
    manta/ex9/estimate   → joint estimate   {p0, v0, p1, v1, p_stddev, v_stddev}

Codegen:

    PYTHONPATH=python/manta_codegen/src \\
        python -m manta_codegen.cli examples/ex9_dual_craft/config.py \\
            --workflow binary
"""

from manta_codegen import (Craft, EKF, MantaConfig, Target, World,
                           publish, subscribe)
from manta_codegen.parts import DVL, IMU, Mass


def _make_drone(name: str):
    """Return (craft, imu, dvl). Both drones are identical save for name."""
    c = Craft(name)
    c.scalar_templated = True   # filter targets need <double>/<Jet> instantiations
    c.add(Mass("body", mass=1.0, moi=(0.05, 0.05, 0.05)))
    imu = IMU("imu", accel_sigma=0.05, gyro_sigma=0.005, rate_hz=100.0)
    dvl = DVL("dvl", velocity_sigma=0.02,                rate_hz=10.0)
    c.add(imu)
    c.add(dvl)
    return c, imu, dvl


def make_config() -> MantaConfig:
    drone_0, imu_0, dvl_0 = _make_drone("drone_0")
    drone_1, imu_1, dvl_1 = _make_drone("drone_1")

    # Both crafts in ONE world. The EKF estimates state[0..12]=drone_0,
    # state[13..25]=drone_1.
    w = (World("ex9")
         .add_craft(drone_0)
         .add_craft(drone_1))

    ekf = EKF(w, measurements=[imu_0, dvl_0, imu_1, dvl_1],
              process_noise=1e-6, initial_covariance=1.0)

    # Each drone's sensor data arrives on its own Zenoh topic.
    subscribe(imu_0.set_measurement, "manta/ex9/imu/0")
    subscribe(dvl_0.set_measurement, "manta/ex9/dvl/0")
    subscribe(imu_1.set_measurement, "manta/ex9/imu/1")
    subscribe(dvl_1.set_measurement, "manta/ex9/dvl/1")

    # Joint estimate publish. The codegen-emitted EKF wrapper exposes per-
    # craft slices via the `position(idx)`/`vel_linear(idx)` accessors;
    # for the default-bound BoundSignals (which read craft 0 only), we
    # expose a representative subset here. Multi-craft output bindings are
    # a future codegen extension.
    publish({
        "p0":       ekf.position,
        "v0":       ekf.vel_linear,
        "p_stddev": ekf.position_stddev,
        "v_stddev": ekf.vel_linear_stddev,
    }, "manta/ex9/estimate")

    return MantaConfig(targets=[
        Target("ex9", drives=[ekf], dt=0.001, sim_rate_mult=1.0),
    ])

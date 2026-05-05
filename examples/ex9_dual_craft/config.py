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
    # state[13..25]=drone_1. The two drones start at distinct positions
    # so the per-craft init API has something concrete to demonstrate.
    w = (World("ex9")
         .add_craft(drone_0, pos=(0.0, 0.0, 0.0))
         .add_craft(drone_1, pos=(5.0, 0.0, 0.0)))

    # Per-craft init knobs. Each *_var field accepts a scalar (broadcast
    # to all crafts), a list-of-scalars (positional, one per craft), or
    # a dict keyed by craft name (others stay at initial_covariance).
    # State fields are inherited from add_craft() above unless
    # overridden here. Below uses each shape once for documentation:
    #   * initial_position_var: scalar — same trust for both crafts.
    #   * initial_attitude_var: dict by name — only drone_0 gets a tight
    #     attitude prior; drone_1 stays at the isotropic default.
    #   * initial_velocity_var: list — different per craft positionally.
    # block_decomposed=True: the two drones don't physically couple
    # (no tether, no contact, no shared field forces beyond gravity
    # which is read-only), so F is block-diagonal and we can compute
    # each craft's 13×13 F-block in its own Jet pass with only 13
    # partials. For NumCrafts ≥ ~5 this is roughly NumCrafts× faster
    # than the monolithic predict; for two crafts it's ~2× — already
    # a measurable win.
    ekf = EKF(w, measurements=[imu_0, dvl_0, imu_1, dvl_1],
              process_noise=1e-6, initial_covariance=1.0,
              initial_position_var=1e-4,
              initial_attitude_var={"drone_0": 1e-4},
              initial_velocity_var=[1e-2, 1e-3],
              block_decomposed=True)

    # Each drone's sensor data arrives on its own Zenoh topic.
    subscribe(imu_0.set_measurement, "manta/ex9/imu/0")
    subscribe(dvl_0.set_measurement, "manta/ex9/dvl/0")
    subscribe(imu_1.set_measurement, "manta/ex9/imu/1")
    subscribe(dvl_1.set_measurement, "manta/ex9/dvl/1")

    # Per-craft signal tree: `ekf.crafts["<name>"]` exposes
    # position/orientation/velocity/stddev BoundSignals scoped to that
    # craft's slice of the joint state. Each one reads
    # `ekf_0.<accessor>(craft_idx)(...)` at codegen time.
    d0 = ekf.crafts["drone_0"]
    d1 = ekf.crafts["drone_1"]
    publish({
        "p0":          d0.position,
        "v0":          d0.vel_linear,
        "p0_stddev":   d0.position_stddev,
        "v0_stddev":   d0.vel_linear_stddev,
        "p1":          d1.position,
        "v1":          d1.vel_linear,
        "p1_stddev":   d1.position_stddev,
        "v1_stddev":   d1.vel_linear_stddev,
    }, "manta/ex9/estimate")

    return MantaConfig(targets=[
        Target("ex9", drives=[ekf], dt=0.001, sim_rate_mult=1.0),
    ])

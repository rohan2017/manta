"""ex8 — Underwater AUV with full sensor suite + sim+EKF in one binary.

Demonstrates:
  * Mass + PointBuoy + Surface1 (drag) + 2× Thruster underwater dynamics.
  * IMU + DVL + Magnetometer driven by sim physics.
  * GravityField + FluidField (seawater 1025 kg/m³) + uniform MagField.
  * Codegen-emitted EKF: predict drives the est craft (Mass + thrusters
    mirroring sim throttle), per-sensor `consume_fresh + update_n` for
    IMU / DVL / Magnetometer.
  * Cross-world `connect()` pipes sim sensor outputs into est
    `set_measurement` hooks each tick; sim throttles mirror onto the
    est-side thrusters so the predict's force model matches.
  * Zenoh I/O: subscribe per-thruster throttle commands; publish truth
    state + EKF estimate.

Single binary `ex8` — both the sim and the EKF run in one process.

Codegen:

    PYTHONPATH=python/manta_codegen/src \\
        python -m manta_codegen.cli examples/ex8_submarine/config.py \\
            --workflow binary
"""

from manta_codegen import (Craft, EKF, MantaConfig, Target, World,
                           connect, publish, subscribe)
from manta_codegen.parts  import (DVL, IMU, Magnetometer, Mass, PointBuoy,
                                  Surface1, Thruster)
from manta_codegen.fields import FluidField, GravityField, MagField


# ---- Shared physical parameters ----
MASS         = 10.0                 # kg
MOI          = (0.05, 1.0, 1.0)     # cylinder along x; lower Ixx
VOLUME       = 0.0098               # m³ — slight negative buoyancy in 1025 kg/m³
WATER_RHO    = 1025.0               # kg/m³
G            = 9.81                 # m/s²
THRUST_FWD   = 50.0                 # N, max
THRUST_VERT  = 50.0                 # N, max

# Drag tensors. v_rel = fluid_v − own_v; positive A_1 makes force point
# along v_rel, which is "drag-like" (opposes own velocity). Streamlined
# along x → low x-drag, higher transverse.
DRAG_FORCE   = (15.0, 80.0, 80.0)   # N / (m/s)
DRAG_TORQUE  = ( 1.0,  6.0,  6.0)   # N·m / (m/s) — yaw/pitch damping

# Magnetic field in the local scene frame (Tesla). Mid-latitude rough:
# horizontal ~25 µT north, vertical ~−45 µT down.
B_FIELD      = (25e-6, 0.0, -45e-6)

# Initial pose: 5 m below surface, facing +x, at rest.
INIT_POS     = (0.0, 0.0, -5.0)


def _sensors(c: "Craft") -> tuple:
    """Add the matched sensor triple to a craft and return the descriptors."""
    imu = IMU("imu", accel_sigma=0.05, gyro_sigma=0.005, rate_hz=100.0)
    dvl = DVL("dvl", velocity_sigma=0.02,                rate_hz=10.0)
    mag = Magnetometer("mag", sigma=2e-7,                 rate_hz=50.0)
    c.add(imu); c.add(dvl); c.add(mag)
    return imu, dvl, mag


def _thrusters(c: "Craft") -> tuple:
    """Add forward + vertical thrusters and return the descriptors."""
    thrust_x = Thruster("thrust_x", max_thrust=THRUST_FWD,  direction=(1.0, 0.0, 0.0))
    thrust_z = Thruster("thrust_z", max_thrust=THRUST_VERT, direction=(0.0, 0.0, 1.0))
    c.add(thrust_x); c.add(thrust_z)
    return thrust_x, thrust_z


def make_config() -> MantaConfig:
    # ---- Sim craft: full physics ----
    sim_c = Craft("ex8")
    sim_c.add(Mass("body", mass=MASS, moi=MOI, apply_gravity=True))
    sim_c.add(PointBuoy("buoy", volume=VOLUME))
    sim_c.add(Surface1("drag",
                       force_tensors=[DRAG_FORCE],
                       torque_tensors=[DRAG_TORQUE]))
    sim_thrust_x, sim_thrust_z = _thrusters(sim_c)
    sim_imu, sim_dvl, sim_mag  = _sensors(sim_c)

    sim_grav  = GravityField(g=(0.0, 0.0, -G))
    sim_fluid = FluidField(density=WATER_RHO)         # seawater everywhere
    sim_mag_field = MagField().add_uniform(B_FIELD)

    sim_w = (World("ex8_sim")
                .add_field(sim_grav)
                .add_field(sim_fluid)
                .add_field(sim_mag_field)
                .add_craft(sim_c, pos=INIT_POS))

    # ---- Est craft: minimal predict-only model ----
    est_c = Craft("ex8_est")
    # Gravity off in the est dynamics — IMU's update absorbs the
    # gravitational signature of the body's motion. Buoyancy + drag are
    # not modeled est-side either; the estimator absorbs the integrated
    # discrepancy through its measurement updates.
    est_c.add(Mass("body", mass=MASS, moi=MOI, apply_gravity=False))
    est_thrust_x, est_thrust_z = _thrusters(est_c)
    est_imu, est_dvl, est_mag  = _sensors(est_c)

    # The est-side Magnetometer's measurement codegen needs a registered
    # MagField on the same world. Use the same uniform-B definition the
    # sim uses; Magnetometer.h(x) reads B at update-time (locally-
    # constant-B approximation).
    est_mag_field = MagField().add_uniform(B_FIELD)
    est_w = (World("ex8_est")
                .add_field(est_mag_field)
                .add_craft(est_c, pos=INIT_POS))

    # Same initial pose as the sim (the EKF picks it up from add_craft
    # automatically). Tight position prior reflects "I know where the
    # vehicle started"; velocity prior moderate; magnetometer plus tight
    # initial attitude lock keeps q observable through the dive.
    ekf = EKF(est_w, measurements=[est_imu, est_dvl, est_mag],
              initial_covariance=1.0,
              initial_position_var=1e-4,
              initial_attitude_var=1e-4,
              initial_velocity_var=1e-2,
              initial_angular_velocity_var=1e-4)

    # ---- Cross-world plumbing ----
    # Sim sensor outputs feed est sensor measurement inputs.
    # Cross-world wiring is owned by the EKF's setup() (StateSpec path).
    # Throttle mirrors so predict's force model matches the sim's input.
    connect(sim_thrust_x.throttle, est_thrust_x.set_throttle)
    connect(sim_thrust_z.throttle, est_thrust_z.set_throttle)

    # ---- Zenoh I/O ----
    subscribe(sim_thrust_x.set_throttle, "manta/ex8/thrust_x/cmd")
    subscribe(sim_thrust_z.set_throttle, "manta/ex8/thrust_z/cmd")
    publish({
        "p": sim_c.position,    "q": sim_c.orientation,
        "v": sim_c.vel_linear,  "w": sim_c.vel_angular,
    }, "manta/ex8/state")
    publish({
        "p":        ekf.position,        "q":        ekf.orientation,
        "v":        ekf.vel_linear,      "w":        ekf.vel_angular,
        "p_stddev": ekf.position_stddev, "v_stddev": ekf.vel_linear_stddev,
    }, "manta/ex8/estimate")

    return MantaConfig(targets=[
        Target("ex8", drives=[sim_w, ekf], dt=0.001, sim_rate_mult=1.0),
    ])

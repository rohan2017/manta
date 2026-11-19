"""Manta INS with editable part models in ins_models.py.

python -m examples.vehicles.ins_standalone --duration 60
python -m examples.vehicles.ins_standalone --export-preintegrator /tmp/imu-firmware

Synthetic measurements follow a prescribed spherical-Earth track across the
dateline. Replace measurements() with real acquisitions; the filter is unchanged.
The host can instead parse packets produced by the same preintegrator on an MCU.
"""
import argparse
from pathlib import Path
import casadi as ca
import numpy as np
from manta import INS, IMUPreintegrator, PreintegratedIMUPacket, TargetCpp, TargetNumpy
from manta.fields import FluidField, GravityField
from manta.ir import Vec3, WorldFrame
from manta.ir._rotation import quat_from_rotmat_np
from manta.planets import Earth
if __package__:
    from .ins_models import IMU, DVL, GPS, Barometer, make_world
else:  # Copy the two files next to each other to run this script directly.
    from ins_models import IMU, DVL, GPS, Barometer, make_world

RATE = 100


def trajectory(planet, t, speed=2.):
    """Independent circular-path oracle in inertial Cartesian coordinates."""
    latitude = np.deg2rad(45.)
    radius = planet.R_EQ + 10.  # Air track: GPS and barometer are both available.
    relative_rate = speed/(radius*np.cos(latitude))
    longitude = np.deg2rad(179.9999) + (planet.omega+relative_rate)*t
    s, c = np.sin(longitude), np.cos(longitude)
    up = np.array([np.cos(latitude)*c, np.cos(latitude)*s, np.sin(latitude)])
    east = np.array([-s, c, 0.])
    north = np.cross(up, east)
    rotation = np.column_stack((east, north, up))
    p = radius*up
    omega = np.array([0., 0., planet.omega+relative_rate])
    v = np.cross(omega, p)
    a = np.cross(omega, v)
    return dict(position=p, velocity=v, orientation=quat_from_rotmat_np(rotation),
                angular_velocity=rotation.T@omega), rotation, a


def preintegrator():
    return IMUPreintegrator(gyro_integration="endpoint", accel_integration="endpoint",
                           sample_rate_hz=RATE, gyro_noise_density=IMU.gyro_noise.sigma/np.sqrt(RATE),
                           accel_noise_density=IMU.accel_noise.sigma/np.sqrt(RATE))


def export_preintegrator(directory):
    """Generate C kernels and a small allocation-free firmware calling shim."""
    model = preintegrator()
    generated = TargetCpp(model, directory, class_name="Preintegrator")
    directory = Path(directory)
    (directory/"imu_packet.h").write_text(PreintegratedIMUPacket.c_header())
    module = model.module()
    initial = np.asarray(module.state.field("x").init).ravel()
    offsets, offset = [], 0
    for field in module.port("u").fields:
        offsets.append(f"#define MANTA_IMU_INPUT_{field.name.upper()} {offset}")
        offset += field.dim
    header = [
        '#ifndef MANTA_IMU_PREINTEGRATOR_H', '#define MANTA_IMU_PREINTEGRATOR_H',
        '#include "preintegrator_kernels.h"', '#include "imu_packet.h"',
        f'#define MANTA_IMU_INPUT_COUNT {offset}', *offsets,
        'typedef struct {',
        f'  double x[{len(initial)}], next_x[{len(initial)}], y[MANTA_IMU_READOUT_COUNT];',
        '  double w[preintegrator_step_SZ_W];',
        '  casadi_int iw[preintegrator_step_SZ_IW];',
        '} ImuPreintegrator;',
        'static inline void imu_preintegrator_reset(ImuPreintegrator *p) {',
        '  static const double initial[] = {'+', '.join(format(v,'.17g') for v in initial)+'};',
        '  memcpy(p->x, initial, sizeof(initial)); memset(p->y, 0, sizeof(p->y));', '}',
        'static inline int imu_preintegrator_step(ImuPreintegrator *p, const double *u, double dt, double t) {',
        '  const double *arg[preintegrator_step_SZ_ARG] = {p->x, u, &dt, &t};',
        '  double *res[preintegrator_step_SZ_RES] = {p->next_x, p->y};',
        '  int status = preintegrator_step(arg, res, p->iw, p->w, 0);',
        '  if (!status) memcpy(p->x, p->next_x, sizeof(p->x));',
        '  return status;', '}', '#endif', '']
    (directory/"imu_preintegrator.h").write_text('\n'.join(header))
    return generated


def build():
    planet = Earth(flattening=0., include_j2=False)
    initial, _, _ = trajectory(planet, 0.)
    world = make_world(planet, initial)
    # Same transform -> runtime pattern as TargetNumpy(EKF(world)) or UKF.
    ins = TargetNumpy(INS(world, imu="imu", propagation="preintegrated",
                          accel_integration="endpoint", covariance="conventional_reset",
                          sensors=["dvl.velocity", "gps.position", "baro.pressure"]))
    prior = ins.module.port("prior_x").spec
    P = np.zeros((prior.tangent_dim, prior.tangent_dim))
    for name, sigma in {"position": .5, "velocity": .02, "orientation": np.deg2rad(.2),
                        "imu.gyro_bias": 1e-6, "imu.accel_bias": .001}.items():
        i = prior.slot("vehicle."+name).tangent_offset
        P[i:i+3, i:i+3] = np.eye(3)*sigma**2
    ins.reset(P=P)
    resolved = world.snapshot()
    p, t = ca.MX.sym("p", 3), ca.MX.sym("t")
    point = Vec3[WorldFrame].from_mx(p)
    gravity = resolved.get_field(GravityField).value_at_sym(point, t).mx
    pressure = resolved.get_field(FluidField).value_at_sym(point, t).pressure
    environment = ca.Function("environment", [p, t], [gravity, pressure])
    return planet, ins, TargetNumpy(preintegrator()), environment


def measurements(planet, environment, t, rng):
    truth, R, acceleration = trajectory(planet, t)
    p, v = truth["position"], truth["velocity"]
    gravity, pressure = environment(p, t)
    gyro = truth["angular_velocity"] + rng.normal(0, IMU.gyro_noise.sigma, 3)
    accel = R.T@(acceleration-np.asarray(gravity).ravel()) + rng.normal(0, IMU.accel_noise.sigma, 3)
    z = {"dvl.velocity": R.T@(v-np.cross(planet.omega_vec_world(), p))+rng.normal(0, DVL.velocity_noise.sigma, 3),
         "gps.position": planet.R_world_from_planet(t).T@p+rng.normal(0, GPS.position_noise.sigma, 3),
         "baro.pressure": float(pressure)+rng.normal(0, Barometer.pressure_noise.sigma)}
    return accel, gyro, z, truth


def run(duration=60.):
    planet, ins, pre, environment = build()
    rng = np.random.default_rng(42)
    left = measurements(planet, environment, 0., rng)
    start, errors = 0., []
    for k in range(1, round(duration*RATE)+1):
        t = k/RATE
        right = measurements(planet, environment, t, rng)
        bias = ins.state_dict()["vehicle"]
        values = pre.step(1/RATE, t=(k-1)/RATE,
                          accel=left[0], gyro=left[1], end_accel=right[0], end_gyro=right[1],
                          accel_bias=bias["imu.accel_bias"], gyro_bias=bias["imu.gyro_bias"])
        if k % 2 == 0 or k == round(duration*RATE):
            packet = PreintegratedIMUPacket(values, start_time=start)
            # Remote alternative: packet = PreintegratedIMUPacket.from_bytes(payload)
            inputs = ins.preintegrated_inputs(packet)  # Reuse endpoint noise for aiding.
            ins.predict(packet)
            for sensor, period in (("dvl.velocity", 20), ("gps.position", 100), ("baro.pressure", 10)):
                if k % period == 0:
                    ins.update(sensor, right[2][sensor], u=inputs)
            errors.append(np.linalg.norm(ins.state_dict()["vehicle"]["position"]-right[3]["position"]))
            pre.reset()
            start = t
        left = right
    print(f"{ins.time:.2f} s; position RMS {np.sqrt(np.mean(np.square(errors))):.3f} m; final {errors[-1]:.3f} m")
    return ins


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--duration", type=float, default=60.)
    parser.add_argument("--export-preintegrator", type=Path)
    args = parser.parse_args()
    if args.export_preintegrator:
        generated = export_preintegrator(args.export_preintegrator)
        print("C kernels:", generated.kernels_c)
    else:
        if not np.isfinite(args.duration) or args.duration < 1/RATE:
            parser.error("duration must be at least one IMU interval")
        run(args.duration)

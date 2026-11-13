"""Manta-only stationary INS example; no simulator, Shiver, or robot package.

Run: python -m examples.vehicles.ins_standalone --duration 120 [--native]
Synthetic stationary measurements demonstrate the API, not qualification.
"""

import argparse

import casadi as ca
import numpy as np

from manta import Craft, INSStream, NavigationEnvironment, World
from manta.ir.frames import WorldFrame
from manta.ir.types import Vec3
from manta.parts import ConstantBiasIMU, Mass, VelocitySensor
from manta.planets import Earth


def build_ins():
    rate = 100
    earth = Earth()
    # Cartesian anchor in PlanetFrame, at geocentric latitude 45 degrees.
    anchor = np.array([1.0, 0.0, 1.0]) * 6378137 / np.sqrt(2)
    # Scene default axes are north, west, up. Keep this convention throughout.
    env = NavigationEnvironment.from_scene(
        earth.scene_at(tuple(anchor)), frame_id="example/site", epoch="t=0"
    )
    craft = Craft("vehicle")
    craft.add(Mass("body", mass=20, moi=(1, 2, 3)))
    # ARW 0.03 deg/sqrt(hour) -> rad/sqrt(second) -> per-sample rad/s.
    gyro_sigma = np.deg2rad(0.03) / 60 * np.sqrt(rate)
    craft.add(
        ConstantBiasIMU(
            "imu", rate=rate, gyro_noise_sigma=gyro_sigma, accel_noise_sigma=0.002
        )
    )
    craft.add(VelocitySensor("dvl", velocity_noise_sigma=0.002))
    world = World("local_estimator").add_field(env.gravity)
    world.add_craft(craft)
    ins = INSStream(
        world,
        imu="imu",
        sample_rate_hz=rate,
        predict_every=2,
        navigation_frame=env.frame,
        sensors=["dvl.velocity"],
    )
    # reset takes a physical product-state prior, before internal error-chart
    # conversion and shared-endpoint augmentation. Build it by named slots.
    prior = ins.filter.module.port("prior_x").spec
    P = np.zeros((prior.tangent_dim, prior.tangent_dim))
    for name, sigma in {
        "position": (1, 1, 1),
        "orientation": np.deg2rad([0.1, 0.1, 5]),
        "velocity": (0.01, 0.01, 0.01),
        "imu.gyro_bias": np.full(3, np.deg2rad(0.05) / 3600),
        "imu.accel_bias": (0.001, 0.001, 0.001),
    }.items():
        slot = prior.slot("vehicle." + name)
        i = slot.tangent_offset
        P[i : i + 3, i : i + 3] = np.diag(np.asarray(sigma) ** 2)
    yaw = np.deg2rad(5)
    ins.reset(
        {"vehicle": {"orientation": [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]}}, P=P
    )
    g = env.gravity.value_at_sym(Vec3[WorldFrame].constant((0, 0, 0)), ca.DM(0))
    stationary_accel = -np.asarray(ca.evalf(g.mx)).ravel()
    return ins, stationary_accel, np.asarray(env.frame.angular_velocity), gyro_sigma


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--duration", type=float, default=120)
    parser.add_argument("--native", action="store_true")
    args = parser.parse_args()
    if not np.isfinite(args.duration) or args.duration <= 0:
        parser.error("duration must be finite and positive")
    ins, specific_force, spin, gyro_sigma = build_ins()
    if args.native:
        ins.compile(sensors=["dvl.velocity"])
    rng = np.random.default_rng(42)
    for k in range(int(args.duration * 100) + 1):
        t = k / 100
        ins.update_imu(
            specific_force + rng.normal(0, 0.002, 3),
            spin + rng.normal(0, gyro_sigma, 3),
            t=t,
        )
        if k and k % 20 == 0:
            ins.update("dvl.velocity", rng.normal(0, 0.002, 3), t=t)
    ins.flush()
    state = ins.state_dict()["vehicle"]
    w, x, y, z = state["orientation"]
    yaw = np.rad2deg(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
    print(f"Elapsed: {ins.time:.2f} s; initial yaw error: 5 deg; final: {yaw:.3f} deg")
    print(f"Position error: {np.linalg.norm(state['position']):.4f} m")
    print("Gyro bias estimate (deg/hour):", np.rad2deg(state["imu.gyro_bias"]) * 3600)


if __name__ == "__main__":
    main()

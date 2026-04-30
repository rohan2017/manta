"""Synthetic sensor publisher for ex7 (Pattern C).

Pretends to be a robot's sensor stack. Publishes:
  - manta/ex7/imu  : 6 floats [ax, ay, az, gx, gy, gz]  (250 Hz)
  - manta/ex7/dvl  : 3 floats [vx, vy, vz]              (50 Hz)

Truth trajectory: 1-D constant-velocity along x at 1.5 m/s, no rotation.
  - true accel  = (0, 0, 0)         → IMU reads N(0, σ_a²) noise
  - true gyro   = (0, 0, 0)         → IMU reads N(0, σ_g²) noise
  - true v      = (1.5, 0, 0)       → DVL reads (1.5 + N(0, σ_v²), ...)

Usage:
    python examples/ex7_real_data_estimator/fake_sensors.py [--seconds N]
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time

import zenoh


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--seconds", type=float, default=0.0,
                   help="Run for N seconds and exit (0 = until SIGINT).")
    p.add_argument("--vx", type=float, default=1.5,
                   help="Truth velocity along x (m/s).")
    p.add_argument("--imu-sigma", type=float, default=0.05)
    p.add_argument("--dvl-sigma", type=float, default=0.02)
    args = p.parse_args()

    cfg = zenoh.Config()
    session = zenoh.open(cfg)

    imu_pub = session.declare_publisher("manta/ex7/imu")
    dvl_pub = session.declare_publisher("manta/ex7/dvl")

    rng = random.Random(7)

    print(f"fake_sensors: publishing manta/ex7/imu (250 Hz), "
          f"manta/ex7/dvl (50 Hz). truth vx={args.vx} m/s.")

    start = time.monotonic()
    next_imu = start
    next_dvl = start
    end_time = start + args.seconds if args.seconds > 0 else float("inf")

    try:
        while time.monotonic() < end_time:
            now = time.monotonic()

            if now >= next_imu:
                ax = rng.gauss(0.0, args.imu_sigma)
                ay = rng.gauss(0.0, args.imu_sigma)
                az = rng.gauss(0.0, args.imu_sigma)
                gx = rng.gauss(0.0, 0.005)
                gy = rng.gauss(0.0, 0.005)
                gz = rng.gauss(0.0, 0.005)
                imu_pub.put(json.dumps([ax, ay, az, gx, gy, gz]))
                next_imu += 1.0 / 250.0

            if now >= next_dvl:
                vx = args.vx + rng.gauss(0.0, args.dvl_sigma)
                vy = 0.0    + rng.gauss(0.0, args.dvl_sigma)
                vz = 0.0    + rng.gauss(0.0, args.dvl_sigma)
                dvl_pub.put(json.dumps([vx, vy, vz]))
                next_dvl += 1.0 / 50.0

            # Sleep until the soonest deadline.
            sleep_until = min(next_imu, next_dvl)
            time.sleep(max(0.0, sleep_until - time.monotonic()))
    except KeyboardInterrupt:
        pass
    finally:
        session.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())

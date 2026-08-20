"""Run IMU preintegration faster than the main INS filter.

This desktop smoke test runs a synthetic IMU stream at 500 Hz, folds ten
ordered samples into each packet, and predicts the main INS at 50 Hz.  A raw
INS running every sample is the reference.  Use ``--emit-cpp`` to lower both
the high-rate recurrence and packet-consuming INS to C/C++.

Run::

    .venv/bin/python -m examples.vehicles.ins_preintegration
    .venv/bin/python -m examples.vehicles.ins_preintegration --emit-cpp build/preintegration
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from manta import INS, IMUPreintegrator, TargetCpp, TargetNumpy

from .ins_vs_ekf import build_world


def _imu_sample(t: float) -> tuple[np.ndarray, np.ndarray]:
    """Smooth sample with changing rotation axes (and therefore coning)."""
    accel = np.array((0.35 * np.sin(1.7 * t),
                      0.22 * np.cos(1.1 * t),
                      9.81 + 0.12 * np.sin(0.8 * t)))
    gyro = np.array((0.55 * np.cos(2.3 * t),
                     0.42 * np.sin(1.9 * t),
                     0.30 + 0.18 * np.sin(1.3 * t)))
    return accel, gyro


def _ins(*, propagation: str):
    return INS(
        build_world(truth=False), imu="auv.imu",
        sensors=["auv.model_force.specific_force"],
        propagation=propagation,
    )


def run(*, duration: float = 2.0, imu_rate: float = 500.0,
        filter_rate: float = 50.0) -> dict[str, float]:
    if duration <= 0.0 or imu_rate <= 0.0 or filter_rate <= 0.0:
        raise ValueError("duration and rates must be positive")
    ratio = imu_rate / filter_rate
    samples_per_packet = round(ratio)
    if samples_per_packet < 1 or not np.isclose(ratio, samples_per_packet):
        raise ValueError("imu_rate must be an integer multiple of filter_rate")

    raw = TargetNumpy(_ins(propagation="raw"))
    packet_ins = TargetNumpy(_ins(propagation="preintegrated"))
    preintegrator = TargetNumpy(IMUPreintegrator(
        # Set these to the calibrated continuous densities for real data.
        accel_noise_density=0.0,
        gyro_noise_density=0.0,
    ))
    imu_dt = 1.0 / imu_rate
    total_samples = round(duration * imu_rate)
    packet_count = 0
    for k in range(total_samples):
        t = k * imu_dt
        accel, gyro = _imu_sample(t)
        raw.predict(imu_dt, t=t, u={"imu.accel": accel, "imu.gyro": gyro})
        packet = preintegrator.step(
            imu_dt, t=t, accel=accel, gyro=gyro,
            accel_bias=(0.0, 0.0, 0.0),
            gyro_bias=(0.0, 0.0, 0.0))
        if (k + 1) % samples_per_packet == 0:
            packet_ins.predict_preintegrated(packet)
            preintegrator.reset()
            packet_count += 1

    raw_state = raw.state_dict()["auv"]
    packet_state = packet_ins.state_dict()["auv"]
    result = {
        "position_difference_m": float(np.linalg.norm(
            raw_state["position"] - packet_state["position"])),
        "velocity_difference_mps": float(np.linalg.norm(
            raw_state["velocity"] - packet_state["velocity"])),
        "quaternion_difference": float(min(
            np.linalg.norm(raw_state["orientation"]
                           - packet_state["orientation"]),
            np.linalg.norm(raw_state["orientation"]
                           + packet_state["orientation"]))),
        "packets": float(packet_count),
    }
    print(f"IMU {imu_rate:g} Hz -> INS {filter_rate:g} Hz "
          f"({samples_per_packet} samples/packet, {packet_count} packets)")
    for name, value in result.items():
        if name != "packets":
            print(f"{name}: {value:.3e}")
    return result


def emit_cpp(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    TargetCpp(IMUPreintegrator(), directory / "mcu",
              class_name="ImuPreintegrator")
    TargetCpp(_ins(propagation="preintegrated"), directory / "main",
              class_name="PreintegratedIns")
    print(f"generated MCU recurrence: {directory / 'mcu'}")
    print(f"generated main filter:    {directory / 'main'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=2.0)
    parser.add_argument("--imu-rate", type=float, default=500.0)
    parser.add_argument("--filter-rate", type=float, default=50.0)
    parser.add_argument("--emit-cpp", type=Path)
    args = parser.parse_args()
    run(duration=args.duration, imu_rate=args.imu_rate,
        filter_rate=args.filter_rate)
    if args.emit_cpp is not None:
        emit_cpp(args.emit_cpp)


if __name__ == "__main__":
    main()

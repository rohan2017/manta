"""Representative 18-state Mako filter replay workload.

The 30-second input schedule matches Shiver's declared stream: 200 Hz raw
accelerometer + gyro, 15 Hz DVL, 10 Hz depth + GPS, 1 Hz SBL, and five 50 Hz
achieved-control channels.  Shiver would own ordering and held controls; this
benchmark materializes that decision as explicit predicts, ordered updates,
and control event boundaries before handing the numeric program to Manta.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import numpy as np

from manta import (
    EKF,
    Craft,
    ReplayBoundary,
    ReplayPredict,
    ReplayUpdate,
    TargetFilterReplay,
    TargetNumpy,
    World,
)
from manta.fields import FluidField, GravityField
from manta.parts import (
    IMU,
    Barometer,
    Mass,
    PositionSensor,
    Thruster,
    VelocitySensor,
)


@dataclass(frozen=True)
class _Event:
    epoch_ns: int
    order: int
    kind: str
    identity: str


def _filter():
    craft = Craft("mako")
    craft.add(Mass("body", mass=45.0, moi=(2.0, 8.0, 8.0)))
    craft.add(
        IMU(
            "imu",
            rate=200.0,
            accel_noise_sigma=0.05,
            gyro_noise_sigma=0.005,
            accel_bias_sigma=1e-4,
            gyro_bias_sigma=1e-5,
        )
    )
    craft.add(VelocitySensor("dvl", rate=15.0, velocity_noise_sigma=0.03))
    craft.add(Barometer("depth", rate=10.0, pressure_noise_sigma=50.0))
    craft.add(PositionSensor("gps", rate=10.0, position_noise_sigma=0.5))
    craft.add(PositionSensor("sbl", rate=1.0, position_noise_sigma=1.0))
    for name, force in (
        ("tail", (80.0, 0.0, 0.0)),
        ("fin0", (0.0, 0.0, 8.0)),
        ("fin1", (0.0, 0.0, -8.0)),
        ("fin2", (0.0, 8.0, 0.0)),
        ("fin3", (0.0, -8.0, 0.0)),
    ):
        craft.add(Thruster(name, force=force))
    world = World(name="mako_filter_replay_benchmark")
    world.add_field(GravityField(g=(0.0, 0.0, -9.81)))
    world.add_field(FluidField().add_flat_ocean())
    world.add_craft(craft, position=(0.0, 0.0, -20.0))
    transform = EKF(world)
    assert transform.spec.tangent_dim == 18
    return transform


def _schedule(duration_s: float) -> list[_Event]:
    events = []
    control_names = ("tail", "fin0", "fin1", "fin2", "fin3")
    for sequence in range(1, round(50 * duration_s) + 1):
        epoch = round(sequence * 1e9 / 50)
        for index, name in enumerate(control_names):
            events.append(_Event(epoch, index, "control", name))
    rates = (
        ("accel", 200, 10),
        ("gyro", 200, 11),
        ("dvl", 15, 12),
        ("depth", 10, 13),
        ("gps", 10, 14),
        ("sbl", 1, 15),
    )
    for name, rate, order in rates:
        for sequence in range(1, round(rate * duration_s) + 1):
            events.append(
                _Event(round(sequence * 1e9 / rate), order, "measurement", name)
            )
    events.sort(key=lambda event: (event.epoch_ns, event.order, event.identity))
    return events


def _measurement(identity: str):
    return {
        "accel": ("mako.imu.accel", (0.0, 0.0, 9.81), np.eye(3) * 0.0025),
        "gyro": ("mako.imu.gyro", (0.0, 0.0, 0.0), np.eye(3) * 0.000025),
        "dvl": ("mako.dvl.velocity", (0.0, 0.0, 0.0), np.eye(3) * 0.0009),
        "depth": ("mako.depth.pressure", (302_000.0,), np.array([[2_500.0]])),
        "gps": ("mako.gps.position", (0.0, 0.0, -20.0), np.eye(3) * 0.25),
        "sbl": ("mako.sbl.position", (0.0, 0.0, -20.0), np.eye(3)),
    }[identity]


def _initial_controls() -> dict[str, float]:
    return {
        f"mako.{name}.throttle": 0.0
        for name in ("tail", "fin0", "fin1", "fin2", "fin3")
    }


def _operations(
    events: list[_Event],
    *,
    start_ns: int = 0,
    initial_controls: dict[str, float] | None = None,
    sequence_offset: int = 0,
):
    controls = dict(initial_controls or _initial_controls())
    operations = []
    current_ns = start_ns
    for local_sequence, event in enumerate(events):
        sequence = sequence_offset + local_sequence
        if event.epoch_ns > current_ns:
            operations.append(
                ReplayPredict(
                    current_ns / 1e9,
                    (event.epoch_ns - current_ns) / 1e9,
                    controls=dict(controls),
                )
            )
            current_ns = event.epoch_ns
        if event.kind == "control":
            controls[f"mako.{event.identity}.throttle"] = ((sequence % 17) - 8) / 20.0
            operations.append(ReplayBoundary(current_ns / 1e9, checkpoint=True))
        else:
            sensor, measurement, covariance = _measurement(event.identity)
            operations.append(
                ReplayUpdate(
                    current_ns / 1e9,
                    sensor,
                    measurement,
                    controls=dict(controls),
                    measurement_covariance=covariance,
                    checkpoint=True,
                )
            )
    return tuple(operations), controls


def _p99_ms(samples_ns: list[int]) -> float:
    return float(np.percentile(np.asarray(samples_ns), 99)) / 1e6


def run(duration_s: float = 30.0, trials: int = 5) -> dict[str, float | int]:
    transform = _filter()
    runtime = TargetNumpy(transform)
    events = _schedule(duration_s)
    expected_events = round(duration_s * 686)
    assert len(events) == expected_events
    operations, _ = _operations(events)
    started = time.perf_counter_ns()
    kernel = TargetFilterReplay(
        transform,
        max_operations=len(operations) + 64,
        max_checkpoints=len(events) + 64,
    )
    compile_ms = (time.perf_counter_ns() - started) / 1e6
    started = time.perf_counter_ns()
    program = kernel.program(runtime.checkpoint(), operations)
    pack_ms = (time.perf_counter_ns() - started) / 1e6
    execution_ns = []
    result = None
    for _ in range(trials):
        started = time.perf_counter_ns()
        result = kernel.run(program)
        execution_ns.append(time.perf_counter_ns() - started)
    assert result is not None
    median_s = float(np.median(execution_ns)) / 1e9

    # Live-safe slices include validation, native execution, materialization,
    # and checkpoint restart.  Each slice consumes exactly 32 incoming events.
    slice_ns = []
    checkpoint = runtime.checkpoint()
    held_controls = _initial_controls()
    for offset in range(0, len(events), 32):
        chunk = events[offset : offset + 32]
        chunk_operations, held_controls = _operations(
            chunk,
            start_ns=round(checkpoint.time * 1e9),
            initial_controls=held_controls,
            sequence_offset=offset,
        )
        started = time.perf_counter_ns()
        chunk_program = kernel.program(checkpoint, chunk_operations)
        chunk_result = kernel.run(chunk_program)
        slice_ns.append(time.perf_counter_ns() - started)
        checkpoint = chunk_result.final
    return {
        "input_events": len(events),
        "numeric_operations": len(operations),
        "state_dimension": transform.spec.tangent_dim,
        "packed_bytes": program.packed_bytes,
        "configured_execution_byte_cap": kernel.max_execution_bytes,
        "estimated_worst_case_bytes": kernel.estimated_worst_case_bytes,
        "compile_ms": compile_ms,
        "pack_ms": pack_ms,
        "full_replay_p50_ms": float(np.percentile(execution_ns, 50)) / 1e6,
        "full_replay_p99_ms": _p99_ms(execution_ns),
        "sustained_input_events_per_s": len(events) / median_s,
        "incoming_input_events_per_s": 686.0,
        "throughput_margin": (len(events) / median_s) / 686.0,
        "live_32_event_slice_p50_ms": float(np.percentile(slice_ns, 50)) / 1e6,
        "live_32_event_slice_p99_ms": _p99_ms(slice_ns),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration-s", type=float, default=30.0)
    parser.add_argument("--trials", type=int, default=5)
    args = parser.parse_args()
    for key, value in run(args.duration_s, args.trials).items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()

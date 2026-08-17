"""Exact generated filter-span execution and failure contracts."""

from __future__ import annotations

import shutil
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from manta import (
    EKF,
    UKF,
    Craft,
    FilterCheckpoint,
    ReplayBoundary,
    ReplayPredict,
    ReplayUpdate,
    TargetFilterReplay,
    TargetNumpy,
    World,
)
from manta.fields import FluidField, GravityField
from manta.parts import IMU, Barometer, Mass, PositionSensor, Thruster

pytestmark = pytest.mark.skipif(shutil.which("cc") is None, reason="cc is required")


def _world() -> World:
    craft = Craft("vehicle")
    craft.add(Mass("body", mass=8.0, moi=(1.0, 1.2, 1.4)))
    craft.add(Thruster("tail", force=(20.0, 0.0, 0.0)))
    craft.add(PositionSensor("gps", position_noise_sigma=0.5))
    craft.add(Barometer("depth", pressure_noise_sigma=25.0))
    craft.add(IMU("imu", accel_noise_sigma=0.05, gyro_noise_sigma=0.005))
    world = World(name="filter_replay_test")
    world.add_field(GravityField(g=(0.0, 0.0, -9.81)))
    world.add_field(FluidField().add_flat_ocean())
    world.add_craft(craft, position=(0.0, 0.0, -4.0))
    return world


def _apply_oracle(runtime, operations):
    updates = []
    checkpoints = []
    for index, operation in enumerate(operations):
        if isinstance(operation, ReplayPredict):
            runtime.predict(
                operation.dt,
                t=operation.time,
                u=dict(operation.controls),
                Q=operation.process_covariance,
            )
        elif isinstance(operation, ReplayUpdate):
            updates.append(
                (
                    index,
                    runtime.update(
                        operation.sensor,
                        operation.measurement,
                        R=operation.measurement_covariance,
                        t=operation.time,
                        u=dict(operation.controls),
                    ),
                )
            )
        if operation.checkpoint:
            checkpoints.append((index, runtime.checkpoint()))
    return runtime.checkpoint(), tuple(updates), tuple(checkpoints)


@pytest.mark.parametrize("estimator", [EKF, UKF])
def test_native_span_has_exact_sequential_oracle_parity(
    estimator, monkeypatch, tmp_path
):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    transform = estimator(_world(), gates={"gps.position": 9.0})
    oracle = TargetNumpy(transform)
    kernel = TargetFilterReplay(
        transform, max_operations=16, max_checkpoints=8, optimization="balanced"
    )
    tangent = oracle.spec.tangent_dim
    operations = (
        ReplayUpdate(
            0.0,
            "vehicle.gps.position",
            (0.1, 0.0, -4.0),
            controls={"vehicle.tail.throttle": 0.2},
            checkpoint=True,
        ),
        ReplayPredict(
            0.0,
            0.02,
            controls={"vehicle.tail.throttle": 0.2},
            checkpoint=True,
        ),
        ReplayBoundary(0.02, checkpoint=True),
        ReplayUpdate(
            0.02,
            "vehicle.depth.pressure",
            (140_000.0,),
            measurement_covariance=np.array([[900.0]]),
            controls={"vehicle.tail.throttle": -0.1},
        ),
        ReplayPredict(
            0.02,
            0.03,
            controls={"vehicle.tail.throttle": -0.1},
            process_covariance=np.eye(tangent) * 1e-8,
        ),
        ReplayUpdate(
            0.05,
            "vehicle.gps.position",
            (100.0, 0.0, -4.0),
            measurement_covariance=np.eye(3) * 0.2,
            checkpoint=True,
        ),
    )
    result = kernel.run(kernel.program(oracle.checkpoint(), operations))
    expected, expected_updates, expected_checkpoints = _apply_oracle(oracle, operations)

    np.testing.assert_allclose(result.final.x, expected.x, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(result.final.P, expected.P, rtol=1e-11, atol=1e-12)
    assert result.final.time == expected.time
    assert [index for index, _ in result.updates] == [
        index for index, _ in expected_updates
    ]
    for (_, actual), (_, wanted) in zip(result.updates, expected_updates, strict=True):
        assert actual.sensor == wanted.sensor
        np.testing.assert_allclose(actual.innovation, wanted.innovation, atol=1e-10)
        np.testing.assert_allclose(
            actual.innovation_covariance,
            wanted.innovation_covariance,
            rtol=1e-11,
            atol=1e-10,
        )
        assert actual.nis == pytest.approx(wanted.nis, rel=1e-11, abs=1e-11)
        assert actual.accepted is wanted.accepted
        assert actual.gate == wanted.gate
        assert actual.covariance_overridden is wanted.covariance_overridden
    assert [item.operation_index for item in result.checkpoints] == [
        index for index, _ in expected_checkpoints
    ]
    for actual, (_, wanted) in zip(
        result.checkpoints, expected_checkpoints, strict=True
    ):
        np.testing.assert_allclose(actual.checkpoint.x, wanted.x, atol=1e-12)
        np.testing.assert_allclose(actual.checkpoint.P, wanted.P, atol=1e-12)
        assert actual.checkpoint.time == wanted.time
    assert not result.updates[-1][1].accepted


def test_program_preserves_mixed_sensor_and_same_epoch_order(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    transform = EKF(_world())
    runtime = TargetNumpy(transform)
    kernel = TargetFilterReplay(transform, max_operations=8, max_checkpoints=2)
    operations = (
        ReplayUpdate(0.0, "vehicle.gps.position", (0.2, 0.0, -4.0)),
        ReplayUpdate(0.0, "vehicle.imu.accel", (0.0, 0.0, 9.81)),
        ReplayUpdate(0.0, "vehicle.imu.gyro", (0.0, 0.0, 0.0)),
        ReplayUpdate(
            0.0,
            "vehicle.depth.pressure",
            (141_000.0,),
            measurement_covariance=np.array([[625.0]]),
        ),
        ReplayUpdate(0.0, "vehicle.gps.position", (-0.1, 0.0, -4.0)),
    )
    result = kernel.run(kernel.program(runtime.checkpoint(), operations))
    expected, expected_updates, _ = _apply_oracle(runtime, operations)
    assert [item.sensor for _, item in result.updates] == [
        "vehicle.gps.position",
        "vehicle.imu.accel",
        "vehicle.imu.gyro",
        "vehicle.depth.pressure",
        "vehicle.gps.position",
    ]
    assert [item.innovation.shape for _, item in result.updates] == [
        (3,),
        (3,),
        (3,),
        (1,),
        (3,),
    ]
    assert [item.innovation_covariance.shape for _, item in result.updates] == [
        (3, 3),
        (3, 3),
        (3, 3),
        (1, 1),
        (3, 3),
    ]
    np.testing.assert_allclose(result.final.x, expected.x, atol=1e-12)
    np.testing.assert_allclose(result.final.P, expected.P, atol=1e-12)
    for (_, actual), (_, wanted) in zip(result.updates, expected_updates, strict=True):
        np.testing.assert_allclose(actual.innovation, wanted.innovation, atol=1e-12)
        np.testing.assert_allclose(
            actual.innovation_covariance, wanted.innovation_covariance, atol=1e-12
        )


def test_checkpoint_is_a_complete_restart_boundary(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    transform = EKF(_world())
    runtime = TargetNumpy(transform)
    kernel = TargetFilterReplay(transform, max_operations=8, max_checkpoints=4)
    first = (
        ReplayUpdate(0.0, "vehicle.gps.position", (0.1, 0.0, -4.0)),
        ReplayPredict(0.0, 0.1, checkpoint=True),
    )
    second = (
        ReplayUpdate(0.1, "vehicle.gps.position", (0.2, 0.0, -4.0)),
        ReplayPredict(0.1, 0.05),
    )
    whole = kernel.run(kernel.program(runtime.checkpoint(), (*first, *second)))
    prefix = kernel.run(kernel.program(runtime.checkpoint(), first))
    resumed = kernel.run(kernel.program(prefix.final, second))
    np.testing.assert_array_equal(prefix.checkpoints[0].checkpoint.x, prefix.final.x)
    np.testing.assert_array_equal(prefix.checkpoints[0].checkpoint.P, prefix.final.P)
    assert prefix.checkpoints[0].checkpoint.time == prefix.final.time
    np.testing.assert_array_equal(resumed.final.x, whole.final.x)
    np.testing.assert_array_equal(resumed.final.P, whole.final.P)
    assert resumed.final.time == whole.final.time


def test_program_is_bounded_owned_and_rejects_invalid_inputs(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    transform = EKF(_world())
    runtime = TargetNumpy(transform)
    kernel = TargetFilterReplay(transform, max_operations=3, max_checkpoints=1)
    initial = runtime.checkpoint()
    program = kernel.program(
        initial,
        (
            ReplayUpdate(0.0, "vehicle.gps.position", (0.0, 0.0, -4.0)),
            ReplayBoundary(0.0, checkpoint=True),
        ),
    )
    assert program.operation_count == 2
    assert program.checkpoint_count == 1
    assert program.packed_bytes > 0
    assert not program._kinds.flags.writeable
    initial.x[:] = 99.0
    assert not np.all(program.initial.x == 99.0)

    with pytest.raises(ValueError, match="configured maximum"):
        kernel.program(initial, (ReplayBoundary(0.0),) * 4)
    with pytest.raises(ValueError, match="configured maximum"):
        kernel.program(
            initial,
            (ReplayBoundary(0.0, checkpoint=True),) * 2,
        )
    with pytest.raises(KeyError, match="unknown sensor"):
        kernel.program(initial, (ReplayUpdate(0.0, "missing", (1.0,)),))
    with pytest.raises(ValueError, match="measurement shape"):
        kernel.program(
            initial,
            (ReplayUpdate(0.0, "vehicle.gps.position", (1.0,)),),
        )
    with pytest.raises(ValueError, match="positive definite"):
        kernel.program(
            initial,
            (
                ReplayUpdate(
                    0.0,
                    "vehicle.gps.position",
                    (0.0, 0.0, -4.0),
                    measurement_covariance=np.zeros((3, 3)),
                ),
            ),
        )
    with pytest.raises(ValueError, match="positive semidefinite"):
        kernel.program(
            initial,
            (
                ReplayPredict(
                    0.0,
                    0.1,
                    process_covariance=-np.eye(runtime.spec.tangent_dim),
                ),
            ),
        )
    with pytest.raises(ValueError, match="nonmonotonic"):
        kernel.program(
            initial,
            (ReplayPredict(0.0, 0.1), ReplayBoundary(0.09)),
        )
    with pytest.raises(ValueError, match="has a gap"):
        kernel.program(initial, (ReplayBoundary(0.1),))
    with pytest.raises(KeyError, match="unknown control"):
        kernel.program(
            initial,
            (ReplayPredict(0.0, 0.1, controls={"missing": 1.0}),),
        )


def test_compile_cache_identity_covers_model_gate_and_bounds(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    transform = EKF(_world(), gates=9.0)
    first = TargetFilterReplay(transform, max_operations=8, max_checkpoints=4)
    second = TargetFilterReplay(transform, max_operations=8, max_checkpoints=4)
    different_bounds = TargetFilterReplay(
        transform, max_operations=9, max_checkpoints=4
    )
    different_gate = TargetFilterReplay(
        EKF(_world(), gates=10.0), max_operations=8, max_checkpoints=4
    )
    assert first.cache_identity == second.cache_identity
    assert first.library_path == second.library_path
    assert first.cache_identity != different_bounds.cache_identity
    assert first.cache_identity != different_gate.cache_identity
    with pytest.raises(ValueError, match="different kernel"):
        different_bounds.run(
            first.program(TargetNumpy(transform).checkpoint(), (ReplayBoundary(0.0),))
        )


def test_constructor_and_checkpoint_fail_locally(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    transform = EKF(_world())
    module = transform.module()
    malformed = replace(
        module,
        entry_points=tuple(
            replace(entry, writes=("P", "x")) if entry.method == "predict" else entry
            for entry in module.entry_points
        ),
    )
    with pytest.raises(TypeError, match="write manifold then covariance"):
        TargetFilterReplay(malformed, max_operations=1, max_checkpoints=0)
    with pytest.raises(ValueError, match="positive integer"):
        TargetFilterReplay(transform, max_operations=0, max_checkpoints=0)
    with pytest.raises(ValueError, match="may not exceed"):
        TargetFilterReplay(transform, max_operations=1, max_checkpoints=2)
    with pytest.raises(ValueError, match="exceeding max_execution_bytes"):
        TargetFilterReplay(
            transform,
            max_operations=100,
            max_checkpoints=100,
            max_execution_bytes=1_024,
        )
    with pytest.raises(ValueError, match="hard native safety ceiling"):
        TargetFilterReplay(
            transform,
            max_operations=1,
            max_checkpoints=0,
            max_execution_bytes=2 * 1024 * 1024 * 1024,
        )
    with pytest.raises(ValueError, match="int32 ABI"):
        TargetFilterReplay(
            transform,
            max_operations=int(np.iinfo(np.int32).max) + 1,
            max_checkpoints=0,
        )
    kernel = TargetFilterReplay(transform, max_operations=2, max_checkpoints=1)
    assert 0 < kernel.estimated_worst_case_bytes <= kernel.max_execution_bytes
    checkpoint = TargetNumpy(transform).checkpoint()
    bad = FilterCheckpoint(checkpoint.x, -np.eye(checkpoint.P.shape[0]), 0.0)
    with pytest.raises(ValueError, match="positive semidefinite"):
        kernel.program(bad, (ReplayBoundary(0.0),))


@pytest.mark.parametrize(
    ("forge", "message"),
    [
        (lambda program: replace(program, operation_count=2), "kinds shape"),
        (
            lambda program: replace(program, _kinds=program._kinds.astype(np.int64)),
            "kinds dtype",
        ),
        (
            lambda program: replace(program, _times=np.zeros((1, 2), dtype=np.float64)),
            "times shape",
        ),
        (
            lambda program: replace(
                program,
                _process_covariances=np.zeros(
                    (1, program.initial.P.shape[0], program.initial.P.shape[0] * 2),
                    dtype=np.float64,
                )[:, :, ::2],
            ),
            "process_covariances must be C-contiguous",
        ),
        (
            lambda program: replace(program, checkpoint_count=1),
            "checkpoint_count says",
        ),
        (
            lambda program: replace(
                program, _checkpoint_flags=np.array([2], dtype=np.uint8)
            ),
            "checkpoint_flags must contain only 0 or 1",
        ),
        (
            lambda program: replace(program, _kinds=np.array([99], dtype=np.int32)),
            "unknown kind",
        ),
        (
            lambda program: replace(
                program,
                _kinds=np.array([1], dtype=np.int32),
                _sensors=np.array([99], dtype=np.int32),
            ),
            "unknown sensor index",
        ),
        (
            lambda program: replace(
                program,
                initial=FilterCheckpoint(
                    program.initial.x.copy(),
                    -np.eye(program.initial.P.shape[0]),
                    program.initial.time,
                ),
            ),
            "positive semidefinite",
        ),
    ],
)
def test_forged_public_program_is_rejected_before_native_call(
    forge, message, monkeypatch, tmp_path
):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    transform = EKF(_world())
    runtime = TargetNumpy(transform)
    kernel = TargetFilterReplay(transform, max_operations=2, max_checkpoints=1)
    program = kernel.program(runtime.checkpoint(), (ReplayBoundary(0.0),))

    native_called = False
    original = kernel._native

    def should_not_run(*args):
        nonlocal native_called
        native_called = True
        return original.run(*args)

    kernel._native = SimpleNamespace(
        identity=original.identity, path=original.path, run=should_not_run
    )
    with pytest.raises((TypeError, ValueError), match=message):
        kernel.run(forge(program))
    assert not native_called


def test_invalid_native_checkpoint_is_not_returned(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    transform = EKF(_world())
    runtime = TargetNumpy(transform)
    kernel = TargetFilterReplay(transform, max_operations=2, max_checkpoints=1)
    program = kernel.program(runtime.checkpoint(), (ReplayBoundary(0.0),))
    original = kernel._native

    def corrupt_final(*args):
        status = original.run(*args)
        args[15][0] = float("nan")  # final_x
        return status

    kernel._native = SimpleNamespace(
        identity=original.identity, path=original.path, run=corrupt_final
    )
    with pytest.raises(ValueError, match="final checkpoint is invalid"):
        kernel.run(program)

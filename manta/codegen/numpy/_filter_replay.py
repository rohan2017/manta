"""Exact generated execution of an already ordered filter operation span.

This target is deliberately narrower than a fixed-lag replay engine.  A
consumer supplies the exact sequence of predicts, measurement updates, and
numeric control-boundary markers.  Manta executes that sequence; it does not
sort it, identify transport samples, select a lag, or infer a control hold.

The native runner removes the Python/CasADi crossing between operations while
calling the same generated predict and per-sensor update kernels as the normal
filter runtime.  It never combines measurements or shortcuts covariance.
"""

from __future__ import annotations

import ctypes
import hashlib
import math
import os
import platform
import shutil
import subprocess
import tempfile
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import casadi as ca
import numpy as np
from numpy.typing import NDArray

from ..._validation import require_positive
from ...ir.module import Hosting, Role, StateRef, entry_ident
from ..target import as_module
from ._compile import CompilationError, _cache_dir
from ._filter import FilterCheckpoint, UpdateResult
from ._runtime import pack_fields


@dataclass(frozen=True)
class ReplayPredict:
    """One explicit predict operation.

    ``time`` is the logical model time at the start of the interval.  The
    controls are resolved independently against the Module's declared
    defaults, exactly like :meth:`NumpyFilter.predict`; Manta does not invent
    a hold/interpolation policy.  ``process_covariance`` overrides the
    model-derived Q for this operation when present.
    """

    time: float
    dt: float
    controls: Mapping[str, Any] = field(default_factory=dict)
    process_covariance: Any | None = None
    checkpoint: bool = False


@dataclass(frozen=True)
class ReplayUpdate:
    """One named sensor update at the current logical model time."""

    time: float
    sensor: str
    measurement: Any
    controls: Mapping[str, Any] = field(default_factory=dict)
    measurement_covariance: Any | None = None
    checkpoint: bool = False


@dataclass(frozen=True)
class ReplayBoundary:
    """A numeric event boundary with no filter equation.

    Shiver can retain achieved-control events in an exact event/checkpoint
    stream without asking Manta to own control identities or hold policy.  A
    boundary never advances logical time; an explicit :class:`ReplayPredict`
    must do that.
    """

    time: float
    checkpoint: bool = False


ReplayOperation = ReplayPredict | ReplayUpdate | ReplayBoundary


@dataclass(frozen=True, eq=False)
class FilterReplayProgram:
    """Validated, bounded native input owned by one kernel identity."""

    kernel_identity: str
    initial: FilterCheckpoint
    operation_count: int
    checkpoint_count: int
    _kinds: np.ndarray = field(repr=False)
    _sensors: np.ndarray = field(repr=False)
    _times: np.ndarray = field(repr=False)
    _dts: np.ndarray = field(repr=False)
    _controls: np.ndarray = field(repr=False)
    _measurements: np.ndarray = field(repr=False)
    _measurement_covariances: np.ndarray = field(repr=False)
    _process_covariances: np.ndarray = field(repr=False)
    _use_measurement_covariance: np.ndarray = field(repr=False)
    _use_process_covariance: np.ndarray = field(repr=False)
    _checkpoint_flags: np.ndarray = field(repr=False)

    @property
    def packed_bytes(self) -> int:
        """Owned numeric storage, useful for capacity accounting."""
        return sum(
            value.nbytes
            for value in (
                self._kinds,
                self._sensors,
                self._times,
                self._dts,
                self._controls,
                self._measurements,
                self._measurement_covariances,
                self._process_covariances,
                self._use_measurement_covariance,
                self._use_process_covariance,
                self._checkpoint_flags,
            )
        )


@dataclass(frozen=True)
class ReplayCheckpointResult:
    """Complete filter checkpoint after one requested operation boundary."""

    operation_index: int
    checkpoint: FilterCheckpoint


@dataclass(frozen=True)
class FilterReplayResult:
    """Final state plus ordered per-update diagnostics and checkpoints."""

    final: FilterCheckpoint
    updates: tuple[tuple[int, UpdateResult], ...]
    checkpoints: tuple[ReplayCheckpointResult, ...]


@dataclass(frozen=True)
class _Sensor:
    index: int
    name: str
    dim: int
    gate: float | None
    diagnostic_entry: Any
    override_entry: Any
    diagnostic_function: ca.Function
    override_function: ca.Function


@dataclass(frozen=True)
class _NativeLibrary:
    identity: str
    path: str
    run: Any


@dataclass(frozen=True)
class _FunctionABI:
    name: str
    arg_slots: int
    result_slots: int
    integer_work: int
    real_work: int


_NATIVE_CACHE: dict[str, _NativeLibrary] = {}
_NATIVE_CACHE_LOCK = threading.Lock()
_DEFAULT_EXECUTION_BYTE_CAP = 512 * 1024 * 1024
_HARD_EXECUTION_BYTE_CAP = 1024 * 1024 * 1024
_INT32_MAX = int(np.iinfo(np.int32).max)


def _validate_checkpoint(module, checkpoint: FilterCheckpoint) -> FilterCheckpoint:
    if not isinstance(checkpoint, FilterCheckpoint):
        raise TypeError("filter replay initial state must be a FilterCheckpoint")
    if checkpoint.artifact_id != module.artifact_id:
        raise ValueError("filter replay checkpoint belongs to a different Module artifact")
    spec = module.spec
    x = np.asarray(checkpoint.x, dtype=float)
    P = np.asarray(checkpoint.P, dtype=float)
    if x.shape != (spec.ambient_dim,):
        raise ValueError(
            f"filter replay x shape {x.shape} does not match {(spec.ambient_dim,)}"
        )
    if P.shape != (spec.tangent_dim, spec.tangent_dim):
        raise ValueError(
            f"filter replay P shape {P.shape} does not match "
            f"{(spec.tangent_dim, spec.tangent_dim)}"
        )
    time = float(checkpoint.time)
    if (
        not np.all(np.isfinite(x))
        or not np.all(np.isfinite(P))
        or not math.isfinite(time)
    ):
        raise ValueError("filter replay checkpoint contains non-finite values")
    if not np.allclose(P, P.T, rtol=1e-10, atol=1e-12):
        raise ValueError("filter replay checkpoint covariance must be symmetric")
    if np.linalg.eigvalsh(P).min() < -1e-12:
        raise ValueError(
            "filter replay checkpoint covariance must be positive semidefinite"
        )
    return FilterCheckpoint(x, P, time, module.artifact_id)


def _validate_covariance(value: Any, dim: int, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape != (dim, dim):
        raise ValueError(f"{name} shape {array.shape} does not match {(dim, dim)}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    if not np.allclose(array, array.T, rtol=1e-10, atol=1e-12):
        raise ValueError(f"{name} must be symmetric")
    try:
        np.linalg.cholesky(array)
    except np.linalg.LinAlgError as exc:
        raise ValueError(f"{name} must be positive definite") from exc
    return array


def _validate_process_covariance(value: Any, dim: int, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape != (dim, dim):
        raise ValueError(f"{name} shape {array.shape} does not match {(dim, dim)}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    if not np.allclose(array, array.T, rtol=1e-10, atol=1e-12):
        raise ValueError(f"{name} must be symmetric")
    if np.linalg.eigvalsh(array).min() < -1e-12:
        raise ValueError(f"{name} must be positive semidefinite")
    return array


def _filter_shape(module) -> None:
    if module.hosting is not Hosting.HELD:
        raise TypeError("TargetFilterReplay requires a HELD filter Module")
    manifolds = tuple(
        field for field in module.state.fields if field.kind == "manifold"
    )
    matrices = tuple(field for field in module.state.fields if field.kind == "matrix")
    if len(manifolds) != 1 or len(matrices) != 1 or module.spec is None:
        raise TypeError(
            "TargetFilterReplay requires exactly one manifold state and one "
            "matrix covariance state"
        )
    if manifolds[0].shape != (module.spec.ambient_dim,) or matrices[0].shape != (
        module.spec.tangent_dim,
        module.spec.tangent_dim,
    ):
        raise TypeError("TargetFilterReplay filter state shapes disagree with its spec")
    try:
        entries = (module.entry("predict"), module.entry("predict_with_Q"))
    except KeyError as exc:
        raise TypeError(
            "TargetFilterReplay requires typed predict and predict_with_Q entries"
        ) from exc
    for entry in entries:
        if entry.writes != (manifolds[0].name, matrices[0].name):
            raise TypeError(
                f"TargetFilterReplay {entry.method} must write manifold then "
                "covariance state"
            )
        _validate_entry_abi(module, entry)


def _validate_entry_abi(module, entry) -> None:
    """Prove a typed entry's CasADi buffers match its declared Module ABI."""
    try:
        function = module.functions[entry.fn]
    except KeyError as exc:
        raise TypeError(
            f"TargetFilterReplay entry {entry.method!r} names missing function "
            f"{entry.fn!r}"
        ) from exc
    if function.n_in() != len(entry.args):
        raise TypeError(
            f"TargetFilterReplay entry {entry.method!r} declares "
            f"{len(entry.args)} inputs but its function has {function.n_in()}"
        )
    for index, argument in enumerate(entry.args):
        if isinstance(argument, StateRef):
            expected = int(np.prod(module.state.field(argument.name).shape))
        else:
            expected = module.port(argument.name).size
        if function.numel_in(index) != expected:
            raise TypeError(
                f"TargetFilterReplay entry {entry.method!r} input {index} has "
                f"{function.numel_in(index)} values, expected {expected}"
            )
    outputs = (*entry.writes, *entry.returns)
    if function.n_out() != len(outputs):
        raise TypeError(
            f"TargetFilterReplay entry {entry.method!r} declares {len(outputs)} "
            f"outputs but its function has {function.n_out()}"
        )
    for index, name in enumerate(outputs):
        expected = (
            int(np.prod(module.state.field(name).shape))
            if name in module.state
            else module.port(name).size
        )
        if function.numel_out(index) != expected:
            raise TypeError(
                f"TargetFilterReplay entry {entry.method!r} output {index} has "
                f"{function.numel_out(index)} values, expected {expected}"
            )


class NativeFilterReplay:
    """Generated native executor for exact sequential EKF/UKF operations.

    The object is stateless.  Each program owns its complete initial
    checkpoint and all numeric inputs, so executions can be retried, compared,
    or handed to another process without an ambient runtime clock.
    """

    def __init__(
        self,
        module,
        *,
        max_operations: int,
        max_checkpoints: int,
        max_execution_bytes: int = _DEFAULT_EXECUTION_BYTE_CAP,
        optimization: str = "runtime",
    ) -> None:
        self.module = as_module(module, "TargetFilterReplay")
        _filter_shape(self.module)
        if not isinstance(max_operations, int) or max_operations <= 0:
            raise ValueError("max_operations must be a positive integer")
        if not isinstance(max_checkpoints, int) or max_checkpoints < 0:
            raise ValueError("max_checkpoints must be a non-negative integer")
        if max_operations > _INT32_MAX or max_checkpoints > _INT32_MAX:
            raise ValueError("filter replay counts must fit the native int32 ABI")
        if max_checkpoints > max_operations:
            raise ValueError("max_checkpoints may not exceed max_operations")
        if not isinstance(max_execution_bytes, int) or max_execution_bytes <= 0:
            raise ValueError("max_execution_bytes must be a positive integer")
        if max_execution_bytes > _HARD_EXECUTION_BYTE_CAP:
            raise ValueError(
                f"max_execution_bytes may not exceed the hard native safety "
                f"ceiling {_HARD_EXECUTION_BYTE_CAP}"
            )
        if optimization not in {"balanced", "runtime"}:
            raise ValueError("optimization must be 'balanced' or 'runtime'")
        self.max_operations = max_operations
        self.max_checkpoints = max_checkpoints
        self.max_execution_bytes = max_execution_bytes
        self.optimization = optimization
        self._u_port = self.module.sole_port(Role.CONTROL)
        self._u_fields = self._u_port.fields if self._u_port is not None else ()
        self._u_dim = sum(field.dim for field in self._u_fields)
        self._input_names = tuple(field.name for field in self._u_fields)
        self._sensors = self._collect_sensors()
        self._sensor_by_name = {sensor.name: sensor for sensor in self._sensors}
        self._max_measurement_dim = max(sensor.dim for sensor in self._sensors)
        self.estimated_worst_case_bytes = self._worst_case_execution_bytes()
        if self.estimated_worst_case_bytes > max_execution_bytes:
            raise ValueError(
                "configured filter replay bounds require up to "
                f"{self.estimated_worst_case_bytes} bytes, exceeding "
                f"max_execution_bytes={max_execution_bytes}; reduce operation/"
                "checkpoint bounds or deliberately raise the byte cap"
            )
        self._native = _compile_native_runner(
            self.module,
            self._sensors,
            max_operations=max_operations,
            max_checkpoints=max_checkpoints,
            max_execution_bytes=max_execution_bytes,
            optimization=optimization,
        )

    def _worst_case_execution_bytes(self) -> int:
        """Peak owned numeric storage for one maximally populated execution.

        Inputs exist twice at the native boundary (the public immutable-ish
        program plus its validated anti-TOCTOU snapshot).  Native outputs and
        their validated/materialized result copies likewise overlap.  Python
        object headers are not counted; the hard cap retains headroom for
        those and for compiler/runtime scratch.
        """
        a = self.module.spec.ambient_dim
        p = self.module.spec.tangent_dim
        z = self._max_measurement_dim
        # Per operation: kind/sensor int32, time/dt float64, control, z, R,
        # Q, and three uint8 flags.
        input_per_operation = (
            2 * np.dtype(np.int32).itemsize
            + (2 + self._u_dim + z + z * z + p * p) * np.dtype(np.float64).itemsize
            + 3 * np.dtype(np.uint8).itemsize
        )
        diagnostic_per_operation = (z + z * z + 2) * np.dtype(np.float64).itemsize
        checkpoint_bytes = (a + p * p + 1) * np.dtype(np.float64).itemsize
        final_bytes = checkpoint_bytes
        return int(
            2 * self.max_operations * input_per_operation
            + 2 * self.max_operations * diagnostic_per_operation
            + 2 * self.max_checkpoints * checkpoint_bytes
            + 2 * final_bytes
        )

    @property
    def cache_identity(self) -> str:
        return self._native.identity

    @property
    def library_path(self) -> str:
        return self._native.path

    @property
    def sensors(self) -> tuple[str, ...]:
        return tuple(sensor.name for sensor in self._sensors)

    @property
    def input_names(self) -> tuple[str, ...]:
        return self._input_names

    def _collect_sensors(self) -> tuple[_Sensor, ...]:
        gates = self.module.metadata.get("nis_gates", {})
        out = []
        for index, port in enumerate(self.module.ports_by_role(Role.MEASUREMENT)):
            ident = entry_ident(port.name)
            diagnostic = self.module.entry(f"update_diagnostic_{ident}")
            override = self.module.entry(f"update_with_R_{ident}")
            for entry in (diagnostic, override):
                _validate_entry_abi(self.module, entry)
                if len(entry.returns) != 4:
                    raise TypeError(
                        f"TargetFilterReplay {entry.method} must return innovation, "
                        "innovation covariance, NIS, and accepted disposition"
                    )
            out.append(
                _Sensor(
                    index=index,
                    name=port.name,
                    dim=port.size,
                    gate=gates.get(port.name),
                    diagnostic_entry=diagnostic,
                    override_entry=override,
                    diagnostic_function=self.module.functions[diagnostic.fn],
                    override_function=self.module.functions[override.fn],
                )
            )
        if not out:
            raise TypeError("TargetFilterReplay requires at least one sensor")
        return tuple(out)

    def _controls(self, values: Mapping[str, Any], *, operation: int) -> np.ndarray:
        if not isinstance(values, Mapping):
            raise TypeError(f"operation {operation}: controls must be a mapping")
        unknown = set(values) - set(self._input_names)
        if unknown:
            raise KeyError(
                f"operation {operation}: unknown control(s) {sorted(unknown)}; "
                f"expected {list(self._input_names)}"
            )
        array = pack_fields(
            self._u_fields,
            values,
            default=lambda item: item.default,
            who=f"operation {operation} controls",
        )
        if not np.all(np.isfinite(array)):
            raise ValueError(
                f"operation {operation}: controls contain non-finite values"
            )
        return array

    def program(
        self,
        initial: FilterCheckpoint,
        operations: Sequence[ReplayOperation],
    ) -> FilterReplayProgram:
        """Validate and pack an already ordered operation span.

        Operation order is preserved exactly.  In particular, Manta never
        sorts same-time sensor updates.  A caller that requires accel before
        gyro must provide that order.
        """
        initial = _validate_checkpoint(self.module, initial)
        initial.x.flags.writeable = False
        initial.P.flags.writeable = False
        if isinstance(operations, (str, bytes)) or not isinstance(operations, Sequence):
            raise TypeError("filter replay operations must be a finite sequence")
        count = len(operations)
        if count == 0:
            raise ValueError("filter replay program needs at least one operation")
        if count > self.max_operations:
            raise ValueError(
                f"filter replay program has {count} operations; configured maximum "
                f"is {self.max_operations}"
            )
        n = count
        p = self.module.spec.tangent_dim
        zdim = self._max_measurement_dim
        kinds: NDArray[np.int32] = np.empty(n, dtype=np.int32)
        sensors: NDArray[np.int32] = np.full(n, -1, dtype=np.int32)
        times: NDArray[np.float64] = np.empty(n, dtype=np.float64)
        dts: NDArray[np.float64] = np.zeros(n, dtype=np.float64)
        controls: NDArray[np.float64] = np.zeros((n, self._u_dim), dtype=np.float64)
        measurements: NDArray[np.float64] = np.zeros((n, zdim), dtype=np.float64)
        measurement_covariances: NDArray[np.float64] = np.zeros(
            (n, zdim * zdim), dtype=np.float64
        )
        process_covariances: NDArray[np.float64] = np.zeros((n, p, p), dtype=np.float64)
        use_r: NDArray[np.uint8] = np.zeros(n, dtype=np.uint8)
        use_q: NDArray[np.uint8] = np.zeros(n, dtype=np.uint8)
        checkpoint_flags: NDArray[np.uint8] = np.zeros(n, dtype=np.uint8)
        logical_time = initial.time
        checkpoint_count = 0

        for index, operation in enumerate(operations):
            if not isinstance(operation, (ReplayPredict, ReplayUpdate, ReplayBoundary)):
                raise TypeError(
                    f"operation {index}: expected ReplayPredict, ReplayUpdate, "
                    f"or ReplayBoundary, got {type(operation).__name__}"
                )
            event_time = float(operation.time)
            if not math.isfinite(event_time):
                raise ValueError(f"operation {index}: time must be finite")
            if not math.isclose(event_time, logical_time, rel_tol=0.0, abs_tol=1e-12):
                relation = "nonmonotonic" if event_time < logical_time else "has a gap"
                raise ValueError(
                    f"operation {index}: {relation} in logical time; operation time "
                    f"{event_time!r} does not match current filter time "
                    f"{logical_time!r}. Insert an explicit ReplayPredict."
                )
            times[index] = event_time
            checkpoint_flags[index] = bool(operation.checkpoint)
            checkpoint_count += int(bool(operation.checkpoint))
            if checkpoint_count > self.max_checkpoints:
                raise ValueError(
                    f"filter replay program requests {checkpoint_count} checkpoints; "
                    f"configured maximum is {self.max_checkpoints}"
                )

            if isinstance(operation, ReplayBoundary):
                kinds[index] = 2
                continue
            controls[index] = self._controls(operation.controls, operation=index)
            if isinstance(operation, ReplayPredict):
                kinds[index] = 0
                dt = float(require_positive(operation.dt, name=f"operation {index} dt"))
                dts[index] = dt
                if operation.process_covariance is not None:
                    process_covariances[index] = _validate_process_covariance(
                        operation.process_covariance,
                        p,
                        name=f"operation {index} process covariance",
                    )
                    use_q[index] = 1
                logical_time = event_time + dt
                continue

            kinds[index] = 1
            sensor = self._sensor_by_name.get(operation.sensor)
            if sensor is None:
                raise KeyError(
                    f"operation {index}: unknown sensor {operation.sensor!r}; "
                    f"expected {list(self.sensors)}"
                )
            sensors[index] = sensor.index
            measurement = np.asarray(operation.measurement, dtype=float).reshape(-1)
            if measurement.shape != (sensor.dim,):
                raise ValueError(
                    f"operation {index} {sensor.name}: measurement shape "
                    f"{measurement.shape} does not match {(sensor.dim,)}"
                )
            if not np.all(np.isfinite(measurement)):
                raise ValueError(
                    f"operation {index} {sensor.name}: measurement contains "
                    "non-finite values"
                )
            measurements[index, : sensor.dim] = measurement
            if operation.measurement_covariance is not None:
                measurement_covariances[index, : sensor.dim * sensor.dim] = (
                    _validate_covariance(
                        operation.measurement_covariance,
                        sensor.dim,
                        name=f"operation {index} {sensor.name} covariance",
                    ).reshape(-1)
                )
                use_r[index] = 1

        arrays = (
            kinds,
            sensors,
            times,
            dts,
            controls,
            measurements,
            measurement_covariances,
            process_covariances,
            use_r,
            use_q,
            checkpoint_flags,
        )
        for array in arrays:
            array.flags.writeable = False
        return FilterReplayProgram(
            kernel_identity=self.cache_identity,
            initial=initial,
            operation_count=count,
            checkpoint_count=checkpoint_count,
            _kinds=kinds,
            _sensors=sensors,
            _times=times,
            _dts=dts,
            _controls=controls,
            _measurements=measurements,
            _measurement_covariances=measurement_covariances,
            _process_covariances=process_covariances,
            _use_measurement_covariance=use_r,
            _use_process_covariance=use_q,
            _checkpoint_flags=checkpoint_flags,
        )

    @staticmethod
    def _native_array(
        value: Any,
        *,
        name: str,
        dtype: np.dtype,
        shape: tuple[int, ...],
    ) -> np.ndarray:
        """Validate then copy one untrusted public-program native buffer.

        A frozen dataclass does not freeze its ndarray members, and
        ``dataclasses.replace`` can forge every field.  Native code therefore
        never receives a public array directly: exact ABI shape/type/layout is
        checked and an owned C-contiguous snapshot closes the validation/call
        race.
        """
        if not isinstance(value, np.ndarray):
            raise TypeError(f"filter replay {name} must be a numpy array")
        if value.dtype != dtype:
            raise ValueError(
                f"filter replay {name} dtype {value.dtype} does not match {dtype}"
            )
        if value.shape != shape:
            raise ValueError(
                f"filter replay {name} shape {value.shape} does not match {shape}"
            )
        if not value.flags.c_contiguous:
            raise ValueError(f"filter replay {name} must be C-contiguous")
        return value.copy(order="C")

    def _validated_native_input(self, program: FilterReplayProgram) -> dict[str, Any]:
        if type(program.operation_count) is not int:
            raise TypeError("filter replay operation_count must be an integer")
        n = program.operation_count
        if n <= 0 or n > self.max_operations:
            raise ValueError(
                f"filter replay operation_count {n} is outside configured "
                f"range [1, {self.max_operations}]"
            )
        if type(program.checkpoint_count) is not int:
            raise TypeError("filter replay checkpoint_count must be an integer")
        if not 0 <= program.checkpoint_count <= self.max_checkpoints:
            raise ValueError(
                f"filter replay checkpoint_count {program.checkpoint_count} is "
                f"outside configured range [0, {self.max_checkpoints}]"
            )
        initial = _validate_checkpoint(self.module, program.initial)
        p = self.module.spec.tangent_dim
        zdim = self._max_measurement_dim
        arrays: dict[str, Any] = {
            "kinds": self._native_array(
                program._kinds, name="kinds", dtype=np.dtype(np.int32), shape=(n,)
            ),
            "sensors": self._native_array(
                program._sensors,
                name="sensors",
                dtype=np.dtype(np.int32),
                shape=(n,),
            ),
            "times": self._native_array(
                program._times,
                name="times",
                dtype=np.dtype(np.float64),
                shape=(n,),
            ),
            "dts": self._native_array(
                program._dts,
                name="dts",
                dtype=np.dtype(np.float64),
                shape=(n,),
            ),
            "controls": self._native_array(
                program._controls,
                name="controls",
                dtype=np.dtype(np.float64),
                shape=(n, self._u_dim),
            ),
            "measurements": self._native_array(
                program._measurements,
                name="measurements",
                dtype=np.dtype(np.float64),
                shape=(n, zdim),
            ),
            "measurement_covariances": self._native_array(
                program._measurement_covariances,
                name="measurement_covariances",
                dtype=np.dtype(np.float64),
                shape=(n, zdim * zdim),
            ),
            "process_covariances": self._native_array(
                program._process_covariances,
                name="process_covariances",
                dtype=np.dtype(np.float64),
                shape=(n, p, p),
            ),
            "use_r": self._native_array(
                program._use_measurement_covariance,
                name="use_measurement_covariance",
                dtype=np.dtype(np.uint8),
                shape=(n,),
            ),
            "use_q": self._native_array(
                program._use_process_covariance,
                name="use_process_covariance",
                dtype=np.dtype(np.uint8),
                shape=(n,),
            ),
            "checkpoint_flags": self._native_array(
                program._checkpoint_flags,
                name="checkpoint_flags",
                dtype=np.dtype(np.uint8),
                shape=(n,),
            ),
        }
        for name in (
            "times",
            "dts",
            "controls",
            "measurements",
            "measurement_covariances",
            "process_covariances",
        ):
            if not np.all(np.isfinite(arrays[name])):
                raise ValueError(f"filter replay {name} contains non-finite values")
        for name in ("use_r", "use_q", "checkpoint_flags"):
            if np.any((arrays[name] != 0) & (arrays[name] != 1)):
                raise ValueError(f"filter replay {name} must contain only 0 or 1")
        actual_checkpoints = int(np.count_nonzero(arrays["checkpoint_flags"]))
        if actual_checkpoints != program.checkpoint_count:
            raise ValueError(
                f"filter replay checkpoint_count says {program.checkpoint_count}, "
                f"but flags request {actual_checkpoints}"
            )

        logical_time = initial.time
        checkpoint_times: list[tuple[int, float]] = []
        for index in range(n):
            kind = int(arrays["kinds"][index])
            if kind not in (0, 1, 2):
                raise ValueError(
                    f"filter replay operation {index}: unknown kind {kind}"
                )
            event_time = float(arrays["times"][index])
            if not math.isclose(event_time, logical_time, rel_tol=0.0, abs_tol=1e-12):
                relation = "nonmonotonic" if event_time < logical_time else "has a gap"
                raise ValueError(
                    f"filter replay operation {index}: {relation} in logical time; "
                    f"{event_time!r} != {logical_time!r}"
                )
            sensor_index = int(arrays["sensors"][index])
            if kind == 0:
                if sensor_index != -1 or arrays["use_r"][index]:
                    raise ValueError(
                        f"filter replay predict operation {index} carries update data"
                    )
                dt = float(arrays["dts"][index])
                if dt <= 0.0:
                    raise ValueError(
                        f"filter replay predict operation {index} dt must be > 0"
                    )
                if arrays["use_q"][index]:
                    _validate_process_covariance(
                        arrays["process_covariances"][index],
                        p,
                        name=f"filter replay operation {index} process covariance",
                    )
                logical_time = event_time + dt
            elif kind == 1:
                if not 0 <= sensor_index < len(self._sensors):
                    raise ValueError(
                        f"filter replay update operation {index} has unknown sensor "
                        f"index {sensor_index}"
                    )
                if arrays["use_q"][index] or arrays["dts"][index] != 0.0:
                    raise ValueError(
                        f"filter replay update operation {index} carries predict data"
                    )
                sensor = self._sensors[sensor_index]
                if arrays["use_r"][index]:
                    packed = arrays["measurement_covariances"][
                        index, : sensor.dim * sensor.dim
                    ]
                    _validate_covariance(
                        packed.reshape(sensor.dim, sensor.dim),
                        sensor.dim,
                        name=(
                            f"filter replay operation {index} {sensor.name} covariance"
                        ),
                    )
            else:
                if (
                    sensor_index != -1
                    or arrays["use_r"][index]
                    or arrays["use_q"][index]
                    or arrays["dts"][index] != 0.0
                ):
                    raise ValueError(
                        f"filter replay boundary operation {index} carries kernel data"
                    )
            if arrays["checkpoint_flags"][index]:
                checkpoint_times.append((index, logical_time))
        arrays["initial"] = initial
        arrays["final_time"] = logical_time
        arrays["checkpoint_times"] = tuple(checkpoint_times)
        return arrays

    def run(self, program: FilterReplayProgram) -> FilterReplayResult:
        if not isinstance(program, FilterReplayProgram):
            raise TypeError("run expects a FilterReplayProgram")
        if program.kernel_identity != self.cache_identity:
            raise ValueError("filter replay program was built for a different kernel")
        native_input = self._validated_native_input(program)
        n = program.operation_count
        a = self.module.spec.ambient_dim
        p = self.module.spec.tangent_dim
        zdim = self._max_measurement_dim
        final_x: NDArray[np.float64] = np.empty(a, dtype=np.float64)
        final_P: NDArray[np.float64] = np.empty((p, p), dtype=np.float64)
        final_time: NDArray[np.float64] = np.empty(1, dtype=np.float64)
        innovation: NDArray[np.float64] = np.full((n, zdim), np.nan, dtype=np.float64)
        innovation_covariance: NDArray[np.float64] = np.full(
            (n, zdim * zdim), np.nan, dtype=np.float64
        )
        nis: NDArray[np.float64] = np.full(n, np.nan, dtype=np.float64)
        accepted: NDArray[np.float64] = np.zeros(n, dtype=np.float64)
        checkpoint_x: NDArray[np.float64] = np.empty(
            (program.checkpoint_count, a), dtype=np.float64
        )
        checkpoint_P: NDArray[np.float64] = np.empty(
            (program.checkpoint_count, p, p), dtype=np.float64
        )
        checkpoint_time: NDArray[np.float64] = np.empty(
            program.checkpoint_count, dtype=np.float64
        )
        failed_operation = ctypes.c_int32(-1)
        dummy: NDArray[np.float64] = np.zeros(1, dtype=np.float64)

        def ptr(array: np.ndarray):
            return array.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

        rc = self._native.run(
            n,
            native_input["kinds"].ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            native_input["sensors"].ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            ptr(native_input["times"]),
            ptr(native_input["dts"]),
            ptr(native_input["controls"])
            if native_input["controls"].size
            else ptr(dummy),
            ptr(native_input["measurements"]),
            ptr(native_input["measurement_covariances"]),
            ptr(native_input["process_covariances"]),
            native_input["use_r"].ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            native_input["use_q"].ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            native_input["checkpoint_flags"].ctypes.data_as(
                ctypes.POINTER(ctypes.c_uint8)
            ),
            ptr(native_input["initial"].x),
            ptr(native_input["initial"].P),
            float(native_input["initial"].time),
            ptr(final_x),
            ptr(final_P),
            ptr(final_time),
            ptr(innovation),
            ptr(innovation_covariance),
            ptr(nis),
            ptr(accepted),
            ptr(checkpoint_x) if checkpoint_x.size else ptr(dummy),
            ptr(checkpoint_P) if checkpoint_P.size else ptr(dummy),
            ptr(checkpoint_time) if checkpoint_time.size else ptr(dummy),
            ctypes.byref(failed_operation),
        )
        if rc != 0:
            raise RuntimeError(
                f"native filter replay failed with status {rc} at operation "
                f"{failed_operation.value}"
            )

        if not math.isclose(
            float(final_time[0]),
            float(native_input["final_time"]),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "native filter replay returned a final logical time that differs "
                "from the validated program"
            )
        try:
            validated_final = _validate_checkpoint(
                self.module, FilterCheckpoint(
                    final_x, final_P, float(final_time[0]),
                    self.module.artifact_id)
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"native filter replay final checkpoint is invalid: {exc}"
            ) from exc

        updates = []
        for index in range(n):
            if int(native_input["kinds"][index]) != 1:
                continue
            sensor = self._sensors[int(native_input["sensors"][index])]
            if (
                not np.all(np.isfinite(innovation[index, : sensor.dim]))
                or not np.all(
                    np.isfinite(innovation_covariance[index, : sensor.dim * sensor.dim])
                )
                or not math.isfinite(float(nis[index]))
                or accepted[index] not in (0.0, 1.0)
            ):
                raise ValueError(
                    f"native filter replay returned invalid diagnostics at "
                    f"operation {index} ({sensor.name})"
                )
            native_s = innovation_covariance[index, : sensor.dim * sensor.dim].reshape(
                sensor.dim, sensor.dim
            )
            if not np.allclose(native_s, native_s.T, rtol=1e-10, atol=1e-12):
                raise ValueError(
                    f"native filter replay returned nonsymmetric innovation "
                    f"covariance at operation {index} ({sensor.name})"
                )
            updates.append(
                (
                    index,
                    UpdateResult(
                        sensor=sensor.name,
                        innovation=innovation[index, : sensor.dim].copy(),
                        innovation_covariance=native_s.copy(),
                        nis=float(nis[index]),
                        accepted=bool(accepted[index]),
                        gate=sensor.gate,
                        covariance_overridden=bool(native_input["use_r"][index]),
                    ),
                )
            )
        checkpoint_results = []
        checkpoint_slot = 0
        expected_checkpoints = dict(native_input["checkpoint_times"])
        for index, requested in enumerate(native_input["checkpoint_flags"]):
            if not requested:
                continue
            returned_time = float(checkpoint_time[checkpoint_slot])
            if not math.isclose(
                returned_time,
                expected_checkpoints[index],
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ValueError(
                    f"native filter replay checkpoint after operation {index} has "
                    "the wrong logical time"
                )
            try:
                validated_checkpoint = _validate_checkpoint(
                    self.module,
                    FilterCheckpoint(
                        checkpoint_x[checkpoint_slot],
                        checkpoint_P[checkpoint_slot],
                        returned_time,
                        self.module.artifact_id,
                    ),
                )
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"native filter replay checkpoint after operation {index} is "
                    f"invalid: {exc}"
                ) from exc
            checkpoint_results.append(
                ReplayCheckpointResult(
                    operation_index=index,
                    checkpoint=validated_checkpoint,
                )
            )
            checkpoint_slot += 1
        return FilterReplayResult(
            final=validated_final,
            updates=tuple(updates),
            checkpoints=tuple(checkpoint_results),
        )


def TargetFilterReplay(
    value,
    *,
    max_operations: int,
    max_checkpoints: int,
    max_execution_bytes: int = _DEFAULT_EXECUTION_BYTE_CAP,
    optimization: str = "runtime",
) -> NativeFilterReplay:
    """Generate a bounded exact-sequential EKF/UKF replay target."""
    return NativeFilterReplay(
        value,
        max_operations=max_operations,
        max_checkpoints=max_checkpoints,
        max_execution_bytes=max_execution_bytes,
        optimization=optimization,
    )


def _kernel_call(
    function: _FunctionABI,
    args: list[str],
    results: list[str],
    *,
    indent: str,
) -> list[str]:
    function_name = function.name
    lines = [
        f"{indent}const double* arg[{max(function.arg_slots, 1)}] = {{0}};",
        *[f"{indent}arg[{index}] = {value};" for index, value in enumerate(args)],
        f"{indent}double* res[{max(function.result_slots, 1)}] = {{0}};",
        *[f"{indent}res[{index}] = {value};" for index, value in enumerate(results)],
        f"{indent}long long iw[{max(function.integer_work, 1)}];",
        f"{indent}double w[{max(function.real_work, 1)}];",
        f"{indent}int kernel_status = {function_name}(arg, res, iw, w, 0);",
        (
            f"{indent}if (kernel_status != 0) {{ *failed_operation = op; "
            "return 100 + kernel_status; }"
        ),
    ]
    return lines


def _runner_source(
    module,
    sensors: tuple[_Sensor, ...],
    functions: Mapping[str, _FunctionABI],
    *,
    max_operations: int,
    max_checkpoints: int,
    max_execution_bytes: int,
) -> str:
    ambient = module.spec.ambient_dim
    tangent = module.spec.tangent_dim
    control = sum(field.dim for field in module.sole_port(Role.CONTROL).fields)
    max_z = max(sensor.dim for sensor in sensors)
    predict = module.entry("predict")
    predict_q = module.entry("predict_with_Q")

    def predict_args(entry, *, override: bool) -> list[str]:
        args = []
        for argument in entry.args:
            if isinstance(argument, StateRef):
                field = module.state.field(argument.name)
                args.append("x" if field.kind == "manifold" else "P")
                continue
            port = module.port(argument.name)
            if port.role is Role.MATRIX:
                if not override:
                    raise TypeError("default predict unexpectedly requires a matrix")
                args.append(f"process_covariances + op * {tangent * tangent}")
            elif port.role is Role.CONTROL:
                args.append(f"controls + op * {control}")
            elif port.role is Role.TIMESTEP:
                args.append("dts + op")
            elif port.role is Role.TIME:
                args.append("times + op")
            else:
                raise TypeError(f"unsupported predict argument role {port.role.name}")
        return args

    def update_args(entry, sensor: _Sensor, *, override: bool) -> list[str]:
        args = []
        for argument in entry.args:
            if isinstance(argument, StateRef):
                field = module.state.field(argument.name)
                args.append("x" if field.kind == "manifold" else "P")
                continue
            port = module.port(argument.name)
            if port.role is Role.MEASUREMENT:
                args.append(f"measurements + op * {max_z}")
            elif port.role is Role.MATRIX:
                if not override:
                    raise TypeError("default update unexpectedly requires a matrix")
                args.append(f"measurement_covariances + op * {max_z * max_z}")
            elif port.role is Role.CONTROL:
                args.append(f"controls + op * {control}")
            elif port.role is Role.TIME:
                args.append("times + op")
            else:
                raise TypeError(f"unsupported update argument role {port.role.name}")
        return args

    lines = [
        "",
        f"/* configured peak execution byte cap: {max_execution_bytes} */",
        "#include <stdint.h>",
        "#include <string.h>",
        "",
        "#if defined(_WIN32)",
        "#define MANTA_REPLAY_EXPORT __declspec(dllexport)",
        "#else",
        '#define MANTA_REPLAY_EXPORT __attribute__((visibility("default")))',
        "#endif",
        "",
        "MANTA_REPLAY_EXPORT int manta_filter_replay_run(",
        "    int32_t operation_count, const int32_t* kinds,",
        "    const int32_t* sensors, const double* times, const double* dts,",
        "    const double* controls, const double* measurements,",
        "    const double* measurement_covariances,",
        "    const double* process_covariances, const uint8_t* use_r,",
        "    const uint8_t* use_q, const uint8_t* checkpoint_flags,",
        "    const double* initial_x, const double* initial_P, double initial_time,",
        "    double* final_x, double* final_P, double* final_time,",
        "    double* innovations, double* innovation_covariances,",
        "    double* nis, double* accepted, double* checkpoint_x,",
        "    double* checkpoint_P, double* checkpoint_time,",
        "    int32_t* failed_operation) {",
        f"    if (operation_count <= 0 || operation_count > {max_operations}) return 1;",
        f"    double x[{ambient}], P[{tangent * tangent}];",
        f"    double x_next[{ambient}], P_next[{tangent * tangent}];",
        f"    memcpy(x, initial_x, sizeof(double) * {ambient});",
        f"    memcpy(P, initial_P, sizeof(double) * {tangent * tangent});",
        "    double logical_time = initial_time;",
        "    int32_t checkpoint_slot = 0;",
        "    *failed_operation = -1;",
        "    for (int32_t op = 0; op < operation_count; ++op) {",
        "        if (kinds[op] == 0) {",
        "            if (use_q[op]) {",
        "                {",
    ]
    lines.extend(
        _kernel_call(
            functions["predict_with_Q"],
            predict_args(predict_q, override=True),
            ["x_next", "P_next"],
            indent="                    ",
        )
    )
    lines.extend(
        [
            "                }",
            "            } else {",
            "                {",
        ]
    )
    lines.extend(
        _kernel_call(
            functions["predict"],
            predict_args(predict, override=False),
            ["x_next", "P_next"],
            indent="                    ",
        )
    )
    lines.extend(
        [
            "                }",
            "            }",
            f"            memcpy(x, x_next, sizeof(double) * {ambient});",
            f"            memcpy(P, P_next, sizeof(double) * {tangent * tangent});",
            "            logical_time = times[op] + dts[op];",
            "        } else if (kinds[op] == 1) {",
            "            switch (sensors[op]) {",
        ]
    )
    for sensor in sensors:
        common_results = [
            "x_next",
            "P_next",
            f"innovations + op * {max_z}",
            f"innovation_covariances + op * {max_z * max_z}",
            "nis + op",
            "accepted + op",
        ]
        lines.extend(
            [
                f"            case {sensor.index}:",
                "                if (use_r[op]) {",
                "                    {",
            ]
        )
        lines.extend(
            _kernel_call(
                functions[f"sensor_{sensor.index}_override"],
                update_args(sensor.override_entry, sensor, override=True),
                common_results,
                indent="                        ",
            )
        )
        lines.extend(
            [
                "                    }",
                "                } else {",
                "                    {",
            ]
        )
        lines.extend(
            _kernel_call(
                functions[f"sensor_{sensor.index}_diagnostic"],
                update_args(sensor.diagnostic_entry, sensor, override=False),
                common_results,
                indent="                        ",
            )
        )
        lines.extend(
            [
                "                    }",
                "                }",
                f"                memcpy(x, x_next, sizeof(double) * {ambient});",
                f"                memcpy(P, P_next, sizeof(double) * {tangent * tangent});",
                "                break;",
            ]
        )
    lines.extend(
        [
            "            default: *failed_operation = op; return 2;",
            "            }",
            "        } else if (kinds[op] != 2) {",
            "            *failed_operation = op; return 3;",
            "        }",
            "        if (checkpoint_flags[op]) {",
            f"            if (checkpoint_slot >= {max_checkpoints}) {{",
            "                *failed_operation = op; return 4;",
            "            }",
            f"            memcpy(checkpoint_x + checkpoint_slot * {ambient}, x,",
            f"                   sizeof(double) * {ambient});",
            f"            memcpy(checkpoint_P + checkpoint_slot * {tangent * tangent}, P,",
            f"                   sizeof(double) * {tangent * tangent});",
            "            checkpoint_time[checkpoint_slot] = logical_time;",
            "            ++checkpoint_slot;",
            "        }",
            "    }",
            f"    memcpy(final_x, x, sizeof(double) * {ambient});",
            f"    memcpy(final_P, P, sizeof(double) * {tangent * tangent});",
            "    *final_time = logical_time;",
            "    return 0;",
            "}",
            "",
        ]
    )
    return "\n".join(lines)


def _compile_native_runner(
    module,
    sensors: tuple[_Sensor, ...],
    *,
    max_operations: int,
    max_checkpoints: int,
    max_execution_bytes: int,
    optimization: str,
) -> _NativeLibrary:
    compiler = shutil.which("cc")
    if compiler is None:
        raise CompilationError(
            "native filter replay requested but no 'cc' compiler is on PATH"
        )
    functions: dict[str, ca.Function] = {}
    function_abis: dict[str, _FunctionABI] = {}

    def add(key: str, function: ca.Function) -> None:
        # Re-wrap under target-owned, collision-free names.  Densification is
        # required for the flat C arrays consumed by the runner.
        name = f"mfr_{key}"
        inputs = [
            ca.MX.sym(function.name_in(i), function.size1_in(i), function.size2_in(i))
            for i in range(function.n_in())
        ]
        output = function(*inputs)
        outputs = [output] if function.n_out() == 1 else list(output)
        dense = [ca.densify(value) for value in outputs]
        wrapped = ca.Function(
            name,
            inputs,
            dense,
            [function.name_in(i) for i in range(function.n_in())],
            [function.name_out(i) for i in range(function.n_out())],
        )
        functions[key] = wrapped
        function_abis[key] = _FunctionABI(
            name=name,
            arg_slots=wrapped.sz_arg(),
            result_slots=wrapped.sz_res(),
            integer_work=wrapped.sz_iw(),
            real_work=wrapped.sz_w(),
        )

    add("predict", module.functions[module.entry("predict").fn])
    add("predict_with_Q", module.functions[module.entry("predict_with_Q").fn])
    for sensor in sensors:
        add(f"sensor_{sensor.index}_diagnostic", sensor.diagnostic_function)
        add(f"sensor_{sensor.index}_override", sensor.override_function)

    try:
        build_dir = tempfile.mkdtemp(prefix="manta-replay-")
    except OSError as exc:
        raise CompilationError(
            f"cannot create replay compilation workspace: {exc}"
        ) from exc
    try:
        generator = ca.CodeGenerator(
            "filter_replay.c",
            {"with_header": True, "with_mem": False, "verbose": False},
        )
        for function in functions.values():
            generator.add(function)
        try:
            generator.generate(build_dir + os.sep)
            source_path = os.path.join(build_dir, "filter_replay.c")
            with open(source_path, encoding="utf-8") as source_file:
                source = source_file.read()
        except (OSError, RuntimeError) as exc:
            raise CompilationError(
                f"native replay code generation failed: {exc}"
            ) from exc
        source += _runner_source(
            module,
            sensors,
            function_abis,
            max_operations=max_operations,
            max_checkpoints=max_checkpoints,
            max_execution_bytes=max_execution_bytes,
        )
        flags = ("-O3", "-march=native") if optimization == "runtime" else ("-O1",)
        identity = hashlib.sha256(
            source.encode()
            + b"\0"
            + "\0".join(flags).encode()
            + b"\0"
            + platform.platform().encode()
            + b"\0native-filter-replay-v1"
        ).hexdigest()
        with _NATIVE_CACHE_LOCK:
            cached = _NATIVE_CACHE.get(identity)
            if cached is not None:
                return cached
            cache_dir = os.path.join(_cache_dir(), "filter_replay")
            try:
                os.makedirs(cache_dir, mode=0o700, exist_ok=True)
            except OSError as exc:
                raise CompilationError(
                    f"cannot create native replay cache {cache_dir!r}: {exc}. "
                    "Set XDG_CACHE_HOME to a writable directory"
                ) from exc
            library_path = os.path.join(cache_dir, f"replay_{identity[:20]}.so")
            if not os.path.exists(library_path):
                with open(source_path, "w", encoding="utf-8") as source_file:
                    source_file.write(source)
                private_path = os.path.join(build_dir, "filter_replay.so")
                try:
                    subprocess.run(
                        [
                            compiler,
                            *flags,
                            "-fPIC",
                            "-shared",
                            source_path,
                            "-o",
                            private_path,
                        ],
                        check=True,
                        capture_output=True,
                        timeout=180,
                    )
                except subprocess.CalledProcessError as exc:
                    stderr = (
                        exc.stderr.decode(errors="replace")
                        if isinstance(exc.stderr, bytes)
                        else (exc.stderr or "")
                    )
                    raise CompilationError(
                        f"native replay compilation failed: "
                        f"{stderr.strip() or 'no compiler output'}"
                    ) from exc
                except subprocess.TimeoutExpired as exc:
                    raise CompilationError(
                        f"native replay compilation exceeded {exc.timeout} seconds"
                    ) from exc
                publish_fd = -1
                publish_path = ""
                try:
                    publish_fd, publish_path = tempfile.mkstemp(
                        prefix=f".{identity[:20]}-", suffix=".so", dir=cache_dir
                    )
                    os.close(publish_fd)
                    publish_fd = -1
                    shutil.copyfile(private_path, publish_path)
                    os.replace(publish_path, library_path)
                    publish_path = ""
                finally:
                    if publish_fd >= 0:
                        os.close(publish_fd)
                    if publish_path:
                        try:
                            os.unlink(publish_path)
                        except OSError:
                            pass
            try:
                library = ctypes.CDLL(library_path)
                run = library.manta_filter_replay_run
            except (OSError, AttributeError) as exc:
                raise CompilationError(
                    f"native replay library {library_path!r} could not be loaded: {exc}"
                ) from exc
            double_pointer = ctypes.POINTER(ctypes.c_double)
            int_pointer = ctypes.POINTER(ctypes.c_int32)
            byte_pointer = ctypes.POINTER(ctypes.c_uint8)
            run.argtypes = [
                ctypes.c_int32,
                int_pointer,
                int_pointer,
                double_pointer,
                double_pointer,
                double_pointer,
                double_pointer,
                double_pointer,
                double_pointer,
                byte_pointer,
                byte_pointer,
                byte_pointer,
                double_pointer,
                double_pointer,
                ctypes.c_double,
                double_pointer,
                double_pointer,
                double_pointer,
                double_pointer,
                double_pointer,
                double_pointer,
                double_pointer,
                double_pointer,
                double_pointer,
                double_pointer,
                int_pointer,
            ]
            run.restype = ctypes.c_int
            compiled = _NativeLibrary(identity=identity, path=library_path, run=run)
            _NATIVE_CACHE[identity] = compiled
            return compiled
    finally:
        shutil.rmtree(build_dir, ignore_errors=True)


__all__ = [
    "FilterReplayProgram",
    "FilterReplayResult",
    "NativeFilterReplay",
    "ReplayBoundary",
    "ReplayCheckpointResult",
    "ReplayOperation",
    "ReplayPredict",
    "ReplayUpdate",
    "TargetFilterReplay",
]

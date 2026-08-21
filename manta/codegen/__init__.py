"""manta backends (codegen targets).

The pipeline:

  1. **Model**: declarative — `World`, `Craft`, fields, parts, planets.
  2. **Transform**: `Sim(world)`, `EKF(world)`, `LQR(world)`, a recurrence
     block — each owns its math and emits a typed `Module`
     (`manta.ir.module`) via `.module()`.
  3. **Target**: lowers a Module to a backend. `TargetNumpy(x)` → the
     native-Python view matching the Module's shape (one kernel engine,
     thin views for sim / filter / recurrence / regulator);
     `TargetCpp(x, …)` → a buildable C++ library via the one generic
     emitter.

Each backend is a self-contained subpackage:

    codegen/
      target.py   as_module — the backend entry-point contract
      numpy/      TargetNumpy + the kernel engine + views + NoiseDriver
      cpp/        TargetCpp + module_emit (the generic emitter)
      wasm/       TargetWasm + the C-ABI shim + JS runtime (browser bundle)
      jax/        TargetJax + the SX→JAX translator (lazy: needs `jax`)
      <future>/   new languages slot in here

Adding a backend = translate a `ca.Function` + one generic lowering of a
`Module`. No per-transform code, anywhere.
"""

from .cpp import TargetCpp
from .numpy import (
    DEFAULT_COMPILATION_TIMEOUT_S,
    CompilationError,
    FilterCheckpoint,
    FilterReplayProgram,
    FilterReplayResult,
    NativeFilterReplay,
    NoiseCheckpoint,
    NoiseDriver,
    NumpyFilter,
    NumpyRecurrence,
    NumpyRegulator,
    NumpyRuntime,
    NumpySim,
    ReplayBoundary,
    ReplayCheckpointResult,
    ReplayOperation,
    ReplayPredict,
    ReplayUpdate,
    SimCheckpoint,
    TargetFilterReplay,
    TargetNumpy,
    UpdateResult,
    compile_functions,
)
from .wasm import TargetWasm


def TargetJax(x):
    """Lower a Module (or any transform exposing `.module()`) to jitted
    JAX kernels by name + a `lax.scan` rollout builder — the functional
    artifacts a training loop wants (see `manta.codegen.jax.JaxModule`).
    `jax` is imported only when this is called, so manta itself never
    requires it."""
    from .jax import TargetJax as _TargetJax
    return _TargetJax(x)

__all__ = [
    "DEFAULT_COMPILATION_TIMEOUT_S",
    "CompilationError",
    "FilterCheckpoint",
    "FilterReplayProgram",
    "FilterReplayResult",
    "NativeFilterReplay",
    "NoiseCheckpoint",
    "NoiseDriver",
    "NumpyFilter",
    "NumpyRecurrence",
    "NumpyRegulator",
    "NumpyRuntime",
    "NumpySim",
    "ReplayBoundary",
    "ReplayCheckpointResult",
    "ReplayOperation",
    "ReplayPredict",
    "ReplayUpdate",
    "SimCheckpoint",
    "TargetCpp",
    "TargetFilterReplay",
    "TargetJax",
    "TargetNumpy",
    "TargetWasm",
    "UpdateResult",
    "compile_functions",
]

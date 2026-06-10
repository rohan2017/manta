"""manta backends (codegen targets).

The pipeline:

  1. **Model**: declarative — `World`, `Craft`, fields, parts, planets.
  2. **Transform**: `Sim(world)`, `EKF(world)`, `LQR(world)`, a recurrence
     block — each owns its math and emits a typed `Module`
     (`manta.ir.module`) via `.module()`.
  3. **Target**: lowers a Module to a backend. `TargetNumpy(x)` → the
     native-Python view matching the Module's shape (one kernel engine,
     four thin views); `TargetCpp(x, …)` → a buildable C++ library via
     the one generic emitter.

Each backend is a self-contained subpackage:

    codegen/
      target.py   as_module — the backend entry-point contract
      numpy/      TargetNumpy + the kernel engine + views + NoiseDriver
      cpp/        TargetCpp + module_emit (the generic emitter)
      jax/        TargetJax + the SX→JAX translator (lazy: needs `jax`)
      <future>/   new languages slot in here

Adding a backend = translate a `ca.Function` + one generic lowering of a
`Module`. No per-transform code, anywhere.
"""

from .cpp import TargetCpp


def TargetJax(x):
    """Lower a Module (or transform) to jitted JAX kernels — see
    `manta.codegen.jax`. Lazy shim: `jax` is imported only when used,
    so manta itself never requires it."""
    from .jax import TargetJax as _TargetJax
    return _TargetJax(x)
from .numpy import (
    NoiseDriver, NumpyFilter, NumpyRecurrence, NumpyRegulator, NumpyRuntime,
    NumpySim, TargetNumpy,
)

__all__ = [
    "TargetNumpy", "TargetCpp", "TargetJax", "NoiseDriver",
    "NumpyRuntime", "NumpySim", "NumpyFilter", "NumpyRecurrence",
    "NumpyRegulator",
]

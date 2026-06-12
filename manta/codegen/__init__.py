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
from .numpy import (
    NoiseDriver, NumpyFilter, NumpyRecurrence, NumpyRegulator, NumpyRuntime,
    NumpySim, TargetNumpy,
)


def TargetJax(x):
    """Lower a Module (or any transform exposing `.module()`) to jitted
    JAX kernels by name + a `lax.scan` rollout builder — the functional
    artifacts a training loop wants (see `manta.codegen.jax.JaxModule`).
    `jax` is imported only when this is called, so manta itself never
    requires it."""
    from .jax import TargetJax as _TargetJax
    return _TargetJax(x)

__all__ = [
    "TargetNumpy", "TargetCpp", "TargetJax", "NoiseDriver",
    "NumpyRuntime", "NumpySim", "NumpyFilter", "NumpyRecurrence",
    "NumpyRegulator",
]

"""manta backends (codegen targets).

The pipeline:

  1. **Model**: declarative — `World`, `Craft`, fields, parts, planets.
  2. **Transform**: `Sim(world)`, `EKF(world)`, `LQR(world)`, a recurrence
     block — each owns its math and emits a typed `Module`
     (`manta.ir.module`) via `.module()`.
  3. **Target**: lowers a Module to a backend. `TargetNumpy(x)` → the one
     native-Python `NumpyRuntime`; `TargetCpp(x, …)` → a buildable C++
     library via the one generic emitter.

Each backend is a self-contained subpackage:

    codegen/
      target.py   as_module — the backend entry-point contract
      numpy/      TargetNumpy + NumpyRuntime + NoiseDriver
      cpp/        TargetCpp + module_emit (the generic emitter)
      <future>/   new languages slot in here

Adding a backend = translate a `ca.Function` + one generic lowering of a
`Module`. No per-transform code, anywhere.
"""

from .cpp import TargetCpp
from .numpy import NoiseDriver, NumpyRuntime, TargetNumpy

__all__ = ["TargetNumpy", "TargetCpp", "NumpyRuntime", "NoiseDriver"]

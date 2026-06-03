"""manta backends (codegen targets).

The pipeline:

  1. **Model**: declarative — `World`, `Craft`, `EKF`, fields, parts,
     planets.
  2. **IR**: result of compiling the model. `Sim(world)` returns
     a `Sim` (a CasADi function + metadata); `EKF(world)`
     returns an `EKF` object that holds the symbolic predict /
     measurement bundle. These are *descriptions*, not runtimes.
  3. **Target**: lowers an IR to a backend-specific runtime. The
     native-Python `TargetNumpy(...)` is the default. `TargetCpp(...)`
     emits C++ source for embedding in firmware / other binaries.

Each backend is a self-contained subpackage under `manta.codegen`:

    codegen/
      module_build.py  to_module(block) → the backend-neutral Module IR
      numpy/           TargetNumpy + NumpyModule (the generic runtime)
      cpp/             TargetCpp + module_emit (the generic emitter)
      <future>/        new languages slot in here

Adding a backend (TensorFlow, raw embedded C, CUDA, …) is one new
`Target*` that translates a `ca.Function` + implements one generic
`lower_module(Module)` — no per-feature code.
"""

from .target import Target
from .cpp    import TargetCpp
from .numpy  import (
    NoiseDriver, NumpyEKF, NumpyRecurrence, NumpyWorld, TargetNumpy,
)

__all__ = ["Target", "TargetNumpy", "TargetCpp",
           "NumpyWorld", "NumpyEKF", "NumpyRecurrence", "NoiseDriver"]

"""manta_next backends (Targets).

The pipeline:

  1. **Model**: declarative — `World`, `Craft`, `EKF`, fields, parts,
     planets.
  2. **IR**: result of compiling the model. `World.compile()` returns
     a `CompiledWorld` (a CasADi function + metadata); `EKF(world)`
     returns an `EKF` object that holds the symbolic predict /
     measurement bundle. These are *descriptions*, not runtimes.
  3. **Target**: lowers an IR to a backend-specific runtime. The
     native-Python `TargetNumpy(...)` is the default. Future
     `TargetCpp(...)` emits C++ source for embedding in firmware /
     other binaries.

The split is explicit on purpose. The model and IR don't carry runtime
state; you choose a target and the target hands you a callable runtime
(`NumpyWorld`, `NumpyEKF`) you `.step()`, `.predict()`, `.update()` on.
"""

from .numpy import NumpyEKF, NumpyWorld, TargetNumpy

__all__ = ["TargetNumpy", "NumpyWorld", "NumpyEKF"]

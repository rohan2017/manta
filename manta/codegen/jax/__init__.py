"""TargetJax — lower a Module to jitted JAX functions.

Requires `jax` (not a core manta dependency); importing this package
enables float64 (`jax_enable_x64`). See `_runtime.JaxModule` for the
surface and `_translate` for the SX-instruction → JAX codegen.
"""

from ._runtime import JaxModule as TargetJax

__all__ = ["TargetJax"]

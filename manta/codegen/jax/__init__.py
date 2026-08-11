"""TargetJax — lower a Module to jitted JAX functions.

Requires `jax` (not a core manta dependency). Importing has no side
effect; float64 (`jax_enable_x64`) is enabled — with a warning — when
the first kernel is *built*, since manta kernels are float64 physics
that float32 tracing would silently corrupt (`_translate._require_x64`).
See `_runtime.JaxModule` for the surface and `_translate` for the
SX-instruction → JAX codegen.
"""

from ._runtime import JaxModule as TargetJax

__all__ = ["TargetJax"]

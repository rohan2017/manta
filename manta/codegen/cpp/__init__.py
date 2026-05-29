"""C++ codegen backend.

`TargetCpp(cw, out_dir, class_name=...)` lowers a `Sim` IR
to a buildable C++ static library project on disk. The emitted class
exposes the world's predict + per-Output measurement functions as
typed Eigen-shaped methods sitting on top of CasADi-generated flat-C
kernels.

Internals:
    extract.py  — Sim → per-function ca.Function objects
    kernels.py  — ca.Function → flat C source
    wrapper.py  — typed Eigen-shaped C++ class
    cmake.py    — CMakeLists.txt fragment
    emit.py     — `_emit_world_cpp()` orchestration (TargetCpp's body)

Today: World only. EKF lowering is a follow-up (the EKF IR carries
the same per-sensor h/H bundles; the wrapper class just needs an
extra layer of mutable state + Joseph-form update).
"""

from __future__ import annotations

from pathlib import Path

from .emit import _emit_world_cpp


def TargetCpp(ir,
              out_dir: str | Path,
              *,
              class_name: str,
              basename: str | None = None,
              namespace: str = "manta_gen"):
    """C++ codegen target.

    Args:
        ir          — `Sim` (today). EKF support is a
                      follow-up.
        out_dir     — destination directory (created if missing).
        class_name  — C++ class name. Conventionally PascalCase.
        basename    — filename stem; defaults to `class_name.lower()`.
        namespace   — C++ namespace enclosing the emitted class.

    Returns:
        `EmitResult` (from `manta.codegen.cpp.emit`) with paths to
        every emitted file plus the `WorldFunctions` bundle.
    """
    from ...world import Sim
    if isinstance(ir, Sim):
        return _emit_world_cpp(ir, out_dir, class_name=class_name,
                               basename=basename, namespace=namespace)
    raise TypeError(
        f"TargetCpp: no handler for IR type {type(ir).__name__}. "
        f"Expected Sim.")


__all__ = ["TargetCpp"]

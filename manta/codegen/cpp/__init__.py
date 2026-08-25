"""C++ codegen backend.

`TargetCpp(x, out_dir, class_name=...)` lowers a typed `Module` — or any
transform exposing `.module()` (`Sim`, `EKF`, `LQR`, a recurrence block) —
to a buildable C++ static-library project on disk, through the ONE generic
emitter. The backend dispatches only on the IR's types (`StateRef`/
`PortRef`, `Port.role`, `Module.hosting`); it contains no per-transform
code.

Internals:
    module_emit.py — the generic `emit_module_cpp(Module)` → typed class
    emit.py        — `emit_module(x)`: as_module → emit_module_cpp → result
    kernels.py     — ca.Function → flat C source
    _structs.py    — shared State pack/unpack emit
    _casadi.py     — densify + flat-C kernel-call boilerplate
    types.py       — Manifold.kind → Eigen type registry
    cmake.py       — CMakeLists.txt fragment
"""

from __future__ import annotations

from pathlib import Path

from .emit import EmitResult, emit_module


def TargetCpp(x,
              out_dir: str | Path,
              *,
              class_name: str,
              basename: str | None = None,
              namespace: str = "manta_gen") -> EmitResult:
    """C++ codegen target.

    Args:
        x           — a `Module`, or a transform with `.module()`.
        out_dir     — destination directory (created if missing).
        class_name  — C++ class name. Conventionally PascalCase.
        basename    — filename stem; defaults to `class_name.lower()`.
        namespace   — C++ namespace enclosing the emitted class.

    Returns:
        `EmitResult` with paths to every emitted file plus a small `funcs`
        summary (world_name / dims).
    """
    return emit_module(x, out_dir, class_name=class_name,
                       basename=basename, namespace=namespace)


__all__ = ["EmitResult", "TargetCpp"]

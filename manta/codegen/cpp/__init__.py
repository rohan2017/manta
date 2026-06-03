"""C++ codegen backend.

`TargetCpp(ir, out_dir, class_name=...)` lowers any IR transform to a
buildable C++ static-library project on disk, all on the same flat-C
kernels + typed Eigen wrapper machinery:

  * evaluator blocks (`Sim`, `LQR`, recurrence) → one generic typed class
    with a method per entry point (Sim: predict + per-Output measurement +
    Jacobians; LQR: `control(x)`; recurrence: stateful `step`).
  * `EKF` → a stateful filter: mutable state `x` + covariance `P`, with
            `predict` (`F P Fᵀ + L Σ Lᵀ`) and per-sensor Joseph-form
            `update_*` (runtime `R = L_h Σ L_hᵀ`, manifold correction via
            a `boxplus` kernel).

Internals:
    module_emit.py — the ONE generic `emit_module_cpp(Module)` → typed class
    emit.py        — `emit_module(block)`: to_module → emit_module_cpp → result
    kernels.py     — ca.Function → flat C source
    _structs.py    — shared State pack/unpack emit
    _casadi.py     — densify + flat-C kernel-call boilerplate
    types.py       — Manifold.kind → Eigen type registry
    cmake.py       — CMakeLists.txt fragment
"""

from __future__ import annotations

from pathlib import Path

from ..target import Target
from .emit import emit_module


class _CppBackend(Target):
    """C++ backend: every IR block lowers through the ONE generic Module
    path — `to_module(block)` → `emit_module_cpp` (`module_emit.py`) — to a
    typed Eigen class. No per-shape code: Sim/EKF/LQR/recurrence all flow
    through `lower_module`. The `lower_evaluator`/`lower_ekf` handlers exist
    only so the `RUNTIME_KIND` dispatch (see `manta.codegen.target`) keeps
    working; both just call the generic emitter."""

    name = "TargetCpp"

    def lower_evaluator(self, block, *, out_dir, class_name,
                        basename=None, namespace="manta_gen", **opts):
        return emit_module(block, out_dir, class_name=class_name,
                           basename=basename, namespace=namespace)

    def lower_ekf(self, ekf, *, out_dir, class_name,
                  basename=None, namespace="manta_gen", **opts):
        return emit_module(ekf, out_dir, class_name=class_name,
                           basename=basename, namespace=namespace)


def TargetCpp(ir,
              out_dir: str | Path,
              *,
              class_name: str,
              basename: str | None = None,
              namespace: str = "manta_gen"):
    """C++ codegen target.

    Args:
        ir          — a `Sim`, `EKF`, `LQR`, or recurrence-block IR.
        out_dir     — destination directory (created if missing).
        class_name  — C++ class name. Conventionally PascalCase.
        basename    — filename stem; defaults to `class_name.lower()`.
        namespace   — C++ namespace enclosing the emitted class.

    Returns:
        `EmitResult` (from `manta.codegen.cpp.emit`) with paths to every
        emitted file plus a small `funcs` summary (world_name / dims).
    """
    return _CppBackend().lower_block(ir, out_dir=out_dir, class_name=class_name,
                                     basename=basename, namespace=namespace)


__all__ = ["TargetCpp", "_CppBackend"]

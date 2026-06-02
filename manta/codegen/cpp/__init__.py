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
    extract.py / extract_ekf.py — IR → per-function ca.Function objects
    kernels.py                  — ca.Function → flat C source
    _structs.py                 — shared State/Inputs + pack/unpack emit
    evaluator_spec.py           — IR → backend-neutral EvaluatorSpec
    evaluator_wrapper.py        — EvaluatorSpec → one generic Eigen class
    ekf_wrapper.py              — the EKF typed Eigen class
    wrapper.py                  — back-compat shim (Sim `emit_wrapper`)
    cmake.py                    — CMakeLists.txt fragment
    emit.py                     — `_emit_{evaluator,ekf}_cpp()` orchestration
"""

from __future__ import annotations

from pathlib import Path

from ..target import Target
from .emit import _emit_evaluator_cpp, _emit_ekf_cpp


class _CppBackend(Target):
    """C++ backend. Two runtime kinds: `evaluator` (Sim, LQR, and the
    recurrence blocks — all pure ``ca.Function`` evaluators, lowered by one
    generic typed-entry-point emitter) and `ekf` (a stateful filter: mutable
    state + Joseph update, native Eigen linear algebra). A block whose kind
    has no handler fails loudly at `lower_block` (see
    `manta.codegen.target`)."""

    name = "TargetCpp"

    def lower_evaluator(self, block, *, out_dir, class_name,
                        basename=None, namespace="manta_gen", **opts):
        return _emit_evaluator_cpp(block, out_dir, class_name=class_name,
                                   basename=basename, namespace=namespace)

    def lower_ekf(self, ekf, *, out_dir, class_name,
                  basename=None, namespace="manta_gen", **opts):
        return _emit_ekf_cpp(ekf, out_dir, class_name=class_name,
                             basename=basename, namespace=namespace)


def TargetCpp(ir,
              out_dir: str | Path,
              *,
              class_name: str,
              basename: str | None = None,
              namespace: str = "manta_gen"):
    """C++ codegen target.

    Args:
        ir          — a `Sim`, `EKF`, or `LQR` IR.
        out_dir     — destination directory (created if missing).
        class_name  — C++ class name. Conventionally PascalCase.
        basename    — filename stem; defaults to `class_name.lower()`.
        namespace   — C++ namespace enclosing the emitted class.

    Returns:
        `EmitResult` (from `manta.codegen.cpp.emit`) with paths to
        every emitted file plus the `WorldFunctions` bundle.
    """
    return _CppBackend().lower_block(ir, out_dir=out_dir, class_name=class_name,
                                     basename=basename, namespace=namespace)


__all__ = ["TargetCpp", "_CppBackend"]

"""C++ codegen backend.

`TargetCpp(ir, out_dir, class_name=...)` lowers any IR transform to a
buildable C++ static-library project on disk, all on the same flat-C
kernels + typed Eigen wrapper machinery:

  * `Sim` → a pure evaluator (predict + per-Output measurement).
  * `EKF` → a stateful filter: mutable state `x` + covariance `P`, with
            `predict` (`F P Fᵀ + L Σ Lᵀ`) and per-sensor Joseph-form
            `update_*` (runtime `R = L_h Σ L_hᵀ`, manifold correction via
            a `boxplus` kernel).
  * `LQR` → a stateless feed-forward `control(x)` over the baked law.

Internals:
    extract.py / extract_ekf.py — IR → per-function ca.Function objects
    kernels.py                  — ca.Function → flat C source
    _structs.py                 — shared State/Inputs + pack/unpack emit
    wrapper.py / ekf_wrapper.py / lqr_wrapper.py — typed Eigen classes
    cmake.py                    — CMakeLists.txt fragment
    emit.py                     — `_emit_{world,ekf,lqr}_cpp()` orchestration
"""

from __future__ import annotations

from pathlib import Path

from ..target import Target
from .emit import (
    _emit_world_cpp, _emit_ekf_cpp, _emit_lqr_cpp, _emit_recurrence_cpp,
)


class _CppBackend(Target):
    """C++ backend. Lowers `Sim` to a pure evaluator, `EKF` to a stateful
    filter (mutable state + Joseph update), and `LQR` to a stateless
    feed-forward `control(x)` law — all on the same flat-C kernels +
    Eigen-typed wrapper machinery. Each `lower_<kind>` handler is the
    backend's support for one runtime kind; a block whose kind has no
    handler fails loudly at `lower_block` (see `manta.codegen.target`)."""

    name = "TargetCpp"

    def lower_sim(self, sim, *, out_dir, class_name,
                  basename=None, namespace="manta_gen", **opts):
        return _emit_world_cpp(sim, out_dir, class_name=class_name,
                               basename=basename, namespace=namespace)

    def lower_ekf(self, ekf, *, out_dir, class_name,
                  basename=None, namespace="manta_gen", **opts):
        return _emit_ekf_cpp(ekf, out_dir, class_name=class_name,
                             basename=basename, namespace=namespace)

    def lower_lqr(self, lqr, *, out_dir, class_name,
                  basename=None, namespace="manta_gen", **opts):
        return _emit_lqr_cpp(lqr, out_dir, class_name=class_name,
                             basename=basename, namespace=namespace)

    def lower_recurrence(self, block, *, out_dir, class_name,
                         basename=None, namespace="manta_gen", **opts):
        return _emit_recurrence_cpp(block, out_dir, class_name=class_name,
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

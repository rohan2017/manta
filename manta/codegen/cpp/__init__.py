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

from ..target import Target
from .emit import _emit_world_cpp


class _CppBackend(Target):
    """C++ backend. Lowers `Sim` to a buildable static-library project;
    `EKF` / `LQR` lowering is a follow-up (the hooks raise so the gap is
    explicit at the call site rather than silent)."""

    name = "TargetCpp"

    def lower_sim(self, sim, *, out_dir, class_name,
                  basename=None, namespace="manta_gen", **opts):
        return _emit_world_cpp(sim, out_dir, class_name=class_name,
                               basename=basename, namespace=namespace)

    def lower_ekf(self, ekf, **opts):
        raise NotImplementedError(
            "TargetCpp: EKF→C++ lowering is not implemented yet. The EKF IR "
            "carries the same Jacobian bundles (via the shared Linearization); "
            "the C++ side needs a mutable-state + Joseph-form-update wrapper.")

    def lower_lqr(self, lqr, **opts):
        raise NotImplementedError(
            "TargetCpp: LQR→C++ lowering is not implemented yet. It needs a "
            "feed-forward emit of the baked gain K and the control law.")


def TargetCpp(ir,
              out_dir: str | Path,
              *,
              class_name: str,
              basename: str | None = None,
              namespace: str = "manta_gen"):
    """C++ codegen target.

    Args:
        ir          — `Sim` (today). EKF / LQR support is a follow-up
                      (those lowerings raise `NotImplementedError`).
        out_dir     — destination directory (created if missing).
        class_name  — C++ class name. Conventionally PascalCase.
        basename    — filename stem; defaults to `class_name.lower()`.
        namespace   — C++ namespace enclosing the emitted class.

    Returns:
        `EmitResult` (from `manta.codegen.cpp.emit`) with paths to
        every emitted file plus the `WorldFunctions` bundle.
    """
    return _CppBackend().lower(ir, out_dir=out_dir, class_name=class_name,
                               basename=basename, namespace=namespace)


__all__ = ["TargetCpp", "_CppBackend"]

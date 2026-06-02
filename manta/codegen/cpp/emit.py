"""Top-level C++ emission.

`_emit_evaluator_cpp(block, …)` (Sim / LQR / recurrence) and
`_emit_ekf_cpp(ekf, …)` each run the spec/extract → kernels → wrapper →
CMake pipeline end-to-end, leaving a buildable C++ static-library project
on disk. These are the bodies of `TargetCpp(...)`'s `lower_evaluator` /
`lower_ekf` handlers.

The output directory layout::

    out_dir/
        <basename>_kernels.c
        <basename>_kernels.h
        <basename>.hpp
        <basename>.cpp
        CMakeLists.txt
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .cmake import emit_cmakelists
from .extract_ekf import EkfFunctions, extract_ekf
from .kernels import emit_kernel_list
from .ekf_wrapper import emit_ekf_wrapper
from .evaluator_spec import build_evaluator_spec
from .evaluator_wrapper import emit_evaluator_wrapper


@dataclass(frozen=True)
class EmitResult:
    """Paths to the files emitted by `TargetCpp`."""
    out_dir:       Path
    kernels_c:     Path
    kernels_h:     Path
    wrapper_hpp:   Path
    wrapper_cpp:   Path
    cmakelists:    Path
    class_name:    str
    funcs:         object       # WorldFunctions (Sim) or EkfFunctions (EKF)


def _emit_evaluator_cpp(block, out_dir: str | Path, *, class_name: str,
                        basename: str | None = None,
                        namespace: str = "manta_gen") -> EmitResult:
    """Emit a buildable C++ library for any evaluator block (Sim / LQR /
    recurrence). The single generic path: build the `EvaluatorSpec`, emit
    its kernels, render the typed wrapper, drop a CMakeLists.

    `EmitResult.funcs` carries the block's `funcs` (a `WorldFunctions` for
    Sim — handy for cross-checking in Python; the IR object otherwise).
    """
    out_dir = Path(out_dir).resolve()
    base = basename or class_name.lower()

    espec = build_evaluator_spec(block)
    kpaths = emit_kernel_list(list(espec.kernels), out_dir, basename=base)
    wpaths = emit_evaluator_wrapper(espec, out_dir, class_name=class_name,
                                    basename=base, namespace=namespace)
    cmake_path = emit_cmakelists(out_dir, library_name=class_name, basename=base)
    return EmitResult(
        out_dir=out_dir, kernels_c=kpaths["c"], kernels_h=kpaths["h"],
        wrapper_hpp=wpaths["hpp"], wrapper_cpp=wpaths["cpp"],
        cmakelists=cmake_path, class_name=class_name, funcs=espec.funcs)


def _emit_ekf_cpp(ekf, out_dir, *, class_name, basename=None,
                  namespace="manta_gen") -> EmitResult:
    """Emit a buildable C++ EKF library (mutable state + Joseph update)."""
    out_dir = Path(out_dir).resolve()
    base = basename or class_name.lower()

    funcs = extract_ekf(ekf)
    fns = [funcs.predict_fn, funcs.F_fn, funcs.boxplus_fn]
    if funcs.L_fn is not None:
        fns.append(funcs.L_fn)
    for s in funcs.sensors:
        fns += [s.h_fn, s.H_fn]
        if s.L_h_fn is not None:
            fns.append(s.L_h_fn)
    kpaths = emit_kernel_list(fns, out_dir, basename=base)
    wpaths = emit_ekf_wrapper(funcs, ekf.world, out_dir,
                              class_name=class_name, basename=base,
                              namespace=namespace)
    cmake_path = emit_cmakelists(out_dir, library_name=class_name, basename=base)
    return EmitResult(
        out_dir=out_dir, kernels_c=kpaths["c"], kernels_h=kpaths["h"],
        wrapper_hpp=wpaths["hpp"], wrapper_cpp=wpaths["cpp"],
        cmakelists=cmake_path, class_name=class_name, funcs=funcs)

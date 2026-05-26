"""Top-level C++ emission for a CompiledWorld.

`_emit_world_cpp(cw, out_dir, class_name)` runs the
extract → kernels → wrapper → CMake pipeline end-to-end, leaving a
buildable C++ static library project on disk. This is the body of
`TargetCpp(cw, ...)` in `manta.targets`.

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
from .extract import WorldFunctions, extract
from .kernels import emit_kernels
from .wrapper import emit_wrapper


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
    funcs:         WorldFunctions


def _emit_world_cpp(cw,
                    out_dir: str | Path,
                    *,
                    class_name: str,
                    basename: str | None = None,
                    namespace: str = "manta_gen") -> EmitResult:
    """Emit a buildable C++ library for the compiled world.

    Args:
        cw          — `CompiledWorld` IR (returned by `World.compile()`).
        out_dir     — destination directory (created if missing).
        class_name  — C++ class name. Conventionally PascalCase.
        basename    — filename stem; defaults to `class_name.lower()`.
        namespace   — C++ namespace enclosing the emitted class.

    Returns:
        `EmitResult` with paths to every emitted file plus the
        `WorldFunctions` bundle (handy if the caller wants to evaluate
        the same functions in Python for cross-checking).
    """
    out_dir = Path(out_dir).resolve()
    base = basename or class_name.lower()

    funcs = extract(cw)
    kpaths = emit_kernels(funcs, out_dir, basename=base)
    wpaths = emit_wrapper(funcs, cw.world, out_dir,
                          class_name=class_name, basename=base,
                          namespace=namespace)
    cmake_path = emit_cmakelists(out_dir,
                                 library_name=class_name, basename=base)

    return EmitResult(
        out_dir=out_dir,
        kernels_c=kpaths["c"],
        kernels_h=kpaths["h"],
        wrapper_hpp=wpaths["hpp"],
        wrapper_cpp=wpaths["cpp"],
        cmakelists=cmake_path,
        class_name=class_name,
        funcs=funcs,
    )

"""Top-level emit_cpp orchestration.

`emit_cpp(craft, out_dir, class_name)` runs the extract → kernels →
wrapper → CMake pipeline end-to-end, leaving a buildable C++ static
library project on disk.

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
from .extract import CraftFunctions, extract
from .kernels import emit_kernels
from .wrapper import emit_wrapper


@dataclass(frozen=True)
class EmitResult:
    """Paths to the files emitted by `emit_cpp`."""
    out_dir:       Path
    kernels_c:     Path
    kernels_h:     Path
    wrapper_hpp:   Path
    wrapper_cpp:   Path
    cmakelists:    Path
    class_name:    str
    funcs:         CraftFunctions


def emit_cpp(craft,
             out_dir: str | Path,
             *,
             class_name: str,
             basename: str | None = None,
             namespace: str = "manta_next_gen",
             gravity_world: tuple[float, float, float] = (0.0, 0.0, -9.81),
             ) -> EmitResult:
    """Emit a complete, buildable C++ library for `craft`.

    Args:
        craft           — the manta_next Craft to export.
        out_dir         — destination directory (created if missing).
        class_name      — C++ class name. Conventionally PascalCase.
        basename        — filename stem; defaults to lowercased class_name.
        namespace       — C++ namespace.
        gravity_world  — world-frame gravity baked into the predict
                           kernel. Must match the sim that generated
                           any data being compared against.

    Returns:
        EmitResult with paths to every emitted file plus the
        CraftFunctions bundle (handy if the caller wants to evaluate
        the same functions in Python for cross-checking).
    """
    out_dir = Path(out_dir).resolve()
    base = basename or class_name.lower()

    funcs = extract(craft, gravity_world=gravity_world)
    kpaths = emit_kernels(funcs, out_dir, basename=base)
    wpaths = emit_wrapper(funcs, craft, out_dir,
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

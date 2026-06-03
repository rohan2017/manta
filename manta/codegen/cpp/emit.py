"""Top-level C++ emission.

`emit_module(x, …)` lowers a typed `Module` (or a transform exposing
`.module()`) to a buildable C++ static-library project on disk via the one
generic path: `emit_module_cpp` (kernels + typed wrapper + CMake). It is
the body of `TargetCpp(...)`.

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
from types import SimpleNamespace

from ..target import as_module
from .module_emit import emit_module_cpp


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
    funcs:         object       # small summary: world_name / dims


def emit_module(x, out_dir: str | Path, *, class_name: str,
                basename: str | None = None,
                namespace: str = "manta_gen") -> EmitResult:
    """Lower one Module to a C++ library."""
    out_dir = Path(out_dir).resolve()
    base = basename or class_name.lower()
    module = as_module(x, "TargetCpp")
    paths = emit_module_cpp(module, out_dir, class_name=class_name,
                            basename=base, namespace=namespace)
    spec = module.spec
    funcs = SimpleNamespace(
        world_name=module.name,
        ambient_dim=spec.ambient_dim if spec else 0,
        tangent_dim=spec.tangent_dim if spec else 0)
    return EmitResult(
        out_dir=out_dir, kernels_c=paths["kernels_c"],
        kernels_h=paths["kernels_h"], wrapper_hpp=paths["hpp"],
        wrapper_cpp=paths["cpp"], cmakelists=paths["cmakelists"],
        class_name=class_name, funcs=funcs)

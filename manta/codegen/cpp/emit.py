"""Top-level C++ emission.

`emit_module(block, …)` lowers any IR block (Sim / EKF / LQR / recurrence)
to a buildable C++ static-library project on disk via the one generic path:
`to_module(block)` → `emit_module_cpp` (kernels + typed wrapper + CMake). It
is the body of `TargetCpp(...)`.

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

from ..module_build import to_module
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
    funcs:         object       # carries world_name / ambient_dim / tangent_dim


def emit_module(block, out_dir: str | Path, *, class_name: str,
                basename: str | None = None,
                namespace: str = "manta_gen") -> EmitResult:
    """Lower one IR block to a C++ library via the generic Module path."""
    out_dir = Path(out_dir).resolve()
    base = basename or class_name.lower()
    module = to_module(block)
    paths = emit_module_cpp(module, out_dir, class_name=class_name,
                            basename=base, namespace=namespace)
    spec = _spec_of(module)
    funcs = SimpleNamespace(
        world_name=module.name,
        ambient_dim=spec.ambient_dim if spec else 0,
        tangent_dim=spec.tangent_dim if spec else 0)
    return EmitResult(
        out_dir=out_dir, kernels_c=paths["kernels_c"],
        kernels_h=paths["kernels_h"], wrapper_hpp=paths["hpp"],
        wrapper_cpp=paths["cpp"], cmakelists=paths["cmakelists"],
        class_name=class_name, funcs=funcs)


def _spec_of(module):
    for f in module.state.fields:
        if f.kind == "manifold":
            return f.manifold
    for p in module.ports:
        if p.manifold is not None:
            return p.manifold
    return None

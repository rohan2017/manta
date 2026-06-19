"""Top-level WASM emission — the one generic lowering of a typed Module.

`emit_module_wasm(x, …)` lowers a Module (or any transform exposing
`.module()`) to a browser-ready bundle on disk via the same path the C++
backend uses for its math — `densify` each kernel, `emit_kernel_list` the
flat-C — and adds only the three WASM-specific artifacts: the C-ABI shim,
the JSON descriptor, the JS runtime, and an Emscripten build script.

Output directory layout::

    out_dir/
        <basename>_kernels.c   # CasADi flat-C (shared with the C++ backend)
        <basename>_kernels.h
        <basename>_abi.c       # the thin flat-double C ABI, KEEPALIVE-exported
        <basename>.descriptor.json
        <basename>.js          # ES-module runtime (Runtime + Sim)
        build.sh               # emcc -> <basename>.mjs + <basename>.wasm

It dispatches only on the IR's types/roles; there is no per-transform code.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from ..cpp._casadi import densify as _densify
from ..cpp.kernels import emit_kernel_list
from ..target import as_module
from .abi import emit_abi_c
from .build import emit_build_sh
from .descriptor import build_descriptor
from .js import emit_js
from .layout import entry_layout


@dataclass(frozen=True)
class WasmEmitResult:
    """Paths to the files emitted by `TargetWasm`."""
    out_dir:     Path
    kernels_c:   Path
    kernels_h:   Path
    abi_c:       Path
    descriptor:  Path
    js:          Path
    build_sh:    Path
    class_name:  str
    descriptor_obj: dict


def emit_module_wasm(x, out_dir, *, class_name: str,
                     basename: str | None = None,
                     namespace: str = "manta_gen") -> WasmEmitResult:
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    base = basename or class_name.lower()
    module = as_module(x, "TargetWasm")

    entries = list(module.entry_points)
    # Distinct symbol spaces: kernels are `k_<base>_<fn>` (internal), shims
    # are `<base>_<method>` (the exported public ABI) — they never collide
    # even when an entry's method and fn names coincide.
    kname = {ep.fn: f"k_{base}_{ep.fn}" for ep in entries}
    kernels = {k: _densify(module.functions[k], sym)
               for k, sym in kname.items()}

    layouts = [entry_layout(module, ep, kernels[ep.fn],
                            kernel=kname[ep.fn],
                            shim=f"{base}_{ep.method}")
               for ep in entries]

    kpaths = emit_kernel_list(list(kernels.values()), out_dir, basename=base)

    abi_path = out_dir / f"{base}_abi.c"
    abi_path.write_text(emit_abi_c(layouts, basename=base))

    descriptor = build_descriptor(module, layouts, class_name=class_name,
                                  basename=base)
    desc_path = out_dir / f"{base}.descriptor.json"
    desc_path.write_text(json.dumps(descriptor, indent=2) + "\n")

    js_path = out_dir / f"{base}.js"
    js_path.write_text(emit_js(descriptor, basename=base))

    build_path = out_dir / "build.sh"
    build_path.write_text(emit_build_sh(basename=base))
    build_path.chmod(0o755)

    return WasmEmitResult(
        out_dir=out_dir, kernels_c=kpaths["c"], kernels_h=kpaths["h"],
        abi_c=abi_path, descriptor=desc_path, js=js_path,
        build_sh=build_path, class_name=class_name,
        descriptor_obj=descriptor)

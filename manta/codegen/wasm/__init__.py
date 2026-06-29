"""WASM codegen backend.

`TargetWasm(x, out_dir, class_name=...)` lowers a typed `Module` — or any
transform exposing `.module()` (`Sim`, `EKF`, `LQR`, a recurrence block) —
to a browser-ready bundle: the CasADi flat-C kernels (shared, verbatim, with
the C++ backend), a thin flat-double **C-ABI shim** exported to WASM, a JSON
descriptor, an ES-module **JS runtime**, and an Emscripten `build.sh`. The
backend dispatches only on the IR's types (`StateRef`/`PortRef`, `Port.role`,
State-field kind); it contains no per-transform code.

The math path is identical to `TargetCpp`'s — `densify` + `emit_kernel_list`
— so the numbers are the same on every backend. WASM adds only the marshalling
glue: a `double* in / double* out` ABI over CasADi's `arg[]`/`res[]` arrays,
plus the JS that packs typed values into the WASM heap.

    out_dir/
      <basename>_kernels.c / .h    CasADi flat-C kernels
      <basename>_abi.c             C-ABI shim (EMSCRIPTEN_KEEPALIVE)
      <basename>.descriptor.json   runtime layout descriptor
      <basename>.js                ES-module runtime (Runtime + Sim/Filter/Regulator)
      build.sh                     emcc -> <basename>.mjs + .wasm

Run `./build.sh` (needs the Emscripten SDK) to produce the `.wasm`, then
`import { load } from "./<basename>.js"` in the browser.
"""

from __future__ import annotations

from pathlib import Path

from .emit import WasmEmitResult, emit_module_wasm


def TargetWasm(x,
               out_dir: str | Path,
               *,
               class_name: str,
               basename: str | None = None,
               namespace: str = "manta_gen") -> WasmEmitResult:
    """WASM codegen target.

    Args:
        x           — a `Module`, or a transform with `.module()`.
        out_dir     — destination directory (created if missing).
        class_name  — public name for the bundle. Conventionally PascalCase.
        basename    — filename stem; defaults to `class_name.lower()`.
        namespace   — reserved for parity with `TargetCpp` (unused in C ABI).

    Returns:
        `WasmEmitResult` with paths to every emitted file plus the descriptor.
    """
    return emit_module_wasm(x, out_dir, class_name=class_name,
                            basename=basename, namespace=namespace)


__all__ = ["TargetWasm", "WasmEmitResult"]

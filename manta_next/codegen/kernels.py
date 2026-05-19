"""Flat-C math kernels emitted by CasADi's CodeGenerator.

Bundles every ca.Function from a `CraftFunctions` (predict, predict
Jacobian, per-Output h/H pairs) into a single C source + header pair.

The emitted C is:
  * standalone (no CasADi runtime needed at link time);
  * row-major arrays of doubles;
  * has `extern int <func>(const double** arg, double** res, ...)` signatures;
  * includes CSE'd dead-code-eliminated expressions.

The typed C++ wrapper (`wrapper.py`) calls into these by packing
Eigen-typed state/inputs into the flat-double arrays and unpacking the
result. The wrapper is the only thing the user touches; the kernels are
an implementation detail.
"""

from __future__ import annotations

import os
from pathlib import Path

import casadi as ca

from .extract import CraftFunctions


def emit_kernels(funcs: CraftFunctions,
                 out_dir: str | Path,
                 *,
                 basename: str | None = None) -> dict[str, Path]:
    """Emit `<basename>_kernels.c` + `<basename>_kernels.h` into `out_dir`.

    Args:
        funcs    — the per-function bundle from `extract.extract(craft)`.
        out_dir  — destination directory; created if missing.
        basename — filename stem. Defaults to `funcs.craft_name`.

    Returns:
        Dict with absolute paths to the emitted `.c` and `.h` files.
    """
    out_dir  = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    base = basename or funcs.craft_name

    gen = ca.CodeGenerator(
        f"{base}_kernels.c",
        {
            "cpp": False,                # plain C, not C++.
            "with_header": True,          # emit .h alongside.
            "with_mem":    False,         # we don't need memory allocators.
            "verbose":     False,
        },
    )
    gen.add(funcs.predict_fn)
    gen.add(funcs.predict_jacobian_fn)
    for o in funcs.outputs:
        gen.add(o.h_fn)
        gen.add(o.H_fn)
    gen.generate(str(out_dir) + os.sep)

    c_path = out_dir / f"{base}_kernels.c"
    h_path = out_dir / f"{base}_kernels.h"
    if not c_path.exists() or not h_path.exists():
        raise RuntimeError(
            f"emit_kernels: CasADi didn't emit expected files at {out_dir}. "
            f"Got: {sorted(p.name for p in out_dir.iterdir())}")

    return {"c": c_path, "h": h_path}


def kernel_function_names(funcs: CraftFunctions) -> dict[str, str]:
    """Return the canonical kernel-function names for one CraftFunctions
    bundle. Keys: 'predict', 'predict_jacobian', and 'h_<part>_<output>',
    'H_<part>_<output>' per Output. Values are the C symbol names
    matching what CasADi's CodeGenerator produces (the ca.Function's name)."""
    out = {
        "predict":          funcs.predict_fn.name(),
        "predict_jacobian": funcs.predict_jacobian_fn.name(),
    }
    for o in funcs.outputs:
        out[f"h_{o.flat_name}"] = o.h_fn.name()
        out[f"H_{o.flat_name}"] = o.H_fn.name()
    return out

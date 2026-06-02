"""Flat-C math kernels emitted by CasADi's CodeGenerator.

Bundles every ca.Function from a `WorldFunctions` (predict, predict
Jacobian, per-Output h/H pairs) into a single C source + header pair.

The emitted C is:
  * standalone (no CasADi runtime needed at link time);
  * row-major arrays of doubles;
  * has `extern int <func>(const double** arg, double** res, ...)` signatures;
  * includes CSE'd dead-code-eliminated expressions.

The typed C++ wrapper calls into these by packing Eigen-typed
state/inputs into the flat-double arrays and unpacking the result. The
wrapper is the only thing the user touches; the kernels are an
implementation detail.
"""

from __future__ import annotations

import os
from pathlib import Path

import casadi as ca


def emit_kernel_list(fns, out_dir: str | Path, *, basename: str) -> dict[str, Path]:
    """Emit an explicit list of `ca.Function`s into one `<basename>_kernels.c/.h`.

    The general entry point used by every transform's backend (Sim adds its
    predict/Jacobian/sensor bundle; the EKF adds `L` + `boxplus`; the LQR
    adds `control`). Any `ca.Function` works."""
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    gen = ca.CodeGenerator(
        f"{basename}_kernels.c",
        {"cpp": False, "with_header": True, "with_mem": False, "verbose": False},
    )
    for fn in fns:
        gen.add(fn)
    gen.generate(str(out_dir) + os.sep)

    c_path = out_dir / f"{basename}_kernels.c"
    h_path = out_dir / f"{basename}_kernels.h"
    if not c_path.exists() or not h_path.exists():
        raise RuntimeError(
            f"emit_kernel_list: CasADi didn't emit expected files at "
            f"{out_dir}. Got: {sorted(p.name for p in out_dir.iterdir())}")
    return {"c": c_path, "h": h_path}

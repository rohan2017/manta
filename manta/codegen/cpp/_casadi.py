"""Shared CasADi ↔ C glue for the C++ backend.

Backend-internal helpers the generic emitter needs: `emit_kernel_call`
(flat-C arg/res/iw/w boilerplate for a baked kernel) and `densify` (dense
row/col-major outputs + a codegen-safe rename for the flat-C / Eigen
interface). Pure CasADi / string plumbing — no IR.
"""

from __future__ import annotations

import casadi as ca


def emit_kernel_call(fn: str, args: list[str],
                     results: list[tuple[int, str]]) -> list[str]:
    """Emit a CasADi flat-C kernel call (arg/res/iw/w boilerplate)."""
    out = ["    {",
           f"        const double* arg[{fn}_SZ_ARG] = {{0}};"]
    out += [f"        arg[{i}] = {a};" for i, a in enumerate(args)]
    out.append(f"        double* res[{fn}_SZ_RES] = {{0}};")
    out += [f"        res[{i}] = {e};" for i, e in results]
    out.append(f"        long long iw[{fn}_SZ_IW > 0 ? {fn}_SZ_IW : 1];")
    out.append(f"        double    w [{fn}_SZ_W  > 0 ? {fn}_SZ_W  : 1];")
    out.append(f"        {fn}(arg, res, iw, w, 0);")
    out.append("    }")
    return out


def densify(fn: ca.Function, name: str) -> ca.Function:
    """Rewrap a `ca.Function` so its outputs are dense (row/col-major fill
    for the flat-C / Eigen interface) and rename to a codegen-safe symbol.
    Dense outputs (state/measurement vectors) pass through unchanged."""
    ins = [ca.MX.sym(fn.name_in(i), fn.size1_in(i), fn.size2_in(i))
           for i in range(fn.n_in())]
    outs = fn(*ins)
    if fn.n_out() == 1:
        outs = [outs]
    outs = [ca.densify(o) for o in outs]
    return ca.Function(name, ins, outs,
                       [fn.name_in(i) for i in range(fn.n_in())],
                       [fn.name_out(i) for i in range(fn.n_out())])

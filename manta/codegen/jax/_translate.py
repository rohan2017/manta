"""CasADi kernel → JAX function, via the SX instruction list.

`ca.Function.expand()` lowers a (pure, Linsol-free) MX kernel to SX —
a flat tape of scalar instructions over a work array. This module walks
that tape once and EMITS PYTHON SOURCE (one line per instruction), so
JAX traces a flat scalar program instead of interpreting a loop:

    w3 = args[0][7] * args[4][0]
    w0 = w0 + w3
    ...

`jax.jit` then compiles it like any hand-written function; `jax.grad`/
`jax.vmap`/`lax.scan` compose over it as usual. Inputs are flattened
column-major (CasADi layout); each output's nonzeros scatter back into
its dense shape through the sparsity pattern's triplets.

Kernels containing runtime-pivoting Linsol nodes (a jointed craft's
joint-space solve) cannot expand — `translate` raises with the reason.
"""

from __future__ import annotations

import casadi as ca

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)   # manta kernels are float64


# Binary/unary scalar ops, by CasADi OP code → source template.
_BINARY = {
    ca.OP_ADD: "{a} + {b}",
    ca.OP_SUB: "{a} - {b}",
    ca.OP_MUL: "{a} * {b}",
    ca.OP_DIV: "{a} / {b}",
    ca.OP_POW: "{a} ** {b}",
    ca.OP_CONSTPOW: "{a} ** {b}",
    ca.OP_ATAN2: "jnp.arctan2({a}, {b})",
    ca.OP_FMIN: "jnp.minimum({a}, {b})",
    ca.OP_FMAX: "jnp.maximum({a}, {b})",
    ca.OP_FMOD: "jnp.fmod({a}, {b})",
    ca.OP_COPYSIGN: "jnp.copysign({a}, {b})",
    ca.OP_IF_ELSE_ZERO: "jnp.where({a} != 0.0, {b}, 0.0)",
    ca.OP_LT: "({a} < {b}) * 1.0",
    ca.OP_LE: "({a} <= {b}) * 1.0",
    ca.OP_EQ: "({a} == {b}) * 1.0",
    ca.OP_NE: "({a} != {b}) * 1.0",
    ca.OP_AND: "jnp.logical_and({a} != 0.0, {b} != 0.0) * 1.0",
    ca.OP_OR: "jnp.logical_or({a} != 0.0, {b} != 0.0) * 1.0",
}

_UNARY = {
    ca.OP_ASSIGN: "{a}",
    ca.OP_NEG: "-{a}",
    ca.OP_EXP: "jnp.exp({a})",
    ca.OP_LOG: "jnp.log({a})",
    ca.OP_SQRT: "jnp.sqrt({a})",
    ca.OP_SQ: "{a} * {a}",
    ca.OP_TWICE: "2.0 * {a}",
    ca.OP_SIN: "jnp.sin({a})",
    ca.OP_COS: "jnp.cos({a})",
    ca.OP_TAN: "jnp.tan({a})",
    ca.OP_ASIN: "jnp.arcsin({a})",
    ca.OP_ACOS: "jnp.arccos({a})",
    ca.OP_ATAN: "jnp.arctan({a})",
    ca.OP_SINH: "jnp.sinh({a})",
    ca.OP_COSH: "jnp.cosh({a})",
    ca.OP_TANH: "jnp.tanh({a})",
    ca.OP_FABS: "jnp.abs({a})",
    ca.OP_SIGN: "jnp.sign({a})",
    ca.OP_FLOOR: "jnp.floor({a})",
    ca.OP_CEIL: "jnp.ceil({a})",
    ca.OP_ERF: "jax.scipy.special.erf({a})",
    ca.OP_INV: "1.0 / {a}",
    ca.OP_NOT: "({a} == 0.0) * 1.0",
}


def translate(fn: ca.Function, *, jit: bool = True):
    """Translate one CasADi kernel to a JAX function.

    The result takes `fn.n_in()` array-likes (any shape; flattened
    column-major to the kernel's layout) and returns a tuple of
    `fn.n_out()` dense `jnp` arrays in the kernel's output shapes.
    """
    try:
        sx = fn.expand()
    except Exception as e:                       # Linsol nodes, etc.
        raise NotImplementedError(
            f"TargetJax: kernel {fn.name()!r} cannot expand to SX "
            f"(typically a jointed craft's runtime-pivoting joint-space "
            f"solve). Underlying error: {e}") from e

    src = ["def _kernel(args):"]
    out_exprs: list[list[str | None]] = [
        [None] * sx.nnz_out(j) for j in range(sx.n_out())]

    for k in range(sx.n_instructions()):
        op = sx.instruction_id(k)
        o = sx.instruction_output(k)
        i = sx.instruction_input(k)
        if op == ca.OP_INPUT:
            src.append(f"    w{o[0]} = args[{i[0]}][{i[1]}]")
        elif op == ca.OP_OUTPUT:
            # Snapshot NOW — the tape reuses work registers, so the
            # value of w{i} at this instruction differs from its value
            # at the end of the tape.
            src.append(f"    o{o[0]}_{o[1]} = w{i[0]}")
            out_exprs[o[0]][o[1]] = f"o{o[0]}_{o[1]}"
        elif op == ca.OP_CONST:
            src.append(f"    w{o[0]} = {float(sx.instruction_constant(k))!r}")
        elif op in _BINARY:
            expr = _BINARY[op].format(a=f"w{i[0]}", b=f"w{i[1]}")
            src.append(f"    w{o[0]} = {expr}")
        elif op in _UNARY:
            expr = _UNARY[op].format(a=f"w{i[0]}")
            src.append(f"    w{o[0]} = {expr}")
        else:
            raise NotImplementedError(
                f"TargetJax: kernel {fn.name()!r} uses unsupported CasADi "
                f"op code {op} — extend the tables in jax/_translate.py.")

    rets = []
    scatter: dict[int, tuple] = {}
    for j in range(sx.n_out()):
        names = out_exprs[j]
        if any(n is None for n in names):
            # An output nonzero never written: structurally zero slot.
            names = [n if n is not None else "0.0" for n in names]
        stack = ("jnp.stack([" + ", ".join(names) + "])" if names
                 else "jnp.zeros(0)")
        sp = sx.sparsity_out(j)
        shape = (sp.size1(), sp.size2())
        if sp.is_dense():
            # Dense CCS nz order == column-major order.
            rets.append(f"{stack}.reshape({shape[1]}, {shape[0]}).T")
        else:
            rows, cols = sp.get_triplet()
            scatter[j] = (tuple(rows), tuple(cols), shape)
            rets.append(
                f"jnp.zeros({shape}).at[_rows{j}, _cols{j}].set({stack})")
    src.append("    return (" + ", ".join(rets) + ("," if rets else "")
               + ")")

    ns = {"jnp": jnp, "jax": jax}
    for j, (rows, cols, _shape) in scatter.items():
        ns[f"_rows{j}"] = jnp.array(rows, dtype=int)
        ns[f"_cols{j}"] = jnp.array(cols, dtype=int)
    exec("\n".join(src), ns)
    kernel = ns["_kernel"]

    def wrapper(*args):
        if len(args) != sx.n_in():
            raise TypeError(
                f"{fn.name()}: expected {sx.n_in()} argument(s) "
                f"({[sx.name_in(i) for i in range(sx.n_in())]}), "
                f"got {len(args)}.")
        flat = [jnp.asarray(a, dtype=jnp.float64).T.ravel() for a in args]
        return kernel(flat)

    wrapper.__name__ = f"jax_{fn.name()}"
    return jax.jit(wrapper) if jit else wrapper

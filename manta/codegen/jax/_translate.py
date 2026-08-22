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
joint-space solve) cannot SX-expand whole. For those, `translate` cuts
the (fully inlined) MX graph at each `OP_SOLVE` node instead: the
pieces between solves become plain SX-expandable stage kernels, and the
solves themselves are mapped to `jnp.linalg.solve` — LAPACK's
partially-pivoted LU, the same class of runtime-pivoting solve CasADi's
Linsol performs. One honest difference: on a genuinely singular matrix
(a configuration-degenerate joint-space system) CasADi's Linsol raises,
while `jnp.linalg.solve` under jit would return inf/NaN and let a scan roll
on. Manta closes that gap: every cut solve is followed by a finiteness
check that raises (`FloatingPointError`, surfaced as a JAX runtime error
under jit/scan) — the same outcome as the numpy/C backend, whose runtime
refuses a kernel that produced non-finite values. The composed function
is jitted / differentiated as one program (JAX differentiates through
`linalg.solve` natively).
"""

from __future__ import annotations

import warnings

import casadi as ca
import numpy as np

import jax
import jax.numpy as jnp


def _require_x64() -> None:
    """Ensure float64 tracing before any kernel is built. manta kernels
    are float64 physics — float32 would silently corrupt them — but
    flipping a process-global JAX flag as an *import* side effect is not
    this module's call; do it at first use, and say so."""
    if not jax.config.jax_enable_x64:
        warnings.warn(
            "TargetJax: enabling jax_enable_x64 (manta kernels are float64 "
            "physics; float32 would silently corrupt them). Set "
            "jax.config.update('jax_enable_x64', True) at startup to make "
            "this explicit.", RuntimeWarning, stacklevel=3)
        jax.config.update("jax_enable_x64", True)


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

    A kernel that cannot SX-expand because it contains Linsol solve
    nodes (a jointed craft) is cut at those nodes and recomposed around
    `jnp.linalg.solve` — see `_translate_around_solves`.
    """
    _require_x64()
    # RuntimeError is what CasADi raises for an inexpandable graph
    # (Linsol nodes). Anything else is a genuine bug and must surface
    # as itself, not be silently rerouted into the solve-cut path.
    try:
        sx = fn.expand()
    except RuntimeError as e:                    # Linsol nodes
        wrapper = _translate_around_solves(fn, cause=e)
        return jax.jit(wrapper) if jit else wrapper
    wrapper = _translate_sx(sx, fn.name())
    return jax.jit(wrapper) if jit else wrapper


_INSTR_WARN = 100_000


def _translate_sx(sx: ca.Function, name: str):
    """The SX-tape walk: emit one Python line per scalar instruction and
    exec it into a function over flattened args (unjitted)."""
    n_instr = sx.n_instructions()
    if n_instr > _INSTR_WARN:
        # One source line per instruction: CPython must parse it and JAX
        # must trace it. There is no interpreted fallback here (unlike
        # the numpy compile path's cap), so warn instead of failing —
        # but the user should know why the first call takes minutes.
        warnings.warn(
            f"TargetJax: kernel {name!r} expands to {n_instr} scalar "
            f"instructions — emitted as one {n_instr}-line Python "
            f"function; expect a long parse/trace on first call. "
            f"Consider a smaller tracked state or the C++ target.",
            RuntimeWarning, stacklevel=2)
    src = ["def _kernel(args):"]
    out_exprs: list[list[str | None]] = [
        [None] * sx.nnz_out(j) for j in range(sx.n_out())]

    for k in range(n_instr):
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
                f"TargetJax: kernel {name!r} uses unsupported CasADi "
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
                f"{name}: expected {sx.n_in()} argument(s) "
                f"({[sx.name_in(i) for i in range(sx.n_in())]}), "
                f"got {len(args)}.")
        flat = [jnp.asarray(a, dtype=jnp.float64).T.ravel() for a in args]
        return kernel(flat)

    wrapper.__name__ = f"jax_{name}"
    return wrapper


# ---------------------------------------------------------------------------
# Linsol-bearing kernels: cut the MX graph at solve nodes
# ---------------------------------------------------------------------------

def _postorder(exprs):
    """Iterative post-order walk of an MX DAG (deps before dependents),
    deduplicated by node identity.

    Dedup keys on `hash(n)`: CasADi's MX hash falls back to the shared
    node pointer, so structurally-distinct live nodes never collide, and
    holding every visited node in `order` keeps them alive (no address
    reuse) for the duration of the walk — an undocumented but load-
    bearing property of the binding."""
    seen: set[int] = set()
    order: list[ca.MX] = []
    stack = [(e, False) for e in exprs]
    while stack:
        n, done = stack.pop()
        if done:
            order.append(n)
            continue
        h = hash(n)
        if h in seen:
            continue
        seen.add(h)
        stack.append((n, True))
        for i in range(n.n_dep()):
            stack.append((n.dep(i), False))
    return order

# Call-node inlining rounds. Each round inlines one nesting level; manta
# kernels nest at most (entry → world tick), so 32 is a generous bound
# against a substitution that fails to make progress.
_MAX_INLINE_ROUNDS = 32


def _inline_calls(outs: list[ca.MX]) -> list[ca.MX]:
    """Replace every embedded function-call node in `outs` with its
    callee's inlined body (repeatedly, until no call nodes remain), so
    solve nodes buried in nested functions surface in one flat graph."""
    for _ in range(_MAX_INLINE_ROUNDS):
        nodes = _postorder(outs)
        calls = {hash(n): n for n in nodes if n.is_call()}
        if not calls:
            return outs
        inlined = {
            h: c.which_function().call(
                [c.dep(i) for i in range(c.n_dep())], True, False)
            for h, c in calls.items()}
        old, new = [], []
        multi: set[int] = set()
        # Multi-output calls are read through output-selector nodes.
        for n in nodes:
            if n.is_output() and hash(n.dep(0)) in inlined:
                h = hash(n.dep(0))
                old.append(n)
                new.append(inlined[h][n.which_output()])
                multi.add(h)
        # Single-output calls ARE the value node.
        for h, c in calls.items():
            if h not in multi:
                old.append(c)
                new.append(inlined[h][0])
        outs = ca.graph_substitute(outs, old, new)
    raise NotImplementedError(
        "TargetJax: function-call inlining did not converge — the kernel "
        "embeds calls nested deeper than expected.")


def _refuse_nonfinite_solve(x, *, kernel: str, index: int) -> None:
    """Raise on a non-finite linear-solve result, under jit/scan/vmap too.

    `jnp.linalg.solve` has no error path: a singular joint-space system
    (a configuration-degenerate mechanism) yields inf/NaN silently. CasADi's
    interpreted Linsol raises on the same input and Manta's numpy runtime
    refuses any kernel output that is not finite, so the JAX lowering must
    fail the same way. The check runs on device; only the failing branch
    calls back into Python to raise, so the healthy path costs one
    reduction. Under `vmap` both branches may run on batched data — the
    callback re-examines the flag it receives, so it never raises
    spuriously.
    """
    finite = jnp.all(jnp.isfinite(x))

    def refuse(flag):
        if not bool(np.all(np.asarray(flag))):
            raise FloatingPointError(
                f"TargetJax: kernel {kernel!r} linear solve {index} produced "
                "non-finite values (singular joint-space system); the "
                "numpy/C backend refuses this configuration as well")

    jax.lax.cond(
        finite,
        lambda flag: None,
        lambda flag: jax.debug.callback(refuse, flag),
        finite,
    )


def _translate_around_solves(fn: ca.Function, *, cause: Exception):
    """Lower a kernel whose whole-graph SX expansion failed.

    Inline the kernel to one flat MX graph and cut it at every
    `OP_SOLVE` (Linsol) node — the reason a jointed craft's tick cannot
    expand. Each cut piece is a plain SX-expandable kernel; the solves
    are re-emitted as `jnp.linalg.solve` between them:

        stage_i : (args, x_0..x_{i-1}) -> A_i, b_i     (SX-translated)
        x_i     = jnp.linalg.solve(A_i, b_i)
        post    : (args, x_0..x_{n-1}) -> outputs      (SX-translated)

    Solves are cut in dependency order, so a solve whose A/b depend on
    an earlier solve's result reads it as a stage argument. A node
    carrying CasADi's internal transpose flag (`Solve<Tr=true>`, from
    autodiffed graphs) solves against Aᵀ. If the graph contains no
    solve nodes the original expansion failure had some other cause —
    re-raise it.

    Singularity semantics match the numpy/C backend by construction: a
    degenerate A is caught by a finiteness check on the solve result and
    raises instead of rolling inf/NaN through the rest of the program.
    """
    ins = [ca.MX.sym(fn.name_in(i), fn.sparsity_in(i))
           for i in range(fn.n_in())]
    outs = _inline_calls(list(fn.call(ins, True, False)))
    solves = [n for n in _postorder(outs) if n.op() == ca.OP_SOLVE]
    if not solves:
        raise NotImplementedError(
            f"TargetJax: kernel {fn.name()!r} cannot expand to SX and "
            f"contains no Linsol solve nodes to cut around. "
            f"Underlying error: {cause}") from cause

    x_syms: list[ca.MX] = []
    stage_fns = []
    for i, s in enumerate(solves):
        b_raw, A_raw = s.dep(0), s.dep(1)   # Solve node: dep0 = b, dep1 = A
        # Sanity-check the (undocumented) CasADi dep ordering we rely
        # on: A square, b shaped like the solution. A silent reversal
        # would produce transposed physics with no error.
        if (A_raw.shape[0] != A_raw.shape[1]
                or tuple(b_raw.shape) != tuple(s.shape)):
            raise NotImplementedError(
                f"TargetJax: solve node {i} of kernel {fn.name()!r} has "
                f"deps b{tuple(b_raw.shape)}, A{tuple(A_raw.shape)} for "
                f"solution {tuple(s.shape)} — CasADi's Solve-node dep "
                f"layout differs from the (b, A) ordering this cut "
                f"assumes. Check the installed CasADi version.")
        info = s.info()
        if "tr" not in info:
            # `.get("tr", False)` here would silently solve against A
            # instead of Aᵀ if CasADi ever renames the flag — wrong
            # numbers, no error. Demand it.
            raise NotImplementedError(
                f"TargetJax: solve node of kernel {fn.name()!r} carries "
                f"no 'tr' flag in info() ({sorted(info)}) — the "
                f"installed CasADi's Solve-node metadata differs from "
                f"what this cut relies on.")
        if info["tr"]:                       # Solve<Tr=true>: x = Aᵀ \ b
            A_raw = A_raw.T
        if x_syms:
            A_raw, b_raw = ca.graph_substitute(
                [A_raw, b_raw], solves[:i], x_syms)
        stage = ca.Function(f"{fn.name()}_linsol_stage{i}",
                            ins + x_syms,
                            [ca.densify(A_raw), ca.densify(b_raw)])
        stage_fns.append(_translate_sx(stage.expand(), stage.name()))
        x_syms.append(ca.MX.sym(f"linsol_x{i}", *s.shape))

    post = ca.Function(f"{fn.name()}_linsol_post", ins + x_syms,
                       ca.graph_substitute(outs, solves, x_syms))
    post_fn = _translate_sx(post.expand(), post.name())
    n_in = fn.n_in()
    name = fn.name()

    def wrapper(*args):
        if len(args) != n_in:
            raise TypeError(
                f"{name}: expected {n_in} argument(s), got {len(args)}.")
        xs: list = []
        for index, stage in enumerate(stage_fns):
            A, b = stage(*args, *xs)
            x = jnp.linalg.solve(A, b)
            _refuse_nonfinite_solve(x, kernel=name, index=index)
            xs.append(x)
        return post_fn(*args, *xs)

    wrapper.__name__ = f"jax_{name}"
    return wrapper

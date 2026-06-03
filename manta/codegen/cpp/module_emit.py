"""The ONE generic C++ lowering of a `Module` (`manta.ir.module`).

`emit_module_cpp(module, …)` renders any Module — Sim, EKF, LQR, or a
recurrence block — to a typed C++ class on the same flat-double kernel ABI:
a `State` struct (manifold-typed slots), an `Inputs`/`Outputs` struct when
the module declares those ports, an `Eigen` covariance member for a matrix
State field, and one method per *deployable* entry point (each pure
pack → call-kernel → unpack/scatter). It reads ONLY the Module — there is no
per-shape code and no `world` — which is the C++ half of the generic backend
contract. This replaces the former per-shape `evaluator_spec` builders +
`evaluator_wrapper` + `ekf_wrapper` + `extract`/`extract_ekf`.

The method shape is derived from each entry:

  * held State (EKF / recurrence) lives as members; otherwise it is threaded
    as a `const State&` parameter (Sim / LQR);
  * a kernel arg in `synthesize_zero` is filled with zero (a measurement's
    `dt`); one in `port_from` is computed by calling another entry's kernel
    (the EKF's `Q` from `process_noise`); the rest are typed parameters;
  * the return is a State (a not-held `writes_state`), an Inputs/Outputs
    struct (a return naming a fielded port), or an Eigen vector/matrix
    (sized from the kernel's output) — e.g. a measurement or a Jacobian.
"""

from __future__ import annotations

from pathlib import Path

from . import _structs as S
from ._casadi import densify as _densify
from ._casadi import emit_kernel_call as _call
from .cmake import emit_cmakelists
from .kernels import emit_kernel_list
from .types import cpp_type_for


# ---------------------------------------------------------------------------
# Module introspection
# ---------------------------------------------------------------------------

def _spec_of(module):
    """The manifold StateSpec this module's State struct is built from —
    a manifold State field, else a manifold port (LQR's `x`)."""
    for f in module.state.fields:
        if f.kind == "manifold":
            return f.manifold, f.init
    for p in module.ports:
        if p.manifold is not None:
            return p.manifold, None
    return None, None


def _matrix_fields(module):
    return [f for f in module.state.fields if f.kind == "matrix"]


def _port(module, name):
    for p in module.ports:
        if p.name == name:
            return p
    return None


# ---------------------------------------------------------------------------
# Struct emission
# ---------------------------------------------------------------------------

def _state_struct(spec, init_vec) -> list[str]:
    out = ["    struct State {"]
    for s in spec.slots:
        cpp = cpp_type_for(s.manifold)
        if init_vec is not None:
            seg = init_vec[s.ambient_offset:s.ambient_offset + s.ambient_dim]
            init = (cpp.literal(float(seg[0])) if s.ambient_dim == 1
                    else cpp.literal([float(v) for v in seg]))
        else:
            init = cpp.zero
        out.append(f"        {cpp.decl} {S.cpp_ident(s.name)} = {init};")
    out.append("    };")
    return out


def _struct_from_fields(name, fields, *, with_init: bool) -> list[str]:
    out = [f"    struct {name} {{"]
    for f in fields:
        tp = S.eigen_vec_type(f.dim)
        if not with_init:
            out.append(f"        {tp} {S.cpp_ident(f.name)};")
        elif f.dim == 1:
            out.append(f"        {tp} {S.cpp_ident(f.name)} = {float(f.default)!r};")
        else:
            out.append(f"        {tp} {S.cpp_ident(f.name)} = {tp}::Zero();")
    if not fields:
        out.append("        // (none)")
    out.append("    };")
    return out


def _pack_inputs(u_port, qcls) -> list[str]:
    out = [f"static void pack_inputs(const {qcls}::Inputs& u, double* uf) {{"]
    if u_port is None or not u_port.fields:
        out.append("    (void)u; (void)uf;")
    else:
        off = 0
        for f in u_port.fields:
            ident = S.cpp_ident(f.name)
            if f.dim == 1:
                out.append(f"    uf[{off}] = u.{ident};")
            else:
                for i in range(f.dim):
                    out.append(f"    uf[{off + i}] = u.{ident}[{i}];")
            off += f.dim
    out.append("}")
    return out


def _scatter_fields(fields, src, dst) -> list[str]:
    out, off = [], 0
    for f in fields:
        ident = S.cpp_ident(f.name)
        if f.dim == 1:
            out.append(f"    {dst}.{ident} = {src}[{off}];")
        else:
            for i in range(f.dim):
                out.append(f"    {dst}.{ident}[{i}] = {src}[{off + i}];")
        off += f.dim
    return out


# ---------------------------------------------------------------------------
# Per-entry method generation
# ---------------------------------------------------------------------------

class _Ctx:
    """Resolved per-module facts the method emitters share."""
    def __init__(self, module, kname):
        self.m = module
        self.kname = kname                  # {fn_key: emitted C symbol}
        self.spec, _ = _spec_of(module)
        self.amb = self.spec.ambient_dim if self.spec else 0
        self.tan = self.spec.tangent_dim if self.spec else 0
        self.held = module.held
        self.mats = _matrix_fields(module)  # e.g. EKF P
        self.fields = set(module.state.names)


def _ret_kind(ep, ctx):
    """Classify an entry's return: 'state' | 'inputs' | 'outputs' |
    ('vec', n) | ('mat', r, c) | 'void'."""
    if not ep.returns:
        return "state" if (ep.writes_state and not ctx.held) else "void"
    name = ep.returns[0]
    port = _port(ctx.m, name)
    if port is not None and port.fields:
        return "inputs" if name == "u" else "outputs"
    fn = ctx.m.functions[ep.fn]
    # return is the last kernel output (after writes_state outputs)
    oi = len(ep.writes_state)
    r, c = int(fn.size1_out(oi)), int(fn.size2_out(oi))
    return ("vec", r) if c == 1 else ("mat", r, c)


def _ret_type(ep, ctx, q=None) -> str:
    """Return-type spelling; nested struct types qualified by `q` (for an
    out-of-class definition) or left bare (for the in-class declaration)."""
    ret = _ret_kind(ep, ctx)
    pre = f"{q}::" if q else ""
    if ret == "state":
        return pre + "State"
    if ret == "inputs":
        return pre + "Inputs"
    if ret == "outputs":
        return pre + "Outputs"
    if ret == "void":
        return "void"
    if ret[0] == "vec":
        return S.eigen_vec_type(ret[1])
    return f"Eigen::Matrix<double, {ret[1]}, {ret[2]}>"


def _const(ep, ctx) -> str:
    return " const" if not ctx.held and _ret_kind(ep, ctx) != "void" else ""


def _method_decl(ep, ctx) -> str:
    """In-class method declaration (with default args)."""
    params = _params(ep, ctx, decl=True)
    return f"{_ret_type(ep, ctx)} {ep.method}({', '.join(params)}){_const(ep, ctx)}"


def _params(ep, ctx, *, decl: bool):
    """Typed parameter list in canonical order: State, z, u, dt, t."""
    out = []
    reads_state_x = any(a == "x" for a in ep.kernel_args)
    x_is_param = reads_state_x and not ctx.held
    if x_is_param:
        out.append("const State& x")
    for a in ep.kernel_args:
        if a in ctx.fields or a == "x":
            continue                         # State / matrix members or x param
        if a in ep.synthesize_zero or a in ep.port_from:
            continue                         # filled internally
        if a == "u":
            out.append("const Inputs& u")
        elif a.startswith("z"):
            port = _port(ctx.m, a)
            out.append(f"const {S.eigen_vec_type(port.size)}& z")
        elif a == "dt":
            out.append("double dt")
        elif a == "t":
            out.append("double t = 0.0" if decl else "double t")
    return out


def _method_body(ep, ctx) -> list[str]:
    ret = _ret_kind(ep, ctx)
    L = []
    # --- buffers ---
    if any(a == "x" for a in ep.kernel_args):
        L.append(f"    double x_in[{ctx.amb}];")
    if "x" in ep.writes_state:
        L.append(f"    double x_out[{ctx.amb}];")
    u_port = _port(ctx.m, "u")
    if any(a == "u" for a in ep.kernel_args):
        n = sum(f.dim for f in u_port.fields) if u_port else 0
        L.append(f"    double u_in[{max(n, 1)}];"
                 + ("" if n else "   // no inputs"))
    for a in ep.synthesize_zero:
        if a == "dt":
            L.append("    double dt = 0.0;")
    # matrix ports filled internally (Q): declare + (port_from) compute
    for a in ep.kernel_args:
        port = _port(ctx.m, a)
        if a in ep.port_from or (a in ep.synthesize_zero and port is not None
                                 and len(port.shape) == 2):
            L.append(f"    Cov {a}m = Cov::Zero();")
    # result buffers for returns
    if ret == "inputs":
        L.append(f"    double u_out[{max(sum(f.dim for f in u_port.fields), 1)}];")
    elif ret == "outputs":
        yp = _port(ctx.m, ep.returns[0])
        L.append(f"    double y_out[{max(sum(f.dim for f in yp.fields), 1)}];")
    elif isinstance(ret, tuple) and ret[0] == "vec":
        L.append("    double yv = 0.0;" if ret[1] == 1
                 else f"    {S.eigen_vec_type(ret[1])} yv = "
                      f"{S.eigen_vec_type(ret[1])}::Zero();")
    elif isinstance(ret, tuple) and ret[0] == "mat":
        L.append(f"    Eigen::Matrix<double, {ret[1]}, {ret[2]}, "
                 f"Eigen::ColMajor> M;")
    matrix_writes = [f for f in ep.writes_state if f != "x"]
    for mf in matrix_writes:
        L.append("    Cov Pout;")
    # --- pack ---
    if any(a == "x" for a in ep.kernel_args):
        L.append("    pack_state(x, x_in);")
    if any(a == "u" for a in ep.kernel_args):
        L.append("    pack_inputs(u, u_in);")
    # --- port_from: compute a port via another entry's kernel ---
    for port_name, src_entry in ep.port_from.items():
        src = next(e for e in ctx.m.entry_points if e.method == src_entry)
        args = [_arg_expr(a, ep, ctx) for a in src.kernel_args]
        L += _call(ctx.kname[src.fn], args, [(0, f"{port_name}m.data()")])
    # --- main kernel call ---
    args = [_arg_expr(a, ep, ctx) for a in ep.kernel_args]
    results = []
    oi = 0
    for w in ep.writes_state:
        results.append((oi, "x_out" if w == "x" else "Pout.data()"))
        oi += 1
    results += _return_results(ret, oi)
    L += _call(ctx.kname[ep.fn], args, results)
    # --- unpack / scatter / return ---
    L += _finish(ep, ctx, ret, matrix_writes)
    return L


def _arg_expr(a, ep, ctx) -> str:
    if a == "x":
        return "x_in"
    if a in ctx.fields:                       # matrix member (P)
        return f"{a}.data()"
    if a == "u":
        return "u_in"
    if a == "dt":
        return "&dt"
    if a == "t":
        return "&t"
    if a in ep.port_from or a in ep.synthesize_zero:
        port = _port(ctx.m, a)
        if port is not None and len(port.shape) == 2:
            return f"{a}m.data()"
        return "&dt" if a == "dt" else f"&{a}"   # scalar zero already declared
    if a.startswith("z"):
        port = _port(ctx.m, a)
        return "&z" if port.size == 1 else "z.data()"
    raise RuntimeError(f"emit: unhandled kernel arg {a!r} in {ep.method}")


def _return_results(ret, oi):
    if ret == "inputs":
        return [(oi, "u_out")]
    if ret == "outputs":
        return [(oi, "y_out")]
    if isinstance(ret, tuple) and ret[0] == "vec":
        return [(oi, "&yv" if ret[1] == 1 else "yv.data()")]
    if isinstance(ret, tuple) and ret[0] == "mat":
        return [(oi, "M.data()")]
    return []


def _finish(ep, ctx, ret, matrix_writes) -> list[str]:
    L = []
    # held writes update members
    if ctx.held and "x" in ep.writes_state:
        L.append("    unpack_state(x_out, x);")
    for mf in matrix_writes:
        L.append(f"    {mf} = Pout;")
    if ret == "state":                        # not-held fresh State return
        L += ["    State out;", "    unpack_state(x_out, out);",
              "    return out;"]
    elif ret == "inputs":
        u_port = _port(ctx.m, "u")
        L.append("    Inputs out;")
        L += _scatter_fields(u_port.fields, "u_out", "out")
        L.append("    return out;")
    elif ret == "outputs":
        yp = _port(ctx.m, ep.returns[0])
        L.append("    Outputs out;")
        L += _scatter_fields(yp.fields, "y_out", "out")
        L.append("    return out;")
    elif isinstance(ret, tuple) and ret[0] in ("vec", "mat"):
        L.append("    return yv;" if ret[0] == "vec" else "    return M;")
    return L


# ---------------------------------------------------------------------------
# Top-level
# ---------------------------------------------------------------------------

def emit_module_cpp(module, out_dir, *, class_name: str,
                    basename: str | None = None,
                    namespace: str = "manta_gen") -> dict:
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    base = basename or class_name.lower()
    q = class_name

    entries = [e for e in module.entry_points if e.deploy]

    # Collect + densify/rename the kernels every emitted method touches
    # (entry kernels + any port_from source kernels).
    needed: dict[str, object] = {}
    for ep in entries:
        needed[ep.fn] = module.functions[ep.fn]
        for src_entry in ep.port_from.values():
            se = next(e for e in module.entry_points if e.method == src_entry)
            needed[se.fn] = module.functions[se.fn]
    kname = {k: f"{base}_{k}" for k in needed}
    kernels = [_densify(fn, kname[k]) for k, fn in needed.items()]

    ctx = _Ctx(module, kname)

    # ---- header ----
    spec = ctx.spec
    H = ["// Generated by manta.codegen. Do not edit by hand.", "",
         "#pragma once", "", "#include <Eigen/Dense>", "",
         f"namespace {namespace} {{", "", f"class {q} {{", "public:"]
    if spec is not None:
        H += [f"    static constexpr int ambient_dim = {ctx.amb};",
              f"    static constexpr int tangent_dim = {ctx.tan};"]
    if ctx.mats:
        H.append("    using Cov = Eigen::Matrix<double, tangent_dim, "
                 "tangent_dim>;")
    H.append("")
    if spec is not None:
        _, init_vec = _spec_of(module)
        H += _state_struct(spec, init_vec) + [""]
    u_port = _port(module, "u")
    if u_port is not None:
        H += _struct_from_fields("Inputs", u_port.fields, with_init=True) + [""]
    y_port = _port(module, "y")
    if y_port is not None:
        H += _struct_from_fields("Outputs", y_port.fields, with_init=False) + [""]
    if ctx.held:
        H.append("    State x;")
        for mf in ctx.mats:
            H.append(f"    Cov {mf.name};")
        H += ["", f"    {q}();"]
        if ctx.mats:
            H.append("    void reset(const State& x0, const Cov& P0);")
        else:
            H.append("    void reset(const State& x0) { x = x0; }")
        H.append("    const State& state() const { return x; }")
        for mf in ctx.mats:
            H.append(f"    const Cov& covariance() const {{ return {mf.name}; }}")
        H.append("")
    elif spec is not None:
        # Threaded (Sim/LQR): a convenience default-state constructor.
        H += ["    State initial_state() const { return State{}; }", ""]
    for ep in entries:
        H.append(f"    {_method_decl(ep, ctx)};")
    H += ["};", "", f"}}  // namespace {namespace}"]
    (out_dir / f"{base}.hpp").write_text("\n".join(H) + "\n")

    # ---- source ----
    C = ["// Generated by manta.codegen. Do not edit by hand.", "",
         f'#include "{base}.hpp"', "", "#include <Eigen/Dense>", "",
         'extern "C" {', f'#include "{base}_kernels.h"', "}", "",
         f"namespace {namespace} {{", ""]
    if spec is not None:
        C += S.pack_state_lines(spec, q) + [""]
        C += S.unpack_state_lines(spec, q) + [""]
    if u_port is not None:
        C += _pack_inputs(u_port, q) + [""]
    if ctx.held:
        init = ["x = State{};"] + [f"{mf.name} = "
                                   f"({_matrix_init(mf)});" for mf in ctx.mats]
        C += [f"{q}::{q}() {{ {' '.join(init)} }}", ""]
        if ctx.mats:
            args = ", ".join(["const State& x0"]
                             + [f"const Cov& {mf.name}0" for mf in ctx.mats])
            sets = " ".join([f"x = x0;"]
                            + [f"{mf.name} = {mf.name}0;" for mf in ctx.mats])
            C += [f"void {q}::reset({args}) {{ {sets} }}", ""]
    for ep in entries:
        head = f"{_method_def_head(ep, ctx, q)} {{"
        C += [head] + _method_body(ep, ctx) + ["}", ""]
    C += [f"}}  // namespace {namespace}"]
    (out_dir / f"{base}.cpp").write_text("\n".join(C) + "\n")

    kpaths = emit_kernel_list(kernels, out_dir, basename=base)
    cmake = emit_cmakelists(out_dir, library_name=class_name, basename=base)
    return {"hpp": out_dir / f"{base}.hpp", "cpp": out_dir / f"{base}.cpp",
            "kernels_c": kpaths["c"], "kernels_h": kpaths["h"],
            "cmakelists": cmake}


def _matrix_init(mf) -> str:
    """Constructor initializer for a held matrix member (EKF P0 = 1e-2·I)."""
    import numpy as np
    M = np.asarray(mf.init, dtype=float)
    if M.ndim == 2 and np.allclose(M, np.diag(np.diag(M))) and \
            np.allclose(np.diag(M), M[0, 0]):
        return f"Cov::Identity() * {float(M[0, 0])!r}"
    rows, cols = M.shape
    body = ", ".join(repr(float(M[i, j])) for i in range(rows)
                     for j in range(cols))
    return f"(Cov() << {body}).finished()"


def _method_def_head(ep, ctx, q) -> str:
    """Out-of-class definition head (nested types qualified, no defaults)."""
    params = _params(ep, ctx, decl=False)
    return (f"{_ret_type(ep, ctx, q=q)} {q}::{ep.method}"
            f"({', '.join(params)}){_const(ep, ctx)}")

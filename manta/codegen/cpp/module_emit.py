"""The ONE generic C++ lowering of a typed `Module`.

`emit_module_cpp(module, …)` renders any Module — a Sim (oracle or deploy),
EKF, LQR, or recurrence — to a typed Eigen class on the flat-double kernel
ABI. It reads ONLY the Module and dispatches ONLY on the IR's types:
`StateRef`/`PortRef` kernel arguments and each Port's `Role`. There is no
name matching, no per-shape code, and no filtering — every entry point
lowers to a method, unconditionally.

  * State struct        ← the manifold spec (a State field's, or a
                          STATE-role port's), defaults from its init.
  * `Inputs`/`Outputs`  ← the CONTROL / OUTPUT ports' named fields.
  * `Cov` member        ← a matrix State field (held covariance).
  * one method / entry  ← HELD state lives as members (+ reset/state/
                          covariance); THREADED state is a `const State&`
                          parameter and a written manifold state is
                          returned fresh. Multi-value entries (an oracle
                          `step` returning state + readings) return an
                          emitted `<Method>Result` struct.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np

from . import _structs as S
from ._casadi import densify as _densify
from ._casadi import emit_kernel_call as _call
from .cmake import emit_cmakelists
from .kernels import emit_kernel_list
from .types import cpp_type_for
from ...ir.module import Hosting, PortRef, Role, StateRef
from ...ir.module import entry_ident as _ident


# ---------------------------------------------------------------------------
# Resolved per-module context
# ---------------------------------------------------------------------------

class _Ctx:
    def __init__(self, module, kname: dict):
        self.m = module
        self.kname = kname                       # {fn key: emitted C symbol}
        self.spec = module.spec
        self.amb = self.spec.ambient_dim if self.spec else 0
        self.tan = self.spec.tangent_dim if self.spec else 0
        self.held = module.hosting is Hosting.HELD
        self.mats = [f for f in module.state.fields if f.kind == "matrix"]
        ctl = module.ports_by_role(Role.CONTROL)
        self.u_port = ctl[0] if ctl else None
        out = module.ports_by_role(Role.OUTPUT)
        self.y_port = out[0] if out else None
        st = module.ports_by_role(Role.STATE)
        self.x_port = st[0] if st else None
        # the manifold state's init vector (struct defaults)
        self.x_init = None
        for f in module.state.fields:
            if f.kind == "manifold":
                self.x_init = f.init
        if self.x_init is None and self.x_port is not None:
            self.x_init = self.x_port.init

    def port(self, name):
        return self.m.port(name)

    def n_u(self) -> int:
        return (sum(f.dim for f in self.u_port.fields)
                if self.u_port is not None else 0)


def _mat_type(r: int, c: int) -> str:
    return f"Eigen::Matrix<double, {r}, {c}>"


# ---------------------------------------------------------------------------
# Struct emission
# ---------------------------------------------------------------------------

def _state_struct(spec, init_vec) -> list[str]:
    out = ["    struct State {"]
    for s in spec.slots:
        cpp = cpp_type_for(s.manifold)
        if init_vec is not None:
            seg = np.asarray(init_vec, dtype=float)[
                s.ambient_offset:s.ambient_offset + s.ambient_dim]
            init = (cpp.literal(float(seg[0])) if s.ambient_dim == 1
                    else cpp.literal([float(v) for v in seg]))
        else:
            init = cpp.zero
        out.append(f"        {cpp.decl} {_ident(s.name)} = {init};")
    out.append("    };")
    return out


def _fields_struct(name, fields, *, with_init: bool) -> list[str]:
    out = [f"    struct {name} {{"]
    for f in fields:
        tp = S.eigen_vec_type(f.dim)
        if not with_init:
            out.append(f"        {tp} {_ident(f.name)};")
        elif f.dim == 1:
            out.append(f"        {tp} {_ident(f.name)} = {float(f.default)!r};")
        else:
            out.append(f"        {tp} {_ident(f.name)} = {tp}::Zero();")
    if not fields:
        out.append("        // (none)")
    out.append("    };")
    return out


def _pack_fields_fn(struct: str, port, qcls: str, fname: str) -> list[str]:
    out = [f"static void {fname}(const {qcls}::{struct}& v, double* flat) {{"]
    if port is None or not port.fields:
        out.append("    (void)v; (void)flat;")
    else:
        off = 0
        for f in port.fields:
            ident = _ident(f.name)
            if f.dim == 1:
                out.append(f"    flat[{off}] = v.{ident};")
            else:
                for i in range(f.dim):
                    out.append(f"    flat[{off + i}] = v.{ident}[{i}];")
            off += f.dim
    out.append("}")
    return out


def _scatter_fields(fields, src, dst) -> list[str]:
    out, off = [], 0
    for f in fields:
        ident = _ident(f.name)
        if f.dim == 1:
            out.append(f"    {dst}.{ident} = {src}[{off}];")
        else:
            for i in range(f.dim):
                out.append(f"    {dst}.{ident}[{i}] = {src}[{off + i}];")
        off += f.dim
    return out


# ---------------------------------------------------------------------------
# Method signatures (typed, from the IR)
# ---------------------------------------------------------------------------

def _param_for(port, decl: bool, ctx) -> str | None:
    """The typed C++ parameter for one PortRef, by Role."""
    r = port.role
    if r is Role.CONTROL:
        return "const Inputs& u"
    if r is Role.MEASUREMENT:
        return f"const {S.eigen_vec_type(port.size)}& z"
    if r is Role.NOISE:
        return f"const {S.eigen_vec_type(max(port.size, 1))}& noise"
    if r is Role.PARAMETER:
        return f"const {S.eigen_vec_type(max(port.size, 1))}& params"
    if r is Role.TIMESTEP:
        return "double dt"
    if r is Role.TIME:
        return "double t = 0.0" if decl else "double t"
    if r is Role.STATE:
        # Named by port. Secondary STATE ports (e.g. an LQR's movable
        # x_ref) additionally get a convenience overload omitting them
        # (`_ref_overload`) — a plain default argument can't be used,
        # since it would need State's member initializers before the
        # enclosing class is complete.
        return ("const State& x" if port is ctx.x_port
                else f"const State& {_ident(port.name)}")
    if r is Role.MATRIX:
        return f"const {_mat_type(*port.shape)}& {_ident(port.name)}"
    raise NotImplementedError(
        f"param for role {r} — update _param_for (and the ARG_ROLES "
        f"contract in manta.ir.module) for the new Role.")


def _params(ep, ctx, *, decl: bool) -> list[str]:
    out = []
    if not ctx.held:
        for a in ep.args:
            if isinstance(a, StateRef):
                out.append("const State& x")
                break
    for a in ep.args:
        if isinstance(a, PortRef):
            out.append(_param_for(ctx.port(a.name), decl, ctx))
    return out


def _ret_type(ep, ctx, q: str | None = None) -> str:
    """Return type. Composite (a written THREADED state plus returns, or
    multiple returns) → the emitted `<Method>Result` struct."""
    pre = f"{q}::" if q else ""
    writes_state = "x" in ep.writes and not ctx.held
    n_out = (1 if writes_state else 0) + len(ep.returns)
    if n_out == 0:
        return "void"
    if n_out > 1:
        return f"{pre}{_result_struct_name(ep)}"
    if writes_state:
        return pre + "State"
    port = ctx.port(ep.returns[0])
    if port.role is Role.CONTROL:
        return pre + "Inputs"
    if port.role is Role.OUTPUT:
        return pre + "Outputs"
    if len(port.shape) == 2:
        return _mat_type(*port.shape)
    return S.eigen_vec_type(port.size)


def _result_struct_name(ep) -> str:
    return "".join(w.capitalize() for w in ep.method.split("_")) + "Result"


def _result_struct(ep, ctx) -> list[str]:
    """Composite-return struct: the fresh State + each returned port."""
    out = [f"    struct {_result_struct_name(ep)} {{"]
    if "x" in ep.writes and not ctx.held:
        out.append("        State x;")
    for name in ep.returns:
        port = ctx.port(name)
        tp = (_mat_type(*port.shape) if len(port.shape) == 2
              else S.eigen_vec_type(port.size))
        out.append(f"        {tp} {_ident(name)};")
    out.append("    };")
    return out


def _const(ep, ctx) -> str:
    return "" if ctx.held and ep.writes else " const"


def _method_decl(ep, ctx) -> str:
    params = ", ".join(_params(ep, ctx, decl=True))
    return f"{_ret_type(ep, ctx)} {ep.method}({params}){_const(ep, ctx)}"


def _method_def_head(ep, ctx, q: str) -> str:
    params = ", ".join(_params(ep, ctx, decl=False))
    return (f"{_ret_type(ep, ctx, q=q)} {q}::{ep.method}({params})"
            f"{_const(ep, ctx)}")


def _param_name(a, ctx) -> str:
    """The C++ parameter name `_param_for` gives one PortRef arg."""
    port = ctx.port(a.name)
    fixed = {Role.CONTROL: "u", Role.MEASUREMENT: "z", Role.NOISE: "noise",
             Role.PARAMETER: "params", Role.TIMESTEP: "dt", Role.TIME: "t"}
    if port.role in fixed:
        return fixed[port.role]
    return "x" if port is ctx.x_port else _ident(port.name)


def _ref_overload(ep, ctx) -> str | None:
    """An inline convenience overload omitting trailing secondary
    STATE-role args (an LQR's movable `x_ref`): they delegate to
    `State{}`, whose member defaults ARE the Module's built operating
    point. Inline body, not a default argument — a default argument may
    not use a nested class's member initializers before the enclosing
    class is complete."""
    args = list(ep.args)
    dropped = 0
    while args and isinstance(args[-1], PortRef):
        port = ctx.port(args[-1].name)
        if port.role is Role.STATE and port is not ctx.x_port:
            args.pop()
            dropped += 1
        else:
            break
    if not dropped:
        return None
    sub = replace(ep, args=tuple(args))
    params = ", ".join(_params(sub, ctx, decl=True))
    names = (["x"] if (not ctx.held
                       and any(isinstance(a, StateRef) for a in args))
             else [])
    names += [_param_name(a, ctx) for a in args if isinstance(a, PortRef)]
    call = ", ".join(names + ["State{}"] * dropped)
    return (f"{_ret_type(ep, ctx)} {ep.method}({params}){_const(ep, ctx)} "
            f"{{ return {ep.method}({call}); }}")


# ---------------------------------------------------------------------------
# Method bodies (typed-arg gather → kernel call → scatter/return)
# ---------------------------------------------------------------------------

def _arg_expr(a, ctx) -> str:
    if isinstance(a, StateRef):
        fld = ctx.m.state.field(a.name)
        return "x_in" if fld.kind == "manifold" else f"{a.name}.data()"
    port = ctx.port(a.name)
    r = port.role
    if r is Role.CONTROL:
        return "u_in"
    if r is Role.MEASUREMENT:
        return "&z" if port.size == 1 else "z.data()"
    if r is Role.NOISE:
        return "noise.data()"
    if r is Role.PARAMETER:
        return "params.data()"
    if r is Role.TIMESTEP:
        return "&dt"
    if r is Role.TIME:
        return "&t"
    if r is Role.STATE:
        return ("x_in" if port is ctx.x_port
                else f"{_ident(port.name)}_in")
    if r is Role.MATRIX:
        return f"{_ident(port.name)}.data()"
    raise NotImplementedError(
        f"arg for role {r} — update _arg_expr (and the ARG_ROLES "
        f"contract in manta.ir.module) for the new Role.")


def _method_body(ep, ctx) -> list[str]:
    L: list[str] = []
    # Manifold reads: (param/member name, kernel buffer) — the held/
    # threaded state plus any secondary STATE-role port (e.g. x_ref).
    manifold_reads: list[tuple[str, str]] = []
    for a in ep.args:
        if isinstance(a, StateRef) \
                and ctx.m.state.field(a.name).kind == "manifold":
            pair = ("x", "x_in")
        elif isinstance(a, PortRef) \
                and ctx.port(a.name).role is Role.STATE:
            port = ctx.port(a.name)
            pair = (("x", "x_in") if port is ctx.x_port
                    else (_ident(port.name), f"{_ident(port.name)}_in"))
        else:
            continue
        if pair not in manifold_reads:
            manifold_reads.append(pair)
    writes_manifold = "x" in ep.writes
    composite = _ret_type(ep, ctx).endswith("Result")

    # buffers
    for _, buf in manifold_reads:
        L.append(f"    double {buf}[{ctx.amb}];")
    if writes_manifold:
        L.append(f"    double x_out[{ctx.amb}];")
    if any(isinstance(a, PortRef) and ctx.port(a.name).role is Role.CONTROL
           for a in ep.args):
        n = ctx.n_u()
        L.append(f"    double u_in[{max(n, 1)}];"
                 + ("" if n else "   // no inputs"))
    matrix_writes = [w for w in ep.writes if w != "x"]
    for w in matrix_writes:
        L.append(f"    Cov {w}_out;")
    ret_bufs: list[tuple[str, str]] = []      # (port name, buffer expr)
    for name in ep.returns:
        port = ctx.port(name)
        buf = f"r_{_ident(name)}"
        if port.role is Role.CONTROL:
            L.append(f"    double {buf}[{max(ctx.n_u(), 1)}];")
            ret_bufs.append((name, buf))
        elif port.role is Role.OUTPUT:
            ydim = sum(f.dim for f in port.fields)
            L.append(f"    double {buf}[{max(ydim, 1)}];")
            ret_bufs.append((name, buf))
        elif len(port.shape) == 2:
            L.append(f"    Eigen::Matrix<double, {port.shape[0]}, "
                     f"{port.shape[1]}, Eigen::ColMajor> {buf};")
            ret_bufs.append((name, f"{buf}.data()"))
        elif port.size == 1:
            L.append(f"    double {buf} = 0.0;")
            ret_bufs.append((name, f"&{buf}"))
        else:
            tp = S.eigen_vec_type(port.size)
            L.append(f"    {tp} {buf} = {tp}::Zero();")
            ret_bufs.append((name, f"{buf}.data()"))

    # pack
    for name, buf in manifold_reads:
        L.append(f"    pack_state({name}, {buf});")
    if any(isinstance(a, PortRef) and ctx.port(a.name).role is Role.CONTROL
           for a in ep.args):
        L.append("    pack_inputs(u, u_in);")

    # kernel call
    results: list[tuple[int, str]] = []
    oi = 0
    for w in ep.writes:
        results.append((oi, "x_out" if w == "x" else f"{w}_out.data()"))
        oi += 1
    for (_, buf) in ret_bufs:
        results.append((oi, buf))
        oi += 1
    L += _call(ctx.kname[ep.fn], [_arg_expr(a, ctx) for a in ep.args],
               results)

    # scatter / return
    if ctx.held and writes_manifold:
        L.append("    unpack_state(x_out, x);")
    for w in matrix_writes:
        L.append(f"    {w} = {w}_out;")

    rt = _ret_type(ep, ctx)
    if rt == "void":
        return L
    if composite:
        L.append(f"    {rt} out;")
        if writes_manifold and not ctx.held:
            L.append("    unpack_state(x_out, out.x);")
        for name, _ in ret_bufs:
            L += _ret_assign(ctx, name, f"out.{_ident(name)}",
                             raw=f"r_{_ident(name)}")
        L.append("    return out;")
        return L
    if writes_manifold and not ctx.held:
        L += ["    State out;", "    unpack_state(x_out, out);",
              "    return out;"]
        return L
    name = ep.returns[0]
    port = ctx.port(name)
    if port.role is Role.CONTROL:
        L.append("    Inputs out;")
        L += _scatter_fields(ctx.u_port.fields, f"r_{_ident(name)}", "out")
        L.append("    return out;")
    elif port.role is Role.OUTPUT:
        L.append("    Outputs out;")
        L += _scatter_fields(port.fields, f"r_{_ident(name)}", "out")
        L.append("    return out;")
    else:
        L.append(f"    return r_{_ident(name)};")
    return L


def _ret_assign(ctx, name, dst, raw) -> list[str]:
    """Copy a raw return buffer into a composite-result member."""
    port = ctx.port(name)
    if port.role in (Role.CONTROL, Role.OUTPUT):     # pragma: no cover
        raise NotImplementedError(
            "struct-valued member inside a composite result")
    return [f"    {dst} = {raw};"]


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

    entries = list(module.entry_points)
    kname = {ep.fn: f"{base}_{ep.fn}" for ep in entries}
    kernels = [_densify(module.functions[k], n) for k, n in kname.items()]
    ctx = _Ctx(module, kname)

    # ---- header ----
    H = ["// Generated by manta.codegen. Do not edit by hand.", "",
         "#pragma once", "", "#include <Eigen/Dense>", "",
         f"namespace {namespace} {{", "", f"class {q} {{", "public:"]
    if ctx.spec is not None:
        H += [f"    static constexpr int ambient_dim = {ctx.amb};",
              f"    static constexpr int tangent_dim = {ctx.tan};"]
    if ctx.mats:
        H.append("    using Cov = Eigen::Matrix<double, tangent_dim, "
                 "tangent_dim>;")
    H.append("")
    if ctx.spec is not None:
        H += _state_struct(ctx.spec, ctx.x_init) + [""]
    if ctx.u_port is not None:
        H += _fields_struct("Inputs", ctx.u_port.fields, with_init=True) + [""]
    if ctx.y_port is not None:
        H += _fields_struct("Outputs", ctx.y_port.fields,
                            with_init=False) + [""]
    for ep in entries:
        if _ret_type(ep, ctx).endswith("Result"):
            H += _result_struct(ep, ctx) + [""]
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
            H.append(f"    const Cov& covariance() const "
                     f"{{ return {mf.name}; }}")
        H.append("")
    elif ctx.spec is not None:
        H += ["    State initial_state() const { return State{}; }", ""]
    for ep in entries:
        H.append(f"    {_method_decl(ep, ctx)};")
        ov = _ref_overload(ep, ctx)
        if ov is not None:
            H.append(f"    {ov}")
    H += ["};", "", f"}}  // namespace {namespace}"]
    (out_dir / f"{base}.hpp").write_text("\n".join(H) + "\n")

    # ---- source ----
    C = ["// Generated by manta.codegen. Do not edit by hand.", "",
         f'#include "{base}.hpp"', "", "#include <Eigen/Dense>", "",
         'extern "C" {', f'#include "{base}_kernels.h"', "}", "",
         f"namespace {namespace} {{", ""]
    if ctx.spec is not None:
        C += S.pack_state_lines(ctx.spec, q) + [""]
        C += S.unpack_state_lines(ctx.spec, q) + [""]
    if ctx.u_port is not None:
        C += _pack_fields_fn("Inputs", ctx.u_port, q, "pack_inputs") + [""]
    if ctx.held:
        init = ["x = State{};"]
        for mf in ctx.mats:
            init.append(f"{mf.name} = ({_matrix_init(mf)});")
        C += [f"{q}::{q}() {{ {' '.join(init)} }}", ""]
        if ctx.mats:
            args = ", ".join(["const State& x0"]
                             + [f"const Cov& {mf.name}0" for mf in ctx.mats])
            sets = " ".join(["x = x0;"]
                            + [f"{mf.name} = {mf.name}0;" for mf in ctx.mats])
            C += [f"void {q}::reset({args}) {{ {sets} }}", ""]
    for ep in entries:
        C += [f"{_method_def_head(ep, ctx, q)} {{"]
        C += _method_body(ep, ctx)
        C += ["}", ""]
    C += [f"}}  // namespace {namespace}"]
    (out_dir / f"{base}.cpp").write_text("\n".join(C) + "\n")

    kpaths = emit_kernel_list(kernels, out_dir, basename=base)
    cmake = emit_cmakelists(out_dir, library_name=class_name, basename=base)
    return {"hpp": out_dir / f"{base}.hpp", "cpp": out_dir / f"{base}.cpp",
            "kernels_c": kpaths["c"], "kernels_h": kpaths["h"],
            "cmakelists": cmake}


def _matrix_init(mf) -> str:
    """Constructor initializer for a held matrix member (e.g. P0 = 1e-2·I)."""
    M = np.asarray(mf.init, dtype=float)
    if M.ndim == 2 and np.allclose(M, np.diag(np.diag(M))) and \
            np.allclose(np.diag(M), M[0, 0]):
        return f"Cov::Identity() * {float(M[0, 0])!r}"
    rows, cols = M.shape
    body = ", ".join(repr(float(M[i, j])) for i in range(rows)
                     for j in range(cols))
    return f"(Cov() << {body}).finished()"

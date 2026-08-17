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

from collections import namedtuple
from dataclasses import replace
from pathlib import Path

import numpy as np

from ...ir.module import Hosting, PortRef, Role, StateRef
from ...ir.module import entry_ident as _ident
from ..target import for_role
from . import _structs as S
from ._casadi import densify as _densify
from ._casadi import emit_kernel_call as _call
from .cmake import emit_cmakelists
from .kernels import emit_kernel_list
from .types import cpp_type_for

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
        self.mats = module.matrix_fields
        self.u_port = module.sole_port(Role.CONTROL)
        self.y_port = module.sole_port(Role.OUTPUT)
        self.x_port = module.sole_port(Role.STATE)
        self.x_init = module.initial_x      # manifold init → struct defaults
        self.has_clock = self.held and any(
            isinstance(a, PortRef) and module.port(a.name).role is Role.TIME
            for ep in module.entry_points for a in ep.args)

    def port(self, name):
        return self.m.port(name)

    def n_u(self) -> int:
        return (sum(f.dim for f in self.u_port.fields)
                if self.u_port is not None else 0)


def _mat_type(r: int, c: int) -> str:
    return f"Eigen::Matrix<double, {r}, {c}>"


def _buf_dim(n: int) -> int:
    """A C scratch buffer (or Eigen vector type) needs ≥1 element even for a
    zero-width port — a control-free craft still declares `double u_in[1]`."""
    return max(int(n), 1)


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

#: One PortRef argument's complete C++ ABI: how it is declared in a method
#: signature (`decl`), the parameter's `name`, and the expression passed to
#: the flat kernel (`arg`). One record per Role keeps all three in lockstep.
_PortABI = namedtuple("_PortABI", ("decl", "name", "arg"))


def _port_abi(port, ctx, *, decl: bool = False) -> _PortABI:
    """The C++ ABI for one PortRef argument, by Role — its typed parameter
    declaration, the parameter name, and the kernel-call expression — from
    ONE source so the three never drift (a decl/name/arg mismatch would emit
    uncompilable C++). `decl` only affects TIME's trailing default argument;
    `name`/`arg` are decl-independent.

    Secondary STATE ports (an LQR's movable `x_ref`) additionally get a
    convenience overload omitting them (`_ref_overload`) — a plain default
    argument can't be used, since it would need State's member initializers
    before the enclosing class is complete."""
    def scalar_or_data(name: str, size: int) -> str:
        # a (buffered) size-1 port is a plain double — take its address;
        # a vector hands over its contiguous storage.
        return f"&{name}" if size == 1 else f"{name}.data()"

    return for_role(port.role, {
        Role.CONTROL: lambda: _PortABI("const Inputs& u", "u", "u_in"),
        Role.MEASUREMENT: lambda: _PortABI(
            f"const {S.eigen_vec_type(port.size)}& z", "z",
            scalar_or_data("z", port.size)),
        Role.NOISE: lambda: _PortABI(
            f"const {S.eigen_vec_type(_buf_dim(port.size))}& noise", "noise",
            scalar_or_data("noise", _buf_dim(port.size))),
        Role.PARAMETER: lambda: _PortABI(
            f"const {S.eigen_vec_type(_buf_dim(port.size))}& params", "params",
            scalar_or_data("params", _buf_dim(port.size))),
        Role.TIMESTEP: lambda: _PortABI("double dt", "dt", "&dt"),
        Role.TIME: lambda: _PortABI(
            ("double t = 0.0" if decl and not ctx.held else "double t"),
            "t", "&t"),
        Role.STATE: lambda: _PortABI(
            "const State& x" if port is ctx.x_port
            else f"const State& {_ident(port.name)}",
            "x" if port is ctx.x_port else _ident(port.name),
            "x_in" if port is ctx.x_port else f"{_ident(port.name)}_in"),
        Role.MATRIX: lambda: _PortABI(
            f"const {_mat_type(*port.shape)}& {_ident(port.name)}",
            _ident(port.name), f"{_ident(port.name)}.data()"),
    }, who="_port_abi")


def _param_for(port, decl: bool, ctx) -> str | None:
    """The typed C++ parameter declaration for one PortRef, by Role."""
    return _port_abi(port, ctx, decl=decl).decl


def _params(ep, ctx, *, decl: bool) -> list[str]:
    out = []
    if not ctx.held:
        for a in ep.args:
            if isinstance(a, StateRef):
                out.append("const State& x")
                break
    for a in ep.args:
        if isinstance(a, PortRef):
            parameter = _param_for(ctx.port(a.name), decl, ctx)
            if parameter is None:
                raise TypeError(
                    f"entry {ep.method!r} port {a.name!r} has no C++ parameter"
                )
            out.append(parameter)
    return out


def _manifold_writes(ep, ctx) -> list[str]:
    """The manifold State fields `ep` writes (kind-based, never by name)."""
    return [w for w in ep.writes if ctx.m.state.field(w).kind == "manifold"]


def _matrix_writes(ep, ctx) -> list[str]:
    """The matrix State fields `ep` writes (an EKF's covariance)."""
    return [w for w in ep.writes if ctx.m.state.field(w).kind == "matrix"]


def _n_out(ep, ctx) -> int:
    """How many values the method hands back: a written THREADED manifold
    state counts (it is returned fresh) plus every declared return."""
    writes_state = bool(_manifold_writes(ep, ctx)) and not ctx.held
    return (1 if writes_state else 0) + len(ep.returns)


def _is_composite(ep, ctx) -> bool:
    """Composite return (→ the emitted `<Method>Result` struct)."""
    return _n_out(ep, ctx) > 1


def _ret_type(ep, ctx, q: str | None = None) -> str:
    """Return type. Composite (a written THREADED state plus returns, or
    multiple returns) → the emitted `<Method>Result` struct."""
    pre = f"{q}::" if q else ""
    writes_state = bool(_manifold_writes(ep, ctx)) and not ctx.held
    n_out = _n_out(ep, ctx)
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
    if _manifold_writes(ep, ctx) and not ctx.held:
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
    return _port_abi(ctx.port(a.name), ctx).name


def _default_expr(port, ctx) -> str | None:
    """The C++ expression for a port's built-in default, or None if it has
    none. A secondary STATE port delegates to `State{}`, whose member
    defaults ARE the Module's built operating point; a MATRIX port with an
    `init` (an LQR's gain / feed-forward) delegates to the emitted
    `<name>_default()` accessor."""
    if port.role is Role.STATE and port is not ctx.x_port:
        return "State{}"
    if port.role is Role.MATRIX and port.init is not None:
        return f"{_ident(port.name)}_default()"
    return None


def _defaulted_matrices(ctx) -> list:
    """MATRIX ports carrying a built-in default, in declaration order."""
    return [p for p in ctx.m.ports_by_role(Role.MATRIX) if p.init is not None]


def _matrix_default_decls(ctx) -> list[str]:
    """Header declarations for each defaulted MATRIX port's accessor."""
    return [f"    static const {_mat_type(*p.shape)}& "
            f"{_ident(p.name)}_default();" for p in _defaulted_matrices(ctx)]


def _matrix_default_defs(ctx, q: str) -> list[str]:
    """Source definitions for the defaulted-MATRIX accessors — a function
    with a static local, so the constant lives in the object file and the
    header stays free of data."""
    L: list[str] = []
    for p in _defaulted_matrices(ctx):
        tp = _mat_type(*p.shape)
        M = np.asarray(p.init, dtype=float).reshape(p.shape)
        # Eigen's comma initializer is ROW-major; `.data()` (what the
        # kernel call hands over) is column-major. Feeding rows here and
        # letting Eigen store it is what keeps the two consistent.
        body = ", ".join(repr(float(v)) for v in M.reshape(-1))
        name = _ident(p.name)
        L += [f"const {tp}& {q}::{name}_default() {{",
              f"    static const {tp} v = ({tp}() << {body}).finished();",
              "    return v;", "}", ""]
    return L


def _ref_overloads(ep, ctx) -> list[str]:
    """Inline convenience overloads omitting trailing args that carry a
    built-in default — an LQR's movable `x_ref`, gain and feed-forward.

    One per drop depth, longest first, so every prefix stays callable:
    `control(x, x_ref, K)` retargets with the built gain, `control(x)`
    flies the built operating point outright. Inline bodies, not default
    arguments: a default argument may not use a nested class's member
    initializers before the enclosing class is complete."""
    args = list(ep.args)
    defaults: list[str] = []
    while args and isinstance(args[-1], PortRef):
        expr = _default_expr(ctx.port(args[-1].name), ctx)
        if expr is None:
            break
        args.pop()
        defaults.insert(0, expr)

    out: list[str] = []
    for k in range(1, len(defaults) + 1):
        kept = list(ep.args)[:len(ep.args) - k]
        sub = replace(ep, args=tuple(kept))
        params = ", ".join(_params(sub, ctx, decl=True))
        names = (["x"] if (not ctx.held
                           and any(isinstance(a, StateRef) for a in kept))
                 else [])
        names += [_param_name(a, ctx) for a in kept if isinstance(a, PortRef)]
        call = ", ".join(names + defaults[len(defaults) - k:])
        out.append(f"{_ret_type(ep, ctx)} {ep.method}({params})"
                   f"{_const(ep, ctx)} {{ return {ep.method}({call}); }}")
    return out


def _clock_overload(ep, ctx) -> str | None:
    """Held modules own logical time; omit a trailing TIME argument by
    forwarding the runtime's clock. Predict-like methods resynchronize and
    advance the clock in their primary method body."""
    if not ctx.has_clock or not ep.args or not isinstance(ep.args[-1], PortRef):
        return None
    if ctx.port(ep.args[-1].name).role is not Role.TIME:
        return None
    kept = replace(ep, args=ep.args[:-1])
    params = ", ".join(_params(kept, ctx, decl=True))
    names = [_param_name(a, ctx) for a in kept.args if isinstance(a, PortRef)]
    call = ", ".join(names + ["logical_time_"])
    ret = _ret_type(ep, ctx)
    invoke = f"{ep.method}({call})"
    body = f"{invoke};" if ret == "void" else f"return {invoke};"
    return f"{ret} {ep.method}({params}){_const(ep, ctx)} {{ {body} }}"


# ---------------------------------------------------------------------------
# Method bodies (typed-arg gather → kernel call → scatter/return)
# ---------------------------------------------------------------------------

def _arg_expr(a, ctx) -> str:
    if isinstance(a, StateRef):
        fld = ctx.m.state.field(a.name)
        return "x_in" if fld.kind == "manifold" else f"{a.name}.data()"
    return _port_abi(ctx.port(a.name), ctx).arg


def _manifold_reads(ep, ctx) -> list[tuple[str, str]]:
    """(member/param name, kernel buffer) for each manifold state read: the
    held/threaded state plus any secondary STATE-role port (an LQR's x_ref)."""
    reads: list[tuple[str, str]] = []
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
        if pair not in reads:
            reads.append(pair)
    return reads


def _has_control_arg(ep, ctx) -> bool:
    return any(isinstance(a, PortRef) and ctx.port(a.name).role is Role.CONTROL
               for a in ep.args)


def _body_buffers(ep, ctx, reads, writes_manifold, matrix_writes):
    """Local scratch declarations; returns `(lines, ret_bufs)` where each
    `ret_buf` is `(port name, buffer expr passed to the kernel)`."""
    L: list[str] = []
    for _, buf in reads:
        L.append(f"    double {buf}[{ctx.amb}];")
    if writes_manifold:
        L.append(f"    double x_out[{ctx.amb}];")
    if _has_control_arg(ep, ctx):
        n = ctx.n_u()
        L.append(f"    double u_in[{_buf_dim(n)}];"
                 + ("" if n else "   // no inputs"))
    for w in matrix_writes:
        L.append(f"    Cov {w}_out;")
    ret_bufs: list[tuple[str, str]] = []
    for name in ep.returns:
        port = ctx.port(name)
        buf = f"r_{_ident(name)}"
        if port.role is Role.CONTROL:
            L.append(f"    double {buf}[{_buf_dim(ctx.n_u())}];")
            ret_bufs.append((name, buf))
        elif port.role is Role.OUTPUT:
            ydim = sum(f.dim for f in port.fields)
            L.append(f"    double {buf}[{_buf_dim(ydim)}];")
            ret_bufs.append((name, buf))
        elif len(port.shape) == 2:
            # Eigen forbids an explicitly column-major fixed row vector.
            # A 1×N matrix has identical contiguous storage either way, so
            # use RowMajor for that shape while retaining CasADi-compatible
            # column-major storage for every genuine matrix.
            storage = ("Eigen::RowMajor"
                       if port.shape[0] == 1 and port.shape[1] != 1
                       else "Eigen::ColMajor")
            L.append(f"    Eigen::Matrix<double, {port.shape[0]}, "
                     f"{port.shape[1]}, {storage}> {buf};")
            ret_bufs.append((name, f"{buf}.data()"))
        elif port.size == 1:
            L.append(f"    double {buf} = 0.0;")
            ret_bufs.append((name, f"&{buf}"))
        else:
            tp = S.eigen_vec_type(port.size)
            L.append(f"    {tp} {buf} = {tp}::Zero();")
            ret_bufs.append((name, f"{buf}.data()"))
    return L, ret_bufs


def _body_pack(ep, ctx, reads) -> list[str]:
    L = [f"    pack_state({name}, {buf});" for name, buf in reads]
    if _has_control_arg(ep, ctx):
        L.append("    pack_inputs(u, u_in);")
    return L


def _body_call(ep, ctx, ret_bufs) -> list[str]:
    manifold = set(_manifold_writes(ep, ctx))
    results: list[tuple[int, str]] = []
    oi = 0
    for w in ep.writes:
        results.append((oi, "x_out" if w in manifold else f"{w}_out.data()"))
        oi += 1
    for (_, buf) in ret_bufs:
        results.append((oi, buf))
        oi += 1
    return _call(ctx.kname[ep.fn], [_arg_expr(a, ctx) for a in ep.args],
                 results)


def _body_return(ep, ctx, reads, writes_manifold, composite, ret_bufs,
                 matrix_writes) -> list[str]:
    L: list[str] = []
    if ctx.held and writes_manifold:
        L.append("    unpack_state(x_out, x);")
    for w in matrix_writes:
        L.append(f"    {w} = {w}_out;")
    roles = {ctx.port(a.name).role for a in ep.args
             if isinstance(a, PortRef)}
    if ctx.has_clock and Role.TIME in roles and Role.TIMESTEP in roles:
        L.append("    logical_time_ = t + dt;")

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
        return L + ["    State out;", "    unpack_state(x_out, out);",
                    "    return out;"]
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


def _method_body(ep, ctx) -> list[str]:
    """The four phases — declare scratch buffers, pack typed args in, call the
    flat-C kernel, scatter/return — each rendered by a helper above."""
    reads = _manifold_reads(ep, ctx)
    writes_manifold = bool(_manifold_writes(ep, ctx))
    composite = _is_composite(ep, ctx)
    matrix_writes = _matrix_writes(ep, ctx)
    L, ret_bufs = _body_buffers(ep, ctx, reads, writes_manifold, matrix_writes)
    L += _body_pack(ep, ctx, reads)
    L += _body_call(ep, ctx, ret_bufs)
    L += _body_return(ep, ctx, reads, writes_manifold, composite, ret_bufs,
                      matrix_writes)
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
        if _is_composite(ep, ctx):
            H += _result_struct(ep, ctx) + [""]
    if ctx.held:
        if ctx.has_clock:
            H += ["    struct Checkpoint {", "        State x;"]
            for mf in ctx.mats:
                H.append(f"        Cov {mf.name};")
            H += ["        double time;", "    };", ""]
        H.append("    State x;")
        for mf in ctx.mats:
            H.append(f"    Cov {mf.name};")
        if ctx.has_clock:
            H.append("    double logical_time_ = 0.0;")
        H += ["", f"    {q}();"]
        if ctx.mats:
            H.append("    void reset(const State& x0, const Cov& P0);")
        else:
            reset = "x = x0;"
            if ctx.has_clock:
                reset += " logical_time_ = 0.0;"
            H.append(f"    void reset(const State& x0) {{ {reset} }}")
        H.append("    const State& state() const { return x; }")
        for mf in ctx.mats:
            H.append(f"    const Cov& covariance() const "
                     f"{{ return {mf.name}; }}")
        if ctx.has_clock:
            H += ["    double time() const { return logical_time_; }",
                  "    Checkpoint checkpoint() const;",
                  "    void restore(const Checkpoint& checkpoint);"]
        H.append("")
    elif ctx.spec is not None:
        H += ["    State initial_state() const { return State{}; }", ""]
    H += _matrix_default_decls(ctx)
    if _defaulted_matrices(ctx):
        H.append("")
    for ep in entries:
        H.append(f"    {_method_decl(ep, ctx)};")
        clock_overload = _clock_overload(ep, ctx)
        if clock_overload is not None:
            H.append(f"    {clock_overload}")
        for ov in _ref_overloads(ep, ctx):
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
    C += _matrix_default_defs(ctx, q)
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
            if ctx.has_clock:
                sets += " logical_time_ = 0.0;"
            C += [f"void {q}::reset({args}) {{ {sets} }}", ""]
        if ctx.has_clock:
            checkpoint_values = ["x"] + [mf.name for mf in ctx.mats]
            checkpoint_values.append("logical_time_")
            C += [f"{q}::Checkpoint {q}::checkpoint() const {{",
                  f"    return Checkpoint{{{', '.join(checkpoint_values)}}};",
                  "}", "", f"void {q}::restore(const Checkpoint& c) {{",
                  "    x = c.x;"]
            for mf in ctx.mats:
                C.append(f"    {mf.name} = c.{mf.name};")
            C += ["    logical_time_ = c.time;", "}", ""]
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

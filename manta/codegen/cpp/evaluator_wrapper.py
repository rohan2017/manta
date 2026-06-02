"""The one generic Eigen C++ wrapper emitter for evaluator blocks.

Renders any `EvaluatorSpec` (Sim, LQR, or a recurrence block) to a typed
C++ class: State/Inputs/(Outputs) structs and one method per entry point,
each packing into flat `double[]`, calling its CasADi kernel via the shared
`_call` boilerplate, and unpacking the typed result. This replaces the
three former per-kind wrappers (`wrapper.py`, `lqr_wrapper.py`,
`recurrence_wrapper.py`) — a new evaluator block needs no new emitter.
"""

from __future__ import annotations

from pathlib import Path

from . import _structs as S
from .ekf_wrapper import _call
from ..evaluator import ArgSource, EvaluatorSpec, ParamKind, ReturnKind

_P = ParamKind
_A = ArgSource
_R = ReturnKind


# ---------------------------------------------------------------------------
# Signature / type helpers
# ---------------------------------------------------------------------------

def _param_decl(p: ParamKind, *, decl: bool) -> str:
    if p is _P.STATE:
        return "const State& x"
    if p is _P.INPUTS:
        return "const Inputs& u"
    if p is _P.DT:
        return "double dt"
    if p is _P.T:
        return "double t = 0.0" if decl else "double t"
    raise ValueError(p)


def _ret_type(ret, qcls: str | None = None) -> str:
    """Return-type spelling. In an out-of-class method *definition* the
    nested struct types must be qualified (`Cls::State`); pass `qcls` there.
    In the in-class *declaration* they're unqualified (`qcls=None`)."""
    pre = f"{qcls}::" if qcls else ""
    k = ret.kind
    if k is _R.STATE:
        return f"{pre}State"
    if k is _R.INPUTS:
        return f"{pre}Inputs"
    if k is _R.OUTPUTS_AND_STATE:
        return f"{pre}Outputs"
    if k is _R.VEC:
        return S.eigen_vec_type(ret.dim)
    if k is _R.MATRIX:
        return f"Eigen::Matrix<double, {ret.rows}, {ret.cols}>"
    raise ValueError(k)


def _struct_lines(name: str, fields, *, empty_comment: str | None) -> list[str]:
    out = [f"    struct {name} {{"]
    for f in fields:
        if f.init_expr is None:
            out.append(f"        {f.type_expr} {f.ident};")
        else:
            out.append(f"        {f.type_expr} {f.ident} = {f.init_expr};")
    if not fields and empty_comment:
        out.append(f"        {empty_comment}")
    out.append("    };")
    return out


# ---------------------------------------------------------------------------
# Method body
# ---------------------------------------------------------------------------

def _scatter_lines(fields, src: str, dst: str) -> list[str]:
    """`dst.<ident> = src[off..]` for each field (reverse of pack)."""
    out, off = [], 0
    for f in fields:
        if f.dim == 1:
            out.append(f"    {dst}.{f.ident} = {src}[{off}];")
        else:
            for i in range(f.dim):
                out.append(f"    {dst}.{f.ident}[{i}] = {src}[{off + i}];")
        off += f.dim
    return out


def _method_lines(ep, espec: EvaluatorSpec, qcls: str) -> list[str]:
    spec = espec.spec
    sig = ", ".join(_param_decl(p, decl=False) for p in ep.params)
    head = (f"{_ret_type(ep.ret, qcls)} {qcls}::{ep.method_name}({sig})"
            + (" const" if ep.const else "") + " {")

    # No-kernel entry (initial_state).
    if ep.fn_name is None:
        return [head, "    return State{};", "}", ""]

    L = [head]
    args = ep.kernel_args
    ret = ep.ret
    n_in = sum(f.dim for f in espec.input_fields)

    needs_x_in = _A.STATE in args
    needs_x_out = ret.kind in (_R.STATE, _R.OUTPUTS_AND_STATE)
    needs_u_in = _A.INPUTS in args

    # --- buffers ---
    if needs_x_in and needs_x_out:
        L.append(f"    double x_in[{spec.ambient_dim}], x_out[{spec.ambient_dim}];")
    elif needs_x_in:
        L.append(f"    double x_in[{spec.ambient_dim}];")
    elif needs_x_out:
        L.append(f"    double x_out[{spec.ambient_dim}];")
    if needs_u_in:
        if n_in > 0:
            L.append(f"    double u_in[{n_in}];")
        else:
            L.append("    double u_in[1] = {0.0};   // no inputs declared")
    if _A.ZERO_DT in args:
        L.append("    double dt = 0.0;")

    # result buffers
    if ret.kind is _R.INPUTS:
        L.append(f"    double u_out[{max(len(espec.input_fields), 1)}];")
    elif ret.kind is _R.VEC:
        if ret.dim == 1:
            L.append("    double y = 0.0;")
        else:
            L.append(f"    {S.eigen_vec_type(ret.dim)} y = "
                     f"{S.eigen_vec_type(ret.dim)}::Zero();")
    elif ret.kind is _R.MATRIX:
        L.append(f"    Eigen::Matrix<double, {ret.rows}, {ret.cols}, "
                 f"Eigen::ColMajor> M;")
    elif ret.kind is _R.OUTPUTS_AND_STATE:
        out_dim = sum(f.dim for f in (espec.output_fields or ())) or 1
        L.append(f"    double y_out[{out_dim}];")

    # --- pack ---
    if needs_x_in:
        L.append("    pack_state(x, x_in);")
    if needs_u_in:
        L.append("    pack_inputs(u, u_in);")

    # --- kernel call ---
    arg_expr = {_A.STATE: "x_in", _A.INPUTS: "u_in",
                _A.DT: "&dt", _A.T: "&t", _A.ZERO_DT: "&dt"}
    call_args = [arg_expr[a] for a in args]
    if ret.kind in (_R.STATE,):
        results = [(0, "x_out")]
    elif ret.kind is _R.INPUTS:
        results = [(0, "u_out")]
    elif ret.kind is _R.VEC:
        results = [(0, "&y" if ret.dim == 1 else "y.data()")]
    elif ret.kind is _R.MATRIX:
        results = [(0, "M.data()")]
    elif ret.kind is _R.OUTPUTS_AND_STATE:
        results = [(ret.state_res, "x_out"), (ret.out_res, "y_out")]
    L += _call(ep.fn_name, call_args, results)

    # --- unpack / return ---
    if ret.kind is _R.STATE:
        L += ["    State out;", "    unpack_state(x_out, out);",
              "    return out;"]
    elif ret.kind is _R.INPUTS:
        L.append("    Inputs out;")
        L += _scatter_lines(espec.input_fields, "u_out", "out")
        L.append("    return out;")
    elif ret.kind is _R.VEC:
        L.append("    return y;")
    elif ret.kind is _R.MATRIX:
        L.append("    return M;")
    elif ret.kind is _R.OUTPUTS_AND_STATE:
        L.append("    unpack_state(x_out, x);")   # held member
        L.append("    Outputs out;")
        L += _scatter_lines(espec.output_fields or (), "y_out", "out")
        L.append("    return out;")
    L += ["}", ""]
    return L


# ---------------------------------------------------------------------------
# Top-level emit
# ---------------------------------------------------------------------------

def emit_evaluator_wrapper(espec: EvaluatorSpec, out_dir: str | Path, *,
                           class_name: str, basename: str | None = None,
                           namespace: str = "manta_gen") -> dict[str, Path]:
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    base = basename or class_name.lower()
    hpp = out_dir / f"{base}.hpp"
    cpp = out_dir / f"{base}.cpp"
    spec = espec.spec
    q = class_name

    # ---- header ----
    H = ["// Generated by manta.codegen. Do not edit by hand.", "",
         "#pragma once", "", "#include <Eigen/Dense>", "",
         f"namespace {namespace} {{", "", f"class {q} {{", "public:"]
    for name, val in espec.consts:
        H.append(f"    static constexpr int {name} = {val};")
    H.append("")
    H += _struct_lines("State", espec.state_fields, empty_comment=None) + [""]
    H += _struct_lines("Inputs", espec.input_fields,
                       empty_comment="// (no inputs declared)") + [""]
    if espec.output_fields is not None:
        H += _struct_lines("Outputs", espec.output_fields,
                           empty_comment=None) + [""]
    if espec.stateful:
        H += ["    State x;          // block state (ambient)", "",
              f"    {q}() {{ x = State{{}}; }}",
              "    void reset(const State& x0) { x = x0; }",
              "    const State& state() const { return x; }", ""]
    for ep in espec.entry_points:
        sig = ", ".join(_param_decl(p, decl=True) for p in ep.params)
        H.append(f"    {_ret_type(ep.ret)} {ep.method_name}({sig})"
                 + (" const" if ep.const else "") + ";")
    H += ["};", "", f"}}  // namespace {namespace}"]
    hpp.write_text("\n".join(H) + "\n")

    # ---- source ----
    returns_state = any(
        ep.ret.kind in (_R.STATE, _R.OUTPUTS_AND_STATE)
        and ep.fn_name is not None
        for ep in espec.entry_points)
    uses_inputs = any(_A.INPUTS in ep.kernel_args for ep in espec.entry_points)

    C = ["// Generated by manta.codegen. Do not edit by hand.", "",
         f'#include "{hpp.name}"', "", "#include <Eigen/Dense>", "",
         'extern "C" {', f'#include "{base}_kernels.h"', "}", "",
         f"namespace {namespace} {{", ""]
    C += S.pack_state_lines(spec, q) + [""]
    if returns_state:
        C += S.unpack_state_lines(spec, q) + [""]
    if uses_inputs:
        C += _pack_inputs_lines(espec.input_fields, q) + [""]
    for ep in espec.entry_points:
        C += _method_lines(ep, espec, q)
    C += [f"}}  // namespace {namespace}"]
    cpp.write_text("\n".join(C) + "\n")
    return {"hpp": hpp, "cpp": cpp}


def _pack_inputs_lines(input_fields, qcls: str) -> list[str]:
    """Concatenate the Inputs struct fields into the flat `u` vector
    (handles both scalar Sim/LQR inputs and vector recurrence ports)."""
    out = [f"static void pack_inputs(const {qcls}::Inputs& u, double* uflat) {{"]
    if not input_fields:
        out.append("    (void)u; (void)uflat;")
    else:
        off = 0
        for f in input_fields:
            if f.dim == 1:
                out.append(f"    uflat[{off}] = u.{f.ident};")
            else:
                for i in range(f.dim):
                    out.append(f"    uflat[{off + i}] = u.{f.ident}[{i}];")
            off += f.dim
    out.append("}")
    return out

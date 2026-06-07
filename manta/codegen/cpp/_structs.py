"""Shared C++ pack/unpack helpers for the generic Module emitter.

The flat-double `pack_state` / `unpack_state` over a `StateSpec` slot layout
plus the small type helper (`eigen_vec_type`) used by `module_emit.py`.
(The former per-wrapper struct/Inputs emitters lived here too; the one
generic emitter now renders those itself, so only the shared slot
pack/unpack remains.) C++ identifiers come from the IR-wide `entry_ident`
convention — names are validated as identifiers at model construction, so
flattening dots is all a C++ symbol needs.
"""

from __future__ import annotations

from ...ir.module import entry_ident


def eigen_vec_type(dim: int) -> str:
    if dim == 1:
        return "double"
    if dim == 3:
        return "Eigen::Vector3d"
    if dim == 4:
        return "Eigen::Vector4d"
    return f"Eigen::Matrix<double, {dim}, 1>"


def pack_state_lines(spec, qcls: str) -> list[str]:
    out = [f"static void pack_state(const {qcls}::State& s, double* x) {{"]
    for slot in spec.slots:
        ident = entry_ident(slot.name)
        if slot.ambient_dim == 1:
            out.append(f"    x[{slot.ambient_offset}] = s.{ident};")
        else:
            for i in range(slot.ambient_dim):
                out.append(f"    x[{slot.ambient_offset + i}] = s.{ident}[{i}];")
    out.append("}")
    return out


def unpack_state_lines(spec, qcls: str) -> list[str]:
    out = [f"static void unpack_state(const double* x, {qcls}::State& s) {{"]
    for slot in spec.slots:
        ident = entry_ident(slot.name)
        if slot.ambient_dim == 1:
            out.append(f"    s.{ident} = x[{slot.ambient_offset}];")
        else:
            for i in range(slot.ambient_dim):
                out.append(f"    s.{ident}[{i}] = x[{slot.ambient_offset + i}];")
    out.append("}")
    return out

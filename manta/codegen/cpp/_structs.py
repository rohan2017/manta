"""Shared C++ struct + pack/unpack emission for the typed wrappers.

The `State` / `Inputs` structs and the `pack_state` / `unpack_state` /
`pack_inputs` helpers are identical for the Sim, EKF, and LQR wrappers —
they all sit on the same flat-double kernel ABI and the same `StateSpec`
slot layout. This module is the single source for them; each wrapper
imports these instead of re-emitting.
"""

from __future__ import annotations

from ...estimation.state_spec import StateSlot
from .types import cpp_type_for


def cpp_ident(name: str) -> str:
    """A manta dot-name → a valid C++ identifier."""
    return name.replace(".", "_")


def eigen_type_for(slot: StateSlot) -> str:
    return cpp_type_for(slot.manifold).decl


def eigen_vec_type(dim: int) -> str:
    if dim == 1:
        return "double"
    if dim == 3:
        return "Eigen::Vector3d"
    if dim == 4:
        return "Eigen::Vector4d"
    return f"Eigen::Matrix<double, {dim}, 1>"


def input_default(input_full_name: str, world) -> float:
    """Resolve `<craft>.<part>.<input>` to its part-instance default."""
    craft_name, rest = input_full_name.split(".", 1)
    part_name, input_name = rest.split(".", 1)
    craft = next(c for c in world.crafts if c.name == craft_name)
    part = next(p for p in craft.parts if p.name == part_name)
    return float(getattr(part, input_name))


def state_field_init(slot: StateSlot, world) -> str:
    """C++ initializer for a State-struct field (manifold default for
    rigid-body slots, declared `init` for user State, zero for RW bias)."""
    cpp = cpp_type_for(slot.manifold)
    head, _, rest = slot.name.partition(".")
    craft = next((c for c in world.crafts if c.name == head), None)
    if craft is not None:
        if "." in rest:
            part_name, sub = rest.split(".", 1)
            part = next(p for p in craft.parts if p.name == part_name)
            if sub in part.state_declarations():
                return cpp.literal(getattr(part, sub))
            from ...parts.base import RandomWalkNoise
            if (sub in part.noise_declarations()
                    and isinstance(part.noise_declarations()[sub],
                                   RandomWalkNoise)):
                return cpp.zero
            raise RuntimeError(
                f"wrapper: per-craft slot {slot.name!r} is not a State "
                f"or RW bias.")
        return cpp.zero
    from ...fields.base import Disturbance
    for field in world.fields:
        for dist in field._disturbances:
            if isinstance(dist, Disturbance) and dist.name == head:
                from ...parts.base import RandomWalkNoise
                if (rest in dist.noise_declarations()
                        and isinstance(dist.noise_declarations()[rest],
                                       RandomWalkNoise)):
                    return cpp.zero
                if rest in dist.state_declarations():
                    return cpp.literal(getattr(dist, rest))
    raise RuntimeError(
        f"wrapper: slot {slot.name!r} doesn't resolve to a known owner.")


# ---------------------------------------------------------------------------
# Struct + pack/unpack line emitters (return lists of source lines)
# ---------------------------------------------------------------------------

def state_struct_lines(spec, world) -> list[str]:
    out = ["    struct State {"]
    for slot in spec.slots:
        out.append(f"        {eigen_type_for(slot)} {cpp_ident(slot.name)} "
                   f"= {state_field_init(slot, world)};")
    out.append("    };")
    return out


def inputs_struct_lines(input_names, world) -> list[str]:
    out = ["    struct Inputs {"]
    for inp in input_names:
        out.append(f"        double {cpp_ident(inp)} = "
                   f"{input_default(inp, world)!r};")
    if not input_names:
        out.append("        // (no inputs declared on this craft)")
    out.append("    };")
    return out


def pack_state_lines(spec, qcls: str) -> list[str]:
    out = [f"static void pack_state(const {qcls}::State& s, double* x) {{"]
    for slot in spec.slots:
        ident = cpp_ident(slot.name)
        if slot.dim == 1:
            out.append(f"    x[{slot.offset}] = s.{ident};")
        else:
            for i in range(slot.dim):
                out.append(f"    x[{slot.offset + i}] = s.{ident}[{i}];")
    out.append("}")
    return out


def unpack_state_lines(spec, qcls: str) -> list[str]:
    out = [f"static void unpack_state(const double* x, {qcls}::State& s) {{"]
    for slot in spec.slots:
        ident = cpp_ident(slot.name)
        if slot.dim == 1:
            out.append(f"    s.{ident} = x[{slot.offset}];")
        else:
            for i in range(slot.dim):
                out.append(f"    s.{ident}[{i}] = x[{slot.offset + i}];")
    out.append("}")
    return out


def pack_inputs_lines(input_names, qcls: str) -> list[str]:
    out = [f"static void pack_inputs(const {qcls}::Inputs& u, double* uflat) {{"]
    if not input_names:
        out.append("    (void)u; (void)uflat;")
    else:
        for i, inp in enumerate(input_names):
            out.append(f"    uflat[{i}] = u.{cpp_ident(inp)};")
    out.append("}")
    return out

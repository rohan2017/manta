"""Surface<N> — generalized drag/lift, N velocity-power tensors.

Each tick the part computes v_rel_part (fluid velocity − own velocity in part
frame) and accumulates:
    F = sum_{k=1..N} A_k * v_rel_part^(k)
    τ = sum_{k=1..N} B_k * v_rel_part^(k)

Surface<1> is plain linear drag/lift; Surface<2> adds a quadratic term;
higher orders are typically overkill (max N = 4).
"""

from __future__ import annotations

from ..._format import cpp_float as _f
from ...core import PartDescriptor


def _emit_mat3(m: tuple[float, ...], scalar: str) -> str:
    """C++ literal for `manta::geom::Mat3<PartFrame, PartFrame, scalar>` from
    a 9-tuple (row-major). Returns a lambda-init expression that compiles for
    any Scalar (the per-component cast `scalar(...)` keeps Jet-typed
    instantiations happy)."""
    if len(m) != 9:
        raise ValueError("force/torque tensor must be a 9-tuple (row-major 3x3)")
    mat3 = f"manta::geom::Mat3<manta::PartFrame, manta::PartFrame, {scalar}>"
    return (
        f"[]{{ {mat3} _M; "
        f"_M.raw()(0,0)={scalar}({_f(m[0])}); _M.raw()(0,1)={scalar}({_f(m[1])}); _M.raw()(0,2)={scalar}({_f(m[2])}); "
        f"_M.raw()(1,0)={scalar}({_f(m[3])}); _M.raw()(1,1)={scalar}({_f(m[4])}); _M.raw()(1,2)={scalar}({_f(m[5])}); "
        f"_M.raw()(2,0)={scalar}({_f(m[6])}); _M.raw()(2,1)={scalar}({_f(m[7])}); _M.raw()(2,2)={scalar}({_f(m[8])}); "
        "return _M; }()"
    )


def _diag_to_full(d: tuple[float, float, float]) -> tuple[float, ...]:
    a, b, c = d
    return (a, 0.0, 0.0,
            0.0, b, 0.0,
            0.0, 0.0, c)


class Surface(PartDescriptor):
    """Pass N orders' worth of force tensors and torque tensors. N is inferred
    from `len(force_tensors)`. Each tensor is either a 3-tuple (diagonal) or a
    9-tuple (full row-major).

    Required fields: FluidField.
    """

    cpp_header = "manta/parts/structure/surface.hpp"
    # `cpp_class` and `cpp_class_template` are set per-instance in __init__
    # since they bake the integer N into the C++ template arguments.

    def __init__(self,
                 name: str,
                 force_tensors:  list[tuple[float, ...]],
                 torque_tensors: list[tuple[float, ...]],
                 **kwargs) -> None:
        super().__init__(name=name, **kwargs)
        if not force_tensors or not torque_tensors:
            raise ValueError("Surface needs at least one force and torque tensor")
        if len(force_tensors) != len(torque_tensors):
            raise ValueError("force_tensors and torque_tensors must have the same length (N)")
        if len(force_tensors) > 4:
            raise ValueError("Surface<N>: N must be in [1, 4]")

        def normalize(t):
            if len(t) == 3:
                return _diag_to_full(tuple(float(x) for x in t))
            elif len(t) == 9:
                return tuple(float(x) for x in t)
            else:
                raise ValueError("each tensor must be a 3-tuple (diagonal) or 9-tuple (full)")

        self.force_tensors  = [normalize(t) for t in force_tensors]
        self.torque_tensors = [normalize(t) for t in torque_tensors]
        self.N = len(force_tensors)
        # Per-instance class names: bake N in.
        self.cpp_class          = f"manta::parts::Surface<{self.N}>"
        self.cpp_class_template = "manta::parts::SurfaceT"

    # SurfaceT has TWO template parameters: int N and class Scalar. Override
    # the default instantiation to splice N in before Scalar.
    def cpp_class_template_instantiation(self, scalar: str) -> str:
        return f"manta::parts::SurfaceT<{self.N}, {scalar}>"

    def emit_constructor_args(self, scalar: str = "manta::Real") -> str:
        n = self.N
        mat3 = f"manta::geom::Mat3<manta::PartFrame, manta::PartFrame, {scalar}>"
        ftype = f"std::array<{mat3}, {n}>"
        f_arr = ", ".join(_emit_mat3(t, scalar) for t in self.force_tensors)
        t_arr = ", ".join(_emit_mat3(t, scalar) for t in self.torque_tensors)
        return (f'"{self.name}", '
                f'{ftype}{{{f_arr}}}, '
                f'{ftype}{{{t_arr}}}')

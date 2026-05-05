"""Surface1..Surface4 — generalized drag/lift, fixed velocity-power count.

Each tick the part computes v_rel_part (fluid velocity − own velocity in part
frame) and accumulates:
    F = sum_{k=1..N} A_k * v_rel_part^(k)
    τ = sum_{k=1..N} B_k * v_rel_part^(k)

Surface1 is plain linear drag/lift; Surface2 adds a quadratic term; Surface3
and Surface4 round out the family. Higher orders are typically overkill — pick
the smallest one that captures the regime you care about.

Tensors are either 3-tuples (diagonal) or 9-tuples (full row-major).
"""

from __future__ import annotations

from ..._format import cpp_float as _f
from ...core import PartDescriptor
from ...fields.fluid_field import FluidField


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


def _normalize(t: tuple[float, ...]) -> tuple[float, ...]:
    if len(t) == 3:
        return _diag_to_full(tuple(float(x) for x in t))
    elif len(t) == 9:
        return tuple(float(x) for x in t)
    else:
        raise ValueError("each tensor must be a 3-tuple (diagonal) or 9-tuple (full)")


class _SurfaceBase(PartDescriptor):
    """Shared logic for SurfaceN. Subclasses set `N` (the fixed velocity-power
    count) and the C++ class names."""

    N: int = 0    # subclasses override

    cpp_header = "manta/parts/structure/surface.hpp"

    # Drag/lift forces are ρ-scaled — Surface is meaningless without
    # a FluidField in the world.
    requires_fields = [FluidField]

    def __init__(self,
                 name: str,
                 force_tensors:  list[tuple[float, ...]],
                 torque_tensors: list[tuple[float, ...]],
                 **kwargs) -> None:
        super().__init__(name=name, **kwargs)
        if len(force_tensors) != self.N or len(torque_tensors) != self.N:
            raise ValueError(
                f"{type(self).__name__} requires exactly {self.N} force_tensors "
                f"and {self.N} torque_tensors (got "
                f"{len(force_tensors)}, {len(torque_tensors)})"
            )
        self.force_tensors  = [_normalize(t) for t in force_tensors]
        self.torque_tensors = [_normalize(t) for t in torque_tensors]

    def emit_constructor_args(self, scalar: str = "manta::Real") -> str:
        n = self.N
        mat3 = f"manta::geom::Mat3<manta::PartFrame, manta::PartFrame, {scalar}>"
        ftype = f"std::array<{mat3}, {n}>"
        f_arr = ", ".join(_emit_mat3(t, scalar) for t in self.force_tensors)
        t_arr = ", ".join(_emit_mat3(t, scalar) for t in self.torque_tensors)
        return (f'"{self.name}", '
                f'{ftype}{{{f_arr}}}, '
                f'{ftype}{{{t_arr}}}')


class Surface1(_SurfaceBase):
    """Linear drag/lift: F = A_1 * v_rel, τ = B_1 * v_rel."""
    N = 1
    cpp_class          = "manta::parts::Surface1"
    cpp_class_template = "manta::parts::Surface1T"


class Surface2(_SurfaceBase):
    """Linear + quadratic: F = A_1*v + A_2*v², τ = B_1*v + B_2*v²."""
    N = 2
    cpp_class          = "manta::parts::Surface2"
    cpp_class_template = "manta::parts::Surface2T"


class Surface3(_SurfaceBase):
    """Up to cubic velocity-power terms."""
    N = 3
    cpp_class          = "manta::parts::Surface3"
    cpp_class_template = "manta::parts::Surface3T"


class Surface4(_SurfaceBase):
    """Up to quartic velocity-power terms."""
    N = 4
    cpp_class          = "manta::parts::Surface4"
    cpp_class_template = "manta::parts::Surface4T"

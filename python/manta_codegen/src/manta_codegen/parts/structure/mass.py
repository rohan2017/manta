"""Mass — point mass with a full 3x3 moment of inertia."""

from __future__ import annotations

from ..._format import cpp_float as _f
from ...core import PartDescriptor


class Mass(PartDescriptor):
    """A lump with both scalar mass AND a configurable MOI tensor (in part
    frame, about the part origin).

    `moi` accepts either:
        * a 3-tuple (Ixx, Iyy, Izz) for a diagonal MOI tensor, or
        * a 9-tuple in row-major order for a full 3x3 MOI tensor.

    Required fields: none.
    Telemetry: none.
    """

    cpp_class          = "manta::parts::Mass"
    cpp_class_template = "manta::parts::MassT"
    cpp_header         = "manta/parts/structure/mass.hpp"

    def __init__(self,
                 name: str,
                 mass: float,
                 moi: tuple[float, ...],
                 **kwargs) -> None:
        super().__init__(name=name, **kwargs)
        if len(moi) not in (3, 9):
            raise ValueError("Mass.moi must be a 3-tuple (diagonal) or 9-tuple (full)")
        self.mass = float(mass)
        self.moi  = tuple(float(x) for x in moi)

    def emit_constructor_args(self, scalar: str = "manta::Real") -> str:
        mat3 = f"manta::geom::Mat3<manta::PartFrame, manta::PartFrame, {scalar}>"
        if len(self.moi) == 3:
            ixx, iyy, izz = self.moi
            moi_expr = (
                f"[]{{ {mat3} m = {mat3}::identity(); "
                f"m.raw()(0,0)={scalar}({_f(ixx)}); m.raw()(1,1)={scalar}({_f(iyy)}); m.raw()(2,2)={scalar}({_f(izz)}); "
                "return m; }()"
            )
        else:
            m = self.moi
            moi_expr = (
                f"[]{{ {mat3} m; "
                f"m.raw()(0,0)={scalar}({_f(m[0])}); m.raw()(0,1)={scalar}({_f(m[1])}); m.raw()(0,2)={scalar}({_f(m[2])}); "
                f"m.raw()(1,0)={scalar}({_f(m[3])}); m.raw()(1,1)={scalar}({_f(m[4])}); m.raw()(1,2)={scalar}({_f(m[5])}); "
                f"m.raw()(2,0)={scalar}({_f(m[6])}); m.raw()(2,1)={scalar}({_f(m[7])}); m.raw()(2,2)={scalar}({_f(m[8])}); "
                "return m; }()"
            )
        return f'"{self.name}", {scalar}({_f(self.mass)}), {moi_expr}'

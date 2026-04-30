"""PointMass — a passive lump of mass at the part origin."""

from __future__ import annotations

from ..._format import cpp_float as _f
from ...core import PartDescriptor


class PointMass(PartDescriptor):
    """A point mass at the part origin.

    By default contributes mass and zero MOI. If a non-zero `moi` is passed,
    a post-construction `set_moi(...)` is emitted so the body has a useful
    rotational inertia (typical for a `body` lump-mass on a quadcopter or
    rocket where the actual mass distribution is complicated).

    `moi` accepts either:
        * a 3-tuple (Ixx, Iyy, Izz) for a diagonal MOI tensor, or
        * a 9-tuple in row-major order for a full 3x3 MOI tensor.

    Required fields: none.
    Telemetry: none.
    """

    cpp_class          = "manta::parts::PointMass"
    cpp_class_template = "manta::parts::PointMassT"
    cpp_header         = "manta/parts/structure/point_mass.hpp"

    def __init__(self,
                 name: str,
                 mass: float,
                 moi: tuple[float, ...] | None = None,
                 **kwargs) -> None:
        super().__init__(name=name, **kwargs)
        self.mass = float(mass)
        if moi is not None and len(moi) not in (3, 9):
            raise ValueError("PointMass.moi must be a 3-tuple (diagonal) or 9-tuple (full)")
        self.moi: tuple[float, ...] | None = (
            tuple(float(x) for x in moi) if moi is not None else None
        )

    def emit_constructor_args(self, scalar: str = "manta::Real") -> str:
        return f'"{self.name}", {scalar}({_f(self.mass)})'

    def emit_post_construction(self, scalar: str = "manta::Real") -> list[str]:
        out = list(super().emit_post_construction(scalar))
        if self.moi is not None:
            mat3 = f"manta::geom::Mat3<manta::PartFrame, manta::PartFrame, {scalar}>"
            if len(self.moi) == 3:
                ixx, iyy, izz = self.moi
                out.append(
                    f"{{ {mat3} _moi = {mat3}::identity(); "
                    f"_moi.raw()(0,0) = {scalar}({_f(ixx)}); "
                    f"_moi.raw()(1,1) = {scalar}({_f(iyy)}); "
                    f"_moi.raw()(2,2) = {scalar}({_f(izz)}); "
                    f"{self.name}_->set_moi(_moi); }}"
                )
            else:
                m = self.moi
                out.append(
                    f"{{ {mat3} _moi; "
                    f"_moi.raw()(0,0)={scalar}({_f(m[0])}); _moi.raw()(0,1)={scalar}({_f(m[1])}); _moi.raw()(0,2)={scalar}({_f(m[2])}); "
                    f"_moi.raw()(1,0)={scalar}({_f(m[3])}); _moi.raw()(1,1)={scalar}({_f(m[4])}); _moi.raw()(1,2)={scalar}({_f(m[5])}); "
                    f"_moi.raw()(2,0)={scalar}({_f(m[6])}); _moi.raw()(2,1)={scalar}({_f(m[7])}); _moi.raw()(2,2)={scalar}({_f(m[8])}); "
                    f"{self.name}_->set_moi(_moi); }}"
                )
        return out

    def render(self, telemetry: dict, path: str) -> None:
        try:
            import rerun as rr
        except ImportError:
            return
        rr.log(path, rr.Points3D(positions=[[0, 0, 0]],
                                 radii=[0.05],
                                 colors=[[180, 180, 180]]))

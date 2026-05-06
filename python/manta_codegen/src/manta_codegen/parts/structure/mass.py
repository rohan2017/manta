"""Mass — point mass with optional MOI tensor and auto-gravity."""

from __future__ import annotations

from ..._format import cpp_float as _f
from ...core import PartDescriptor


class Mass(PartDescriptor):
    """A lump with scalar mass and (optional) full 3x3 moment of inertia in
    part frame. Replaces the deleted `PointMass`: pass `moi=None` (the default)
    to model a point mass with zero MOI, or supply a tuple for a body whose
    rotational inertia matters.

    `moi` accepts either:
        * a 3-tuple (Ixx, Iyy, Izz) for a diagonal MOI tensor,
        * a 9-tuple in row-major order for a full 3x3 MOI tensor,
        * None for zero MOI (point-mass shorthand).

    If a `GravityField` is registered on the world (or the craft), the part
    queries it at its CoM each tick and applies m·g to itself. Pass
    `apply_gravity=False` to opt out (e.g. for ballast inside a sealed
    compartment whose buoyancy is already modeled, or estimator crafts that
    don't represent gravity).

    Required fields: none.
    Telemetry: none.
    """

    cpp_class          = "manta::parts::Mass"
    cpp_class_template = "manta::parts::MassT"
    cpp_header         = "manta/parts/structure/mass.hpp"

    def __init__(self,
                 name: str,
                 mass: float,
                 moi: tuple[float, ...] | None = None,
                 apply_gravity: bool = True,
                 **kwargs) -> None:
        super().__init__(name=name, **kwargs)
        if moi is not None and len(moi) not in (3, 9):
            raise ValueError("Mass.moi must be a 3-tuple (diagonal) or 9-tuple (full)")
        self.mass: float = float(mass)
        self.moi: tuple[float, ...] | None = (
            tuple(float(x) for x in moi) if moi is not None else None
        )
        self.apply_gravity: bool = bool(apply_gravity)

    def emit_constructor_args(self, scalar: str = "manta::MFloat") -> str:
        ag = "true" if self.apply_gravity else "false"
        if self.moi is None:
            return f'"{self.name}", {scalar}({_f(self.mass)}), {ag}'
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
        return f'"{self.name}", {scalar}({_f(self.mass)}), {moi_expr}, {ag}'

    def render(self, telemetry: dict, path: str) -> None:
        try:
            import rerun as rr
        except ImportError:
            return
        rr.log(path, rr.Points3D(positions=[[0, 0, 0]],
                                 radii=[0.05],
                                 colors=[[180, 180, 180]]))

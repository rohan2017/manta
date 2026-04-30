"""PropThruster — Thruster with propeller reaction torque."""

from __future__ import annotations

from ..._format import cpp_float as _f
from .thruster import Thruster


class PropThruster(Thruster):
    """A Thruster with an additional reaction torque along the thrust direction:
        τ = sign · kt · thrust
    where sign is +1 for clockwise propellers (viewed from the thrust direction)
    and -1 for counter-clockwise.

    Required fields: none.
    Telemetry: throttle (float).
    """

    cpp_class          = "manta::parts::PropThruster"
    cpp_class_template = "manta::parts::PropThrusterT"
    cpp_header         = "manta/parts/actuator/prop_thruster.hpp"

    def __init__(self,
                 name: str,
                 max_thrust: float,
                 kt: float = 0.02,
                 cw: bool = False,
                 direction: tuple[float, float, float] = (0.0, 0.0, 1.0),
                 **kwargs) -> None:
        super().__init__(name=name, max_thrust=max_thrust, direction=direction, **kwargs)
        self.kt = float(kt)
        self.cw = bool(cw)

    def emit_constructor_args(self, scalar: str = "manta::Real") -> str:
        d = self.direction
        cw = "true" if self.cw else "false"
        return (f'"{self.name}", {scalar}({_f(self.max_thrust)}), '
                f'{scalar}({_f(self.kt)}), {cw}, '
                f'manta::geom::Vec3<manta::PartFrame, {scalar}>{{'
                f'{scalar}({_f(d[0])}), {scalar}({_f(d[1])}), {scalar}({_f(d[2])})}}')

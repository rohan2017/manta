"""GimbaledThruster — Thruster on a 2-axis gimbal."""

from __future__ import annotations

from ..._format import cpp_float as _f
from .thruster import Thruster


class GimbaledThruster(Thruster):
    """A thruster mounted on a 2-axis gimbal. Effective thrust direction is the
    base direction rotated by the configured gimbal angles, clamped to
    ±max_angle on each axis.

    Required fields: none.
    Telemetry: throttle (float).
    """

    cpp_class          = "manta::parts::GimbaledThruster"
    cpp_class_template = "manta::parts::GimbaledThrusterT"
    cpp_header         = "manta/parts/actuator/gimbaled_thruster.hpp"

    def __init__(self,
                 name: str,
                 max_thrust: float,
                 max_angle: float = 0.15,
                 base_direction: tuple[float, float, float] = (0.0, 0.0, 1.0),
                 **kwargs) -> None:
        super().__init__(name=name, max_thrust=max_thrust, direction=base_direction, **kwargs)
        self.max_angle = float(max_angle)

    def emit_constructor_args(self, scalar: str = "manta::Real") -> str:
        d = self.direction
        return (f'"{self.name}", {scalar}({_f(self.max_thrust)}), '
                f'{scalar}({_f(self.max_angle)}), '
                f'manta::geom::Vec3<manta::PartFrame, {scalar}>{{'
                f'{scalar}({_f(d[0])}), {scalar}({_f(d[1])}), {scalar}({_f(d[2])})}}')

"""GimbaledThruster — Thruster on a 2-axis gimbal."""

from __future__ import annotations

from ..._format import cpp_float as _f
from ...signal import Signal, scalar_in_signal, scalar_out_signal
from .thruster import Thruster


class GimbaledThruster(Thruster):
    """A thruster mounted on a 2-axis gimbal. Effective thrust direction is the
    base direction rotated by the configured gimbal angles, clamped to
    ±max_angle on each axis.

    Bindable signals (in addition to the inherited Thruster signals
    `throttle` / `set_throttle`):
      * `pitch`       (out, 1 float) — current gimbal pitch (rad).
      * `yaw`         (out, 1 float) — current gimbal yaw   (rad).
      * `set_gimbal`  (in,  2 floats) — sets [pitch, yaw], clamped internally.

    Required fields: none.
    """

    cpp_class          = "manta::parts::GimbaledThruster"
    cpp_class_template = "manta::parts::GimbaledThrusterT"
    cpp_header         = "manta/parts/actuator/gimbaled_thruster.hpp"

    signals = [
        *Thruster.signals,
        scalar_out_signal("pitch", "pitch"),
        scalar_out_signal("yaw",   "yaw"),
        Signal(
            name="set_gimbal",
            direction="in",
            n_floats=2,
            cpp_write_stmt="{accessor}.set_gimbal({v0}, {v1});",
        ),
    ]

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

"""Motor — 1-DOF revolute joint articulated part."""

from __future__ import annotations

from ..._format import cpp_float as _f
from ...core import PartDescriptor
from ...signal import scalar_in_signal, scalar_out_signal


def _emit_axis(axis: tuple[float, float, float]) -> str:
    if len(axis) != 3:
        raise ValueError("axis must be a 3-tuple")
    return (f"manta::geom::Vec3<manta::PartFrame>::from_raw("
            f"Eigen::Matrix<manta::MFloat, 3, 1>("
            f"{_f(axis[0])}, {_f(axis[1])}, {_f(axis[2])}))")


class Motor(PartDescriptor):
    """1-DOF motor (revolute joint). Subclass of ArticulatedPart.

    Required fields: none.
    Telemetry: angle (rad), rate (rad/s), accel (rad/s²).

    Children added via `motor.add(...)` in the descriptor tree become
    children of the joint's output frame. The framework rotates them by
    the current joint angle each tick.
    """

    cpp_class  = "manta::parts::Motor"
    cpp_header = "manta/parts/articulation/motor.hpp"

    signals = [
        scalar_out_signal("angle",     "angle"),
        scalar_out_signal("rate",      "rate"),
        scalar_out_signal("accel",     "accel"),
        scalar_in_signal ("set_torque", "set_torque"),
    ]

    def __init__(self,
                 name: str,
                 axis: tuple[float, float, float] = (0.0, 0.0, 1.0),
                 stall_torque: float = 0.0,
                 damping: float = 0.0,
                 **kwargs) -> None:
        super().__init__(name=name, **kwargs)
        self.axis = (float(axis[0]), float(axis[1]), float(axis[2]))
        self.stall_torque = float(stall_torque)
        self.damping = float(damping)

    def emit_constructor_args(self) -> str:
        return (f'"{self.name}", '
                f'{_emit_axis(self.axis)}, '
                f'{_f(self.stall_torque)}, '
                f'{_f(self.damping)}')

    def telemetry(self) -> list[tuple[str, str, str]]:
        return [("angle", "manta::MFloat", f"craft.{self.name}().angle()"),
                ("rate",  "manta::MFloat", f"craft.{self.name}().rate()"),
                ("accel", "manta::MFloat", f"craft.{self.name}().accel()")]

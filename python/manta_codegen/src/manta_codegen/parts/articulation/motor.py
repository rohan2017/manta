"""Motor — 1-DOF revolute joint articulated part."""

from __future__ import annotations

from ..._format import cpp_float as _f
from ...core import PartDescriptor


def _emit_axis(axis: tuple[float, float, float]) -> str:
    if len(axis) != 3:
        raise ValueError("axis must be a 3-tuple")
    return (f"manta::geom::Vec3<manta::PartFrame>::from_raw("
            f"Eigen::Matrix<manta::Real, 3, 1>("
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

    def __init__(self,
                 name: str,
                 axis: tuple[float, float, float] = (0.0, 0.0, 1.0),
                 stall_torque: float = 0.0,
                 damping: float = 0.0,
                 publish_state: bool = True,
                 **kwargs) -> None:
        super().__init__(name=name, publish_state=publish_state, **kwargs)
        self.axis = (float(axis[0]), float(axis[1]), float(axis[2]))
        self.stall_torque = float(stall_torque)
        self.damping = float(damping)

    def emit_constructor_args(self) -> str:
        return (f'"{self.name}", '
                f'{_emit_axis(self.axis)}, '
                f'{_f(self.stall_torque)}, '
                f'{_f(self.damping)}')

    def telemetry_fields(self) -> list[tuple[str, str]]:
        return [("angle", "manta::Real"),
                ("rate",  "manta::Real"),
                ("accel", "manta::Real")]

    def emit_telemetry_reads(self) -> list[tuple[str, str]]:
        return [("angle", f"craft.{self.name}().angle()"),
                ("rate",  f"craft.{self.name}().rate()"),
                ("accel", f"craft.{self.name}().accel()")]

    def emit_command_apply(self, part_accessor: str, payload_var: str) -> str:
        # Single-float payload: commanded torque.
        return (f"if (!{payload_var}.empty()) "
                f"{part_accessor}.set_torque({payload_var}[0]);")

"""Thruster — applies a force along a direction in part frame, scaled by throttle."""

from __future__ import annotations

from ..._format import cpp_float as _f
from ...core import PartDescriptor
from ...signal import scalar_in_signal, scalar_out_signal


class Thruster(PartDescriptor):
    """A thruster that applies a force along `direction` in part frame, scaled
    by throttle ∈ [0, 1].

    Bindable signals:
      * `throttle`     (out, 1 float) — current throttle value.
      * `set_throttle` (in,  1 float) — sets throttle (clamped to [0,1] internally).

    Required fields: none.
    """

    cpp_class          = "manta::parts::Thruster"
    cpp_class_template = "manta::parts::ThrusterT"
    cpp_header         = "manta/parts/actuator/thruster.hpp"

    signals = [
        scalar_out_signal("throttle",     "throttle"),
        scalar_in_signal ("set_throttle", "set_throttle"),
    ]

    def __init__(self,
                 name: str,
                 max_thrust: float,
                 direction: tuple[float, float, float] = (0.0, 0.0, 1.0),
                 **kwargs) -> None:
        super().__init__(name=name, **kwargs)
        self.max_thrust = float(max_thrust)
        self.direction  = tuple(float(x) for x in direction)

    def emit_constructor_args(self, scalar: str = "manta::Real") -> str:
        d = self.direction
        return (f'"{self.name}", {scalar}({_f(self.max_thrust)}), '
                f'manta::geom::Vec3<manta::PartFrame, {scalar}>{{'
                f'{scalar}({_f(d[0])}), {scalar}({_f(d[1])}), {scalar}({_f(d[2])})}}')

    def telemetry_fields(self) -> list[tuple[str, str]]:
        return [("throttle", "float")]

    def emit_telemetry_reads(self) -> list[tuple[str, str]]:
        return [("throttle", f"craft.{self.name}().throttle()")]

    def render(self, telemetry: dict, path: str) -> None:
        try:
            import rerun as rr
        except ImportError:
            return
        rr.log(path, rr.Boxes3D(half_sizes=[[0.04, 0.04, 0.06]],
                                colors=[[200, 100, 50]]))
        thr = float(telemetry.get("throttle", 0.0))
        if thr > 0:
            d = self.direction
            length = 0.5 * thr
            rr.log(f"{path}/plume",
                   rr.Arrows3D(origins=[[0, 0, 0]],
                               vectors=[[-d[0]*length, -d[1]*length, -d[2]*length]],
                               colors=[[255, 100, 50]]))
        rr.log(f"{path}/throttle", rr.Scalar(thr))

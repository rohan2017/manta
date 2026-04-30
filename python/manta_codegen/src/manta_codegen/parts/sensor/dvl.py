"""DVL — Doppler velocity log (body-frame velocity sensor)."""

from __future__ import annotations

from ..._format import cpp_float as _f
from ...core import PartDescriptor


class DVL(PartDescriptor):
    """A Doppler Velocity Log. Reports the part's absolute linear velocity in
    its own (part) frame, with optional white-Gaussian noise.

    Required fields: none.
    Telemetry: velocity (Vec3<PartFrame>).
    """

    cpp_class          = "manta::parts::DVL"
    cpp_class_template = "manta::parts::DVLT"
    cpp_header         = "manta/parts/sensor/dvl.hpp"

    def __init__(self,
                 name: str,
                 velocity_sigma: float = 0.0,
                 publish_state: bool = True,
                 **kwargs) -> None:
        super().__init__(name=name, publish_state=publish_state, **kwargs)
        self.velocity_sigma = float(velocity_sigma)

    def emit_constructor_args(self, scalar: str = "manta::Real") -> str:
        return (f'"{self.name}", '
                f'manta::parts::DvlNoiseParams{{{_f(self.velocity_sigma)}}}')

    def telemetry_fields(self) -> list[tuple[str, str]]:
        return [("velocity", "manta::geom::Vec3<manta::PartFrame>")]

    def emit_telemetry_reads(self) -> list[tuple[str, str]]:
        return [("velocity", f"craft.{self.name}().last_velocity()")]

    def emit_measurement_decode(self, part_accessor: str, payload_var: str) -> str:
        # DVL measurement payload: 3 floats — [vx, vy, vz].
        return (
            f"if ({payload_var}.size() >= 3) {{ "
            f"{part_accessor}.set_measurement("
            f"manta::geom::Vec3<manta::PartFrame>{{"
            f"{payload_var}[0], {payload_var}[1], {payload_var}[2]}}); "
            f"}}"
        )

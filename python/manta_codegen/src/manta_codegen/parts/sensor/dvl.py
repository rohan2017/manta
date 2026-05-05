"""DVL — Doppler velocity log (body-frame velocity sensor)."""

from __future__ import annotations

from ..._format import cpp_float as _f
from ...core import PartDescriptor
from ...signal import Signal, vec3_out_signal


class DVL(PartDescriptor):
    """A Doppler Velocity Log. Reports the part's absolute linear velocity in
    its own (part) frame, with optional white-Gaussian noise.

    Bindable signals:
      * `last_velocity`   (out, 3 floats) — most recent velocity sample.
      * `set_measurement` (in,  3 floats) — feed external [vx, vy, vz] into
        the part's `set_measurement(velocity)`.

    Required fields: none.
    """

    cpp_class          = "manta::parts::DVL"
    cpp_class_template = "manta::parts::DVLT"
    cpp_header         = "manta/parts/sensor/dvl.hpp"

    signals = [
        vec3_out_signal("last_velocity", "last_velocity"),
        Signal(
            name="set_measurement",
            direction="in",
            n_floats=3,
            cpp_write_stmt=(
                "{accessor}.set_measurement("
                "manta::geom::Vec3<manta::PartFrame>{{{v0}, {v1}, {v2}}});"
            ),
        ),
    ]

    def __init__(self,
                 name: str,
                 velocity_sigma: float = 0.0,
                 rate_hz: float = 0.0,
                 **kwargs) -> None:
        super().__init__(name=name, **kwargs)
        self.velocity_sigma = float(velocity_sigma)
        self.rate_hz        = float(rate_hz)

    def emit_constructor_args(self, scalar: str = "manta::Real") -> str:
        return (f'"{self.name}", '
                f'{_f(self.velocity_sigma)}, '
                f'manta::Real({_f(self.rate_hz)})')

    def telemetry_fields(self) -> list[tuple[str, str]]:
        return [("velocity", "manta::geom::Vec3<manta::PartFrame>")]

    def emit_telemetry_reads(self) -> list[tuple[str, str]]:
        return [("velocity", f"craft.{self.name}().last_velocity()")]

"""DVL — Doppler velocity log (body-frame velocity sensor)."""

from __future__ import annotations

from ..._format import cpp_float as _f, cpp_mfloat as _mf
from ...core import NoiseChannel, PartDescriptor
from ...signal import Signal, vec3_out_signal


class DVL(PartDescriptor):
    """A Doppler Velocity Log. Reports the part's absolute linear velocity in
    its own (part) frame, with optional white-Gaussian noise.

    Bindable signals:
      * `last_velocity` (out, 3 floats) — most recent velocity sample.

    Reading injection flows through `ekf.measure(...)`, not a sensor-input signal.

    Required fields: none.
    """

    cpp_class_template = "manta::parts::DVLT"
    cpp_header         = "manta/parts/sensor/dvl.hpp"

    signals = [
        vec3_out_signal("last_velocity", "last_velocity"),
    ]

    def __init__(self,
                 name: str,
                 velocity_sigma: float = 0.0,
                 rate_hz: float = 0.0,
                 **kwargs) -> None:
        super().__init__(name=name, **kwargs)
        self.velocity_sigma = float(velocity_sigma)
        self.rate_hz        = float(rate_hz)

    def emit_constructor_args(self, scalar: str = "manta::MFloat") -> str:
        return (f'"{self.name}", '
                f'{_f(self.velocity_sigma)}, '
                f'{_mf(self.rate_hz)}')

    def noise_channels(self) -> list[NoiseChannel]:
        # σ ≥ 0 ⇒ register a 3-axis white-Gaussian on the velocity reading.
        # Mirrors `DVLT::register_noise()` exactly.
        return [NoiseChannel("velocity_noise", "white_3d", self.velocity_sigma)]

    def telemetry(self) -> list[tuple[str, str, str]]:
        return [("velocity", "manta::geom::Vec3<manta::PartFrame>",
                 f"craft.{self.name}().last_velocity()")]

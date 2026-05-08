"""Magnetometer — 3-axis magnetic-field sensor, queries a MagField."""

from __future__ import annotations

from ..._format import cpp_float as _f, cpp_mfloat as _mf
from ...core import NoiseChannel, PartDescriptor
from ...fields.mag_field import MagField
from ...signal import Signal, vec3_out_signal


class Magnetometer(PartDescriptor):
    """A 3-axis magnetometer. Reads the registered `MagField` (any concrete
    implementation — `DipoleMagField`, future IGRF, etc.) at the part's
    scene-frame position, rotates the field vector into part frame, and
    optionally injects white-Gaussian noise.

    Bindable signals:
      * `last_b`           (out, 3 floats) — most recent magnetic-field
                                              sample in part frame (Tesla).
      * `set_measurement`  (in,  3 floats) — feed external [bx, by, bz]
                                              for estimator workflows.

    Required fields: `MagField` (registered by Earth via dipole_moment > 0,
    or any user-supplied magnetic-model field).
    """

    cpp_class_template = "manta::parts::MagnetometerT"
    cpp_header         = "manta/parts/sensor/magnetometer.hpp"

    # Hard requirement — Magnetometer is meaningless without a MagField.
    # Codegen validates this at config time and the C++ side has a
    # mirroring `MANTA_PART_REQUIRES_FIELD` static_assert.
    requires_fields    = [MagField]

    signals = [
        vec3_out_signal("last_b", "last_b"),
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
                 sigma: float = -1.0,
                 rate_hz: float = 0.0,
                 **kwargs) -> None:
        super().__init__(name=name, **kwargs)
        self.sigma   = float(sigma)
        self.rate_hz = float(rate_hz)

    def emit_constructor_args(self, scalar: str = "manta::MFloat") -> str:
        return (f'"{self.name}", '
                f'{_f(self.sigma)}, '
                f'{_mf(self.rate_hz)}')

    def noise_channels(self) -> list[NoiseChannel]:
        return [NoiseChannel("noise", "white_3d", self.sigma)]

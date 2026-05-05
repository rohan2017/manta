"""IMU — inertial measurement unit (angular rate + acceleration, with noise)."""

from __future__ import annotations

from ..._format import cpp_float as _f
from ...core import PartDescriptor
from ...signal import Signal, vec3_out_signal


class IMU(PartDescriptor):
    """An IMU samples the kinematic-pass acceleration and angular velocity each
    tick, with optional white-Gaussian noise on each.

    Bindable signals:
      * `last_accel` (out, 3 floats) — most recent acceleration sample.
      * `last_gyro`  (out, 3 floats) — most recent angular-rate sample.
      * `set_measurement` (in, 6 floats) — feed external [ax, ay, az, gx, gy, gz]
        into the part's `set_measurement(accel, gyro)`. Used by the real_data
        workflow to drive an estimator off real sensor topics.

    Required fields: none.
    """

    cpp_class          = "manta::parts::IMU"
    cpp_class_template = "manta::parts::IMUT"
    cpp_header         = "manta/parts/sensor/imu.hpp"

    signals = [
        vec3_out_signal("last_accel", "last_accel"),
        vec3_out_signal("last_gyro",  "last_gyro"),
        Signal(
            name="set_measurement",
            direction="in",
            n_floats=6,
            cpp_write_stmt=(
                "{accessor}.set_measurement("
                "manta::geom::Vec3<manta::PartFrame>{{{v0}, {v1}, {v2}}}, "
                "manta::geom::Vec3<manta::PartFrame>{{{v3}, {v4}, {v5}}});"
            ),
        ),
        # Per-channel inputs — used when wiring sim→est via single-signal
        # connect() calls (one for accel, one for gyro). Either form sets
        # the IMU's freshness bit.
        Signal(
            name="set_measurement_accel",
            direction="in",
            n_floats=3,
            cpp_write_stmt=(
                "{accessor}.set_measurement_accel("
                "manta::geom::Vec3<manta::PartFrame>{{{v0}, {v1}, {v2}}});"
            ),
        ),
        Signal(
            name="set_measurement_gyro",
            direction="in",
            n_floats=3,
            cpp_write_stmt=(
                "{accessor}.set_measurement_gyro("
                "manta::geom::Vec3<manta::PartFrame>{{{v0}, {v1}, {v2}}});"
            ),
        ),
    ]

    def __init__(self,
                 name: str,
                 accel_sigma: float = 0.0,
                 gyro_sigma:  float = 0.0,
                 rate_hz: float = 0.0,
                 **kwargs) -> None:
        super().__init__(name=name, **kwargs)
        self.accel_sigma = float(accel_sigma)
        self.gyro_sigma  = float(gyro_sigma)
        # rate_hz=0 (default) means "refresh every tick" — backward compat
        # with code from before the rate-cap landing.
        self.rate_hz     = float(rate_hz)

    def emit_constructor_args(self, scalar: str = "manta::Real") -> str:
        return (f'"{self.name}", '
                f'{_f(self.accel_sigma)}, '
                f'{_f(self.gyro_sigma)}, '
                f'manta::Real({_f(self.rate_hz)})')

    # Vec3 telemetry types are not yet handled by the scalar-only JSON encoder.
    # Declaring the fields anyway so the struct shape is right; values won't
    # currently appear in the JSON. To be revisited when the encoder grows
    # vector support.
    def telemetry_fields(self) -> list[tuple[str, str]]:
        return [
            ("accel", "manta::geom::Vec3<manta::PartFrame>"),
            ("gyro",  "manta::geom::Vec3<manta::PartFrame>"),
        ]

    def emit_telemetry_reads(self) -> list[tuple[str, str]]:
        return [
            ("accel", f"craft.{self.name}().last_accel()"),
            ("gyro",  f"craft.{self.name}().last_gyro()"),
        ]

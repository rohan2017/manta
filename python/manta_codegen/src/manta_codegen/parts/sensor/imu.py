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
    ]

    def __init__(self,
                 name: str,
                 accel_sigma: float = 0.0,
                 gyro_sigma:  float = 0.0,
                 **kwargs) -> None:
        super().__init__(name=name, **kwargs)
        self.accel_sigma = float(accel_sigma)
        self.gyro_sigma  = float(gyro_sigma)

    def emit_constructor_args(self, scalar: str = "manta::Real") -> str:
        # ImuNoiseParams uses raw float fields (sim-side noise parameters);
        # they don't change with Scalar.
        return (f'"{self.name}", '
                f'manta::parts::ImuNoiseParams{{{_f(self.accel_sigma)}, {_f(self.gyro_sigma)}}}')

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

    def emit_measurement_decode(self, part_accessor: str, payload_var: str) -> str:
        # IMU measurement payload: 6 floats — [ax, ay, az, gx, gy, gz].
        return (
            f"if ({payload_var}.size() >= 6) {{ "
            f"{part_accessor}.set_measurement("
            f"manta::geom::Vec3<manta::PartFrame>{{"
            f"{payload_var}[0], {payload_var}[1], {payload_var}[2]}}, "
            f"manta::geom::Vec3<manta::PartFrame>{{"
            f"{payload_var}[3], {payload_var}[4], {payload_var}[5]}}); "
            f"}}"
        )

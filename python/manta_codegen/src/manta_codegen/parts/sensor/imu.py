"""IMU — inertial measurement unit (angular rate + acceleration, with noise)."""

from __future__ import annotations

from ..._format import cpp_float as _f
from ...core import PartDescriptor


class IMU(PartDescriptor):
    """An IMU samples the kinematic-pass acceleration and angular velocity each
    tick, with optional white-Gaussian noise on each.

    Required fields: none.
    Telemetry: gyro (Vec3<PartFrame>), accel (Vec3<PartFrame>) — currently
    not serialized to JSON (vector-typed telemetry serializer pending).
    """

    cpp_class          = "manta::parts::IMU"
    cpp_class_template = "manta::parts::IMUT"
    cpp_header         = "manta/parts/sensor/imu.hpp"

    def __init__(self,
                 name: str,
                 accel_sigma: float = 0.0,
                 gyro_sigma:  float = 0.0,
                 publish_state: bool = True,
                 **kwargs) -> None:
        super().__init__(name=name, publish_state=publish_state, **kwargs)
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

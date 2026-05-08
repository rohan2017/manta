"""IMU — inertial measurement unit (Kalibr 4-parameter noise model).

Mirrors https://github.com/ethz-asl/kalibr/wiki/IMU-Noise-Model:

    ω̃ = ω + b_g + n_g    ḃ_g = n_bg
    ã = a + b_a + n_a    ḃ_a = n_ba

with n_*, n_b* white-Gaussian. Pass any subset of the four σ values; any
left at the default (-1.0) is skipped — both for sim-side noise sampling
and for EKF/UKF noise registration. Setting only `accel_sigma` and
`gyro_sigma` reproduces the old "simple" IMU; adding `accel_bias_sigma`
and `gyro_bias_sigma` registers the biases as augmented filter states.
"""

from __future__ import annotations

from ..._format import cpp_float as _f, cpp_mfloat as _mf
from ...core import NoiseChannel, PartDescriptor
from ...signal import Signal, vec3_out_signal


class IMU(PartDescriptor):
    """IMU with Kalibr's 4-parameter noise model.

    Args (all σ default <0 → skip registration):
        accel_sigma:      accel white-noise density [m/s²/√Hz]
        gyro_sigma:       gyro white-noise density [rad/s/√Hz]
        accel_bias_sigma: accel bias random-walk PSD [m/s³/√Hz]
        gyro_bias_sigma:  gyro bias random-walk PSD [rad/s²/√Hz]
        rate_hz:          sample-rate cap (0 = unrated)

    Bias channels register as random walks; when σ ≥ 0 they become
    augmented filter states (BiasDim grows by 3 per enabled bias).

    Bindable signals:
      * `last_accel`           (out, 3 floats)
      * `last_gyro`             (out, 3 floats)
      * `set_measurement`       (in, 6 floats — both at once)
      * `set_measurement_accel` (in, 3 floats — single channel)
      * `set_measurement_gyro`  (in, 3 floats — single channel)
    """

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
                 accel_sigma:      float = -1.0,
                 gyro_sigma:       float = -1.0,
                 accel_bias_sigma: float = -1.0,
                 gyro_bias_sigma:  float = -1.0,
                 rate_hz:          float = 0.0,
                 **kwargs) -> None:
        super().__init__(name=name, **kwargs)
        self.accel_sigma      = float(accel_sigma)
        self.gyro_sigma       = float(gyro_sigma)
        self.accel_bias_sigma = float(accel_bias_sigma)
        self.gyro_bias_sigma  = float(gyro_bias_sigma)
        self.rate_hz          = float(rate_hz)

    def emit_constructor_args(self, scalar: str = "manta::MFloat") -> str:
        return (f'"{self.name}", '
                f'{_f(self.accel_sigma)}, '
                f'{_f(self.gyro_sigma)}, '
                f'{_f(self.accel_bias_sigma)}, '
                f'{_f(self.gyro_bias_sigma)}, '
                f'{_mf(self.rate_hz)}')

    def noise_channels(self) -> list[NoiseChannel]:
        return [
            NoiseChannel("accel_noise", "white_3d", self.accel_sigma),
            NoiseChannel("gyro_noise",  "white_3d", self.gyro_sigma),
            NoiseChannel("accel_bias",  "rw_3d",    self.accel_bias_sigma),
            NoiseChannel("gyro_bias",   "rw_3d",    self.gyro_bias_sigma),
        ]

    def telemetry(self) -> list[tuple[str, str, str]]:
        return [
            ("accel", "manta::geom::Vec3<manta::PartFrame>",
             f"craft.{self.name}().last_accel()"),
            ("gyro",  "manta::geom::Vec3<manta::PartFrame>",
             f"craft.{self.name}().last_gyro()"),
        ]

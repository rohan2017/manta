"""Portable preintegrated IMU values, independent of a transport or device."""

from collections.abc import Mapping
from dataclasses import dataclass
import math
import struct
from types import MappingProxyType

import numpy as np


PREINTEGRATION_PACKET_SCHEMA = 4
# Canonical wire order; matrices are column-major, like CasADi/C codegen.
PACKET_LAYOUT = (
    ("delta_orientation", (4,)),
    ("delta_velocity", (3,)),
    ("delta_position", (3,)),
    ("covariance", (9, 9)),
    ("delta_start_gyro_cross_covariance", (9, 3)),
    ("delta_end_gyro_cross_covariance", (9, 3)),
    ("start_end_gyro_correlation", (3, 3)),
    ("start_gyro_noise_sigma", (3,)),
    ("end_gyro_noise_sigma", (3,)),
    ("delta_start_accel_cross_covariance", (9, 3)),
    ("delta_end_accel_cross_covariance", (9, 3)),
    ("start_end_accel_correlation", (3, 3)),
    ("start_accel_noise_sigma", (3,)),
    ("end_accel_noise_sigma", (3,)),
    ("bias_jacobian", (9, 6)),
    ("gyro_bias_reference", (3,)),
    ("accel_bias_reference", (3,)),
    ("start_gyro", (3,)),
    ("end_accel", (3,)),
    ("end_gyro", (3,)),
    ("duration", ()),
    ("accel_endpoint", ()),
    ("sample_count", ()),
    ("velocity_time_moments", (4,)),
    ("position_time_moments", (4,)),
)
PACKET_FIELDS = tuple(name for name, _ in PACKET_LAYOUT)
_HEADER = struct.Struct("<4sId")
_COUNT = sum(math.prod(shape) for _, shape in PACKET_LAYOUT)


@dataclass(frozen=True)
class PreintegratedIMUPacket(Mapping):
    """One nonoverlapping IMU interval, ready for ``filter.predict(packet)``.

    ``data`` accepts the preintegrator's readouts or decoded firmware fields.
    Matrix inputs may be 2-D or flattened column-major. Values are copied and
    validated. ``start_time`` uses the same seconds/epoch as the filter and
    planet model; clock synchronization and sensor identity belong to the
    transport. Bias references are sensor-frame rad/s and m/s². Deltas are
    expressed in the IMU axes at packet start, not planet/navigation axes.

    Optional wire encoding: 4 bytes ``MIPK``, uint32 schema, float64 start_time,
    then PACKET_LAYOUT float64 values, all little-endian, without padding.
    ``wire_size`` is fixed. No CRC or transport framing is included. Firmware
    can send this encoding or use any transport that preserves these fields.
    """

    data: Mapping
    start_time: float
    schema: int = PREINTEGRATION_PACKET_SCHEMA
    wire_size = _HEADER.size + 8 * _COUNT

    def __post_init__(self):
        if not isinstance(self.data, Mapping):
            raise TypeError("packet data must be a mapping of preintegrator readouts")
        if (
            not isinstance(self.schema, (int, np.integer))
            or self.schema != PREINTEGRATION_PACKET_SCHEMA
        ):
            raise ValueError(f"unsupported IMU packet schema {self.schema}")
        if not math.isfinite(self.start_time):
            raise ValueError("packet start_time must be finite")
        missing = set(PACKET_FIELDS) - self.data.keys()
        if missing:
            raise ValueError(f"IMU packet missing {sorted(missing)}")
        owned = {}
        for name, shape in PACKET_LAYOUT:
            value = np.asarray(self.data[name], dtype=float)
            if value.shape not in (shape, (math.prod(shape),)):
                raise ValueError(
                    f"packet {name} requires shape {shape} or its flat vector"
                )
            if not np.isfinite(value).all():
                raise ValueError(f"packet {name} must be finite")
            flat = value.reshape(-1, order="F")
            owned[name] = (
                float(flat[0])
                if not shape
                else np.frombuffer(flat.astype(np.float64).tobytes(), dtype=np.float64)
            )
        if owned["duration"] <= 0 or not math.isfinite(
            self.start_time + owned["duration"]
        ):
            raise ValueError("packet duration must be positive with finite end_time")
        count = owned["sample_count"]
        if count < 1 or not count.is_integer():
            raise ValueError("packet sample_count must be a positive integer")
        if owned["accel_endpoint"] not in (0.0, 1.0):
            raise ValueError("packet accel_endpoint must be 0 or 1")
        q = owned["delta_orientation"]
        if abs(q @ q - 1) > 1e-8:
            raise ValueError("packet delta_orientation must be a unit quaternion")
        for name in PACKET_FIELDS:
            if name.endswith("noise_sigma") and np.any(owned[name] < 0):
                raise ValueError(f"packet {name} must be nonnegative")
        covariance = owned["covariance"].reshape(9, 9, order="F")
        tolerance = 1e-12 + 1e-10 * np.linalg.norm(covariance, ord=2)
        if (
            not np.allclose(covariance, covariance.T, atol=tolerance, rtol=0)
            or np.linalg.eigvalsh((covariance + covariance.T) / 2).min() < -tolerance
        ):
            raise ValueError(
                "packet covariance must be symmetric positive semidefinite"
            )

        # Endpoint cross-covariances must describe a possible joint distribution.
        def matrix(name, shape):
            return owned[name].reshape(shape, order="F")

        start = np.column_stack(
            [
                matrix(f"delta_start_{sensor}_cross_covariance", (9, 3))
                for sensor in ("gyro", "accel")
            ]
        )
        end = np.column_stack(
            [
                matrix(f"delta_end_{sensor}_cross_covariance", (9, 3))
                for sensor in ("gyro", "accel")
            ]
        )
        correlation = np.zeros((6, 6))
        for i, sensor in enumerate(("gyro", "accel")):
            correlation[3 * i : 3 * i + 3, 3 * i : 3 * i + 3] = matrix(
                f"start_end_{sensor}_correlation", (3, 3)
            )
        joint = np.block(
            [
                [covariance, start, end],
                [start.T, np.eye(6), correlation],
                [end.T, correlation.T, np.eye(6)],
            ]
        )
        scale = np.sqrt(np.maximum(np.diag(joint), 1e-30))
        normalized = joint / scale[:, None] / scale[None, :]
        if np.linalg.eigvalsh((normalized + normalized.T) / 2).min() < -1e-8:
            raise ValueError(
                "packet endpoint correlations do not form a positive semidefinite joint covariance"
            )
        object.__setattr__(self, "data", MappingProxyType(owned))
        object.__setattr__(self, "start_time", float(self.start_time))

    @property
    def duration(self):
        return self["duration"]

    @property
    def end_time(self):
        return self.start_time + self.duration

    def __getitem__(self, name):
        return self.data[name]

    def __iter__(self):
        return iter(PACKET_FIELDS)

    def __len__(self):
        return len(PACKET_FIELDS)

    def to_bytes(self):
        payload = np.concatenate([np.atleast_1d(self[name]) for name in PACKET_FIELDS])
        return (
            _HEADER.pack(b"MIPK", self.schema, self.start_time)
            + payload.astype("<f8").tobytes()
        )

    @staticmethod
    def c_header():
        """C99/C++ encoder for flattened codegen readouts; no heap or Eigen."""
        offsets, offset = [], 0
        for name, shape in PACKET_LAYOUT:
            offsets.append(f"#define MANTA_IMU_{name.upper()} {offset}")
            offset += math.prod(shape)
        return "\n".join(
            [
                "/* Generated by Manta. Matrix readouts are column-major. */",
                "#ifndef MANTA_IMU_PACKET_H",
                "#define MANTA_IMU_PACKET_H",
                "#include <stdint.h>",
                "#include <string.h>",
                "#include <float.h>",
                "#if DBL_MANT_DIG != 53 || DBL_MAX_EXP != 1024 || FLT_RADIX != 2",
                '#error "Manta IMU packets require IEEE-754 binary64 doubles"',
                "#endif",
                "typedef char manta_double_must_be_8_bytes[(sizeof(double)==8)?1:-1];",
                f"#define MANTA_IMU_PACKET_SCHEMA {PREINTEGRATION_PACKET_SCHEMA}",
                f"#define MANTA_IMU_PACKET_SIZE {PreintegratedIMUPacket.wire_size}",
                f"#define MANTA_IMU_READOUT_COUNT {_COUNT}",
                *offsets,
                "static inline void manta_imu_f64(uint8_t *out, double value) {",
                "    uint64_t bits; memcpy(&bits, &value, 8);",
                "    for (int k=0; k<8; ++k) out[k]=(uint8_t)(bits >> (8*k));",
                "}",
                "static inline void manta_imu_packet_encode(uint8_t *out, double start_time, const double *readouts) {",
                '    memcpy(out, "MIPK", 4);',
                "    for (int k=0; k<4; ++k) out[4+k]=(uint8_t)((uint32_t)MANTA_IMU_PACKET_SCHEMA >> (8*k));",
                "    manta_imu_f64(out+8, start_time);",
                "    for (int k=0; k<MANTA_IMU_READOUT_COUNT; ++k) manta_imu_f64(out+16+8*k, readouts[k]);",
                "}",
                "#endif",
                "",
            ]
        )

    @classmethod
    def from_bytes(cls, data):
        if len(data) != cls.wire_size:
            raise ValueError(f"IMU packet requires exactly {cls.wire_size} bytes")
        magic, schema, start = _HEADER.unpack_from(data)
        if magic != b"MIPK":
            raise ValueError("invalid IMU packet magic")
        payload = np.frombuffer(data, dtype="<f8", offset=_HEADER.size)
        values, offset = {}, 0
        for name, shape in PACKET_LAYOUT:
            count = math.prod(shape)
            values[name] = payload[offset : offset + count].reshape(shape, order="F")
            offset += count
        return cls(values, start_time=start, schema=schema)

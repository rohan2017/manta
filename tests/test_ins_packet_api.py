"""Device/host packet boundary and the common filter runtime."""

from dataclasses import replace
from pathlib import Path
import shutil
import subprocess
import numpy as np
import pytest
from manta import INS, IMUPreintegrator, PreintegratedIMUPacket, TargetCpp, TargetNumpy
from manta.estimation.imu_packet import PACKET_LAYOUT
from tests.test_imu_preintegrator import _world


def block():
    return IMUPreintegrator(
        gyro_integration="endpoint",
        accel_integration="endpoint",
        sample_rate_hz=100,
        gyro_noise_density=0.0003,
        accel_noise_density=0.002,
    )


def step(pre, k):
    return pre.step(
        0.01,
        accel=[0.4, -0.2, 9.81],
        gyro=[0.3 + 0.01 * k, -0.1, 0.5],
        end_accel=[0.4, -0.2, 9.81],
        end_gyro=[0.3 + 0.01 * (k + 1), -0.1, 0.5],
        accel_bias=[0, 0, 0],
        gyro_bias=[0, 0, 0],
    )


def packet(start=0.0):
    return PreintegratedIMUPacket(step(TargetNumpy(block()), 0), start_time=start)


def test_packet_roundtrip_owns_data_and_matches_readout_layout():
    values = step(TargetNumpy(block()), 0)
    p = PreintegratedIMUPacket(values, start_time=123.0)
    decoded = PreintegratedIMUPacket.from_bytes(p.to_bytes())
    assert decoded.end_time == 123.01
    assert len(p.to_bytes()) == p.wire_size
    for key in p:
        np.testing.assert_array_equal(decoded[key], p[key])
    values["delta_velocity"][0] = 12345
    assert p["delta_velocity"][0] != 12345
    with pytest.raises(ValueError):
        p["delta_velocity"][0] = 1
    assert tuple(f.name for f in block().module().port("y").fields) == tuple(p)


@pytest.mark.parametrize(
    "change",
    [
        {"schema": 999},
        {"start_time": float("nan")},
        {"duration": 0},
        {"sample_count": 1.5},
        {"accel_endpoint": 2},
        {"end_gyro": [0, 0]},
        {"gyro_bias_reference": [0, 0, float("inf")]},
        {"delta_orientation": [1, 1, 0, 0]},
        {"end_accel_noise_sigma": [-1, 0, 0]},
        {"covariance": -np.eye(9)},
        {"delta_end_gyro_cross_covariance": np.ones((9, 3))},
        {"start_end_accel_correlation": np.eye(3) * 2},
    ],
)
def test_bad_packets_rejected(change):
    p = packet()
    if set(change) <= {"schema", "start_time"}:
        with pytest.raises(ValueError):
            replace(p, **change)
    else:
        with pytest.raises(ValueError):
            replace(p, data={**p, **change})


def test_truncated_or_unknown_wire_packet_is_rejected():
    wire = packet().to_bytes()
    for bad in (
        wire[:-1],
        wire + b"X",
        b"WRNG" + wire[4:],
        wire[:4] + b"\xff" * 4 + wire[8:],
    ):
        with pytest.raises(ValueError):
            PreintegratedIMUPacket.from_bytes(bad)


def test_common_predict_accepts_packet_and_rejects_gaps_atomically():
    model = INS(
        _world(),
        imu="imu",
        sensors=[],
        propagation="preintegrated",
        accel_integration="endpoint",
        covariance="conventional_reset",
    )
    a, b = TargetNumpy(model), TargetNumpy(model)
    a.reset(t=100.0)
    b.reset(t=100.0)
    p = packet(100.0)
    a.predict(p)
    b.predict_preintegrated(dict(p), t=100.0)
    np.testing.assert_array_equal(a.x, b.x)
    np.testing.assert_array_equal(a.P, b.P)
    saved = a.checkpoint()
    for bad in (p, replace(p, start_time=100.02)):
        with pytest.raises(ValueError, match="start at filter.time"):
            a.predict(bad)
        np.testing.assert_array_equal(a.x, saved.x)
        np.testing.assert_array_equal(a.P, saved.P)
    assert a.time == 100.01


@pytest.mark.cpp
def test_generated_preintegrator_c_packet_roundtrip(tmp_path):
    if (
        not shutil.which("cc")
        or not shutil.which("c++")
        or not Path("/usr/include/eigen3").exists()
    ):
        pytest.skip("C/C++ and Eigen are needed for the generated-code test")
    generated = TargetCpp(block(), tmp_path, class_name="Preintegrator")
    (tmp_path / "imu_packet.h").write_text(PreintegratedIMUPacket.c_header())
    assignments = []
    for name, shape in PACKET_LAYOUT:
        count = int(np.prod(shape))
        for i in range(count):
            value = f"o.{name}[{i}]" if shape else f"o.{name}"
            assignments.append(f"flat[MANTA_IMU_{name.upper()}+{i}]={value};")
    (tmp_path / "main.cpp").write_text(
        """
#include "preintegrator.hpp"
#include "imu_packet.h"
#include <cstdio>
int main() {
  manta_gen::Preintegrator pre;
  manta_gen::Preintegrator::Inputs u;
  manta_gen::Preintegrator::Outputs o;
  u.accel << .4, -.2, 9.81; u.end_accel = u.accel;
  u.accel_bias.setZero(); u.gyro_bias.setZero();
  for (int k=0;k<3;k++) {
    u.gyro << .3+.01*k, -.1, .5;
    u.end_gyro << .3+.01*(k+1), -.1, .5;
    o=pre.step(u,.01);
  }
  double flat[MANTA_IMU_READOUT_COUNT];
"""
        + "\n".join(assignments)
        + """
  uint8_t wire[MANTA_IMU_PACKET_SIZE];
  manta_imu_packet_encode(wire,123.,flat);
  return std::fwrite(wire,1,sizeof(wire),stdout)==sizeof(wire) ? 0 : 1;
}
"""
    )
    commands = [
        ["cc", "-O1", "-c", str(generated.kernels_c), "-o", str(tmp_path / "kernel.o")],
        [
            "c++",
            "-std=c++17",
            "-O1",
            "-I/usr/include/eigen3",
            f"-I{tmp_path}",
            str(tmp_path / "main.cpp"),
            str(generated.wrapper_cpp),
            str(tmp_path / "kernel.o"),
            "-o",
            str(tmp_path / "run"),
        ],
    ]
    for command in commands:
        subprocess.run(command, check=True, capture_output=True, timeout=120)
    wire = subprocess.check_output([str(tmp_path / "run")])
    actual = PreintegratedIMUPacket.from_bytes(wire)
    pre = TargetNumpy(block())
    for k in range(3):
        expected = step(pre, k)
    for name in actual:
        np.testing.assert_allclose(
            actual[name], expected[name], atol=2e-13, rtol=1e-10, err_msg=name
        )
    assert actual.start_time == 123.0

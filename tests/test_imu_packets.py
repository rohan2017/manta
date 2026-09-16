"""Batching must preserve high-rate motion and its joint packet noise."""
import copy
from functools import reduce

import numpy as np
import pytest

from manta import (
    IMUPreintegrator,
    TargetNumpy,
    compose_preintegrated_packets,
    frame_preintegrated_packet,
)
from manta.estimation.imu_preintegrator import PACKET_FIELDS


def _packets():
    rng = np.random.default_rng(713)
    dts = rng.uniform(.003, .005, 12)
    accel = rng.normal(size=(13, 3)) + (0, 0, 9.81)
    gyro = rng.normal(size=(13, 3))*.4
    bias_a, bias_g = (.01, -.02, .03), (.001, .002, -.001)
    pre = TargetNumpy(IMUPreintegrator(accel_noise_density=.002, gyro_noise_density=.0002))
    full = TargetNumpy(IMUPreintegrator(accel_noise_density=.002, gyro_noise_density=.0002))
    packets = []
    for i, dt in enumerate(dts):
        args = {
            "accel": accel[i], "gyro": gyro[i],
            "accel_bias": bias_a, "gyro_bias": bias_g,
        }
        packet = pre.step(float(dt), **args)
        complete = full.step(float(dt), **args)
        frame = {
            "end_accel": accel[i+1], "end_gyro": gyro[i+1],
            "end_gyro_noise_sigma": np.full(
                3, .0002/np.sqrt(dts[min(i+1, len(dts)-1)])
            ),
        }
        packets.append(frame_preintegrated_packet(packet, **frame))
        pre.reset()
    return packets, frame_preintegrated_packet(complete, **frame)


def test_composition_matches_raw_recurrence_including_noise_bias_and_earth_moments():
    packets, reference = _packets()
    before = copy.deepcopy(packets)
    combined = reduce(compose_preintegrated_packets, packets)
    for field in PACKET_FIELDS:
        np.testing.assert_allclose(combined[field], reference[field], rtol=2e-10, atol=1e-13,
            err_msg=field)
    for original, packet in zip(before, packets):
        for field in PACKET_FIELDS:
            np.testing.assert_array_equal(original[field], packet[field])


def test_transport_batch_grouping_does_not_change_composed_packet():
    packets, _ = _packets()
    direct = reduce(compose_preintegrated_packets, packets)
    groups = [reduce(compose_preintegrated_packets, packets[i:i+3]) for i in range(0, 12, 3)]
    grouped = reduce(compose_preintegrated_packets, groups)
    for field in PACKET_FIELDS:
        np.testing.assert_allclose(grouped[field], direct[field], rtol=2e-10, atol=1e-13)


@pytest.mark.parametrize("field", ["gyro_bias_reference", "accel_bias_reference", "start_gyro",
    "start_gyro_noise_sigma", "delta_end_gyro_cross_covariance", "start_end_gyro_correlation"])
def test_incompatible_or_nonindependent_packets_are_refused(field):
    packets, _ = _packets()
    left, right = packets[:2]
    right[field] = np.asarray(right[field]).copy() + .1
    with pytest.raises(ValueError):
        compose_preintegrated_packets(left, right)

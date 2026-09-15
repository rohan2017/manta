"""Composition of contiguous, independently framed preintegration intervals.

The producer owns acquisition identities, timestamps, clock epochs and schema
validation. This operation owns the motion, bias and noise algebra. A batch
may retain a packet at every raw IMU boundary while a consumer combines only
the intervals up to its next aiding or publication boundary.
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np

from ..ir._rotation import quat_mul_np, quat_to_rotmat_np
from .imu_preintegrator import PACKET_FIELDS


def compose_preintegrated_packets(
    left: Mapping[str, Any], right: Mapping[str, Any],
) -> dict[str, Any]:
    """Compose adjacent packets with the same fixed bias reference.

    Both must have fresh independent right gyro boundaries, as produced by
    ``frame_preintegrated_packet``. The left end and right start must name the
    same physical gyro observation (the caller validates its identity/time).
    Their integrated samples are disjoint, although their boundary sample is
    shared. Overlapping cumulative prefixes are not valid inputs.

    Returns new arrays; neither input is mutated. Noise and bias sensitivities
    are transported through the same first-order increment composition used
    by the preintegrator. This does not run the outer INS.
    """
    for packet in (left, right):
        missing = set(PACKET_FIELDS) - packet.keys()
        if missing:
            raise ValueError(f"preintegration packet missing {sorted(missing)}")

    def vector(packet: Mapping[str, Any], key: str, size: int) -> np.ndarray:
        value = np.asarray(packet[key], dtype=float).reshape(-1)
        if value.size != size or not np.isfinite(value).all():
            raise ValueError(f"packet {key} must contain {size} finite values")
        return value

    def matrix(packet: Mapping[str, Any], key: str, rows: int, cols: int) -> np.ndarray:
        return vector(packet, key, rows * cols).reshape((rows, cols), order="F")

    for packet in (left, right):
        for key in ("delta_end_gyro_cross_covariance", "start_end_gyro_correlation"):
            size = 27 if key.startswith("delta") else 9
            if np.any(vector(packet, key, size)):
                raise ValueError("packet composition requires fresh independent right boundaries")
        count = float(vector(packet, "sample_count", 1)[0])
        if count < 1 or not count.is_integer():
            raise ValueError("packet sample_count must be a positive integer")
        for key in ("start_gyro_noise_sigma", "end_gyro_noise_sigma"):
            if np.any(vector(packet, key, 3) < 0):
                raise ValueError("packet gyro noise sigma must be nonnegative")
        for key in ("gyro_bias_reference", "accel_bias_reference"):
            vector(packet, key, 3)
        vector(packet, "start_gyro", 3)
        vector(packet, "end_gyro", 3)
        vector(packet, "end_accel", 3)

    for key in ("gyro_bias_reference", "accel_bias_reference"):
        if not np.array_equal(vector(left, key, 3), vector(right, key, 3)):
            raise ValueError("packet composition requires identical bias references")
    for end, start in (("end_gyro", "start_gyro"),
                       ("end_gyro_noise_sigma", "start_gyro_noise_sigma")):
        if not np.array_equal(vector(left, end, 3), vector(right, start, 3)):
            raise ValueError("packet gyro boundary is discontinuous")

    ta, tb = (float(vector(p, "duration", 1)[0]) for p in (left, right))
    if ta <= 0 or tb <= 0:
        raise ValueError("packet durations must be positive")
    qa, qb = (vector(p, "delta_orientation", 4) for p in (left, right))
    if any(abs(float(q @ q) - 1) > 1e-9 for q in (qa, qb)):
        raise ValueError("packet orientation must be a unit quaternion")
    ra, rb = quat_to_rotmat_np(qa), quat_to_rotmat_np(qb)
    va, vb = (vector(p, "delta_velocity", 3) for p in (left, right))
    pa, pb = (vector(p, "delta_position", 3) for p in (left, right))

    def skew(v: np.ndarray) -> np.ndarray:
        x, y, z = v
        return np.array(((0., -z, y), (z, 0., -x), (-y, x, 0.)))

    a, b = np.zeros((9, 9)), np.zeros((9, 9))
    a[:3, :3] = rb.T
    a[3:6, :3] = -ra @ skew(vb)
    a[3:6, 3:6] = np.eye(3)
    a[6:9, :3] = -ra @ skew(pb)
    a[6:9, 3:6] = tb * np.eye(3)
    a[6:9, 6:9] = np.eye(3)
    b[:3, :3] = np.eye(3)
    b[3:6, 3:6] = ra
    b[6:9, 6:9] = ra
    cov = (a @ matrix(left, "covariance", 9, 9) @ a.T
           + b @ matrix(right, "covariance", 9, 9) @ b.T)
    jac = a @ matrix(left, "bias_jacobian", 9, 6) + b @ matrix(right, "bias_jacobian", 9, 6)
    start_cross = a @ matrix(left, "delta_start_gyro_cross_covariance", 9, 3)
    # Validate the unused right/start block too. That boundary lies inside the
    # combined packet; its noise is already in the right increment covariance.
    matrix(right, "delta_start_gyro_cross_covariance", 9, 3)

    shift = np.array([[math.comb(j, k) * ta**(j-k) if k <= j else 0.
                       for k in range(4)] for j in range(4)])
    vma = vector(left, "velocity_time_moments", 4)
    vm = vma + shift @ vector(right, "velocity_time_moments", 4)
    pm = (vector(left, "position_time_moments", 4) + tb*vma
          + shift @ vector(right, "position_time_moments", 4))
    result = {key: np.asarray(value).copy() for key, value in left.items()}
    result.update({
        "delta_orientation": quat_mul_np(qa, qb),
        "delta_velocity": va + ra @ vb,
        "delta_position": pa + tb*va + ra @ pb,
        "covariance": ((cov + cov.T)*.5).reshape(-1, order="F"),
        "bias_jacobian": jac.reshape(-1, order="F"),
        "delta_start_gyro_cross_covariance": start_cross.reshape(-1, order="F"),
        "delta_end_gyro_cross_covariance": np.zeros(27),
        "start_end_gyro_correlation": np.zeros(9),
        "end_accel": vector(right, "end_accel", 3).copy(),
        "end_gyro": vector(right, "end_gyro", 3).copy(),
        "end_gyro_noise_sigma": vector(right, "end_gyro_noise_sigma", 3).copy(),
        "duration": ta + tb,
        "sample_count": float(left["sample_count"]) + float(right["sample_count"]),
        "velocity_time_moments": vm,
        "position_time_moments": pm,
    })
    return result

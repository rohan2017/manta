"""High-rate IMU preintegration as a deployable recurrence block.

The block folds an ordered stream of accelerometer and gyro samples into one
lower-rate packet.  Rotation increments are composed on SO(3), so coning
information present in the high-rate sample order is retained when the packet
is later consumed by :class:`~manta.estimation.ins.INS`.

The packet convention is

    R_j = R_i DeltaR
    v_j = v_i + g dt + R_i Deltav
    p_j = p_i + v_i dt + 0.5 g dt^2 + R_i Deltap

where the deltas are expressed in the sensor frame at the beginning of the
packet.  ``covariance`` is the 9x9 covariance of ``[dtheta, dv, dp]`` and
``bias_jacobian`` is the 9x6 derivative with respect to
``[d gyro_bias, d accel_bias]``.  Both matrices are flattened in CasADi/Eigen
column-major order.
"""

from __future__ import annotations

import math
from functools import lru_cache

import casadi as ca
import numpy as np

from ..ir._rotation import quat_conj, quat_mul, so3_exp, so3_log
from ..ir.manifold import R3Manifold, RnManifold, ScalarManifold, SO3Manifold
from ..recurrence import RecurrenceBlock

PACKET_FIELDS = (
    "delta_orientation",
    "delta_velocity",
    "delta_position",
    "covariance",
    "bias_jacobian",
    "gyro_bias_reference",
    "accel_bias_reference",
    "start_accel",
    "start_gyro",
    "end_accel",
    "end_gyro",
    "duration",
    "sample_count",
)


@lru_cache(maxsize=32)
def _single_sample_kernel(accel_density: float, gyro_density: float):
    block = IMUPreintegrator(
        accel_noise_density=accel_density,
        gyro_noise_density=gyro_density)
    return block.update_fn, block.x0.copy(), tuple(block.outputs)


def _single_sample_packet(*, accel, gyro, dt: float,
                          accel_noise_sigma: float,
                          gyro_noise_sigma: float) -> dict[str, object]:
    """Build the one-sample packet used by truth-backed analysis tools.

    IMU Part white-noise sigmas are per tick, while ``IMUPreintegrator``
    accepts densities. Multiplication by ``sqrt(dt)`` makes the integrated
    packet covariance identical to raw INS for this sample interval.
    """
    accel_density = float(accel_noise_sigma) * math.sqrt(float(dt))
    gyro_density = float(gyro_noise_sigma) * math.sqrt(float(dt))
    fn, x0, outputs = _single_sample_kernel(accel_density, gyro_density)
    u = np.concatenate((np.asarray(accel, dtype=float).reshape(3),
                        np.asarray(gyro, dtype=float).reshape(3),
                        np.zeros(6)))
    _x_next, y = fn(x0, u, float(dt), 0.0)
    flat = np.asarray(y, dtype=float).reshape(-1)
    packet: dict[str, object] = {}
    off = 0
    for port in outputs:
        value = flat[off:off + port.dim].copy()
        packet[port.name] = float(value[0]) if port.dim == 1 else value
        off += port.dim
    return packet


def _finite_nonnegative(value: float, *, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and >= 0, got {value!r}")
    return value


def _normalise(q):
    return q / ca.sqrt(ca.dot(q, q) + 1e-30)


def _local_error(q, dv, dp, q_ref, dv_ref, dp_ref):
    return ca.vertcat(
        so3_log(quat_mul(quat_conj(q_ref), q)),
        dv - dv_ref,
        dp - dp_ref,
    )


class IMUPreintegrator(RecurrenceBlock):
    """Accumulate high-rate IMU samples into an INS prediction packet.

    Args:
        accel_noise_density: accelerometer white-noise density in
            ``m/s^2/sqrt(Hz)``.
        gyro_noise_density: gyro white-noise density in ``rad/s/sqrt(Hz)``.
        name: codegen basename / default C++ class stem.

    ``accel_bias`` and ``gyro_bias`` are latched from the first sample after
    reset and become the packet's bias reference.  Later values supplied
    before the next reset are intentionally ignored; one packet must have one
    linearization point.
    """

    def __init__(self, *, accel_noise_density: float = 0.0,
                 gyro_noise_density: float = 0.0,
                 name: str = "imu_preintegrator") -> None:
        self.accel_noise_density = _finite_nonnegative(
            accel_noise_density, name="accel_noise_density")
        self.gyro_noise_density = _finite_nonnegative(
            gyro_noise_density, name="gyro_noise_density")
        sigma_a = self.accel_noise_density
        sigma_g = self.gyro_noise_density

        def rec(x, u, dt, t):
            del t
            q = x["delta_orientation"]
            dv = x["delta_velocity"]
            dp = x["delta_position"]
            covariance = ca.reshape(x["covariance"], 9, 9)
            bias_jacobian = ca.reshape(x["bias_jacobian"], 9, 6)
            first = x["sample_count"] < 0.5

            gyro_bias_ref = ca.if_else(
                first, u["gyro_bias"], x["gyro_bias_reference"])
            accel_bias_ref = ca.if_else(
                first, u["accel_bias"], x["accel_bias_reference"])
            accel = u["accel"] - accel_bias_ref
            gyro = u["gyro"] - gyro_bias_ref

            def compose(q0, dv0, dp0, a, w):
                # Left-endpoint specific-force integration, matching the raw
                # INS recurrence. Ordered quaternion products preserve coning.
                from ..ir._rotation import quat_to_rotmat
                a0 = quat_to_rotmat(q0) @ a
                return (
                    _normalise(quat_mul(q0, so3_exp(w * dt))),
                    dv0 + a0 * dt,
                    dp0 + dv0 * dt + 0.5 * a0 * dt * dt,
                )

            q_next, dv_next, dp_next = compose(q, dv, dp, accel, gyro)

            # Error transition A over [dtheta, dv, dp], obtained from the
            # recurrence itself rather than maintained as hand-coded algebra.
            error = ca.MX.sym("preint_error", 9, 1)
            q_pert = quat_mul(q, so3_exp(error[0:3]))
            qp, dvp, dpp = compose(
                q_pert, dv + error[3:6], dp + error[6:9], accel, gyro)
            propagated_error = _local_error(
                qp, dvp, dpp, q_next, dv_next, dp_next)
            A = ca.substitute(ca.jacobian(propagated_error, error), error,
                              ca.MX.zeros(9, 1))

            # Bias sensitivities use [gyro bias, accel bias]. A packet can be
            # corrected at the main filter's current bias without replaying
            # the high-rate samples while the first-order approximation holds.
            db = ca.MX.sym("preint_db", 6, 1)
            qb, dvb, dpb = compose(
                q, dv, dp, accel - db[3:6], gyro - db[0:3])
            bias_error = _local_error(
                qb, dvb, dpb, q_next, dv_next, dp_next)
            B = ca.substitute(ca.jacobian(bias_error, db), db,
                              ca.MX.zeros(6, 1))
            J_next = A @ bias_jacobian + B

            # The constructor takes continuous white-noise densities. A held
            # rate sample has sigma_density/sqrt(dt); autodiff maps that sample
            # uncertainty through the same nonlinear composition.
            white = ca.MX.sym("preint_white", 6, 1)
            sqrt_dt = ca.sqrt(dt)
            qn, dvn, dpn = compose(
                q, dv, dp,
                accel + (sigma_a / sqrt_dt) * white[3:6],
                gyro + (sigma_g / sqrt_dt) * white[0:3])
            noise_error = _local_error(
                qn, dvn, dpn, q_next, dv_next, dp_next)
            G = ca.substitute(ca.jacobian(noise_error, white), white,
                              ca.MX.zeros(6, 1))
            C_next = A @ covariance @ A.T + G @ G.T
            C_next = 0.5 * (C_next + C_next.T)

            start_accel = ca.if_else(first, u["accel"], x["start_accel"])
            start_gyro = ca.if_else(first, u["gyro"], x["start_gyro"])
            nxt = {
                "delta_orientation": q_next,
                "delta_velocity": dv_next,
                "delta_position": dp_next,
                "covariance": ca.reshape(C_next, 81, 1),
                "bias_jacobian": ca.reshape(J_next, 54, 1),
                "gyro_bias_reference": gyro_bias_ref,
                "accel_bias_reference": accel_bias_ref,
                "start_accel": start_accel,
                "start_gyro": start_gyro,
                "end_accel": u["accel"],
                "end_gyro": u["gyro"],
                "duration": x["duration"] + dt,
                "sample_count": x["sample_count"] + 1.0,
            }
            return nxt, dict(nxt)

        zero3 = np.zeros(3)
        self._build_recurrence(
            name=name,
            state=[
                ("delta_orientation", SO3Manifold()),
                ("delta_velocity", R3Manifold()),
                ("delta_position", R3Manifold()),
                ("covariance", RnManifold(81)),
                ("bias_jacobian", RnManifold(54)),
                ("gyro_bias_reference", R3Manifold()),
                ("accel_bias_reference", R3Manifold()),
                ("start_accel", R3Manifold()),
                ("start_gyro", R3Manifold()),
                ("end_accel", R3Manifold()),
                ("end_gyro", R3Manifold()),
                ("duration", ScalarManifold()),
                ("sample_count", ScalarManifold()),
            ],
            inputs=[("accel", 3), ("gyro", 3),
                    ("accel_bias", 3), ("gyro_bias", 3)],
            outputs=[
                ("delta_orientation", 4),
                ("delta_velocity", 3),
                ("delta_position", 3),
                ("covariance", 81),
                ("bias_jacobian", 54),
                ("gyro_bias_reference", 3),
                ("accel_bias_reference", 3),
                ("start_accel", 3),
                ("start_gyro", 3),
                ("end_accel", 3),
                ("end_gyro", 3),
                ("duration", 1),
                ("sample_count", 1),
            ],
            x0={
                "delta_orientation": (1.0, 0.0, 0.0, 0.0),
                "delta_velocity": zero3,
                "delta_position": zero3,
                "covariance": np.zeros(81),
                "bias_jacobian": np.zeros(54),
                "gyro_bias_reference": zero3,
                "accel_bias_reference": zero3,
                "start_accel": zero3,
                "start_gyro": zero3,
                "end_accel": zero3,
                "end_gyro": zero3,
                "duration": 0.0,
                "sample_count": 0.0,
            },
            recurrence=rec,
        )

    def __repr__(self) -> str:
        return ("<IMUPreintegrator "
                f"accel_noise_density={self.accel_noise_density} "
                f"gyro_noise_density={self.gyro_noise_density}>")

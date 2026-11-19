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
column-major order. These equations describe inertial/non-rotating axes;
a planet-attached INS applies the companion-side navigation-frame correction.
Packet schema 4 carries velocity/position sampling-time moments
(order zero through three), allowing that correction to preserve the actual
left-held quadrature without teaching this recurrence about the planet.

The packet also carries the joint covariance between its delta and its two
gyro boundaries in standardized coordinates.  That information is required
when an IMU is displaced from the craft origin: the same noisy boundary gyro
both participates in preintegration and removes the rigid-body lever velocity.
A 9x9 delta covariance alone loses that correlation and can be severely
overconfident. ``gyro_integration='left_hold'`` describes held rate inputs;
its end boundary is its last integrated sample. A framer replacing that end
with a fresh independent acquisition must zero the corresponding correlations.
``gyro_integration='endpoint'`` instead integrates two real point samples with
ordered half rotations and retains their shared error across intervals. Its
right endpoint already belongs to the packet: DO NOT pass it to the independent
endpoint framer. Accelerometer integration is separately selectable:
``accel_integration='endpoint'`` uses Simpson force quadrature, coning and
six-component gyro/accelerometer boundary noise. Schema 4 adds accelerometer
boundary cross-covariances, sigmas and the ``accel_endpoint`` mode tag. Its
consumer must retain both boundary errors; a schema-3 consumer is insufficient.
"""

from __future__ import annotations

import math
from dataclasses import replace
from functools import lru_cache

import casadi as ca
import numpy as np

from ..ir._rotation import quat_conj, quat_mul, so3_exp, so3_log
from ..ir.manifold import R3Manifold, RnManifold, ScalarManifold, SO3Manifold
from ..recurrence import RecurrenceBlock

from .imu_packet import PACKET_FIELDS, PREINTEGRATION_PACKET_SCHEMA


@lru_cache(maxsize=32)
def _single_sample_kernel(accel_density: float, gyro_density: float):
    block = IMUPreintegrator(
        accel_noise_density=accel_density,
        gyro_noise_density=gyro_density)
    return block.update_fn, block.x0.copy(), tuple(block.outputs)


def _single_sample_packet(*, accel, gyro, end_accel, end_gyro, dt: float,
                          accel_noise_sigma: float,
                          gyro_noise_sigma: float) -> dict[str, object]:
    """Build one left-held interval for truth-backed analysis tools.

    IMU Part white-noise sigmas are per tick, while ``IMUPreintegrator``
    accepts densities. Multiplication by ``sqrt(dt)`` makes the integrated
    packet covariance identical to raw INS for this sample interval.

    ``accel``/``gyro`` are the sample held over the interval. ``end_accel`` and
    ``end_gyro`` are the independently sampled right boundary. Requiring both
    prevents a single sample from being labeled as two physical observations.
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
    return frame_preintegrated_packet(
        packet, end_accel=end_accel, end_gyro=end_gyro
    )


def frame_preintegrated_packet(
    packet: dict[str, object],
    *,
    end_accel,
    end_gyro,
    end_gyro_noise_sigma=None,
) -> dict[str, object]:
    """Attach a fresh, independent right IMU boundary to a packet.

    ``IMUPreintegrator.step`` consumes left-held intervals, so its immediate
    readout describes the last *integrated* gyro as the recurrence end. A
    deployable packet spanning ``[t0, t1]`` instead needs the fresh IMU sample
    acquired at ``t1``. This helper changes the samples and, inseparably, the
    joint covariance contract: the new endpoint is independent of the delta
    and start boundary.

    At a fixed IMU cadence the recurrence's last per-sample gyro sigma is also
    the endpoint sigma. A variable-rate framer must pass the endpoint's
    three-axis ``end_gyro_noise_sigma`` explicitly.
    """
    if bool(np.asarray(packet.get("accel_endpoint",0)).item()):
        raise ValueError("an integrated accelerometer endpoint cannot be reframed as independent")
    out = dict(packet)
    missing = [name for name in PACKET_FIELDS if name not in out]
    if missing:
        raise KeyError(f"preintegration packet is missing {missing}")

    def vector3(value, *, name: str, nonnegative: bool = False):
        array = np.asarray(value, dtype=float).reshape(-1)
        if array.shape != (3,) or not np.isfinite(array).all():
            raise ValueError(f"{name} must contain three finite values")
        if nonnegative and np.any(array < 0.0):
            raise ValueError(f"{name} must be nonnegative")
        return array.copy()

    out["end_accel"] = vector3(end_accel, name="end_accel")
    out["end_gyro"] = vector3(end_gyro, name="end_gyro")
    if end_gyro_noise_sigma is not None:
        out["end_gyro_noise_sigma"] = vector3(
            end_gyro_noise_sigma,
            name="end_gyro_noise_sigma",
            nonnegative=True,
        )
    else:
        out["end_gyro_noise_sigma"] = vector3(
            out["end_gyro_noise_sigma"],
            name="end_gyro_noise_sigma",
            nonnegative=True,
        )
    out["delta_end_accel_cross_covariance"] = np.zeros(27)
    out["start_end_accel_correlation"] = np.zeros(9)
    out["delta_end_gyro_cross_covariance"] = np.zeros(27)
    out["start_end_gyro_correlation"] = np.zeros(9)
    return out


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
        gyro_integration: ``left_hold`` for held inputs, or ``endpoint`` for
            timestamped point samples at both interval boundaries.
        accel_integration: ``left_hold`` or ``endpoint`` (requires endpoint gyro).
        sample_rate_hz: acquisition rate, required for ``endpoint``. Converts
            density to a fixed per-acquisition sigma independently of elapsed
            integration time, so one shared sample is never rescaled by jitter.
        name: codegen basename / default C++ class stem.

    ``accel_bias`` and ``gyro_bias`` are latched from the first sample after
    reset and become the packet's bias reference.  Later values supplied
    before the next reset are intentionally ignored; one packet must have one
    linearization point.
    """

    def __init__(self, *, accel_noise_density: float = 0.0,
                 gyro_noise_density: float = 0.0,
                 gyro_integration: str = "left_hold",
                 accel_integration: str = "left_hold",
                 sample_rate_hz: float | None = None,
                 name: str = "imu_preintegrator") -> None:
        self.accel_noise_density = _finite_nonnegative(
            accel_noise_density, name="accel_noise_density")
        self.gyro_noise_density = _finite_nonnegative(
            gyro_noise_density, name="gyro_noise_density")
        if gyro_integration not in {"left_hold", "endpoint"}:
            raise ValueError("gyro_integration must be 'left_hold' or 'endpoint'")
        endpoint = gyro_integration == "endpoint"
        if endpoint:
            if (sample_rate_hz is None or not math.isfinite(sample_rate_hz)
                    or sample_rate_hz <= 0):
                raise ValueError("endpoint gyro integration requires positive sample_rate_hz")
        elif sample_rate_hz is not None:
            raise ValueError("sample_rate_hz requires endpoint gyro integration")
        if accel_integration not in {"left_hold", "endpoint"}:
            raise ValueError("unknown accelerometer integration")
        if accel_integration == "endpoint" and not endpoint:
            raise ValueError("endpoint acceleration requires endpoint gyro")
        self.accel_integration = accel_integration
        accel_endpoint = accel_integration == "endpoint"
        self.gyro_integration = gyro_integration
        self.sample_rate_hz = float(sample_rate_hz) if endpoint else None
        sigma_a = self.accel_noise_density
        sigma_g = self.gyro_noise_density

        def rec(x, u, dt, t):
            del t
            q = x["delta_orientation"]
            dv = x["delta_velocity"]
            dp = x["delta_position"]
            covariance = ca.reshape(x["covariance"], 9, 9)
            delta_start_cross = ca.reshape(
                x["delta_start_gyro_cross_covariance"], 9, 3)
            bias_jacobian = ca.reshape(x["bias_jacobian"], 9, 6)
            first = x["sample_count"] < 0.5

            gyro_bias_ref = ca.if_else(
                first, u["gyro_bias"], x["gyro_bias_reference"])
            accel_bias_ref = ca.if_else(
                first, u["accel_bias"], x["accel_bias_reference"])
            accel = u["accel"] - accel_bias_ref
            gyro = u["gyro"] - gyro_bias_ref
            end_gyro = u["end_gyro"] - gyro_bias_ref if endpoint else gyro

            end_accel = u["end_accel"] - accel_bias_ref if accel_endpoint else accel

            def compose(q0, dv0, dp0, a, w, w_end, a_end):
                from ..ir._rotation import quat_to_rotmat
                rotation = (so3_exp((w+w_end)*(.5*dt)+ca.cross(w,w_end)*dt*dt/12)
                            if accel_endpoint else
                            quat_mul(so3_exp(w*(.5*dt)), so3_exp(w_end*(.5*dt)))
                            if endpoint else so3_exp(w*dt))
                q1 = _normalise(quat_mul(q0, rotation))
                f0 = quat_to_rotmat(q0) @ a
                if accel_endpoint:
                    qm = quat_mul(q0, so3_exp((.75*w+.25*w_end)*(.5*dt)))
                    fm = quat_to_rotmat(qm) @ (.5*(a+a_end))
                    f1 = quat_to_rotmat(q1) @ a_end
                    # Simpson force integral and its first time moment.
                    dv1 = dv0 + dt*(f0+4*fm+f1)/6
                    dp1 = dp0 + dv0*dt + dt*dt*(f0+2*fm)/6
                else:
                    dv1, dp1 = dv0+f0*dt, dp0+dv0*dt+.5*f0*dt*dt
                return q1, dv1, dp1

            q_next, dv_next, dp_next = compose(q, dv, dp, accel, gyro, end_gyro, end_accel)

            # Error transition A over [dtheta, dv, dp], obtained from the
            # recurrence itself rather than maintained as hand-coded algebra.
            error = ca.MX.sym("preint_error", 9, 1)
            q_pert = quat_mul(q, so3_exp(error[0:3]))
            qp, dvp, dpp = compose(
                q_pert, dv + error[3:6], dp + error[6:9], accel, gyro, end_gyro, end_accel)
            propagated_error = _local_error(
                qp, dvp, dpp, q_next, dv_next, dp_next)
            A = ca.substitute(ca.jacobian(propagated_error, error), error,
                              ca.MX.zeros(9, 1))

            # Bias sensitivities use [gyro bias, accel bias]. A packet can be
            # corrected at the main filter's current bias without replaying
            # the high-rate samples while the first-order approximation holds.
            db = ca.MX.sym("preint_db", 6, 1)
            qb, dvb, dpb = compose(
                q, dv, dp, accel - db[3:6], gyro - db[0:3], end_gyro - db[0:3], end_accel - db[3:6])
            bias_error = _local_error(
                qb, dvb, dpb, q_next, dv_next, dp_next)
            B = ca.substitute(ca.jacobian(bias_error, db), db,
                              ca.MX.zeros(6, 1))
            J_next = A @ bias_jacobian + B

            # Held intervals use density/sqrt(dt). Point acquisitions instead
            # use the declared sample rate, even when timestamp spacing varies.
            # Autodiff maps each original sample through the same composition.
            white = ca.MX.sym("preint_white", 12 if accel_endpoint else 9 if endpoint else 6, 1)
            sqrt_dt = ca.sqrt(dt)
            noise_scale = math.sqrt(self.sample_rate_hz) if endpoint else 1 / sqrt_dt
            gyro_sigma = sigma_g * noise_scale
            accel_sigma = sigma_a * noise_scale
            qn, dvn, dpn = compose(
                q, dv, dp,
                accel + accel_sigma * white[3:6],
                gyro + gyro_sigma * white[0:3],
                end_gyro + gyro_sigma * white[6:9] if endpoint else end_gyro,
                end_accel + accel_sigma * white[9:12] if accel_endpoint else end_accel)
            noise_error = _local_error(
                qn, dvn, dpn, q_next, dv_next, dp_next)
            G = ca.substitute(ca.jacobian(noise_error, white), white,
                              ca.MX.zeros(white.shape))
            C_next = A @ covariance @ A.T + G @ G.T
            if endpoint:
                # The old packet end is this interval's left acquisition.
                # Its error is correlated with the accumulated deltas, so
                # the cross terms cannot be discarded or counted as fresh.
                delta_left_cross = ca.reshape(
                    x["delta_end_gyro_cross_covariance"], 9, 3)
                cross = A @ delta_left_cross @ G[:, 0:3].T
                C_next += cross + cross.T
            if accel_endpoint:
                cross = A @ ca.reshape(x["delta_end_accel_cross_covariance"], 9, 3) @ G[:, 3:6].T
                C_next += cross + cross.T
            C_next = 0.5 * (C_next + C_next.T)

            # Retain the correlations which a lower-rate filter cannot
            # reconstruct from C_next.  The boundary coordinates are unit
            # normal variables; the matching physical rate sigmas travel in
            # the packet. Left-held mode first reuses its start as its end;
            # endpoint mode always has distinct start/right acquisitions.
            G_gyro = G[:, 0:3]
            if endpoint:
                left_start = ca.if_else(
                    first, ca.MX.eye(3),
                    ca.reshape(x["start_end_gyro_correlation"], 3, 3).T)
                delta_start_cross_next = A @ delta_start_cross + G_gyro @ left_start
                delta_end_cross_next = G[:, 6:9]
                start_end_correlation_next = ca.MX.zeros(3, 3)
            else:
                delta_start_cross_next = (
                    A @ delta_start_cross
                    + ca.if_else(first, G_gyro, ca.MX.zeros(9, 3))
                )
                delta_end_cross_next = G_gyro
                start_end_correlation_next = ca.if_else(
                    first, ca.MX.eye(3), ca.MX.zeros(3, 3))
            gyro_sample_sigma = ca.repmat(gyro_sigma, 3, 1)
            start_gyro_sigma_next = ca.if_else(
                first, gyro_sample_sigma, x["start_gyro_noise_sigma"])

            start_gyro = ca.if_else(first, u["gyro"], x["start_gyro"])
            # Deterministic left-hold quadrature, independent of the planet.
            # V[j] = sum h*t_left**j;
            # P[j] = sum h*(T-t_left-h/2)*t_left**j.
            # The companion uses these to rotate inertial force integrals.
            elapsed = x["duration"]
            powers = ca.vertcat(1, elapsed, elapsed**2, elapsed**3)
            velocity_moments = x["velocity_time_moments"] + dt * powers
            position_moments = (x["position_time_moments"]
                                + dt * x["velocity_time_moments"]
                                + 0.5 * dt * dt * powers)
            if accel_endpoint:
                # Same quadrature nodes as the force integration, including
                # the weighted time moment for the position integral.
                mid, right = elapsed+.5*dt, elapsed+dt
                mp = ca.vertcat(1,mid,mid**2,mid**3)
                rp = ca.vertcat(1,right,right**2,right**3)
                velocity_moments = x["velocity_time_moments"] + dt*(powers+4*mp+rp)/6
                position_moments = x["position_time_moments"] + dt*x["velocity_time_moments"] + dt*dt*(powers+2*mp)/6
            astart = ca.reshape(x["delta_start_accel_cross_covariance"],9,3)
            if accel_endpoint:
                left_start = ca.if_else(first, ca.MX.eye(3), ca.reshape(x["start_end_accel_correlation"],3,3).T)
                astart = A@astart + G[:,3:6]@left_start
                aend = G[:,9:12]
                acorr = ca.MX.zeros(3,3)
            else:
                astart = A@astart + ca.if_else(first,G[:,3:6],ca.MX.zeros(9,3))
                aend = G[:,3:6]
                acorr = ca.if_else(first,ca.MX.eye(3),ca.MX.zeros(3,3))
            nxt = {
                "delta_orientation": q_next,
                "delta_velocity": dv_next,
                "delta_position": dp_next,
                "covariance": ca.reshape(C_next, 81, 1),
                "delta_start_gyro_cross_covariance": ca.reshape(
                    delta_start_cross_next, 27, 1),
                "delta_end_gyro_cross_covariance": ca.reshape(
                    delta_end_cross_next, 27, 1),
                "start_end_gyro_correlation": ca.reshape(
                    start_end_correlation_next, 9, 1),
                "start_gyro_noise_sigma": start_gyro_sigma_next,
                "end_gyro_noise_sigma": gyro_sample_sigma,
                "bias_jacobian": ca.reshape(J_next, 54, 1),
                "gyro_bias_reference": gyro_bias_ref,
                "accel_bias_reference": accel_bias_ref,
                "start_gyro": start_gyro,
                "end_accel": u["end_accel"] if accel_endpoint else u["accel"],
                "delta_start_accel_cross_covariance": ca.vec(astart),
                "delta_end_accel_cross_covariance": ca.vec(aend),
                "start_end_accel_correlation": ca.vec(acorr),
                "start_accel_noise_sigma": ca.if_else(first,ca.repmat(accel_sigma,3,1),x["start_accel_noise_sigma"]),
                "end_accel_noise_sigma": ca.repmat(accel_sigma,3,1),
                "end_gyro": u["end_gyro"] if endpoint else u["gyro"],
                "duration": x["duration"] + dt,
                "accel_endpoint": float(accel_endpoint),
                "sample_count": x["sample_count"] + 1.0,
                "velocity_time_moments": velocity_moments,
                "position_time_moments": position_moments,
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
                ("delta_start_gyro_cross_covariance", RnManifold(27)),
                ("delta_end_gyro_cross_covariance", RnManifold(27)),
                ("start_end_gyro_correlation", RnManifold(9)),
                ("start_gyro_noise_sigma", R3Manifold()),
                ("end_gyro_noise_sigma", R3Manifold()),
                ("delta_start_accel_cross_covariance", RnManifold(27)),
                ("delta_end_accel_cross_covariance", RnManifold(27)),
                ("start_end_accel_correlation", RnManifold(9)),
                ("start_accel_noise_sigma", RnManifold(3)),
                ("end_accel_noise_sigma", RnManifold(3)),
                ("bias_jacobian", RnManifold(54)),
                ("gyro_bias_reference", R3Manifold()),
                ("accel_bias_reference", R3Manifold()),
                ("start_gyro", R3Manifold()),
                ("end_accel", R3Manifold()),
                ("end_gyro", R3Manifold()),
                ("duration", ScalarManifold()),
                ("accel_endpoint", ScalarManifold()),
                ("sample_count", ScalarManifold()),
                ("velocity_time_moments", RnManifold(4)),
                ("position_time_moments", RnManifold(4)),
            ],
            inputs=[("accel", 3), ("gyro", 3),
                    ("accel_bias", 3), ("gyro_bias", 3)]
                   + ([("end_gyro", 3)] if endpoint else [])
                   + ([("end_accel", 3)] if accel_endpoint else []),
            outputs=[
                ("delta_orientation", 4),
                ("delta_velocity", 3),
                ("delta_position", 3),
                ("covariance", 81),
                ("delta_start_gyro_cross_covariance", 27),
                ("delta_end_gyro_cross_covariance", 27),
                ("start_end_gyro_correlation", 9),
                ("start_gyro_noise_sigma", 3),
                ("end_gyro_noise_sigma", 3),
                ("delta_start_accel_cross_covariance", 27),
                ("delta_end_accel_cross_covariance", 27),
                ("start_end_accel_correlation", 9),
                ("start_accel_noise_sigma", 3),
                ("end_accel_noise_sigma", 3),
                ("bias_jacobian", 54),
                ("gyro_bias_reference", 3),
                ("accel_bias_reference", 3),
                ("start_gyro", 3),
                ("end_accel", 3),
                ("end_gyro", 3),
                ("duration", 1),
                ("accel_endpoint", 1),
                ("sample_count", 1),
                ("velocity_time_moments", 4),
                ("position_time_moments", 4),
            ],
            x0={
                "delta_orientation": (1.0, 0.0, 0.0, 0.0),
                "delta_velocity": zero3,
                "delta_position": zero3,
                "covariance": np.zeros(81),
                "delta_start_gyro_cross_covariance": np.zeros(27),
                "delta_end_gyro_cross_covariance": np.zeros(27),
                "start_end_gyro_correlation": np.zeros(9),
                "start_gyro_noise_sigma": zero3,
                "end_gyro_noise_sigma": zero3,
                "delta_start_accel_cross_covariance": np.zeros(27),
                "delta_end_accel_cross_covariance": np.zeros(27),
                "start_end_accel_correlation": np.zeros(9),
                "start_accel_noise_sigma": np.zeros(3),
                "end_accel_noise_sigma": np.zeros(3),
                "bias_jacobian": np.zeros(54),
                "gyro_bias_reference": zero3,
                "accel_bias_reference": zero3,
                "start_gyro": zero3,
                "end_accel": zero3,
                "end_gyro": zero3,
                "duration": 0.0,
                "accel_endpoint": float(accel_endpoint),
                "sample_count": 0.0,
                "velocity_time_moments": np.zeros(4),
                "position_time_moments": np.zeros(4),
            },
            recurrence=rec,
        )

    def __repr__(self) -> str:
        return ("<IMUPreintegrator "
                f"accel_noise_density={self.accel_noise_density} "
                f"gyro_noise_density={self.gyro_noise_density} "
                f"gyro_integration={self.gyro_integration}>")

    def module(self):
        module = super().module()
        return replace(module, metadata={
            **module.metadata,
            "preintegration_packet_schema": PREINTEGRATION_PACKET_SCHEMA,
            "rotation_reference": "inertial",
            "integration_quadrature": self.accel_integration,
            "accel_integration": self.accel_integration,
            "gyro_integration": self.gyro_integration,
            "sample_rate_hz": self.sample_rate_hz,
        })


__all__ = ["PACKET_FIELDS", "PREINTEGRATION_PACKET_SCHEMA",
           "IMUPreintegrator", "frame_preintegrated_packet"]

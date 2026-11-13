"""Small, ordered point-sample driver for the Manta INS kernels."""

from __future__ import annotations

import math

import numpy as np

from ..codegen import TargetNumpy
from .imu_preintegrator import IMUPreintegrator
from .ins import INS
from .navigation_frame import MAX_PACKET_FRAME_ROTATION_RAD


class INSStream:
    """Run an INS from timestamped, paired accelerometer/gyro acquisitions.

    This convenience driver uses the same ``INS`` and ``IMUPreintegrator``
    kernels as the low-level API. It selects conventional covariance and
    endpoint integration explicitly; the low-level INS defaults are unchanged.
    ``sample_rate_hz`` defines the acquisition noise bandwidth. ``predict_every``
    sets the maximum number of intervals per prediction. Aiding updates flush
    a partial packet and must coincide with the latest observed IMU timestamp.

    Inputs are specific force (m/s²) and inertial angular rate (rad/s), in the
    declared IMU's axes. Samples must be independent point acquisitions, not
    delta-angle/delta-velocity increments or held/interpolated duplicates.
    The first pair establishes an endpoint and does not advance the filter.

    ModelForce updates are never automatic: explicitly call ``update`` once
    with the current accelerometer sample when model aiding is appropriate.
    The application owns control freshness, submergence, and sensor validity.
    """

    def __init__(
        self,
        world,
        *,
        imu,
        sample_rate_hz: float,
        predict_every: int = 1,
        covariance: str = "conventional",
        **ins_options,
    ):
        rate = float(sample_rate_hz)
        if not math.isfinite(rate) or rate <= 0:
            raise ValueError("sample_rate_hz must be finite and positive")
        if (
            isinstance(predict_every, bool)
            or not isinstance(predict_every, int)
            or predict_every < 1
        ):
            raise ValueError("predict_every must be a positive integer")
        self.model = INS(
            world,
            imu=imu,
            propagation="preintegrated",
            accel_integration="endpoint",
            covariance=covariance,
            **ins_options,
        )
        self.filter = TargetNumpy(self.model)
        sensor = self.model.sys.imu
        self.preintegrator = TargetNumpy(
            IMUPreintegrator(
                gyro_integration="endpoint",
                accel_integration="endpoint",
                sample_rate_hz=rate,
                gyro_noise_density=float(sensor.gyro_noise_sigma) / math.sqrt(rate),
                accel_noise_density=float(sensor.accel_noise_sigma) / math.sqrt(rate),
            )
        )
        self.predict_every = predict_every
        self._clear()

    @property
    def spec(self):
        return self.filter.spec

    @property
    def x(self):
        return self.filter.x

    @property
    def P(self):
        return self.filter.P

    @property
    def time(self):
        """Timestamp of the estimated state (None before the first sample)."""
        return self._state_time

    def state_dict(self):
        return self.filter.state_dict()

    def _clear(self):
        self._last = None
        self._packet = None
        self._count = 0
        self._state_time = None
        self._has_prediction = False
        self._boundary_sigmas = {"gyro": np.zeros(3), "accel": np.zeros(3)}

    def reset(self, state=None, *, P=None):
        """Reset state/covariance and discard all pending IMU history."""
        self.filter.reset(state, P=P)
        self.preintegrator.reset()
        self._clear()

    def _bias(self, channel):
        try:
            slot = self.spec.slot(f"{self.model.sys.imu_name}.{channel}_bias")
        except KeyError:
            return np.zeros(3)
        return self.x[slot.ambient_offset : slot.ambient_offset + 3].copy()

    def update_imu(self, accel, gyro, *, t: float, u=None):
        """Accept one observed pair; return True if a prediction was emitted.

        Controls ``u`` are passed to a prediction emitted by this call. When
        controls change within a packet, flush first at an observed boundary.
        """
        t = float(t)
        a, w = (np.asarray(v, dtype=float).reshape(-1) for v in (accel, gyro))
        if (
            not math.isfinite(t)
            or a.shape != (3,)
            or w.shape != (3,)
            or not np.isfinite(a).all()
            or not np.isfinite(w).all()
        ):
            raise ValueError(
                "IMU pair requires a finite timestamp and two finite 3-vectors"
            )
        if self._last is None:
            self._last = (t, a.copy(), w.copy())
            self._state_time = t
            return False
        start, left_a, left_w = self._last
        if t <= start:
            raise ValueError("IMU timestamps must be strictly increasing")
        frame = self.model.sys.navigation_frame
        span = t - self._state_time
        if (
            frame is not None
            and np.linalg.norm(frame.angular_velocity) * span
            > MAX_PACKET_FRAME_ROTATION_RAD
        ):
            raise ValueError(
                "packet spans too much frame rotation; increase prediction rate"
            )
        packet = self.preintegrator.step(
            t - start,
            t=start,
            accel=left_a,
            gyro=left_w,
            end_accel=a,
            end_gyro=w,
            accel_bias=self._bias("accel"),
            gyro_bias=self._bias("gyro"),
        )
        self._last = (t, a.copy(), w.copy())
        self._packet = packet
        self._count += 1
        if self._count >= self.predict_every:
            self.flush(u=u)
            return True
        return False

    def flush(self, *, u=None):
        """Predict through the latest observed endpoint, if a packet is pending."""
        if self._packet is None:
            return False
        packet = self._packet
        self.filter.predict_preintegrated(packet, t=self._state_time, u=u)
        self._state_time = self._last[0]
        self._has_prediction = True
        self._boundary_sigmas = {
            channel: np.asarray(packet[f"end_{channel}_noise_sigma"]).reshape(3).copy()
            for channel in ("gyro", "accel")
        }
        self.preintegrator.reset()
        self._packet = None
        self._count = 0
        return True

    def update(self, sensor: str, value, *, t: float, u=None, R=None):
        """Correct at the latest IMU endpoint, flushing a partial packet first.

        No interpolation, extrapolation, or delayed-measurement replay is
        performed. Off-boundary observations must be synchronized by the caller.
        """
        if self._last is None or not math.isfinite(t) or abs(t - self._last[0]) > 1e-9:
            raise ValueError(
                "aiding timestamp must match the latest observed IMU endpoint"
            )
        self.flush(u=u)
        if not self._has_prediction:
            raise ValueError("aiding requires at least one integrated IMU interval")
        inputs = dict(u or {})
        inputs[self.model.sys.accel_input] = self._last[1]
        inputs[self.model.sys.gyro_input] = self._last[2]
        for channel, sigma in self._boundary_sigmas.items():
            inputs[
                self.model.preintegration_input_map[f"end_{channel}_noise_sigma"]
            ] = sigma
        return self.filter.update(sensor, value, t=t, u=inputs, R=R)

    def compile(
        self,
        *,
        sensors=(),
        covariance="model",
        optimization="balanced",
        max_instructions=20000,
        **options,
    ):
        """Compile hot kernels; e.g. ``optimization='balanced'`` selects O1.

        Pass the aiding outputs to compile and choose ``covariance='per_sample'``
        when supplying R at runtime. Compilation failures propagate to the caller.
        """
        options = dict(
            options, optimization=optimization, max_instructions=max_instructions
        )
        self.preintegrator.compile_functions(("step",), **options)
        self.filter.compile_functions(("predict",), **options)
        if sensors:
            self.filter.compile_sensor_updates(
                sensors, covariance=covariance, **options
            )
        return self

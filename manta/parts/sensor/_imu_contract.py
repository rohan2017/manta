"""Structural strapdown contract shared by INS and force observations."""

from ...ir.frames import PartFrame
from ..base import Part


def require_strapdown_imu(part):
    """Accept user-authored Parts without requiring the built-in IMU model.

    The selected part promises calibrated sensor-frame specific force and
    inertial rate with additive bias and isotropic white acquisition noise.
    Calibration, scale, and coloured acquisition effects must be removed
    before strapdown or represented by a different mechanization.
    """
    if (
        not isinstance(part, Part)
        or not {"accel", "gyro"} <= part.output_declarations().keys()
    ):
        raise TypeError("strapdown IMU must be a Part declaring accel and gyro outputs")
    noises = part.noise_declarations()
    for name in ("accel_noise", "gyro_noise"):
        noise = noises.get(name)
        if (
            noise is None
            or noise.kind != "white"
            or noise.signal_manifold.ambient_dim != 3
            or getattr(noise.signal_manifold, "frame", None) is not PartFrame
        ):
            raise TypeError(
                f"strapdown IMU requires {name}: WhiteNoise('R3', frame=PartFrame)"
            )

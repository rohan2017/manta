"""Dynamics-predicted specific force as an ordinary pseudo-sensor.

``ModelForce`` is the measurement side of Manta's strapdown INS. It asks the
compiled world model for the specific force at its mount and adds the
model's *identified* error: the held-out residual bias, a white per-axis
floor, and a first-order Gauss–Markov (time-correlated) per-axis component.
The part has no estimator knowledge: EKF, UKF, INS, NoiseFit, observability,
and NEES all see the same ordinary ``Output``/``Noise`` contract.

Mount this part at the IMU proof-mass frame. The normal kinematic pass then
includes the complete rigid lever-arm acceleration, while actuator, fluid,
buoyancy, and estimable disturbance states reach the output through the
compiled wrench model.

Error model and where each evidence term enters
-----------------------------------------------

The doctrine requires the held-out residual bias and the time-correlated
process covariance to be consumed explicitly. ``ModelForce(evidence=...)``
builds its error model from a `FitEvidence` (`manta.fit`) computed on the
colocated accelerometer's held-out residual ``z − h``:

* **Held-out residual bias → deterministic correction.** The measured
  per-axis mean residual is added to the predicted sample (``h' = h + b``).
  It is *not* a filter state: the innovation ``r_f = f_IMU − f_model``
  already observes the accelerometer bias with ``∂r_f/∂δb_a = −I``, and a
  second constant offset in the same residual is indistinguishable from it
  (the pair is unobservable). The measured value is therefore applied
  deterministically, and the remaining uncertainty of that measurement
  (``residual_bias_stderr``) is a constant the IMU ``accel_bias`` state
  absorbs — the statistically honest placement.
* **White per-axis floor → measurement noise.** ``model_error_<axis>``
  (`WhiteNoise`, R1) carries ``white_sigma``; it is the ``R`` of the
  pseudo-measurement.
* **Time-correlated component → Gauss–Markov filter state.**
  ``model_error_correlated_<axis>`` (`GaussMarkovNoise`, R1) carries the
  fitted ``τ``/``σ``; the framework synthesizes its state slot with the
  exact ``exp(−dt/τ)`` transition and matching process noise, so the
  filter's covariance sees the correlation instead of treating the error
  as white. An axis whose evidence chose the white model (recorded
  fallback) leaves the channel inert (σ = 0, no slot).
* A ``random_walk`` evidence model is **refused**: a random-walk model
  error is indistinguishable from the accelerometer bias random walk.

Without ``evidence`` the part is an exploratory pseudo-sensor whose
per-axis white σ you set by hand (``model_error_x_sigma=...`` or the
isotropic shorthand ``model_error_sigma=...``); its bias is zero and its
correlated channels are inert. That form is fine for EKF/UKF experiments —
a model-aided ``INS`` refuses it, naming the missing evidence.
"""

from __future__ import annotations

import casadi as ca

from ...fields import gravity_at
from ...fit._evidence import FitEvidence
from ...ir.frames import PartFrame, WorldFrame
from ...ir.types import Vec3
from ...ir.wrench import Wrench
from .._declarations import (
    GaussMarkovNoise,
    Output,
    Parameter,
    PartUpdate,
    WhiteNoise,
)
from ..base import Part, PartRole
from .imu import IMU

_AXES = ("x", "y", "z")


class ModelForce(Part):
    """Model-predicted sensor-frame specific force.

    Args:
        imu      — the colocated `IMU` whose raw accelerometer sample is the
                   observation for ``specific_force`` (its cadence is the
                   default ``rate``).
        evidence — `FitEvidence` for that IMU's ``accel`` channel. Builds
                   the complete error model (see module docstring) and
                   refuses any hand-set error override alongside it.
        model_error_sigma — isotropic shorthand for the three white
                   ``model_error_<axis>_sigma`` values (evidence-less use).

    Per-axis channels (1-σ, m/s²): ``model_error_{x,y,z}`` (white) and
    ``model_error_correlated_{x,y,z}`` (Gauss–Markov, with
    ``model_error_correlated_<axis>_tau`` seconds). ``residual_bias`` is the
    deterministic held-out bias correction in the sensor frame.
    """

    role = PartRole.SENSOR

    accelerometer = Parameter(None, numeric=False)
    rate: float | None = Parameter(None)
    evidence = Parameter(None, numeric=False)
    residual_bias = Parameter((0.0, 0.0, 0.0))
    model_error_x = WhiteNoise("R1", sigma=0.0)
    model_error_y = WhiteNoise("R1", sigma=0.0)
    model_error_z = WhiteNoise("R1", sigma=0.0)
    model_error_correlated_x = GaussMarkovNoise("R1", sigma=0.0, tau=1.0)
    model_error_correlated_y = GaussMarkovNoise("R1", sigma=0.0, tau=1.0)
    model_error_correlated_z = GaussMarkovNoise("R1", sigma=0.0, tau=1.0)
    specific_force = Output()

    def __init__(self, name: str, *, imu: IMU,
                 evidence: FitEvidence | None = None, **overrides) -> None:
        if not isinstance(imu, IMU):
            raise TypeError("ModelForce: imu must be an IMU part")
        if "model_error_sigma" in overrides:
            iso = overrides.pop("model_error_sigma")
            for axis in _AXES:
                key = f"model_error_{axis}_sigma"
                if key in overrides:
                    raise TypeError(
                        f"ModelForce({name!r}): model_error_sigma and {key} "
                        "both given")
                overrides[key] = iso
        if evidence is not None:
            overrides.update(self._overrides_from_evidence(
                name, imu, evidence, overrides))
        # The pseudo-reading is sourced from this accelerometer, so its
        # default cadence must agree.  A caller may still override ``rate``
        # explicitly for a downsampled observer channel.
        overrides.setdefault("rate", imu.rate)
        super().__init__(name, accelerometer=imu, evidence=evidence,
                         **overrides)

    @staticmethod
    def _overrides_from_evidence(name: str, imu: IMU, evidence,
                                 overrides: dict) -> dict:
        who = f"ModelForce({name!r})"
        if not isinstance(evidence, FitEvidence):
            raise TypeError(
                f"{who}: evidence must be a manta.fit.FitEvidence, got "
                f"{type(evidence).__name__}")
        channel = evidence.channel
        expected = f"{imu.name}.accel"
        if not (channel == expected or channel.endswith(f".{expected}")):
            raise ValueError(
                f"{who}: evidence channel {channel!r} is not the colocated "
                f"IMU's accelerometer ({expected!r}); the model error is the "
                "held-out residual of that sensor")
        if tuple(a.axis for a in evidence.axes) != _AXES:
            raise ValueError(
                f"{who}: evidence must carry axes {_AXES}, got "
                f"{[a.axis for a in evidence.axes]}")
        owned = {"residual_bias", *(
            f"model_error_{axis}_sigma" for axis in _AXES), *(
            f"model_error_correlated_{axis}_{attr}"
            for axis in _AXES for attr in ("sigma", "tau"))}
        clash = sorted(owned & set(overrides))
        if clash:
            raise TypeError(
                f"{who}: {clash} cannot be set alongside evidence — the "
                "error model is built from the evidence alone")
        out: dict = {
            "residual_bias": tuple(float(a.residual_bias)
                                   for a in evidence.axes),
        }
        for axis, ax in zip(_AXES, evidence.axes):
            out[f"model_error_{axis}_sigma"] = float(ax.white_sigma)
            model = ax.noise_model
            if model.kind == "gauss_markov":
                out[f"model_error_correlated_{axis}_sigma"] = float(model.sigma)
                out[f"model_error_correlated_{axis}_tau"] = float(model.tau)
            elif model.kind == "white":
                out[f"model_error_correlated_{axis}_sigma"] = 0.0
            else:
                raise ValueError(
                    f"{who}: evidence axis {axis!r} fitted a "
                    f"{model.kind!r} model error; a random-walk model error "
                    "is indistinguishable from the accelerometer bias random "
                    "walk and is refused (fit a Gauss–Markov or white model)")
        return out

    @property
    def white_sigmas(self) -> tuple[float, float, float]:
        """Per-axis white model-error σ (the pseudo-measurement's R)."""
        return tuple(float(getattr(self, f"model_error_{axis}_sigma"))
                     for axis in _AXES)

    @property
    def correlated_sigmas(self) -> tuple[float, float, float]:
        """Per-axis Gauss–Markov stationary σ (0 ⇒ inert channel)."""
        return tuple(float(getattr(self, f"model_error_correlated_{axis}_sigma"))
                     for axis in _AXES)

    @property
    def correlated_taus(self) -> tuple[float, float, float]:
        """Per-axis Gauss–Markov correlation time, seconds."""
        return tuple(float(getattr(self, f"model_error_correlated_{axis}_tau"))
                     for axis in _AXES)

    def _axis_vector(self, prefix: str) -> Vec3:
        return Vec3[PartFrame].from_mx(ca.vertcat(*(
            getattr(self, f"{prefix}_{axis}")._mx for axis in _AXES)))

    def update(self, ctx) -> PartUpdate:
        zero = Vec3[PartFrame].constant((0.0, 0.0, 0.0))
        g_world = gravity_at(ctx, ctx.position[WorldFrame])
        predicted = ctx.orientation.conjugate().apply(
            ctx.acceleration[WorldFrame] - g_world)
        # The observation supplied to this pseudo-sensor is the selected
        # accelerometer's raw sample. Include that IMU's estimable bias in h
        # so innovation r = z - h has dr/db_a = -I. White IMU noise belongs
        # to INS propagation Q; the per-axis white channels alone own this
        # R, the Gauss–Markov channels are filter state, and the held-out
        # bias is a deterministic correction (see module docstring).
        expected_sample = (predicted + self.accelerometer.accel_bias
                           + Vec3[PartFrame].coerce(self.residual_bias)
                           + self._axis_vector("model_error_correlated"))
        return PartUpdate(
            wrench=Wrench(force=zero, torque=zero),
            outputs={"specific_force":
                     expected_sample + self._axis_vector("model_error")},
            rates={"specific_force": self.rate},
        )


__all__ = ["ModelForce"]

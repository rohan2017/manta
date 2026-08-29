"""manta.fit — system identification over a World.

Two complementary fitters, one data format (`Window`):

* `Fit` — MAP fit of PHYSICAL parameters (thruster gains, mounts,
  masses; any promotable `Parameter`) by windowed prediction error
  with Gaussian priors, structural ties (`Tied`/`Free` — identical
  actuators, mirrored mounts, shared geometry), and hard box bounds
  (`Prior(lower=, upper=)`). See `_map.py`.
* `NoiseFit` — maximum-likelihood fit of NOISE σ values (IMU white
  noise, bias random walks, actuator jitter) by EKF-innovation NLL —
  the statistics an L2 loss can't see. See `_nll.py`.

Typical workflow::

    training, held_out = hold_out(windows, fraction=0.3)      # untouched set
    result = Fit(world, parameters={...}).solve(training)      # dynamics
    physics = result.derive(
        evidence=result.evidence(held_out, sensor="imu.accel"))
    nresult = NoiseFit(physics, noise={...}).solve(training)   # then σ
    evidence = nresult.evidence(held_out, sensor="imu.accel")
    model = nresult.derive(evidence=evidence)
    # model is an immutable, replayable ModelArtifact carrying the typed
    # held-out evidence (bias, τ/σ², acceptance). `ModelForce(evidence=...)`
    # consumes it; a model-aided INS refuses a model without it. For an
    # exploratory authoring loop, result.apply() still writes back to the
    # editable World.
"""

from ._common import (
    DEFAULT_FILL_POLICY_ID,
    FitDefaultFill,
    Free,
    Prior,
    Tied,
    Window,
)
from ._evidence import (
    NOISE_KINDS,
    AcceptanceCheck,
    AxisFitEvidence,
    FitAcceptanceCriteria,
    FitEvidence,
    FitEvidenceBinding,
    HeldOutWindow,
    ProcessNoiseModel,
    held_out_evidence,
    hold_out,
    window_digest,
)
from ._map import Fit, FitResult
from ._nll import NoiseFit, NoiseFitResult
from ._report import FitDerivationReport
from ._residuals import ResidualStatistics, bartlett_hac_residual_statistics

__all__ = [
    "DEFAULT_FILL_POLICY_ID",
    "NOISE_KINDS",
    "AcceptanceCheck",
    "AxisFitEvidence",
    "Fit",
    "FitAcceptanceCriteria",
    "FitDefaultFill",
    "FitDerivationReport",
    "FitEvidence",
    "FitEvidenceBinding",
    "FitResult",
    "Free",
    "HeldOutWindow",
    "NoiseFit",
    "NoiseFitResult",
    "Prior",
    "ProcessNoiseModel",
    "ResidualStatistics",
    "Tied",
    "Window",
    "bartlett_hac_residual_statistics",
    "held_out_evidence",
    "hold_out",
    "window_digest",
]

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

    result = Fit(world, parameters={...}).solve(windows)   # dynamics
    physics = result.derive(validation={"accepted": True, ...})
    nresult = NoiseFit(physics, noise={...}).solve(windows)  # then σ
    model = nresult.derive(validation={"accepted": True, ...})
    # model is an immutable, replayable ModelArtifact. For an exploratory
    # authoring loop, result.apply() still writes back to the editable World.
"""

from ._common import Free, Prior, Tied, Window
from ._map import Fit, FitResult
from ._nll import NoiseFit, NoiseFitResult
from ._report import FitDerivationReport

__all__ = ["Fit", "FitResult", "FitDerivationReport", "Free", "NoiseFit", "NoiseFitResult",
           "Prior", "Tied", "Window"]

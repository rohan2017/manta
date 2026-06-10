"""manta.fit — system identification over a World.

Two complementary fitters, one data format (`Window`):

* `Fit` — MAP fit of PHYSICAL parameters (thruster gains, mounts,
  masses; any promotable `Parameter`) by windowed prediction error
  with Gaussian priors. See `_map.py`.
* `NoiseFit` — maximum-likelihood fit of NOISE σ values (IMU white
  noise, bias random walks, actuator jitter) by EKF-innovation NLL —
  the statistics an L2 loss can't see. See `_nll.py`.

Typical workflow::

    result = Fit(world, parameters={...}).solve(windows)   # dynamics
    result.apply()
    nresult = NoiseFit(world, noise={...}).solve(windows)  # then σ
    nresult.apply()
    # world now carries the identified physics AND noise model:
    # Sim(world), EKF(world) (auto-Q/R from the fitted σ), TargetCpp...
"""

from ._common import Prior, Window
from ._map import Fit, FitResult
from ._nll import NoiseFit, NoiseFitResult

__all__ = ["Fit", "FitResult", "NoiseFit", "NoiseFitResult", "Prior",
           "Window"]

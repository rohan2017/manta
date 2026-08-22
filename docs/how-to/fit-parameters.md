# Fit parameters from a log

!!! note "Draft"
    This guide is scaffolded. The outline below marks what it should cover.

[`Fit`][manta.Fit] minimizes windowed prediction error against logged
controls + measurements to recover physical parameters.

## To cover

- Marking a `Parameter` promotable with a `manifold=` (thruster gains,
  `Mass.mass`, mount transforms).
- Assembling [`Window`][manta.Window]s from a log (`x0`, `u`, `z`, `dt`).
- Adding a [`Prior`][manta.Prior] for MAP regularization — mean +
  uncertainty in, posterior σ out (`post/prior ≈ 1` ⇒ the data never
  informed that number).
- Enforcing structure so the fit stays the declared vehicle:
  [`Tied`][manta.Tied] for identical/mirrored parameters (four motors,
  one gain), [`Free`][manta.Free] for shared geometry (one arm length
  sourcing all four mount positions), `Prior(lower=, upper=)` for hard
  physical bounds.
- Running `Fit(world, parameters={...})`, reading `FitResult.converged`
  and the recovered values.
- Fitting the **noise model** instead with [`NoiseFit`][manta.NoiseFit]
  (innovation-NLL σ) — and why σ can't be L2-fit.
- Pitfalls: whiten sensors before fitting; `OP_OUTPUT` must snapshot.

## Held-out evidence and the derived artifact

Split the log **before** fitting and never let the held-out tail into a
fit; the evidence is computed there alone:

```python
from manta import Fit, NoiseFit, hold_out
from manta.fit import FitAcceptanceCriteria

training, held_out = hold_out(windows, fraction=0.3)
result = Fit(world, parameters={...}).solve(training)
physics = result.derive(
    evidence=result.evidence(held_out, sensor="imu.accel"))
nresult = NoiseFit(physics, noise={...}).solve(training)
evidence = nresult.evidence(
    held_out, sensor="imu.accel",
    criteria=FitAcceptanceCriteria(max_bias_ratio=0.5,
                                   max_autocorrelation_rmse=0.15,
                                   min_samples=200))
print(evidence.summary())
model = nresult.derive(evidence=evidence)     # ModelArtifact, hashed with it
```

[`FitEvidence`][manta.FitEvidence] records, per residual axis, the held-out
mean residual (bias) with its standard error, the white per-sample floor,
and the fitted process-noise model: a Gauss–Markov `tau`/`sigma` when the
residual is time-correlated, otherwise a white model with the fallback and
its reason written down (`white_fallback_reason`). `accepted` is computed
from the declared [`FitAcceptanceCriteria`][manta.FitAcceptanceCriteria] —
it cannot be set by hand, and a window that entered the fit is refused as
held-out data. `derive()` without evidence still works for exploratory
loops but yields a visibly unaccepted revision; a model-aided
[`INS`][manta.INS] refuses a [`ModelForce`][manta.parts.ModelForce] built
without accepted evidence.

## Known limitation: every `Window` needs a trusted `x0`

The fit's decision vector contains **only the promoted parameters** —
each window's initial state is a fixed constant of the problem. The
predicted trajectory is the oracle `step` kernel folded from `x0` over
the recorded controls, so any error in `x0` is misattributed to the
parameters being fitted.

Where to get a trustworthy `x0`:

- **Synthetic recoverability runs** — capture `sim.state` from the
  truth sim; it is exact.
- **Real logs** — seed each window from the estimator's output
  (`ekf.state_dict()` at the window start). This is approximate: the
  estimate's own error leaks into the fit, so prefer short windows
  started at low-dynamics moments (hover, steady cruise) where the
  estimator is well-converged, and weight sensors accordingly.

The principled fix is **multiple shooting**: promote each window's
initial state (or a subset of its slots) into the decision vector,
priored by the estimator's covariance. The NLP plumbing (`_FitBlock`,
bounds, priors, IPOPT) already supports extra decision blocks — the
per-window `x0` blocks and their manifold handling (orientation slots
box⊕ on SO(3)) are simply **not built yet**. Until then, treat a fit
whose windows have doubtful initial states with suspicion — check
`result.summary()`'s posterior σ before trusting recovered values.

## Source material

- Reference: [System identification](../reference/fit.md)
- Code: `manta/fit/`
- Tutorial: [System identification — drone](../tutorials/sysid-drone.md)

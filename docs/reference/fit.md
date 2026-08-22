# System identification

Fit a model's physical parameters (`Fit`) or its noise model (`NoiseFit`)
to recorded data. See [Fit parameters from a log](../how-to/fit-parameters.md)
for a worked recipe.

## Parameter fit

::: manta.Fit

::: manta.FitResult

`FitResult.evidence(held_out, sensor=...)` computes the typed held-out
evidence (below) on windows the fit never saw; `FitResult.derive(evidence=...)`
returns an immutable `ModelArtifact` with the source revision, objective,
fitted values, and that evidence. `apply()` remains the mutable alternative
for iterative authoring.

## Noise fit

::: manta.NoiseFit

::: manta.NoiseFitResult

Noise fits support the same `derive()` / `apply()` split.

::: manta.FitDerivationReport

## Held-out evidence

The doctrine's artifact channel: a fitted model's held-out residual bias and
its time-correlated process covariance are typed evidence that model-aided
estimators consume explicitly — never an implicit zero. `hold_out` splits the
log, `held_out_evidence` (or `FitResult.evidence` / `NoiseFitResult.evidence`)
computes the artifact, `FitAcceptanceCriteria` declares the thresholds that
decide `FitEvidence.accepted`, and `ModelForce(evidence=...)` consumes it.

::: manta.fit.hold_out

::: manta.fit.held_out_evidence

::: manta.FitEvidence

::: manta.AxisFitEvidence

::: manta.ProcessNoiseModel

::: manta.HeldOutWindow

::: manta.FitAcceptanceCriteria

::: manta.fit.AcceptanceCheck

::: manta.fit.window_digest

## Inputs

::: manta.Window

::: manta.Prior

## Structure

Symmetry and sanity are declared, not hoped for: tie identical or
mirrored parameters to one decision variable (`Tied`), introduce shared
geometry as an auxiliary variable (`Free`), and wall off absurd values
with `Prior(lower=, upper=)` box bounds.

::: manta.Tied

::: manta.Free

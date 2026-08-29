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

`FitEvidence.binding` scopes that decision to the exact evaluated fitted model
and artifact, pre-fit source model and artifact, opaque configuration and
profile IDs, disjoint training/selection/acceptance dataset digests, and the
qualified Manta channel shape/cadence contract. An integration layer can also
provide `channel_contract_id` for its schema/frame/unit contract.
`FitResult.derive()` and `NoiseFitResult.derive()` reject evidence issued for
another result; model-aided INS rejects unbound evidence when consuming it.

Manta deliberately remains permissive when a window omits initial-state or
control fields: it fills them from the model. Every such substitution is a
typed `FitDefaultFill` in the derivation report and evidence, including its
training/selection/acceptance role, window digest, source, field name, shape,
and exact finite numeric value. These records are canonical artifact identity
and never acceptance checks; `FitEvidence.accepted` still depends only on the
declared residual criteria. `dt` and `t0` are not substitutions: every Window
already has concrete values for both and `window_digest` binds them exactly.

::: manta.FitDefaultFill

::: manta.fit.hold_out

::: manta.fit.held_out_evidence

::: manta.FitEvidence

::: manta.FitEvidenceBinding

::: manta.AxisFitEvidence

::: manta.ProcessNoiseModel

::: manta.HeldOutWindow

::: manta.FitAcceptanceCriteria

::: manta.fit.AcceptanceCheck

::: manta.fit.window_digest

## Residual covariance

`bartlett_hac_residual_statistics` is the dimension-generic mathematical
boundary used by fitting and reduction pipelines after they produce residual
sequences. It keeps independent windows separate, reports bias explicitly, and
returns both instantaneous sample covariance and a positive-semidefinite
Bartlett/Newey–West long-run covariance suitable as white-equivalent process
noise. Vehicle replay, acceptance thresholds, and release policy remain with
the caller.

::: manta.bartlett_hac_residual_statistics

::: manta.ResidualStatistics

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

# System identification

Fit a model's physical parameters (`Fit`) or its noise model (`NoiseFit`)
to recorded data. See [Fit parameters from a log](../how-to/fit-parameters.md)
for a worked recipe.

## Parameter fit

::: manta.Fit

::: manta.FitResult

## Noise fit

::: manta.NoiseFit

::: manta.NoiseFitResult

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

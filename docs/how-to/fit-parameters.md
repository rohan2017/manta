# Fit parameters from a log

!!! note "Draft"
    This guide is scaffolded. The outline below marks what it should cover.

[`Fit`][manta.Fit] minimizes windowed prediction error against logged
controls + measurements to recover physical parameters.

## To cover

- Marking a `Parameter` promotable with a `manifold=` (thruster gains,
  `Mass.mass`, mount transforms).
- Assembling [`Window`][manta.Window]s from a log (`x0`, `u`, `z`, `dt`).
- Adding a [`Prior`][manta.Prior] for MAP regularization.
- Running `Fit(world, parameters={...})`, reading `FitResult.converged`
  and the recovered values.
- Fitting the **noise model** instead with [`NoiseFit`][manta.NoiseFit]
  (innovation-NLL σ) — and why σ can't be L2-fit.
- Pitfalls: whiten sensors before fitting; `OP_OUTPUT` must snapshot.

## Source material

- Reference: [System identification](../reference/fit.md)
- Code: `manta/fit/`
- Tutorial: [System identification — drone](../tutorials/sysid-drone.md)

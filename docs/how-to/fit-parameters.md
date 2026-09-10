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
- Observing long solves through `Fit.solve(progress=...)`. Each
  [`FitProgress`][manta.FitProgress] carries the retained-best ambient
  parameter values with ties resolved. Atomically checkpoint those values;
  returning `False` requests an orderly stop at that iteration boundary.
  `compute_posterior=False` avoids building the full residual Jacobian until
  the parameter solve is worth diagnosing.
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

For an interruptible exploratory fit:

```python
def checkpoint(update):
    write_atomically(update.values, update.best_objective)
    print(update.iteration, update.best_objective)
    return not operator_requested_stop()

result = Fit(world, parameters={...}).solve(
    training,
    progress=checkpoint,
    compute_posterior=False,
)
```

The callback runs after every accepted IPOPT iteration, including iteration
zero. `values` is the best finite iterate seen so far, so an interrupted
process leaves a usable incumbent rather than only the most recent trial.
Run a later diagnostic pass with `compute_posterior=True` when posterior
contraction is required for acceptance.

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

## Initial states, asynchronous observations, and control defaults

By default, each window's initial state is fixed. For a real log, mark the
uncertain slots with `x0_sigma`; Manta then adds a window-local tangent-space
multiple-shooting variable around that explicit `x0` prior mean:

```python
window = Window(
    x0=seeded_state,
    x0_sigma={
        "mako.velocity": (0.2, 0.2, 0.2),       # m/s
        "mako.orientation": (0.03, 0.03, 0.06), # rotation vector, rad
    },
    u=controls,
    z=measurements,
    z_mask=availability,
    dt=plant_dt,
)
```

The initial-state delta appears in `result.summary()` and
`result.window_initial_state_deltas`; it never appears in the fitted model
artifact. SO(3) uses a three-component rotation-vector perturbation, not four
independent quaternion components. Use a covariance-derived sigma: an
arbitrary loose prior can let a window absorb physical model error into its
initial condition.

`z_mask` maps sensor names to boolean `(K,)` availability arrays. A false row
is only a storage placeholder: `Fit` does not score it and `NoiseFit` skips the
Kalman measurement update while still running the process transition at the
base `dt`. This permits GPS, DVL, pressure, and IMU traces to share one window
without fabricating repeated measurements. Prediction inputs needed by an
estimator transition cannot be masked.

When composing separately sourced datasets, `Fit.solve(...,
window_weights=[...])` applies one positive scalar to each complete window's
data term. Normalize real and synthetic groups independently in the caller
(for example, real weights summing to 0.7 and synthetic weights summing to
0.3); do not let the amount of cheaply generated synthetic data decide its
authority. Priors are applied once and are not multiplied by window weight.

Manta permits partial `x0` and `u` mappings for exploratory and sparse-log
workflows. Missing fields use the model's initial state or declared control
default. Every substituted field and exact finite value is retained as
`FitDefaultFill` provenance in the result's derivation report and held-out
evidence; it does not change the residual acceptance decision. Prefer explicit
data whenever it exists. `dt` and `t0` are always concrete Window values and
already participate directly in the window digest, so they are not default-fill
records.

Where to get the `x0` prior mean:

- **Synthetic recoverability runs** — capture `sim.state` from the
  truth sim; it is exact.
- **Real logs** — seed each window from the estimator's output
  (`ekf.state_dict()` at the window start), and copy the relevant estimator
  covariance into `x0_sigma`. Prefer short windows so one local initial-state
  correction cannot disguise sustained model mismatch. Check both physical
  and window-local posterior contraction before trusting recovered values.

## Source material

- Reference: [System identification](../reference/fit.md)
- Code: `manta/fit/`
- Tutorial: [System identification — drone](../tutorials/sysid-drone.md)

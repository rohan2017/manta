# INS covariance investigation — 2026-09-08

**Diagnosis: the product-manifold INS loses consistency under finite
relinearization of coupled attitude, velocity, and bias uncertainty. No
validated estimator fix has been applied.** The Earth-rate weak-bias failure
and an existing noisy-gyro failure both remain release blockers. This work
adds reproducible diagnostics and regression checks in the isolated Manta
branch; it changes no Shiver code or deployed artifact.

## What was checked

Independent central differences at a moving, tilted state with nonzero biases
agree with the generated Jacobians:

| Quantity | Maximum absolute difference |
| --- | --- |
| State transition `F` | 1.35e-10 |
| Process-noise map `L` | 1.35e-10 |
| DVL measurement `H` | 5.50e-11 |
| Product-manifold covariance reset | 8.87e-11 |

The raw covariance also has the expected `density² * dt` attitude and velocity
noise variance at 100, 200, and 500 Hz. Per-sample sigma is
`density / sqrt(dt)`; there is no missing or duplicated rate conversion here.
The harness uses an independent DVL sample and consumes each gyro sample only
as a process input. Its fixed truth is an exact stationary solution of the
noise-free predictor. Quaternion errors use Manta's left perturbation in
navigation axes; bias errors use their declared sensor axes.

The ANEES calculation now scales units before a Cholesky solve, without
regularizing the covariance, and reports separate attitude, gyro-bias, and
accelerometer-bias marginals. The large joint ANEES is not removed by the
better-scaled calculation. A matrix being symmetric and positive definite
or its velocity ANIS passing does not establish consistency.

## Deterministic reproduction

For this stationary, root-mounted, constant-bias fixture, the gyro reading is

```text
y_g = Rᵀ Ω + b_g
```

Let `D(ψ)` rotate around navigation vertical. These states give identical
IMU and DVL readings:

```text
R(ψ)   = D(ψ) R(0)
b_g(ψ) = b_g(0) + R(0)ᵀ Ω − R(ψ)ᵀ Ω
v(ψ)   = 0
```

Rotation about gravity leaves the accelerometer reading unchanged. The
family's derivative with respect to heading, in the current tangent layout,
is

```text
δθ = vertical
δb_g = Rᵀ (vertical × Ω)
δp = δv = δb_a = 0
```

At headings −10°, 0°, 5°, and 10°, each state's own linearization satisfies
`F N = N` to 2.23e-16 and `H N = 0` exactly in the audit. Propagating the
corresponding full nonlinear state leaves it stationary to 2.71e-23.
Consequently, the local Earth-rate Jacobian is not accidentally declaring
this direction observable.

The direction **depends on attitude**, however. The existing product reset
rotates the attitude block but leaves Euclidean bias coordinates additive.
For a finite 5° change it does not transport the tangent of the stationary
family to the tangent at the new state. The actual bias displacement also
has a roughly `2.19e-7 rad/s` component omitted by its first-order heading
approximation. Repeated tight updates make that curvature consequential.

A covariance-only diagnostic holds the reference at one exact stationary
state, or alternates between two such states every ten seconds. Both use
the emitted predict/update kernels and identical zero-innovation data for
60 seconds:

| Reference handling | Heading sigma |
| --- | --- |
| Fixed reference | 4.46694° |
| Alternate 0° / 5° with the existing injection reset | 2.83552° |

This is an intentionally controlled **linearization probe**, not a simulated
estimator trajectory: reference changes are imposed explicitly. It shows
where the covariance approximation acquires extra confidence. The ordinary
Monte Carlo below demonstrates that real updates encounter the failure too.
The product reset is correct for its declared retraction; replacing its
sign or deleting it is not a correction to this problem.

This stationary ambiguity is not a global symmetry of every moving,
Earth-relative trajectory. Permanently projecting it out would risk discarding
real heading and bias information during motion.

## Controlled Monte Carlo and baseline comparison

All rows use 16 seeds, master seed 82719, 100 Hz, independent DVL samples,
and the same physical noise conventions as the main qualification. The
nine-dimensional final attitude/bias ANEES interval is `[7.042, 11.195]`.

| Case | Duration | Heading prior | Heading RMSE | Heading sigma | Joint ANEES |
| --- | --- | --- | --- | --- | --- |
| Calibrated bias, Earth rotation | 60 s | 5° | 0.01607° | 0.02235° | 9.09 |
| Weak bias, Earth rotation | 60 s | 5° | 4.42446° | 2.53949° | **254.64** |
| Same weak bias/noise, smaller heading prior | 60 s | 0.1° | 0.07725° | 0.10543° | 8.75 |
| Noisy gyro, zero Earth rotation, draft | 300 s | 5° | 24.98066° | 1.64219° | **294.05** |
| Same noisy fixture, unchanged baseline | 300 s | 5° | 24.98066° | 1.64219° | **294.05** |

The weak-bias row's individual three-dimensional marginal ANEES values are
6.30 for attitude, **154.82 for gyro bias**, and 2.60 for accelerometer bias.
Shrinking only the heading prior removes this failure in the small-error
control. It is not a proposed operational workaround.

The baseline is commit `b0734a1`, the snapshot preceding the Earth-relative
implementation. It has no `NavigationFrame` INS feature. A standalone copy
of the same harness supplied zero-spin truth to that baseline, with its
frame constructor argument removed. The baseline and draft noisy results
agree to approximately 1e-9 degrees. This establishes that the noisy-gyro
problem predates the new frame mechanics; disabling Earth rotation cannot
resolve it.

Additional temporary ablations did not justify a localized production patch:
removing the reset still failed, as did evaluating the DVL Jacobian at zero
velocity. Using truth-state Jacobians improved heading statistics but did
not make the joint product-coordinate ANEES consistent.

## What a fix must cover

Temporary coupled-retraction experiments support the diagnosis, but are not
accepted implementations. An Earth-rate-specific retraction improved the
weak-bias result when scored in its own coordinates and broke the calibrated
prior case. A more general retraction coupling navigation and bias corrections
retained calibrated performance and improved the weak-bias case, but still
failed with noisy gyros, including zero Earth rotation.

Changing the finite-error coordinates also changes the meaning of the joint
covariance and of NEES. A lower score obtained with a different `boxminus`
is not, by itself, proof of a corrected physical covariance. Prior
initialization, correction/reset, physical uncertainty reporting, and
validation must all use a consistent contract. These experiments remain
outside the estimator source and its public API.

The next implementation needs to:

1. Preserve the coupled attitude/velocity/bias uncertainty through prediction
   and updates. Evaluate a consistent linearization strategy or a fully
   coupled navigation/bias error formulation; a quaternion-reset-only change
   is insufficient. General invariant-filter literature also distinguishes
   geometric navigation errors from the treatment of IMU biases; see
   [van Goor and Mahony's biased INS filter](https://arxiv.org/abs/2202.02058).
2. Specify its error coordinates and prior interpretation explicitly. Carry
   that contract through `StateSpec`, covariance reset, module identity,
   analysis tools, and any conversion to physical marginal uncertainty.
3. Pass calibrated, weak-bias, and noisy-gyro cases with and without Earth
   rotation. The calibrated case must retain legitimate north-seeking
   information. Include multiple heading priors and independent seed batches.
4. Verify motion, gyro random walks, mounts, and aiding changes before
   claiming general INS consistency. Only then extend the corrected
   covariance/update design through packet bias Jacobians and boundary
   Schmidt cross-covariances and qualify raw/packet statistical agreement.

No empirical covariance floor, changed sensor noise, stationary
pseudo-measurement, or sensor-grade threshold was introduced.

## Reproduction and tests

With this checkout on `PYTHONPATH` and a writable cache:

```bash
python -m examples.qualification.earth_ins_debug --output /tmp/earth-ins/debug.json
python -m examples.qualification.earth_ins --duration 60 --output /tmp/earth-ins/calibrated60.json
python -m examples.qualification.earth_ins --duration 60 --bias-sigma 1e-5 --output /tmp/earth-ins/weak60.json
python -m examples.qualification.earth_ins --duration 60 --bias-sigma 1e-5 --yaw-sigma-deg 0.1 --output /tmp/earth-ins/small-heading60.json
python -m examples.qualification.earth_ins --bias-sigma 1e-3 --gyro-density 1e-3 --spin 0 --output /tmp/earth-ins/noisy-zero300.json
python -m pytest tests/test_ins_covariance_audit.py tests/test_so3_state.py tests/test_ins.py tests/test_imu_preintegrator.py tests/test_ins_navigation_frame.py
```

The failed Monte Carlo commands still exit with status 2 after writing their
reports. All **116** tests in the command above pass, including seven new
covariance-audit tests. Changed Python files pass Ruff. These tests validate
the diagnosed math/noise contracts; they do not turn the failed statistical
qualification into a pass.

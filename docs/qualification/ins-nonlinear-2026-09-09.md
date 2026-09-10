# Nonlinear INS investigation — September 9, 2026

**Current status: the v2 covariance chart passes the synthetic consistency
cases listed below.** The first v1 implementation still failed weak-bias
qualification; those failures and controls are preserved later in this report.
Defaults remain linearized and Shiver is untouched. Work is durable in the
workspace worktree `.worktrees/manta-ins-gyrocompass`, branch
`fix/ins-gyrocompass`.

## Current v2 result

The final gyro-bias retraction is

```text
bg_true = bg + delta_bg + R_sensor.T (Omega - D.T Omega - theta × Omega)
```

It makes stationary gyro observations affine in joint attitude/bias error,
including away from a zero-residual stationary family. The previous chart's
bias transport preserved that family but curved gyro observations at nonzero
residuals; repeated noisy conditioning still gained false heading information.
The accelerometer chart, joint state/noise quadrature, physical-prior mapping,
full reset and estimated packet endpoint error remain necessary parts of the
estimator. Sensor noise, physical truth priors and acceptance intervals were
not changed to obtain these results.

Fresh held-out trials use seed 104729, which was not used to develop the chart.
Each run lasts 300 seconds with a known displaced/rotated IMU, a 100 Hz
acquisition rate, 10 Hz packets and independent 10 Hz DVL observations:

| Case | Trials | Physical heading RMSE / sigma | Heading ANEES (expected 1) | Joint attitude/bias ANEES (expected 9) |
| --- | ---: | ---: | ---: | ---: |
| Calibrated bias 1e-8, gyro density 1e-7 | 64 | 0.02111° / 0.01954° | 1.167 | 9.511 |
| Weak bias 1e-5, gyro density 1e-7 | 256 | 4.33239° / 4.34204° | 0.9995 | 9.146 |
| Noisy bias 0.001, gyro density 0.001 | 128 | 16.38811° / 17.89843° | 0.8385 | 8.802 |

All three pass their final 95% heading and joint ANEES intervals. Their joint
12-dimensional tests including endpoint gyro error also pass. These are three
separate sensor-grade models, not noise values adjusted against the same data.

Additional controls pass: zero Earth rotation (64 trials, heading 18.74° / 17.93°,
ANEES 9.022), prescribed rotation (64 trials, 0.00729° / 0.00628°, ANEES 9.437),
southern latitude -60° with 200 Hz IMU, and the equator with 500 Hz IMU.
The final 6/12/24-hour numerical check also passes. The wheel builds and an
extracted-wheel prediction/update smoke test passes with the v2 estimator.
The complete suite passes 1305 tests with five skips and the same five existing
EKF/UKF failures as the recovered baseline. Matched raw/packet heading RMSEs
differ by 0.022° and reported sigmas by 0.008° after 300 seconds; both joint
ANEES checks pass. Data filenames prefixed `v2-` identify the accepted chart.

A final optimized-native smoke check exposed cancellation in a one-sample
conditional covariance. Factoring the joint covariance with the start boundary
first fixes it without a variance floor. Both `-O1` and `-O3 -march=native`
parity tests pass. A 300-second mounted one-sample-packet Monte Carlo run under
the runtime compiler profile also passes (16 trials, ANEES 8.667; heading
19.00° / 17.90°), as does the repeated 24-hour numerical check. The full compiled
wheel runtime smoke test passes with `max_instructions=50000`.

This qualifies the synthetic acquisition/model contract, not a particular
hardware IMU, vehicle mission or Shiver deployment. Nonlinear raw propagation
requires a colocated IMU; displaced IMUs use framed packets, including
one-sample packets. The chart is local and refuses branch-crossing sigma
points instead of wrapping a broad heading distribution into false confidence.

## What changed

- Recovered fixed Earth-relative strapdown mechanics from the lost session.
- Added a coupled finite attitude/velocity/bias chart, joint sigma-point
  prediction, physical-prior moment mapping and full covariance reset.
- Estimated the transient gyro endpoint error in the nonlinear packet filter.
  Aiding can learn it, and the next packet reuses its conditional distribution.
- Preserved static Schmidt installation uncertainty and checkpoint semantics.
- Added native physical-prior initialization, including the packet filter's
  internal boundary state, with NumPy/C++ parity checks.
- Rejected local-chart branch crossings and the unqualified single-sample raw
  displaced-IMU combination in the new mode.
- Rewrote opposed Earth/body quaternion increments and mount conjugation to
  avoid cancelling small increments through O(1) quaternion multiplications.

The previous installation-uncertainty refactor is present. It solves reuse of
correlated mount errors, a different failure from the colocated/no-mount-error
cases reproduced here.

## Earlier v1 matched statistical evidence (superseded)

The physical truth prior and declared sensor noise agree: independent constant
IMU biases, gyro per-sample sigma equal to density × sqrt(sample rate), and
independent 0.001 m/s DVL samples. There is no invented bias random walk or
stationary pseudo-measurement. These synthetic tests do not identify actual
vehicle sensor performance.

All table rows span 300 seconds. Packet tests use the corrected acquisition
fixture: a fresh right endpoint with matching covariance metadata becomes the
next packet's left sample. The recovered fixture previously reused a correlation
with an unrelated sample; its old packet results are invalidated.

| Case | Trials | Heading chart RMSE / sigma | Joint attitude/bias ANEES | Result |
| --- | ---: | ---: | ---: | --- |
| Mounted, calibrated bias (sigma 1e-8), gyro density 1e-7 | 64 | 0.01184° / 0.01179° | 9.359 | Pass |
| Mounted, noisy gyro and bias (both 0.001) | 64 | 19.19° / 17.85° | 9.137 | Pass |
| Colocated, noisy gyro and bias (both 0.001) | 64 | 20.22° / 17.86° | 9.198 | Pass |
| Mounted, weak bias (sigma 1e-5), gyro density 1e-7 | 64 | 5.29° / 3.78° | 10.236 | **Fail** |
| Same weak-bias case, independent larger sample | 256 | 4.80° / 3.83° | 9.912 | **Fail** |
| Equator, calibrated bias, 500 Hz IMU | 16 | 0.00934° / 0.00924° | 7.958 | Pass |

For 64 trials, the 9-dimensional final ANEES 95% interval is
[7.9905, 10.0687]. With 256 trials it is narrower; the larger sample confirms
that weak-bias failure was not just an unlucky initial 64-trial batch.
The mounted noisy run also passes the 12-dimensional joint test including
boundary error (ANEES 12.076). The table's heading metric is the chart's vertical
attitude coordinate; physical heading/coverage diagnostics also exist in the
raw qualification runner. Passing the joint 9-dimensional test alone should
not conceal a failing heading marginal.

Before active boundary estimation, the mounted packet case had ANEES 33.585,
mostly accelerometer bias (28.610 for its three coordinates). Active boundary
estimation reduces that bias marginal to 2.491 in the 64-trial run. The old raw
mounted path remains unqualified (ANEES 17.265), and is explicitly excluded
from the new mode rather than hidden by changing noise.

A negative control understating gyro noise by ten times fails strongly. The
estimator is not made consistent by blanket covariance inflation.

## Earlier v1 issue and bounded controls (resolved by v2)

A noiseless stationary control retains the analytically expected roughly 4.47°
heading sigma for a 5° initial heading prior and 1e-5 rad/s gyro-bias prior.
With matching noisy inputs the weak-bias filter instead falls to about 3.8°.
That residual effect concerned the noisy finite posterior and repeated
conditioning, not an error in the assumed sensor grade.

A complete unscented posterior reset, in place of the reset Jacobian, changes
weak-bias ANEES from 10.2364 to 10.2350: it does not fix this failure. Extending
the gyro chart to preserve the full Earth-rate/bias stationary family also has
negligible effect. These are rejected controls, not production changes.

## Long-duration numerical evidence

The mounted packet estimator passes 6, 12 and 24 simulated hours with a 1 kHz
IMU and 10 Hz packets/aiding. Inputs are a stationary noiseless oracle with
nonzero declared covariance, so this is a numerical durability check, not a
Monte Carlo acceptance result.

Before the stable quaternion arithmetic, the unaided 24-hour oracle accumulated
0.42 m position drift and failed its 1e-7 m/s velocity tolerance. After the
arithmetic correction, with the same tolerance, drift is 6.24e-6 m and velocity
error is 1.38e-10 m/s. The orientation is unchanged in the tested double-precision
representation. The aided covariance stays finite and symmetric, with positive
scaled eigenvalues; quaternion norm error is at most 2.22e-16. Generated native
and interpreted one-step results agree.

## Computation

Final joint-factor version, Intel Core Ultra 9 386H, Linux/WSL, generated
double-precision C kernels, `-O1`. Median of seven batches of 1000 evaluations through a CasADi map loop:

| Kernel | Linearized | Nonlinear |
| --- | ---: | ---: |
| Raw predict | 14.1 microseconds | 74.6 microseconds |
| Raw DVL update | 11.6 microseconds | 18.3 microseconds |
| Packet predict | 31.1 microseconds | 140.8 microseconds |
| Packet DVL update | 17.8 microseconds | 26.6 microseconds |

The nonlinear predictor uses 43 sigma points for this raw model and 61 for the
packet model. The nonlinear packet covariance has 18 state coordinates versus
15 physical coordinates plus three Schmidt cross columns in the old path.
At 10 packet predictions and 10 DVL updates per second, the nonlinear kernels
consume about 1.7 ms of CPU time per second on this host. These timings exclude
compilation, prior initialization, packet construction,
Python validation and I/O. They are not embedded-target timing qualification.

## Reproduction

Run from this checkout with its package on `PYTHONPATH` and the Manta Python
environment. Keep `XDG_CACHE_HOME` in a durable workspace directory.

```sh
python -m examples.qualification.earth_ins_split --covariance nonlinear --mounted --gyro-density .001 --bias-sigma .001 --seed 104729 --seeds 128 --output noisy.json
python -m examples.qualification.earth_ins_split --covariance nonlinear --mounted --bias-sigma 1e-5 --seed 104729 --seeds 256 --output weak.json
python -m examples.qualification.earth_ins_long --output long.json
python -m examples.qualification.earth_ins_native_packet --output native-single.json
python -m examples.qualification.earth_ins_verify docs/qualification/data/ins-nonlinear-2026-09-09
python -m examples.qualification.earth_ins_covariance_benchmark --output benchmark.json
pytest tests/test_ins_nonlinear.py tests/test_ins_navigation_frame.py tests/test_imu_preintegrator.py tests/test_filter_runtime.py
```

Detailed data, including failed trials, are in
[data/ins-nonlinear-2026-09-09](data/ins-nonlinear-2026-09-09).
The full recovered-baseline suite and the first nonlinear checkpoint have the
same five pre-existing failures (EKF/UKF consistency/tracking tests). The final
version passes 1305 tests, with five skips and exactly those same five existing
failures. No new suite failures were introduced. Its 17 nonlinear covariance
contract tests include prior moments, branch rejection, static Schmidt
uncertainty, shared-boundary estimation and both native optimization profiles.

## Durable source and merge boundary

The v2 covariance implementation is `ce015bb`; the optimized conditional-factor
refinement is `7c0cefd`. Raw data and final provenance are committed here.
The original Manta branches `main` and `feature/leviathan-contact` were both at
`4b52e95`; neither contained a separate recoverable version of the lost `/tmp`
repository. The earlier installation-uncertainty refactor is already in that
baseline.

`6c3df65` records the user's pre-existing dirty Manta workspace. Review the
feature as `git diff 6c3df65..fix/ins-gyrocompass` so that snapshot is not confused
with INS changes. Shiver integration and artifact regeneration remain a
separate merge step. The new covariance mode is opt-in; the legacy default has
not been silently changed.

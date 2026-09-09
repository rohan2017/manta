# Nonlinear INS investigation — September 9, 2026

**Status: experimental; weak-bias statistical acceptance still fails.**
The severe original covariance collapse is substantially reduced, but this
branch must not be described as a fully validated estimator fix. Defaults remain
linearized and Shiver is untouched. Work is durable in the workspace worktree
`.worktrees/manta-ins-gyrocompass`, branch `fix/ins-gyrocompass`.

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

## Matched statistical evidence

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

## Remaining issue and bounded controls

A noiseless stationary control retains the analytically expected roughly 4.47°
heading sigma for a 5° initial heading prior and 1e-5 rad/s gyro-bias prior.
With matching noisy inputs the weak-bias filter instead falls to about 3.8°.
The unresolved effect therefore concerns the noisy finite posterior and repeated
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

Intel Core Ultra 9 386H, Linux/WSL, generated double-precision C kernels, `-O1`.
Median of seven batches of 1000 evaluations through a CasADi map loop:

| Kernel | Linearized | Nonlinear |
| --- | ---: | ---: |
| Raw predict | 15.1 microseconds | 89.7 microseconds |
| Raw DVL update | 11.2 microseconds | 23.6 microseconds |
| Packet predict | 26.4 microseconds | 151.7 microseconds |
| Packet DVL update | 16.9 microseconds | 29.4 microseconds |

The nonlinear predictor uses 43 sigma points for this raw model and 61 for the
packet model. The nonlinear packet covariance has 18 state coordinates versus
15 physical coordinates plus three Schmidt cross columns in the old path.
These timings exclude compilation, prior initialization, packet construction,
Python validation and I/O. They are not embedded-target timing qualification.

## Reproduction

Run from this checkout with its package on `PYTHONPATH` and the Manta Python
environment. Keep `XDG_CACHE_HOME` in a durable workspace directory.

```sh
python -m examples.qualification.earth_ins_split --covariance nonlinear --mounted --gyro-density .001 --bias-sigma .001 --seed 19273 --seeds 64 --output noisy.json
python -m examples.qualification.earth_ins_split --covariance nonlinear --mounted --bias-sigma 1e-5 --seed 190447 --seeds 256 --output weak.json
python -m examples.qualification.earth_ins_long --output long.json
python -m examples.qualification.earth_ins_covariance_benchmark --output benchmark.json
pytest tests/test_ins_nonlinear.py tests/test_ins_navigation_frame.py tests/test_imu_preintegrator.py tests/test_filter_runtime.py
```

Detailed data, including failed trials, are in
[data/ins-nonlinear-2026-09-09](data/ins-nonlinear-2026-09-09).
The full recovered-baseline suite and the first nonlinear checkpoint have the
same five pre-existing failures (EKF/UKF consistency/tracking tests); the newest
boundary/arithmetic changes are still undergoing final regression validation.

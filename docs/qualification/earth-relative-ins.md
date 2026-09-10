# Earth-relative INS qualification — 2026-09-08

**Result: mechanics draft only; release acceptance fails.** The new equations
pass the analytical and numerical checks below. The current strapdown
error-state filter acquires unjustified confidence with weak gyro-bias priors
and noisy gyros. This branch must not be described as a qualified replacement
for a deployed estimator.

The subsequent [covariance investigation](earth-ins-covariance-debug.md)
isolates the finite-relinearization problem, verifies the local Jacobians and
noise scaling, and reproduces the noisy-gyro failure on the unchanged Manta
baseline. It adds diagnostic evidence, not a validated covariance fix.

The branch was developed against an exact snapshot of the existing, modified
Manta workspace. No Shiver source or shared Manta checkout was changed.

## Delivered

- Immutable Cartesian `NavigationFrame`, explicit gravity convention, and
  reference metadata bound into transform/module identities.
- Shared companion-side Earth rotation and Coriolis mechanics for raw and
  preintegrated INS; relative rates at displaced sensor boundaries.
- Inertial packet schema 2 with deterministic sampling-time moments; existing
  delta/boundary stochastic covariance retained.
- Ordinary IMU truth oracle in a rotating Manta Earth world, independent
  Cartesian axis projection, and synthetic calibrated/weak/noisy IMU fixtures.
- Native/interpreted parity, SX expansion, one-sample state/covariance parity,
  variable-cadence moments, motion refinement, invalid packet rejection, and
  6/12/24-hour state-kernel runs at 100/200/500 Hz.
- Reproducible Monte Carlo, long-duration, and desktop benchmark commands.

## Five-minute Monte Carlo

Each row uses 16 independent seeds from master seed 82719. The initial yaw
prior is 5 degrees; tilt is 0.1 degree. Gyro/accelerometer biases are estimated
constants through ordinary `ConstantBiasIMU` states. The independent velocity
sensor has 0.001 m/s per-sample white sigma. Accelerometer density is
`1e-5 m/s²/sqrt(Hz)` and its bias prior sigma is `1e-5 m/s²`.

Gyro white-noise density is converted to per-sample sigma by `density*sqrt(hz)`
exactly once. No gyro sample is used as an ordinary measurement. These are
synthetic grades, **not AHRS-10P calibration claims**.

| Case | Latitude / Hz | Gyro density | Bias prior sigma (rad/s) | Heading RMSE | Reported heading sigma | Attitude/bias ANEES |
| --- | --- | --- | --- | --- | --- | --- |
| Calibrated | 37.78° / 100 | 1e-7 | 1e-8 | 0.00858° | 0.01167° | 9.34 |
| Calibrated southern | −60° / 200 | 1e-7 | 1e-8 | 0.01451° | 0.01841° | 9.23 |
| Calibrated equatorial | 0° / 500 | 1e-7 | 1e-8 | 0.00897° | 0.00919° | 9.18 |
| Zero Earth rotation | 37.78° / 100 | 1e-7 | 1e-8 | 4.30° | 3.62° | 9.90 |
| Weak bias | 37.78° / 100 | 1e-7 | 1e-5 | 4.50° | **0.98°** | **1960.08** |
| Noisy gyro, weak bias | 37.78° / 100 | 1e-3 | 1e-3 | **25.01°** | **1.64°** | **294.77** |

Final-epoch nine-dimensional attitude/bias marginal ANEES has a 95% interval
of `[7.042, 11.195]` for 16 independent runs. This is one marginal consistency
gate, not whole-filter or whole-release acceptance. ANIS at the final epoch
remains within its `[1.922, 4.314]` interval even for the failed weak-bias cases:
normal-looking velocity innovations alone do not establish heading consistency.
All six runs retained symmetric, positive-definite covariance at reported
checkpoints; numerical health alone also does not establish consistency.

The nominal 300-second sigma-horizon analysis distinguishes the ambiguity:
orientation sigma decreases from 5° to approximately 0.0117° with calibrated
bias, stays around 4.47° with a `1e-5 rad/s` bias prior, and remains 5° when
Earth rotation is zero. The realized weak-bias filter shrinks uncertainty much
more than that stationary nominal analysis supports. Relinearization and
observability preservation require investigation; an additional stationary
measurement or empirical covariance floor is not justified by these results.

## Long-duration oracle

A native fold executes the actual state recurrence, retaining every sample
step; it only removes Python call overhead. This is a deterministic state
check, not a long-duration covariance or Monte Carlo claim. The main packet
contains ten physical IMU samples.

| IMU rate | Raw position drift at 24 h | Packet position drift at 24 h |
| --- | --- | --- |
| 100 Hz | 0 m in this oracle | 3.18e-6 m |
| 200 Hz | 0 m in this oracle | 4.95e-6 m |
| 500 Hz | 0 m in this oracle | 5.14e-6 m |

The 6- and 12-hour checkpoints also remain finite and stationary. Packet
velocity error stays below `1.2e-10 m/s`; quaternion error stays below
`6e-16`, with no measured quaternion-norm error. Separate tests cover tilted
attitudes, displaced/rotated mounts, both hemispheres, near-polar latitude,
and cardinal/noncardinal headings.

## Reproduction

Use this branch on `PYTHONPATH`, its dependencies, and a writable cache:

```bash
export XDG_CACHE_HOME=/tmp/manta-earth-cache
python -m pytest tests/test_ins.py tests/test_imu_preintegrator.py tests/test_ins_navigation_frame.py
python -m examples.qualification.earth_ins --output /tmp/earth-ins/nav.json
python -m examples.qualification.earth_ins --latitude -60 --rate 200 --output /tmp/earth-ins/south.json
python -m examples.qualification.earth_ins --latitude 0 --rate 500 --output /tmp/earth-ins/fast.json
python -m examples.qualification.earth_ins --spin 0 --output /tmp/earth-ins/zero.json
python -m examples.qualification.earth_ins --sigma-horizon --bias-sigma 1e-5 --output /tmp/earth-ins/weak.json
python -m examples.qualification.earth_ins --bias-sigma 1e-3 --gyro-density 1e-3 --output /tmp/earth-ins/noisy.json
python -m examples.qualification.earth_ins_long_duration --rate 500 --propagation preintegrated --output /tmp/earth-ins/long.json
python -m examples.qualification.earth_ins_benchmark --output /tmp/earth-ins/benchmark
```

The weak/noisy commands deliberately return status 2 after writing their
failed consistency reports. Do not turn that result into a passing release.
The JSON includes seeds, rate, grade, thresholds, covariance health, marginal
ANEES bounds, ANIS, rejected updates, and time-series summaries. Raw generated
reports/build products belong in artifact storage, not source commits.

## Repository validation

- All 99 targeted INS/preintegration tests pass, including generated native
  round trips and the new analytical contracts.
- The complete core run finished with 1276 passed, 5 skipped, and 5 failures.
  All five failures reproduce on the unmodified snapshot: the consistency
  verdict test, EKF/UKF Gauss-Markov drift tests, gyro random-walk bias test,
  and UKF/EKF attitude comparison. The added motion/SX checks were also run
  separately as part of the 99-test final target run.
- Outside the default test path, Mako server smoke tests give 8 passed and
  1 pre-existing failure (missing explicit gravity in the noise-slot fixture),
  also reproduced on the unmodified snapshot.
- Changed-file Ruff checks and diff whitespace checks pass. Whole-tree Ruff
  reports the same 75 findings as the unmodified snapshot.
- Both source distribution and wheel build; the wheel imports from its own
  archive and executes a stationary rotating-frame INS smoke test.

## Integration and remaining gates

Shiver integration must derive the rate vector and full Cartesian anchor/basis
at its geodetic boundary, retain frame identity and gravity convention in its
artifact, carry both new packet-moment vectors, and regenerate estimator and
preintegrator artifacts together. Published angular rates must remove the
frame rotation at the reported attitude. The water-surface datum stays separate.

Before release, resolve weak-bias/noisy-gyro overconfidence; qualify moving
trajectories, gyro random walks, sustained covariance health, GNSS dropout,
DVL/depth aiding, and displaced dynamic mounts. Extend Monte Carlo latitude,
heading, motion, and noise sweeps with confidence intervals. Actual AHRS-10P
calibration, Jetson/MCU budgets, transport ABI deployment, and stationary
hardware recording are not qualified here. No field artifact was regenerated.

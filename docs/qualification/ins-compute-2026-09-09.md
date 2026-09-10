# INS compute investigation — September 9, 2026

This is the historical prototype comparison at commit `f944935`. Its temporary
predictor override has since been replaced by the public `covariance="geometric"`
option. The [implementation report](ins-geometric-2026-09-09.md) records the
subsequent mean-transport correction and newly discovered reference failures.
The commands below reproduce the old experiment from `f944935`; current
qualification runners accept `--covariance geometric --expand` directly.

The accepted sigma-point implementation establishes a consistency reference,
not a minimum compute requirement. Its qualification tested the combined
coordinate, prior, reset, boundary-state and prediction changes. It did not
establish that sigma-point prediction was necessary for every sensor regime.

## Bounded comparison

`examples.qualification.earth_ins_prediction_ablation` retains the accepted
v2 coordinates, physical-prior initialization, active packet boundary, and
measurement/reset code. It replaces prediction with the existing analytic
first derivatives: `P_next = F P F.T + Q`, and propagates static nuisance
cross covariance with `F`. Packet conditional noise still uses the stable
joint factor introduced for optimized native execution. There are no
prediction sigma points and no noise retuning or covariance inflation.

The finite chart's zero-error differential is the same as the physical error
chart, so its local process Jacobians are already available. This is a
first-order approximation of the finite error dynamics, not a claim that
the complete biased, displaced-IMU INS has exactly linear error evolution.

Comparisons reuse seed 104729 from the accepted reference, making these
paired development comparisons, not a new untouched release acceptance set.
All runs span 300 seconds, with 100 Hz acquisition, 10 Hz packets, 10 Hz
independent DVL, a known displaced/rotated IMU, and the same physical priors.
The 16-trial exploratory batches use seed 82719 and are also retained.

The calibrated original linearized INS passes both its heading and joint
attitude/bias gates in 64 trials. Its heading RMSE is 0.012879 degrees versus
0.011795 degrees reported sigma; joint ANEES is 9.590 (expected 9).
The cheap v2 predictor also passes that 64-trial case: 0.021111 degrees versus
0.019541 degrees, joint ANEES 9.513. The different finite prior/correction
representations need not produce identical estimates.

The cheap predictor also passes all three gates in the 256-trial weak-bias
comparison: heading RMSE 4.336240 degrees, sigma 4.335478 degrees, heading
ANEES 1.00471, joint attitude/bias ANEES 9.15071, and joint including boundary
ANEES 12.31365. The respective 95% intervals are [0.83428, 1.18051],
[8.48772, 9.52707] and [11.40732, 12.60748].

The 128-trial noisy case rejects the cheap v2 predictor: joint ANEES is
187.37 (expected 9), mostly accelerometer bias (175.75 for three dimensions).
Its heading error is 14.94 degrees against 17.85 degrees sigma. Inspecting
heading alone would conceal the joint failure. The small exploratory
calibrated batch also misses the lower 95% heading bound (0.419 versus 0.432),
although the larger paired 64-trial case passes all three gates. No failed
result is removed or its interval widened.

## Sensor interpretation

The calibrated fixture assumes residual gyro bias sigma `1e-8 rad/s`
(approximately 0.0021 degrees/hour) and white gyro noise density `1e-7`
(approximately 0.00034 degrees/sqrt(hour) in the fixture's convention).
These are idealized qualification inputs, not a generic FOG specification.
The weak-bias fixture assumes about 2.06 degrees/hour residual bias sigma;
the noisy fixture assumes about 206 degrees/hour. These labels describe
models, not actual devices.

FOG alone does not specify residual calibration uncertainty, startup heading,
noise bandwidth, installation errors or operating motion. A tighter justified
bias prior and successful initial alignment may make the original linearized
path sufficient. That needs qualification using the actual installation's
assumptions. Bias instability and turn-on bias uncertainty are distinct; a
datasheet stability figure is not automatically the initial bias prior.

## Compute direction

Both existing paths already integrate nominal rotation with analytic
quaternion/SO(3) operations. The additional cost is propagating uncertainty.
Increasing sample rate reduces the time-step error but does not shrink an
uncertain heading distribution. Reusing measurements at a higher correction
rate adds no independent information.

The preferred next step is a deployment-specific linearized baseline and a
qualified analytic predictor in the corrected coordinates. If the intended
operating envelope needs higher moments, target those terms analytically or
use conditional quadrature over only the genuinely nonlinear variables.
Position/velocity and other conditionally affine variables need not always
be sampled independently. Reducing point count without preserving their
cross correlations is not a valid shortcut.

For example, a Gaussian uncertain yaw has analytic sine/cosine moments:
`E[cos(yaw)] = cos(mean_yaw) * exp(-variance_yaw / 2)` and the corresponding
sine expression. Quadratic lever-arm acceleration also has analytic moments.
These are possible building blocks, not proof that independent yaw moments
alone suffice: full 3D rotations, biases and reused endpoint errors remain
coupled, and all relevant joint moments must be transported consistently.

Geometric analytic filtering is an established approach; see
[Barrau and Bonnabel's invariant EKF analysis](https://arxiv.org/abs/1410.1465).
That paper's exact error structure and stability results have specific model
assumptions and do not certify this Manta approximation. Conditional
sigma-point reduction is described by Morelande and Moran in
[An Unscented Transformation for Conditionally Linear Models](https://dihana.cps.unizar.es/proceedings/ICASSP/2007/pdfs/0301417.pdf).
For the relationship between sensor bias, latitude and heading accuracy, see
[VectorNav's gyrocompassing explanation](https://www.vectornav.com/resources/inertial-navigation-primer/theory-of-operation/theory-gyros).

The doctrine already provides split-rate preintegration: keep high-rate IMU
integration and run the full covariance filter at an appropriate lower rate.
Actual packet construction, model aiding, runtime overhead and output latency
must be included in the hardware budget. Laptop kernel measurements alone
cannot qualify a Jetson or MCU deployment.

## Measured generated-code cost

Same laptop, native double-precision `-O1`, seven batches of 1000 calls, with
scalar expansion applied to **all** prediction and update kernels. Expanded
native outputs are checked against the original symbolic outputs, including
covariance comparison in correlation units. All checks pass.

| Propagation | Estimator | Prediction | Update | One predict + one update |
| --- | --- | ---: | ---: | ---: |
| Raw | Original linearized | 4.85 us | 5.17 us | 10.02 us |
| Raw | Analytic v2 candidate | 4.93 us | 6.41 us | 11.33 us |
| Raw | Full sigma-point v2 | 24.86 us | 6.32 us | 31.18 us |
| Packet | Original linearized | 10.15 us | 8.34 us | 18.49 us |
| Packet | Analytic v2 candidate | 14.01 us | 10.94 us | 24.94 us |
| Packet | Full sigma-point v2 | 64.59 us | 11.21 us | 75.80 us |

The candidate is approximately 1.1x the linearized raw cycle and 1.35x the
linearized packet cycle in this small model, substantially below the earlier
3.5x cost. Its statistics are qualified only to the limited comparisons above.
These are kernel timings, excluding preintegration, initialization, validation,
I/O and compilation. Prior quadrature is still performed at initialization.

The initial unexpanded analytic packet kernel took 155 us merely to predict.
Using the simpler local derivative reduced that to 134 us; scalar expansion
then removed most nested derivative-call overhead. Expanding the existing
linearized and full sigma-point kernels improves those as well, so reporting
only the candidate's gain against unoptimized code would be misleading.
The fair timings above apply the same expansion to all alternatives.

Expansion increases scalar instruction counts even while reducing measured
runtime. The full sigma-point packet predictor expands to approximately
228,000 scalar instructions; this benchmark uses an explicit 250,000 build
ceiling and 120-second compiler timeout. Its first cold O1 build took about
21 seconds on this host. This is an explicit experiment, not a change to the
default backend build policy or a promise of better instruction-cache behavior
on a different processor.

All retained comparisons and timing stages are in
[data/ins-compute-2026-09-09](data/ins-compute-2026-09-09).

## Reproduction and status

Run with the feature checkout on `PYTHONPATH` and the Manta virtual environment:

```sh
python -m examples.qualification.earth_ins_prediction_ablation --bias-sigma 1e-8 --seeds 64 --seed 104729 --output analytic-calibrated64.json
python -m examples.qualification.earth_ins_prediction_ablation --seeds 256 --seed 104729 --output analytic-weak256.json
python -m examples.qualification.earth_ins_prediction_ablation --bias-sigma .001 --gyro-density .001 --seeds 128 --seed 104729 --output analytic-noisy128.json
python -m examples.qualification.earth_ins_prediction_ablation --benchmark --output analytic-benchmark.json
python -m examples.qualification.earth_ins_prediction_ablation --benchmark --expand --output analytic-expanded-benchmark.json
python -m examples.qualification.earth_ins_covariance_benchmark --expand --output reference-expanded-benchmark.json
python -m examples.qualification.earth_ins_split --mounted --bias-sigma 1e-8 --seeds 64 --seed 104729 --output linear-calibrated64.json
```

The ablation is a research runner. No production covariance mode, default,
Shiver source, or deployment artifact has changed. It still needs motion,
long-duration and generated-backend qualification within an explicit sensor
and initialization envelope before becoming a deployment option. The
sigma-point implementation remains the accepted broader reference.

Generated-code scalar expansion is tested separately from the algorithmic
change. Computing the packet residual Jacobian in the physical chart gives
the same zero-error derivative without evaluating the finite inverse chart.
The expanded candidate reproduces the original 16-trial weak-bias batch to
roundoff (heading RMSE changes by less than 1e-10 degrees). Its four NumPy/C++
prior/predict/update/checkpoint parity cases pass, including raw and packet
propagation under `-O1` and `-O3 -march=native`. The override tests must run
with `pytest -n 0` so worker processes do not lose the scoped predictor override.
These are focused prototype checks; the full production suite was not rerun
because production code is unchanged.

# Finite-chart INS covariance

`INS(..., covariance="geometric", expand=True)` selects analytic covariance
prediction with a quadratic bias-mean transport correction. It uses no
prediction sigma points. The finite prior, coupled measurement/reset and
active packet-boundary contracts below also apply to this mode. It is an
available API, **not broadly statistically qualified**: fresh calibrated and
motion cohorts expose a confidence gap shared with the sigma-point reference.
See the [geometric implementation report](../qualification/ins-geometric-2026-09-09.md).

**Historical sigma-point qualification.** The v2 chart passes the
matched calibrated, weak-bias and noisy-gyro trials. See the
[qualification report](../qualification/ins-nonlinear-2026-09-09.md) for the
measured scope, independent trials and retained earlier failures.

The newer failures above limit that earlier evidence; passing its selected
cohorts was not a guarantee over every calibration and initialization case.

`INS(..., covariance="nonlinear")` selects a gravity-referenced finite error
model and nonlinear uncertainty propagation. `covariance="linearized"` retains
the existing implementation for compatibility. This option changes covariance
handling and correction coordinates; both use the same strapdown mechanics.

## Why the error representation changes

Stationary gravity and Earth-rate measurements couple attitude with sensor
bias. A Gaussian over independent attitude and additive bias errors cannot
retain that curved relationship through repeated corrections. In the recovered
fixture, the old estimator reported 1.64 degrees of heading uncertainty while
its error was about 25 degrees, even with Earth rotation disabled and matching
sensor noise. This is separate from the earlier Schmidt installation-uncertainty
refactor, which is retained.

The new chart separates relative twist around a fixed reference vertical from
relative swing. Navigation vectors use the same rotation; Earth-rate/gyro-bias and gravity/accelerometer-bias curvature are carried in
sensor axes. Both stationary inertial observations are affine in the joint
error coordinates, including at nonzero gyro residuals. The differential
at zero retains the existing tangent units and state layout. Finite covariance
is local to the complete chart, rather than an independent quaternion block.
See `_ins_error.py` for the forward and inverse maps. The reference vector is
opposite effective gravity evaluated at the declared initial position and time
zero; the angular reference is the navigation frame's declared rotation. They
define coordinates and do not replace the world gravity or frame model.

The full covariance reset is the derivative of

```text
boxminus(boxplus(x, correction + error), boxplus(x, correction))
```

with respect to `error` at zero. Joseph and Schmidt updates use that same map,
including navigation/bias cross blocks and the nuisance cross covariance.

## Prediction and acquisition correlations

Positive-weight augmented quadrature propagates state and process uncertainty
through the ordinary strapdown function. Packet deltas, left/right gyro
boundaries, and retained static Schmidt variables form a joint Gaussian. The
nonlinear packet filter adds three active, transient coordinates for the
standardized error of the current gyro endpoint. Aiding updates their mean and
covariance before the next packet reuses that sample. The next endpoint
replaces these coordinates using the packet's conditional noise law. They are
not vehicle angular-rate dynamics or a persistent gyro bias. Holding that
endpoint error at zero mean as a Schmidt nuisance failed the displaced-IMU
consistency test; estimating it resolves that failure. Installation uncertainty
remains a static Schmidt parameter and is not averaged away. The covariance square root
supports zero-variance directions, removes only correlation-scale floating-point
roundoff, and adds no physical variance floor.

A packet must describe the samples actually used. In particular, a fresh right
boundary must replace both the recurrence endpoint reading and its correlation
metadata, using `frame_preintegrated_packet`. That exact right sample becomes
the next packet's left sample. Reusing a correlation while supplying an unrelated
sample is an incorrect sensor model, not evidence of estimator inconsistency.
The recovered split-rate qualification fixture had this bug; its old packet
results are invalidated and must be replaced with fixture-schema-2 results.

## Initialization and checkpoints

`NumpyFilter.reset(state=..., P=...)` accepts a physical product-Gaussian prior.
Its mean and covariance are transformed together using conditional Gauss-Hermite
integration over attitude. The other conditional Euclidean coordinates are
integrated analytically. Copying only the supplied covariance into the new
chart is incorrect: gravity curvature also changes the chart mean.

The generated `initialize_prior(prior_x, prior_P)` entry provides the same
operation to native callers. Generated C++ `reset(State{}, P0)` uses it too.
For a basic bias-tracking packet INS, `PriorCov` is 15×15 and stored `Cov` is
18×18. The extra endpoint error starts independent, with zero mean and unit
covariance; callers supply only the physical prior. `State{}` describes the
physical initial state; a constructed filter holds the
mapped chart state. Initial Schmidt variables are independent of that prior.
Restoring a checkpoint restores its already-mapped state, covariance, nuisance
cross covariance, and time directly; it does not map the prior a second time.

## Native execution and integration

For a world containing `craft.imu` and independent DVL aiding:

```python
ins = INS(world, imu="craft.imu", sensors=["dvl.velocity"],
          navigation_frame=frame, propagation="preintegrated",
          covariance="nonlinear")
runtime = TargetNumpy(ins, compile=True, max_instructions=50000)
runtime.reset(state=initial_state, P=physical_prior_covariance)
```

The larger instruction limit is explicit because nonlinear kernels exceed the
backend's default 3000-instruction cold-build guard. The tested full NumPy
native build uses its default `-O3 -march=native` profile. For an `-O1` hot
subset, construct `TargetNumpy(ins)`, use its public `compile_functions`
method for selected prediction kernels, and use `compile_sensor_updates` with
semantic sensor names for correction kernels. Apply the same size limit.
Generated C++ does not use that NumPy size gate; both optimization profiles
are covered by the runtime parity tests.

Condition packet noise by factoring the joint `[start, delta, end]` covariance
with `start` first. Its trailing factor is the conditional residual root.
Explicit subtraction followed by normalization of the conditional block can
amplify FMA roundoff in one-sample packets. The joint factor supports
zero conditional directions without adding a variance floor.

When publishing a body-rate estimate from the nonlinear packet filter, include
the estimated endpoint error as well as gyro bias:

```text
relative_body_rate = R_body_from_imu @ (end_gyro - gyro_bias + end_sigma * eta)
                     - R_nav_from_body.T @ frame.angular_velocity
```

Here `eta` is the `gyro_boundary_error_state` named in artifact metadata.
The framer must continue with the same physical endpoint sample after a
checkpoint restore; the filter checkpoint contains its estimated error, while
the acquisition/replay layer owns the actual sample and packet boundaries.

## Interpretation and scope

Score joint errors with `spec.boxminus_sym(truth, estimate)`. Finite boxminus
is not antisymmetric. A Gaussian in this chart maps to a curved distribution in
ordinary physical attitude/bias coordinates; converting it to another Gaussian
and applying the same chi-square gate does not preserve that test's assumptions.
The qualification also reports physical heading and bias errors separately.

This remains a local Gaussian approximation. Relative swing has a singularity
at 180 degrees, and twist has an angular branch cut. Prior and prediction
quadrature reject points crossing those branches by producing nonfinite
outputs. NumPy rejects those atomically; native hosts must check finiteness.
It is not a global
multi-hypothesis attitude estimator. Linear observability and sigma-horizon
utilities remain local analyses, not certificates of nonlinear consistency.

A displaced IMU requires `propagation="preintegrated"` in this mode. Framed
one-sample packets are allowed. The legacy single-sample raw path uses the
vehicle model's angular acceleration in its lever correction and still fails
the noisy displaced-IMU consistency test. This restriction does not change the
legacy linearized API. A rotated IMU colocated with the craft origin can use raw
propagation.

Prediction uses augmented unscented quadrature; measurement updates still use
Jacobians and the Joseph/Schmidt recursion, followed by the coupled reset.
This is a hybrid, not the package's conventional `UKF` transform. The final
native benchmark measured approximately five times the prediction cost and
3.5–4 times the predict-plus-update cost of linearized INS. See the qualification
report for timings, model, hardware, exclusions and reproducible commands.

That qualification tests the combined implementation, not the necessity of
sigma points in every sensor regime. The separate
[compute investigation](../qualification/ins-compute-2026-09-09.md) isolates
analytic prediction from the finite coordinates and identifies a cheaper
candidate's passing calibrated/weak-bias cases and failing noisy-sensor case.

No Shiver artifact, wire adapter, or deployed estimator has been changed.

## Analytic mode and explicit scalar expansion

```python
ins = INS(world, imu="craft.imu", sensors=["dvl.velocity"],
          navigation_frame=frame, propagation="preintegrated",
          covariance="geometric", expand=True)
runtime = TargetNumpy(ins, compile=True, max_instructions=150000)
runtime.reset(state=initial_state, P=physical_prior_covariance)
```

The analytic predictor retains `F P F.T + Q` and the conditional packet noise
factor. It also transports the physical bias mean when orientation or attitude
covariance changes. A zero mean in the finite chart includes quadratic
rotation/bias offsets, so holding only the nominal bias constant during a turn
is incorrect. Closed-form second moments supply that correction. The remaining
covariance propagation is first order; this is not an exact nonlinear filter.

`expand=True` is independent of the covariance algorithm and can also be used
with `linearized` or `nonlinear`. It expands predict/update kernels, including
noise overrides and diagnostic updates, before the common backend lowers the
artifact. Initialization remains a separate quadrature operation. No compiler
optimization flags, floating-point precision or runtime noise values change.
The option is explicit because expansion can help some graphs and enlarge
others. An unsupported scalar operation fails with its kernel name. The
artifact records expansion and prediction semantics in metadata.

The complete expanded geometric example has roughly 124,000 instructions
summed across its six entry functions; its prediction alone is about 29,000.
The explicit 150,000 instruction build ceiling above covers that example,
including all updates and overrides. This is a build budget, not an estimate
of executed instructions or a universal ceiling for larger vehicle models.
Existing defaults remain `covariance="linearized", expand=False`.

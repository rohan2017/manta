# Geometric INS implementation — September 9, 2026

**Implemented and technically tested; broad statistical acceptance remains
incomplete.** The cheaper predictor is now a public Manta option. Fresh
calibrated and prescribed-motion cohorts also expose confidence failures in
the sigma-point reference. Those results constrain the earlier qualification;
they are not removed, explained away by sensor noise mismatch, or converted
to passes by relaxing the intervals.

## Public contract

```python
ins = INS(world, imu="craft.imu", sensors=["dvl.velocity"],
          navigation_frame=frame, propagation="preintegrated",
          covariance="geometric", expand=True)
runtime = TargetNumpy(ins, compile=True, max_instructions=150000)
runtime.reset(state=initial_state, P=physical_prior_covariance)
```

`geometric` retains the v2 finite attitude/velocity/bias coordinates, physical
prior initialization, full measurement covariance reset, active gyro boundary
and static Schmidt installation covariance. Prediction uses analytic first
derivatives plus a quadratic physical-bias mean correction. It uses no
prediction sigma points. It is not advertised as a textbook invariant EKF or
an exact nonlinear posterior representation. Prior quadrature still runs at
initialization, not every IMU tick.

`expand=True` independently expands the predict/update expressions into scalar
CasADi functions. It covers diagnostics and Q/R overrides as well as ordinary
entry points. It works with all three INS covariance choices and records the
choice in artifact metadata. Unsupported expansion fails explicitly with the
kernel name. Defaults remain `linearized` and `expand=False`; Shiver source
and deployment artifacts are untouched.

The 150,000 build ceiling is explicit: this small expanded geometric module
has about 124,000 instructions summed over all entry functions. That sum is a
build guard, not a count of instructions executed per prediction. The former
50,000 full-module budget refuses this expanded batch; raising the measured
budget allows its O3/native wheel smoke test to complete.

## Analytic bias-mean transport

The first prototype omitted a mean term: a zero finite-chart error mean has
a nonzero quadratic physical bias offset. When the body turns or attitude
covariance changes, the new nominal bias must preserve that physical mean.

Let `U = up up.T`, `T = I-U`, and `A` be attitude covariance. To second order:

```text
M_swing = (T A T - trace(T A T) I) / 2
M_rotation = M_swing + (U A U - trace(U A U) I) / 2 + U A T
gyro_offset_mean = -R_sensor.T M_rotation Omega
accel_offset_mean = -R_sensor.T M_swing g
```

The predictor carries the old offset through the bias transition and subtracts
the offset in the predicted chart. Only bias means change in this correction;
the chart axes and first-order covariance do not. No variance floor, noise
retuning, fictitious bias random walk or stationary pseudo-observation is added.
A separate 7-node-per-axis quadrature test verifies conservation of physical
bias means during a turn. It fails without the analytic correction.

## New evidence and limits

These acquisition fixtures use matched physical priors and sensor noise.
Seed 196613 is independent of the earlier development/acceptance seed 104729.
Each Monte Carlo run lasts 300 seconds. Packet cases use a known displaced
and rotated IMU with 100 Hz acquisition and 10 Hz packets/DVL; the prescribed
rotation control is raw and colocated. All raw data, including failures and
the earlier version without mean transport, are retained below.

- Final geometric weak-bias case, 256 trials: heading RMSE 4.63979 degrees
  versus 4.34553 degrees sigma; heading ANEES 1.15870, joint attitude/bias
  ANEES 8.66216 and joint including boundary 11.71199. All three final gates pass.
- Final geometric noisy case, 128 trials: heading RMSE 18.02257 degrees versus
  17.86061 degrees sigma; heading ANEES 1.01842, joint 9.72502, joint including
  boundary 12.50562. The three aggregate gates pass. The accelerometer-bias
  marginal remains high (4.057 for three dimensions), so this is not evidence
  that every marginal is perfectly represented.
- Final calibrated 64-trial case: heading ANEES 1.72399 fails its 95% interval
  [0.6840, 1.3751]. The result before the mean-transport addition is 1.72398.
  The sigma-point reference
  on exactly the same cohort returns 1.72393 and also fails. Both joint tests
  pass, demonstrating why the heading marginal must not be hidden.
- An independent 256-trial calibrated follow-up at seed 262147 returns heading
  ANEES 0.76873, below its 95% interval [0.8343, 1.1805], despite passing the
  joint tests. The direction differs from the 64-trial failure. We do not
  interpret either result as proof of a particular remaining cause.
- Final geometric prescribed-motion case, 64 trials: joint ANEES 11.31501
  fails [7.9905, 10.0687]; the sigma-point reference returns 11.38368 and also
  fails. Before the analytic mean correction the cheaper predictor returned
  12.47955. Gyro-bias uncertainty dominates the remaining joint discrepancy.
- Final 6/12/24-hour numerical oracle passes: finite symmetric covariance,
  positive scaled eigenvalues, maximum quaternion norm error 2.22e-16. At
  24 hours the unaided mechanization has 6.24e-6 m position drift and
  1.38e-10 m/s velocity error. This is a noiseless-input numerical check with
  declared covariance, not statistical acceptance.

The earlier selected cohorts passed; the new failures show why that did not
establish consistency across the entire calibration/initialization domain.
The remaining cause requires investigation; initialization and finite
posterior approximations are candidates, not a proved diagnosis. Actual
sensor residual bias, noise bandwidth, startup heading information and motion
requirements are still needed for deployment qualification. FOG alone does
not specify those inputs.

The acquisition runners now include heading and joint/boundary ANEES in the
top-level verdict. Historical files may say `acceptance: pass` under their
explicit older joint-only scope. `acceptance-audit.json` recomputes the full
available criteria without changing the original files. No threshold changed.

## Generated-code overhead

The final geometric packet predictor emits 4,081,747 bytes (136,465 lines) of
C without expansion and 602,779 bytes (29,141 lines) with expansion. The
unexpanded source has 820 copy and 1,087 clear helper call sites and seven
generated function symbols; the expanded source has no such helper calls and
one function symbol. These are static source metrics, not profiler samples or
counts of helper calls executed per tick.

The symbolic instruction count increases from 6,557 MX operations to 28,879
scalar operations while the generated code becomes smaller and faster. MX
nodes can hide large nested derivative graphs; comparing their counts directly
with scalar counts is misleading. The native timings in the accompanying data
hold the compiler profile and math constant when comparing graph expansion.

[CasADi documents MX-to-SX expansion](https://web.casadi.org/docs/#converting-mx-to-sx)
as a possible speed improvement with potential memory cost. Here it exposes
constant operations and smaller scalar expressions before C compilation.
This is evidence of inefficient graph lowering in these kernels, not a claim
that all CasADi code or the deployed vehicle's complete estimator is slow for
the same reason. Large model-aiding and MPC graphs need their own profiles;
blanket expansion can worsen code size and instruction-cache behavior.

Final native O1 timings, median of seven batches of 1000 evaluations, with the
same expansion applied to all alternatives:

| Propagation | Linearized predict + update | Geometric predict + update | Sigma-point predict + update |
| --- | ---: | ---: | ---: |
| Raw | 9.99 us | 10.81 us | 28.61 us |
| Packet | 17.32 us | 25.29 us | 72.28 us |

The final geometric packet cycle is about 1.46x this run's optimized linear
cycle (roughly 1.4–1.5x across the measurements). Raw is about 1.08x. These
exclude preintegration, Python validation, I/O, initialization and compilation
and do not establish timing on the target vehicle processor. The small analytic
mean correction is included in these final costs. Every timed native kernel
passes an output parity check against its original symbolic function.

## Verification and reproduction

The complete Manta suite has **1321 passed, 5 skipped and the same 5 existing
failures** as the recovered baseline. Tests cover the new physical-bias mean
contract, expansion and noise overrides, physical prior/reset/checkpoint,
static Schmidt and transient boundary covariance, and NumPy/C++ parity under
O1 and O3/native. The wheel builds; importing the extracted wheel and running
the complete O3/native geometric runtime passes with the explicit 150,000
instruction build budget. Shiver's environment supplies only the existing
wheel-builder dependency; no Shiver application source is edited.

The separate `server/mako/test_smoke.py` suite has 8 passes and one failure:
`test_noise_slot_expects_pre_scaled_draws` constructs a world without declaring
gravity. That exact failure reproduces on the recovered baseline worktree.

```sh
python -m examples.qualification.earth_ins_split --covariance geometric --expand --mounted --bias-sigma 1e-5 --seeds 256 --seed 196613 --output weak.json
python -m examples.qualification.earth_ins_split --covariance geometric --expand --mounted --bias-sigma 1e-8 --seeds 64 --seed 196613 --output calibrated.json
python -m examples.qualification.earth_ins --covariance geometric --expand --motion --seeds 64 --seed 196613 --output motion.json
python -m examples.qualification.earth_ins_long --covariance geometric --expand --output long.json
python -m examples.qualification.earth_ins_covariance_benchmark --expand --output benchmark.json
pytest tests/test_ins_geometric.py tests/test_ins_nonlinear.py
```

The former scoped monkeypatch experiment is removed; its source remains in
commit `f944935`. Qualification now exercises the public API and the same
emitted entry points used by backends.

Data and provenance: [data/ins-geometric-2026-09-09](data/ins-geometric-2026-09-09).

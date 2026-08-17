# State estimation runtime

[`EKF(world)`][manta.EKF] builds an **error-state** Extended Kalman Filter
over every craft and every state-bearing disturbance. Covariance and
updates live in the tangent space, so orientation never leaves the unit
quaternion.

[`UKF(world)`][manta.UKF] is the **unscented** twin: same constructor, same
held `x`/`P`, same auto-assembled `Q`/`R`, and the same emitted `Module`
(so it lowers to every backend through the EKF's path). It replaces the
EKF's linearized covariance push with a sigma-point sample of the
*nonlinear* `f`/`h` retracted onto the manifold — no Jacobians. With the
default tight sigma spread the two filters agree closely on a near-linear
model.

## Model-derived filters

- **Why error-state** — the rigid-body state lives on a manifold
  (`SO(3)`); the filter carries manifold-correct boxplus/boxminus, and
  the covariance is over the tangent dimension, not the ambient one.
- **Auto-assembled Q** — process-noise contributions are picked up from
  declared `Noise` channels by autodiff (`L·Σ·Lᵀ`); RW biases get
  `dt·σ²` on their slot diagonal automatically.
- **Auto-assembled R** — per-sensor measurement covariance from the noise
  channels feeding each `Output`.
- **Auto-built state spec** — walking every craft + disturbance to lay out
  the estimated slots.
- **The update/predict surface** — you own the loop. A measurement sampled
  at the interval start is folded before predicting that interval. Both
  operations are deterministic: `update` never advances logical time and
  `predict(dt)` advances it by exactly `dt`. Joseph-form updates preserve
  covariance symmetry.
- **Manifold-aware updates** — SO(3) tangent for orientation, R3 for
  vec3 states, R1 for scalars.
- **Analysis tools** — observability and `sigma_horizon` covariance
  analysis, NEES consistency (see the [estimation reference](../reference/estimation.md)).

## Measurement diagnostics and gating

Every sensor update returns an `UpdateResult` containing the innovation,
innovation covariance, normalized innovation squared (NIS), and whether the
sample was accepted. Configure a generated NIS gate when constructing either
filter:

```python
runtime = TargetNumpy(EKF(
    world,
    gates={"imu.gyro": 16.3, "gps.position": 11.34},
))
result = runtime.update("gps.position", position)
if not result.accepted:
    log_rejection(result.sensor, result.nis, result.gate)
```

Thresholds are embedded in the Module, so NumPy and deployed generated code
make the same decision. A rejected measurement still returns diagnostics but
leaves both `x` and `P` bit-for-bit unchanged. Manta reports the result; policy
for repeated rejection or sensor health belongs in Shiver.

The model-derived covariance remains the default. A driver with a trustworthy
per-sample covariance may override it:

```python
result = runtime.update("gps.position", position, R=receiver_covariance)
```

`R` must have the sensor's exact square shape, be finite, symmetric, and
positive definite. The override travels through a typed
`update_with_R_<sensor>` Module entry point and therefore exists in generated
C++ as well as NumPy; it is not NumPy-only post-processing.

## Checkpoint and restore

`runtime.checkpoint()` owns copies of nominal state `x`, tangent covariance
`P`, and logical filter time. `restore()` validates the complete snapshot
before changing any live state. Generated C++ filters expose the equivalent
`Checkpoint`, `checkpoint()`, `restore()`, and `time()` API. This is the
primitive required for rewind and replay; covariance or time must never be
restored independently of the nominal state.

## Fixed-lag replay boundary

Manta does not prescribe a replay container. A correct container needs a
declared control-hold/interpolation policy, timestamp epoch,
same-time control-versus-observation ordering, and ownership of duplicate
transport identities. Those are Shiver/Fathom runtime contracts, not Kalman
math, and choosing them here would create a second clock and transport model.

The downstream fixed-lag runtime should use Manta checkpoints as sparse replay
anchors, maintain a bounded stable-ordered event history, deduplicate by the
transport sample identity, reject measurements older than its retained
checkpoint and restore the preceding checkpoint. It may replay through the
ordinary `update`/`predict` calls as the independent NumPy oracle, or compile
its already ordered numeric span with `TargetFilterReplay`. The native target
executes explicit `ReplayPredict`, `ReplayUpdate`, and `ReplayBoundary`
operations sequentially through the same generated EKF/UKF kernels. It never
sorts, deduplicates, infers a control hold, combines measurements, or owns a
clock identity.

`TargetFilterReplay` requires startup bounds for operations, requested complete
checkpoints, and peak execution bytes. Construction computes the worst-case
program snapshot, native output, and validated materialized-result storage and
refuses a configuration above the byte cap before compiling or allocating.
The public program is treated as untrusted at the native boundary: every count,
dtype, shape, layout, opcode, sensor index, covariance, and logical-time edge is
validated and copied before C execution. Native final and requested checkpoint
state/covariance/time are validated again before they are returned.

### Representative replay evidence

The pre-change Shiver research runtime was profiled before selecting this
architecture. Reproduction on the same development host measured 27.98 ms p99
for a 32-operation Python/CasADi replay service call and 9.27 s wall time to
apply one delayed correction through 12,000 retained raw-IMU updates. That
showed the numerical call boundary—not Shiver's ordered container—as the
dominant cost and motivated a native sequential loop rather than a new replay
policy or an approximate measurement collapse.

`benchmarks/filter_replay.py` builds an 18-tangent-state Mako-like EKF and the
declared 30-second Shiver workload: 200 Hz accelerometer + gyro, 15 Hz DVL,
10 Hz depth + GPS, 1 Hz SBL, and five 50 Hz achieved-control channels. On the
2026-08-17 development host, 20,580 incoming events became 26,880 explicit
numeric/boundary operations (predicts are explicit because Manta does not own
the hold policy). The program occupied 74,054,400 bytes; the startup capacity
calculation reported a 268,126,976-byte worst-case execution peak under the
configured 512 MiB cap.

After a 9.13 s cold one-time compile and 388 ms program pack, complete `run()`
latency was 1.190 s p50 / 1.241 s p99 over five trials: 17,290 incoming
events/s, 25.20 times the declared 686 events/s arrival rate. Sequential
32-incoming-event live slices, including program validation/packing, native
execution, result validation/materialization, and checkpoint restart, measured
2.63 ms p50 / 4.45 ms p99. This clears the 20 ms live-slice deadline with
margin; rerun the benchmark on each deployment processor rather than treating
development-host timing as a hardware guarantee.

## Source material

- Reference: [Transforms](../reference/transforms.md),
  [Estimation](../reference/estimation.md)
- Code: `manta/estimation/ekf.py`, `manta/estimation/ukf.py`,
  `manta/estimation/_kalman.py`, `manta/codegen/numpy/_filter.py`,
  `manta/codegen/numpy/_filter_replay.py`
- Tutorial: [camera interceptor](../tutorials/interceptor.md)

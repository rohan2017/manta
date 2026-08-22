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

[`INS(world, imu=...)`][manta.INS] is the strapdown sibling. It emits the same
held filter Module and runtime surface, but uses the selected IMU's raw
accelerometer and gyro as prediction inputs instead of propagating navigation
state from vehicle dynamics. There is no angular-velocity estimate and no gyro
or torque-model residual. The state contains navigation, selected IMU bias,
and structurally relevant disturbance slots.

## INS force disturbance observer

Add a colocated [`ModelForce`][manta.parts.ModelForce] pseudo-part to expose
the compiled model's specific-force prediction as an ordinary noisy Output.
Its error model is not a hand-picked sigma: it is built from the fit
pipeline's typed held-out evidence for that IMU's accelerometer channel
(see [System identification](../reference/fit.md)), and a model-aided INS
refuses a part without accepted evidence:

```python
imu = IMU("imu", accel_noise_sigma=6.9e-3,
          gyro_noise_sigma=1e-3, accel_bias_sigma=1e-4,
          gyro_bias_sigma=1e-5)
craft.add(imu)
evidence = nresult.evidence(held_out_windows, sensor="imu.accel")
craft.add(ModelForce("model_force", imu=imu, evidence=evidence))

ins_ir = INS(world, imu="imu",
             sensors=["model_force.specific_force"])
ins = TargetNumpy(ins_ir)

sample = {"imu.accel": accel, "imu.gyro": gyro}
ins.update("model_force.specific_force", accel, u=sample)
ins.predict(dt, u=sample)
```

The doctrine's two evidence terms — held-out residual bias and
time-correlated process covariance — each enter where they are
statistically right, and the part's docstring is the contract:

- **Held-out residual bias → deterministic correction.** The per-axis mean
  residual is added to the predicted sample. It is not a filter state: the
  innovation already observes the accelerometer bias with
  `∂r_f/∂δb_a = -I`, and a second constant offset in the same residual
  would be unobservable against it. The measured value is applied; what
  remains of its uncertainty (`residual_bias_stderr`) is a constant the
  `accel_bias` state absorbs.
- **White per-axis floor → measurement noise.** `model_error_<axis>`
  (`WhiteNoise`, R1) carries `white_sigma` — the pseudo-measurement's `R`.
- **Time-correlated component → Gauss–Markov filter state.**
  `model_error_correlated_<axis>` ([`GaussMarkovNoise`][manta.parts.GaussMarkovNoise],
  R1) carries the fitted `tau`/`sigma`; the framework synthesizes its state
  slot with the exact `exp(-dt/tau)` transition and `(1 - exp(-2dt/tau))·σ²`
  process noise in the IR, so every backend sees the correlation instead of
  a white approximation. An axis whose evidence fell back to the white model
  (the fallback and its reason are recorded in the artifact) leaves the
  channel inert.
- A `random_walk` evidence model is refused — a random-walk model error is
  indistinguishable from the accelerometer bias random walk.

`INS` construction refuses a `ModelForce` with no evidence, or whose
`FitEvidence.accepted` is false, naming the missing artifact or the failed
acceptance checks; the consumed evidence is stored in the Module metadata
(`model_force_evidence`) so the estimator artifact records exactly which
held-out set admitted it.

This is a disturbance observer, not generic model aiding. For innovation
`r_f = f_IMU - f_model`, `∂r_f/∂δb_a = -I`, directly constraining the
accelerometer bias. When IMU and model agree otherwise, the residual is the
unmodeled external force. Declared random-walk force states such as
`CraftWindBubble.wind` therefore re-enter the estimate through the ordinary
sensor dependency graph.

The accelerometer sample drives both propagation and the pseudo-measurement,
so those noises are formally correlated. INS drops that cross term, which is
sound only while `rho = accel_noise_sigma / white model-error sigma` (the
quietest axis of the evidence's white floor; the Gauss–Markov component is
state, not `R`) is small: the neglected contribution to the innovation
covariance and gain scales like `rho²`. Construction therefore enforces a
documented ceiling rather than dropping the term silently:
`rho > MODEL_FORCE_RHO_CEILING` (0.5, a 25 % correction) raises
`ValueError`; `rho > MODEL_FORCE_RHO_WARNING` (0.1, a 1 % correction) builds
but emits a `RuntimeWarning` carrying the value. The artifact metadata stores
`rho_by_sensor`, both thresholds, and `rho_warned_sensors`; the NumPy runtime
logs the ratio (at warning level above the warning threshold). A model whose
white error is below the accelerometer's own noise is outside the regime
this pseudo-measurement is valid in, and the refusal says so. Consumers
should still validate the ratio against their own model, sensor bandwidth,
and operating envelope. Convert density units to Manta's effective
per-sample sigma at the configuration boundary.

Lever-arm compensation is part of the symbolic tick: the measured gyro drives
the centripetal term, so a sensor 10 cm off-axis at 3 rad/s does not inject the
roughly 0.9 m/s² (about 90 mg) acceleration into the craft origin. CasADi
autodiff derives `F` from that same tick. Any analytic Jacobian is only a
finite-difference test oracle.

Because `ModelForce` is a normal sensor declaration, R assembly, sensor
selection, gating, `observability`, `observability_trajectory`, NEES, and
`NoiseFit` reuse the common estimator paths. World-level trajectory tools read
the selected IMU from their truth simulation; open-loop `sigma_horizon` callers
must include accel and gyro samples in `control`.

## High-rate preintegration

When the IMU is sampled faster than the main estimator loop, use
[`IMUPreintegrator`][manta.IMUPreintegrator] as a high-rate recurrence and
construct the filter with `INS(..., propagation="preintegrated")`. Each packet
accumulates ordered rotation products on SO(3), specific-force velocity and
position deltas, a 9×9 covariance over `[δθ, δv, δp]`, and a 9×6 Jacobian with
respect to `[δb_g, δb_a]`. Bias values are latched on the first sample after
reset; the Jacobian lets the main INS correct a packet at its current bias
estimate without replaying the samples.

```python
pre = TargetNumpy(IMUPreintegrator(
    accel_noise_density=accel_density,
    gyro_noise_density=gyro_density,
))
ins = TargetNumpy(INS(world, imu="imu", sensors=[...],
                      propagation="preintegrated"))

for accel, gyro, sample_dt in high_rate_samples:
    packet = pre.step(sample_dt, accel=accel, gyro=gyro,
                      accel_bias=bias_ref_a, gyro_bias=bias_ref_g)

packet_u = ins.preintegrated_inputs(packet, u=actuator_commands)
ins.predict_preintegrated(packet, u=actuator_commands)
ins.update("model_force.specific_force", packet["end_accel"], u=packet_u,
           t=ins.time)
pre.reset()
```

The deltas live in the sensor frame at the packet start. Boundary gyro samples
let INS propagate the sensor origin and transform back to the craft origin at
both endpoints, so a nonzero lever arm is retained without differentiating the
gyro or invoking the torque model. The endpoint accelerometer remains the
ordinary `ModelForce` measurement source. Packet covariance is intrinsic and
is added in both automatic-Q and explicit-Q prediction; the latter overrides
only the separate model/bias process covariance.

The packet spans an interval, so predict it first and then fold endpoint
measurements. This differs from the raw sample loop's update-at-interval-start
convention. The packet carries no absolute timestamps — transport framing owns
`t0`/`t1` — but it does carry its own span, `duration`, and that span is part
of the kernel contract: the packet-consuming `predict` must advance by exactly
`duration` (relative tolerance `1e-9`), because the gravity and lever-arm
terms are scaled by the predict `dt` while the deltas were integrated over the
packet. `predict_preintegrated` takes `dt` from the packet. A direct caller
(the NumPy `predict`, or generated C++ `predict(u, dt)`) that passes a
different `dt` gets a NaN navigation state from the kernel rather than a
mis-scaled one; the NumPy runtime names the mismatch before the kernel runs,
and the generated `Inputs` header documents the invariant on the `duration`
field.

`IMUPreintegrator` and the packet-consuming INS are independent Modules. Lower
the recurrence to an MCU with `TargetCpp`, and lower or run the main filter on
the companion computer through any filter backend. The packet matrices use
column-major CasADi/Eigen flattening. Transport framing still owns timestamps,
sequence numbers, health/saturation flags, calibration identity, and CRC.

## Model-derived filters

- **Why error-state** — the rigid-body state lives on a manifold
  (`SO(3)`); the filter carries manifold-correct boxplus/boxminus, and
  the covariance is over the tangent dimension, not the ambient one.
- **Auto-assembled Q** — process-noise contributions are picked up from
  declared `Noise` channels by autodiff (`L·Σ·Lᵀ`); RW biases get
  `dt·σ²` on their slot diagonal automatically, and Gauss–Markov channels
  (`GaussMarkovNoise(tau=, sigma=)`) get the exact discrete
  `(1 - exp(-2dt/tau))·σ²` alongside their `exp(-dt/tau)` transition in
  `F` — the correlated error is filter state in EKF, UKF, and INS alike.
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
sample was accepted. Configure a generated NIS gate when constructing any
filter. A consistent filter's NIS is χ²-distributed with the measurement
dimension as its degrees of freedom, so derive gates from a chosen
false-rejection rate with [`chi2_gate`][manta.estimation.chi2_gate]
(scipy-free) rather than remembering quantiles:

```python
from manta.estimation import chi2_gate

runtime = TargetNumpy(EKF(
    world,
    gates={"imu.gyro": chi2_gate(3, 0.999),      # 16.27
           "gps.position": chi2_gate(3, 0.99)},  # 11.34
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
  `manta/estimation/ins.py`,
  `manta/estimation/_kalman.py`, `manta/codegen/numpy/_filter.py`,
  `manta/codegen/numpy/_filter_replay.py`
- Tutorial: [camera interceptor](../tutorials/interceptor.md)

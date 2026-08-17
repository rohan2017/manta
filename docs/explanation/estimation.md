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

Manta intentionally does not yet prescribe a replay container. A correct
container needs a declared control-hold/interpolation policy, timestamp epoch,
same-time control-versus-observation ordering, and ownership of duplicate
transport identities. Those are Shiver/Fathom runtime contracts, not Kalman
math, and choosing them here would create a second clock and transport model.

The downstream fixed-lag runtime should use Manta checkpoints as sparse replay
anchors, maintain a bounded stable-ordered event history, deduplicate by the
transport sample identity, reject measurements older than its retained
checkpoint, restore the preceding checkpoint, and replay through the same
`update`/`predict` calls. Manta can grow a generic container once a second
consumer demonstrates a common clock-free event contract.

## Source material

- Reference: [Transforms](../reference/transforms.md),
  [Estimation](../reference/estimation.md)
- Code: `manta/estimation/ekf.py`, `manta/estimation/ukf.py`,
  `manta/estimation/_kalman.py`, `manta/codegen/numpy/_filter.py`
- Tutorial: [camera interceptor](../tutorials/interceptor.md)

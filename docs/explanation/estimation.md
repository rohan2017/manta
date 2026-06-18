# State estimation (error-state EKF)

!!! note "Draft"
    This page is scaffolded. The outline below marks what it should cover.

[`EKF(world)`][manta.EKF] builds an **error-state** Extended Kalman Filter
over every craft and every state-bearing disturbance. Covariance and
updates live in the tangent space, so orientation never leaves the unit
quaternion.

## To cover

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
- **The update/predict surface** — you own the loop; the
  **update-then-predict** order and why it matters. Joseph-form updates.
- **Manifold-aware updates** — SO(3) tangent for orientation, R3 for
  vec3 states, R1 for scalars.
- **Analysis tools** — observability and `sigma_horizon` covariance
  analysis, NEES consistency (see the [estimation reference](../reference/estimation.md)).

## Source material

- Reference: [Transforms](../reference/transforms.md),
  [Estimation](../reference/estimation.md)
- Code: `manta/estimation/ekf.py`, `manta/estimation/_kalman.py`
- Tutorial: [camera interceptor](../tutorials/interceptor.md)

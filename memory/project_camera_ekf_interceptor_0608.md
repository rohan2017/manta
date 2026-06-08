---
name: project_camera_ekf_interceptor_0608
description: EKF-able camera (cross-craft bbox measurement) + TrajectoryEndpoint part + rocket-intercepts-ballistic-target example; emits_field rename
metadata:
  type: project
---

Session 2026-06-08, 5 LOCAL commits on main (085558d..81e42a7), 481 tests green.

**emits_field rename** (085558d): `FieldSource.disturbance_field` → `emits_field` (pairs with read-side `requires_fields`; see [[project_optical_field_sources_0607]]). `requires_fields` is an ASSERTION (fail compile if absent), NOT "fields I read" — reading is undeclared via `ctx.field(Cls)`; `requires_fields`/`has_field`/`get_field` already cover multi-field + optional (compile-time-static `if ctx.has_field(X)` selects traced terms, zero runtime cost).

**TrajectoryEndpoint** (cbef9f5, `manta/parts/attachment/trajectory_endpoint.py`): SE(3) spring-damper "playback" actuator — slews an actuator-free craft along a prescribed pose path so the world generates fake-but-consistent sensor data. Reference is a symbolic callable `f(t)->TrajectorySample` (Inputs are SCALAR-ONLY, so the path bakes into the tick; symbol-pure → EKF/LQR see an unperturbed plant). Position spring + SO(3) box-minus attitude spring + optional mass-based gravity/accel feedforward (exact tracking). `LinearTrajectory`/`hover` helpers. NOT actuator-limited (that's the point; use LQR-vs-ref for realism). Mount at craft origin (root). KEY frame gotcha: `ctx.orientation` is `Quat[WorldFrame, PartFrame]` even at the root (tag is PartFrame, not CraftFrame) — reinterpret the user's q_ref into PartFrame via `Quat[WorldFrame,PartFrame].from_mx(q._mx)` or boxminus FrameErrors.

**EKF-able Camera** (6138016): a camera's box of a target ellipsoid is ALREADY a differentiable function of BOTH craft poses — only missing piece was a noise channel. Opt-in `Camera(bbox_sigma=...)`: declares one white-noise channel per box edge per target (dynamic `noise_declarations()` override mirroring `output_declarations()`; sets `<name>_sigma` attrs because tick_signature.py:152 reads them). bbox_sigma==0 → byte-identical (no channels). Select the 4 edge outputs as EKF sensors (NOT `_vis`, no noise→singular) ⇒ ONE joint filter block (measurement spans both craft tangent blocks, union-find merges them; Joseph update already global, no EKF-internal change). Recovers target 3-D position: bearing from box center, range from apparent size given known semi-axes. Camera also gained `transform=`. Cross-craft EKF facts: `update("h_sym", z, R=)` low-level path exists; per-sensor R must be nonzero or EKF raises; index P/Jacobians by `StateSlot.tangent_offset`; predict has `predict_with_Q`; read estimate via `ekf.state_dict()`, covariance via `ekf.P`. EKF numpy filter has NO `.state`/`.P` setter — use `reset(state=, P=)`.

**camera_tracking example rewrite** (81e42a7): TVC rocket (rocket.py LQR machine, legs dropped, launches powered) with nose Camera+IMU+GPS intercepts a ballistic projectile (Mass+DragSurface+OpticalSource) via the joint EKF. 11/12 noise seeds intercept (<1.5 m). Hard-won control/estimation lessons (all in-file comments):
- Target is BALLISTIC → tight process Q (KNOWN dynamics); a loose Q lets the weakly-observed monocular cross-range run away → filter diverges.
- Acquire→engage gating: climb + hold camera UP until target-pos covariance converges for N frames, THEN pursue. Chasing a not-yet-converged fix saturates tilt and dumps the rocket (chicken-and-egg divergence).
- TVC throttle band: gimbal→body-torque gain ∝ thrust but LQR trims at ONE throttle; wide swings (>~2× trim) make the attitude loop too hot → tips. BUT THR_MIN must dip BELOW hover or the rocket can't brake a climb (coasts past).
- Intercept = altitude-HOLD + lateral pursuit (target falls through), NOT vertical chase (thrust-limited rocket can't arrest the overshoot).
- Monocular bearing/size: ill-conditioned at long range / overhead; well-conditioned CLOSING (range shrinks → observability improves). Acceleration-command/attitude guidance FAILS here (gimbal attitude response too slow for large commanded tilts; LQR linearized upright).
- IMU accel as an EKF sensor DIVERGED the merged block (don't); gps+gyro+camera is stable.
- Measurement min-miss must be tracked at the physics substep rate (250 Hz), not the 50 Hz control rate, or a fast crossing is missed / mis-reported.

All 5 commits LOCAL on main, not pushed.

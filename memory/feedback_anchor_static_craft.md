---
name: feedback_anchor_static_craft
description: A "static" craft (ground sensor array, fixed station) FREE-FALLS under GravityField unless anchored — anchor it or truth/filter desync
metadata:
  type: feedback
---

A craft that is meant to be STATIONARY infrastructure (a ground camera array, a fixed beacon/tracker) still has a `Mass`, so a registered `GravityField` makes it **free-fall** — there is no implicit ground. Anchor it explicitly: a `Collider` resting on a `CollisionField` `HalfSpace` (like the rocket pad), or otherwise cancel its weight.

**Why:** I spent a very long debugging session convinced the bearings-only EKF was "fundamentally unstable" tracking a ballistic target from a ground centroid-camera array — velocity diverged from a perfect init, wider baselines didn't help, etc. The real cause: the tracker craft was unanchored, so in the TRUTH sim the whole camera array fell ~1.6 km over 18 s, while in the EKF the tracker was pinned at the origin. The camera measurements (which depend on the camera's world pose) were systematically inconsistent between truth and filter → the filter "diverged." A constant ~165 px offset in the centroid that matched a camera at z=−½gt² gave it away. Anchoring the tracker (Collider on a ground HalfSpace) made the EKF track cleanly.

**How to apply:** Whenever a craft should not move, give it a physical support or it will fall under gravity. When an estimator pins a craft's pose (low P/Q) as "known/surveyed", the TRUTH craft MUST actually hold that pose, or every measurement that reads its pose desyncs. Symptom of this class of bug: a filter that's stable only at *exactly* truth and diverges from any perturbation, with the error growing at roughly the target's speed (a frozen/biased estimate while truth moves). See [[project_camera_ekf_interceptor_0608]].

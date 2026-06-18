# Estimation — camera interceptor

!!! note "Draft"
    This tutorial is scaffolded. It narrates `examples/vehicles/interceptor.py`,
    which is complete and runnable today.

A controllable interceptor tracks a maneuvering target using a camera's
bearing measurements, fused in a **joint cross-craft** EKF.

## To cover

- Two crafts in one world; a camera observing the target craft.
- How bbox/centroid measurement noise becomes a joint filter block across
  both crafts.
- The adaptive substep and the lean EKF used in the fast regime.
- Closing the intercept with proportional navigation / LQR.
- Recording to `.rrd` for Rerun visualization.

## Run it

```bash
python -m examples.vehicles.interceptor
```

## Source material

- Code: `examples/vehicles/interceptor.py`
- Concepts: [State estimation (error-state EKF)](../explanation/estimation.md)

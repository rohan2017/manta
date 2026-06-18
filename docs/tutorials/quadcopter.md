# Closed-loop control — quadcopter

!!! note "Draft"
    This tutorial is scaffolded. It narrates `examples/vehicles/quadcopter.py`,
    which is complete and runnable today.

Build a quadcopter, synthesize an [`LQR`][manta.LQR] regulator about a
hover trim, and fly it to a position setpoint with the estimate from an
EKF in the loop.

## To cover

- Declaring the quadcopter (four thrusters, IMU, position sensor).
- Choosing the operating point (`x_ref` / `u_ref`) and the controllable
  subspace (`regulate=`) — a single-axis-thrust craft regulates position
  *through attitude*.
- Picking `Q` / `R` cost weights.
- The closed loop: `ekf.state_dict()` → `lqr.control(...)` →
  `sim.step(...)`.
- Retargeting the setpoint mid-flight.

## Run it

```bash
python -m examples.vehicles.quadcopter
```

## Source material

- Code: `examples/vehicles/quadcopter.py`
- Concepts: [Control synthesis (LQR)](../explanation/control.md)

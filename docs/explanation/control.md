# Control synthesis (LQR)

!!! note "Draft"
    This page is scaffolded. The outline below marks what it should cover.

[`LQR(world, …)`][manta.LQR] synthesizes a state-feedback regulator about
an operating point — the third sibling transform. It regulates a
controllable subset and freezes the rest.

## To cover

- **The operating point** — `x_ref` / `u_ref` (trim), and why a free
  rigid body is underactuated so full-state LQR isn't stabilizable.
- **`regulate=`** — selecting the controllable subspace (e.g.
  `["drone.position", "drone.velocity"]`) and freezing the remainder at
  the operating point.
- **Q / R cost weights** — distinct from the EKF's process/measurement
  noise; how they shape the gain.
- **The runtime surface** — `lqr.control(state_dict) → {input: u}` and
  `retarget(...)` to move the setpoint.
- **Under-actuation through attitude** — how a single-thruster craft
  regulates position via attitude (the quadcopter demo).

## Source material

- Reference: [Transforms](../reference/transforms.md)
- Code: `manta/control/lqr.py`
- Tutorial: [closed-loop quadcopter](../tutorials/quadcopter.md)

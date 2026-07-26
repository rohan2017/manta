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
- **The runtime surface** — `lqr.control(state_dict) → {input: u}`.
- **Under-actuation through attitude** — how a single-thruster craft
  regulates position via attitude (the quadcopter demo).

## Moving the operating point

A gain is solved *about a point*. Two ways to move it, and the
difference is not cosmetic:

`retarget(x_ref)` moves the reference and keeps the gain. That is exact
wherever the dynamics are invariant along the move — a translation of a
hover setpoint under uniform gravity — and cheap enough to do every
tick.

It is **wrong for a heading change.** The tangent error uses
world-frame position and velocity, so `K` permanently encodes the
actuator→world-force map at the attitude it was solved at. Retarget the
reference heading by Δψ and every translational feedback comes out
rotated by Δψ: at 90° the feedback is perpendicular to the error and the
craft orbits its target; at 180° it is positive feedback and the craft
accelerates away.

[`resolve_at`][manta.LQR.resolve_at] is the general answer. It
re-evaluates `A`, `B` at the new reference and re-runs the Riccati
solve, returning an [`LQRSolution`][manta.LQRSolution] — gain,
feed-forward and reference as plain arrays:

```python
sol = lqr.resolve_at(x_ref={"sub": {"position": p, "orientation": q}})
ctrl.reprogram(sol)          # gain, trim and reference move together
```

The symbolic linearization is already compiled, so this is a matrix
evaluation plus a small dense DARE — µs plus ms on a ~12-dim tangent,
not a rebuild.

The law's `K` and `u_ff` are **Ports**, not baked constants, defaulting
to the built solve. So a regulator lowered to *any* backend can be
reprogrammed in place, and nothing changes for a caller that ignores
them:

```js
// wasm: the same triple, straight off a retarget endpoint as JSON
reg.reprogram(await (await fetch("/api/retarget", …)).json());
```

```cpp
// C++: control(x) flies the built point; the gain is an argument
auto u = lqr.control(x, ref, K, u_ff);
```

Two limits worth knowing. Only slots this LQR **regulates** can move —
the frozen complement is baked into `A`/`B` as a constant, and
`resolve_at` raises rather than silently ignore a move there. And the
trim `u_ref` is attitude-dependent in general; solving for it is a
root-solve and stays yours (pass it as `u_ref=`).

## Source material

- Reference: [Transforms](../reference/transforms.md)
- Code: `manta/control/lqr.py`
- Tutorial: [closed-loop quadcopter](../tutorials/quadcopter.md)

# System identification — drone

!!! note "Draft"
    This tutorial is scaffolded. It narrates `examples/vehicles/sysid_drone.py`,
    which is complete and runnable today.

Fit a drone's physical parameters (mass, thruster gains, mount
transforms) to a recorded log using [`Fit`][manta.Fit].

## To cover

- Declaring promotable `Parameter`s (those with a `manifold=`) and what
  promotion does (`Sim(world, parameters=[...])` → a `params` port).
- Building [`Window`][manta.Window]s from logged controls + measurements.
- Running the MAP fit, reading `FitResult`, checking `converged`.
- Pitfalls: whiten sensors; σ is not L2-fittable (use
  [`NoiseFit`][manta.NoiseFit] for that); joint-space A can be
  legitimately singular.

## Run it

```bash
python -m examples.vehicles.sysid_drone
```

## Structured fits — `examples/vehicles/sysid_quad_tied.py`

The companion demo fits a symmetric X-quad as a *design* rather than an
airframe: one [`Free`][manta.Free] arm length sourcing all four mount
positions, one thrust curve and one yaw coefficient shared by four
rotors via [`Tied`][manta.Tied], and `Prior(lower=, upper=)` sanity
rails — 40 decision variables collapsed to 7. It then fits the same log
with everything free and scores both models on a *second airframe*
neither has seen. The unstructured fit wins the training log and loses
the fleet test, which is the whole argument for tying.

It also shows two things that are not about the fitter at all: designing
excitation per mixer axis (a dedicated yaw doublet is what makes the
drag coefficient identifiable), and why inertia and arm length must not
be freed together (the gyro sees only their product).

```bash
python -m examples.vehicles.sysid_quad_tied
```

## Source material

- Code: `examples/vehicles/sysid_drone.py`,
  `examples/vehicles/sysid_quad_tied.py`
- Reference: [System identification](../reference/fit.md)
- How-to: [Fit parameters from a log](../how-to/fit-parameters.md)

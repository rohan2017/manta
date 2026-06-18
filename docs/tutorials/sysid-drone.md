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

## Source material

- Code: `examples/vehicles/sysid_drone.py`
- Reference: [System identification](../reference/fit.md)
- How-to: [Fit parameters from a log](../how-to/fit-parameters.md)

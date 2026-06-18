# Write a custom Part

!!! note "Draft"
    This guide is scaffolded. The outline below marks what it should cover.

Subclass [`Part`][manta.parts.Part] to add a new behavior with its own
parameters, state, inputs, and per-tick wrench contribution.

## To cover

- Declaring channels at class scope (`Parameter` / `State` / `Input` /
  `Output` / `Noise`) — see [Parts](../explanation/parts.md).
- Implementing `update(self, ctx) -> PartUpdate` (or returning a bare
  `Wrench` for a stateless part): reading `ctx` views (pose, twist,
  frames), returning a frame-tagged `Wrench` and any state derivatives.
- Choosing a manifold for a `State` (R1 / R3 / SO3).
- Declaring `requires_fields` / `requires_planet` when the part samples a
  field.
- Making a `Parameter` fittable (give it a `manifold=`).
- Testing the part in isolation.

## Worked example skeleton

```python
from manta.parts import Part, Parameter, Input, Wrench
# from manta.ir.frames import CraftFrame

class MyThruster(Part):
    gain = Parameter(1.0)
    throttle = Input(0.0)

    def update(self, ctx):
        ...  # return PartUpdate(wrench=Wrench(force=..., frame=CraftFrame))
```

## Source material

- Reference: [Parts](../reference/parts.md)
- Code: `manta/parts/actuation/thruster.py` (a simple worked part)

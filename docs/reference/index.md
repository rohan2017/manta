# API reference

This reference is generated directly from the source docstrings. It is
organized by the same layers as the [architecture](../explanation/architecture.md):

- **[Model](model.md)** — `World`, `Craft`, `Coupling`: the declarative
  layer.
- **[Parts](parts.md)** — the stock parts you add to a craft.
- **[Fields](fields.md)** — gravity, fluid, magnetic, collision, optical
  fields and their disturbances.
- **[Planets](planets.md)** — `Planet` / `Earth` world-level entities.
- **[Transforms](transforms.md)** — `Sim`, `EKF`, `UKF`, `INS`, `LQR`: the compile-time
  siblings.
- **[Estimation](estimation.md)** — attitude filters and observability /
  consistency analysis.
- **[System identification](fit.md)** — `Fit`, `NoiseFit`.
- **[Targets](targets.md)** — `TargetNumpy` / `TargetCpp` / `TargetWasm`
  / `TargetJax` and the runtimes they produce.
- **[Simulation-only plants](simulation.md)** — detailed truth models, such
  as battery cells and BMS state, excluded from estimation and codegen.
- **[IR primitives](ir.md)** — frames, types, manifolds, state spec.

Most users work through the top-level `manta` namespace:

```python
from manta import World, Craft, Sim, EKF, LQR, TargetNumpy, TargetCpp
```

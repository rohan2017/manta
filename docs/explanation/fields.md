# Fields and disturbances

!!! note "Draft"
    This page is scaffolded. The outline below marks what it should cover.

Each [`Field`][manta.fields.Field] is a single concrete physical kind
(`GravityField`, `FluidField`, `MagField`, `CollisionField`,
`OpticalField`). Per-source variation is expressed by attaching different
[`Disturbance`][manta.fields.Disturbance] subclasses to the same field
instance.

## To cover

- **One field per physical kind** — why `GravityField` is a single class
  and a planet's pull, a uniform background, and a body-pull are all
  *disturbances* on it.
- **The combining flags** — how overlapping disturbances combine:
    - `"additive"` (default) — linear sum (gravity, B-field).
    - `"averaged"` — mean of the running additive sum + every averaged
      contribution (overlapping wind bubbles compromise on the mean).
    - `"projected"` — Gram-Schmidt residual; only the component extending
      the running sum is added.
- **Estimable disturbances** — how a disturbance carrying `State`/`Noise`
  (e.g. `CraftWindBubble`) becomes a state the
  [EKF](estimation.md) estimates.
- **Body-anchored disturbances** — sources that ride a craft via a
  `FieldSource` part.

## Source material

- Reference: [Fields](../reference/fields.md)
- Code: `manta/fields/base.py` (combining logic), `manta/fields/*.py`

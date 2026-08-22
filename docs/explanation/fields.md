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
- **Gravity is declared, never inferred** — a World refuses to resolve
  without a `GravityField` (a planet registers one). Zero-g is a modelling
  decision, so a free-floating rigid body or an orbital test declares it
  with `GravityField.none()`; an absent field is a configuration error, not
  a weightless default.
- **Combining** — most fields are a plain linear superposition (sum) of
  their disturbances (gravity, B-field). `FluidField` is richer: each
  disturbance has a `combining` role and a smooth spatial `membership`,
  and the field folds them per [`FluidField.value_at_sym`][manta.fields.FluidField]:
    - `"baseline"` — a regime medium (`Ocean`, `Atmosphere`) that *layers*
      by membership in insertion order (`base ← (1−w)·base + w·value`), so
      a background overlaid by a pocket is an alpha-composite override —
      density is selected by region, never summed.
    - `"averaged"` — a membership-weighted self-mean among the averaged
      disturbances (overlapping wind bubbles agree on the mean).
    - `"additive"` — a membership-weighted perturbation summed on top
      (currents, thruster wakes, explosions).
- **Estimable disturbances** — how a disturbance carrying `State`/`Noise`
  (e.g. `CraftWindBubble`) becomes a state the
  [EKF](estimation.md) estimates.
- **Body-anchored disturbances** — sources that ride a craft via a
  `FieldSource` part.

## Source material

- Reference: [Fields](../reference/fields.md)
- Code: `manta/fields/base.py` (combining logic), `manta/fields/*.py`

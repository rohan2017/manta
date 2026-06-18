# Write a custom Field or Disturbance

!!! note "Draft"
    This guide is scaffolded. The outline below marks what it should cover.

Most extensions are a new [`Disturbance`][manta.fields.Disturbance] on an
existing [`Field`][manta.fields.Field] — a new gravity source, a new flow.
A genuinely new physical kind is a new `Field` subclass.

## To cover

- **New disturbance (common case)** — subclass `Disturbance`, implement
  `contribute_at_sym(point, t)`, set the `combining` flag
  (additive / averaged / projected), add it to the field instance.
- **Estimable disturbance** — carry `State` / `Noise` declarations so the
  disturbance becomes an EKF-estimated state (as `CraftWindBubble` does).
- **Body-anchored disturbance** — emit it from a `FieldSource` part so it
  rides a craft.
- **New field kind (rare)** — subclass `Field` / `SuperposedField`,
  define the value type and the `_scale` / `_project_combine` hooks if the
  value is compound (see `FluidField`/`FluidState`).

## Source material

- Reference: [Fields](../reference/fields.md)
- Concepts: [Fields and disturbances](../explanation/fields.md)
- Code: `manta/fields/base.py`, `manta/fields/gravity.py` (simple
  disturbances), `manta/fields/fluid.py` (compound value)

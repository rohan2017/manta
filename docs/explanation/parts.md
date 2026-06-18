# Parts and the declaration model

!!! note "Draft"
    This page is scaffolded. The outline below marks what it should cover;
    the [Parts reference](../reference/parts.md) already renders every
    stock part from its docstring.

A [`Part`][manta.parts.Part] is an atomic unit of behavior on a craft, and
a craft is a tree of them. Parts declare their interface **at class
scope** using five sentinels, and the framework walks those declarations
at compile time to build the symbolic graph.

## To cover

- **The five declaration sentinels** — what each is and when to reach for
  it:
    - `Parameter(default, manifold=…)` — frozen at construction; baked
      into the graph (promotable for [Fit](../reference/fit.md)).
    - `State(init, manifold="R1"|"R3"|SO3Manifold(...))` — mutable
      per-tick state with manifold-correct boxplus.
    - `Input(default)` — per-tick user value.
    - `Output(shape)` — per-tick observable (a sensor reading).
    - `Noise(shape, kind="white"|"rw", sigma)` — white noise or an
      RW-bias state that synthesizes its own slot + driver.
- **The craft tree** — composite parts, mounting transforms, frames
  (`PartFrame`, `ParentFrame`, `CraftFrame`).
- **`update()` and `PartUpdate`** — how a part contributes a `Wrench`
  and/or state derivatives each tick.
- **How declarations feed the transforms** — a `Noise` channel becomes a
  Q/R contribution in the [EKF](estimation.md); an `Output` becomes a
  measurement; a manifold `Parameter` becomes a fit variable.

## Source material

- Reference: [Parts](../reference/parts.md)
- Code: `manta/parts/base.py`, `manta/parts/_declarations.py`

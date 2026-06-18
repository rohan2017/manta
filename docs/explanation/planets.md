# Planets

!!! note "Draft"
    This page is scaffolded. The outline below marks what it should cover.

A [`Planet`][manta.Planet] (and the [`Earth`][manta.planets.Earth] preset)
is a world-level entity that ties a planet-fixed frame to the shared
fields.

## To cover

- **The planet-fixed frame** — axis + rotation rate, and the symbolic +
  numpy transforms between `PlanetFrame` and `WorldFrame`.
- **Standing disturbances** — what `Earth` auto-registers on the world's
  shared fields: point-mass (+ optional J2) gravity, ocean + ISA
  atmosphere via `PlanetFrameFluid`, a dipole magnetic field.
- **Initial-state factories** — `earth.position(...)`, `earth.velocity(...)`,
  `earth.at_rest()` and how they resolve to WorldFrame seeds at compile
  time.
- **Multiple planets** in one world, superposing into the shared fields.
- **Earth's rotation and the gyrocompass** — sensing Earth-rate (see the
  gyrocompass demo).

## Source material

- Reference: [Planets](../reference/planets.md)
- Code: `manta/planets/*.py`

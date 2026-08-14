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
  shared fields: point-mass (+ optional J2) gravity, an `Ocean` + ISA
  `Atmosphere` pair of fluid regimes (baseline media built on
  `PlanetFrameFluid`), a dipole magnetic field.
- **Initial-state factories** — `earth.position(...)`, `earth.velocity(...)`,
  `earth.velocity(0, 0, 0)` and how they resolve to WorldFrame seeds at compile time.
- **`Scene`** — `earth.scene_at(planet_point)` returns a local East/North/Up
  frame fixed in the planet's body frame, used as numeric I/O glue while the
  dynamics stay in WorldFrame: `scene.at_rest(...)` for full rigid-attachment
  placement (orbital velocity + body spin rate + local-tangent attitude) in
  small scene-local coordinates, `scene.relative(state)` to report a craft's
  pose/velocity scene-relative, and `scene.world_pose(t)` to anchor rendering
  (publish the scene's world pose, log children relative to it, track it) so
  the planet can stay at its true scale at the world origin.
- **Multiple planets** in one world, superposing into the shared fields.
- **Earth's rotation and the gyrocompass** — sensing Earth-rate (see the
  gyrocompass demo).

## Source material

- Reference: [Planets](../reference/planets.md)
- Code: `manta/planets/*.py`

# Planets

!!! note "Draft"
    This page is scaffolded. The outline below marks what it should cover.

A [`Planet`][manta.Planet] (and the [`Earth`][manta.planets.Earth] preset)
is a world-level entity that ties a planet-fixed frame to the shared
fields.

## To cover

- **The planet-fixed frame** — axis + rotation rate, and the symbolic +
  numpy transforms between `PlanetFrame` and `WorldFrame`.
- **The reference shape** — a `Planet` is a sphere or an oblate spheroid
  (`equatorial_radius`, `flattening`) about its spin axis. `Earth` is the
  WGS-84 ellipsoid: "up" is the geodetic normal, altitude is height above
  the ellipsoid, and `earth.ecef_from_geodetic(lat, lon, alt)` /
  `earth.geodetic_from_ecef(p)` / `earth.scene_at_geodetic(lat, lon, alt)`
  use the same geodetic lat/lon/alt a GNSS receiver reports (PlanetFrame
  axes = WGS-84 ECEF for the default spin axis). A spherical Earth is
  `Earth(flattening=0.0)`.
- **Standing disturbances** — what `Earth` auto-registers on the world's
  shared fields: point-mass + J2 gravity (J2 on by default for the oblate
  Earth — with the spinning frame's centrifugal term it makes the
  ellipsoid an equipotential, so gravity is normal to the sea surface), an
  `Ocean` + ISA `Atmosphere` pair of fluid regimes (baseline media built on
  `PlanetFrameFluid`) split at the ellipsoid, the sea surface as a solid
  `Ellipsoid` collision obstacle, a dipole magnetic field.
- **Initial-state factories** — `earth.position(...)`, `earth.velocity(...)`,
  `earth.velocity(0, 0, 0)` and how they resolve to WorldFrame seeds at compile time.
- **`Scene`** — `earth.scene_at(planet_point)` / `scene_at_geodetic(lat, lon)`
  returns a local East/North/Up
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

# Model a displacement hull

Use `DisplacementHull` for a low-speed craft that crosses a free surface. It
is a `CompositePart`: each quadrature point contains an ordinary `PointBuoy`
and anisotropic `DragSurface`. Consequently, the model works unchanged with a
flat ocean, `Earth`/`SeaWaves`, spatial currents, and every supported backend.

```python
from manta import Craft
from manta.parts import DisplacementHull, Mass

boat = Craft("boat")
boat.add(Mass(
    "structure",
    mass=95.0,
    moi=(18.0, 24.0, 32.0),
    mount_offset=(0.0, 0.0, -0.08),
))
boat.add(DisplacementHull(
    "port_hull",
    dimensions=(1.20, 0.22, 0.30),
    displacement_volume=0.052,
    hydrostatic_offset=(0.0, 0.0, 0.015),
    drag_coefficients=(0.20, 0.85, 1.0),
    sample_resolution=(5, 2, 8),
    mount_offset=(0.0, 0.36, 0.0),
))
```

For a catamaran, add a second instance at the mirrored lateral mount. This is
geometry and physics composition only; steering and surface-trajectory policy
belong outside Manta.

## Calibrate it

`dimensions` defines the distribution envelope. By default its ellipsoid
volume is the full-submersion displacement. Prefer a measured or CAD-derived
`displacement_volume` when available; it intentionally may differ from that
idealized volume. Shift `hydrostatic_offset` to match the measured centre of
buoyancy. Keep the `Mass` parts at measured centres of mass: their separation
from the wet sample centroid is what produces the righting arm.

The default drag reference areas are the ellipsoid's projected frontal,
lateral, and planform areas. Supply `reference_areas=(Ax, Ay, Az)` when tow
tests or a better geometric estimate are available, then fit or calibrate the
three `drag_coefficients` separately. Offset drag points see the local
`omega × r` velocity, so they also produce roll, pitch, and yaw damping.

At the desired calm draft, check that

```text
rho_water * displaced wet volume = craft mass
```

`hull.displaced_volume_below(z)` is a hard-cut, flat-water calibration helper.
Runtime physics instead samples the world's smooth and possibly moving fluid
boundary.

## Check resolution

`sample_resolution=(axial, radial, circumferential)` controls the product
quadrature. Total displacement and drag area are conserved exactly at every
resolution, but draft and righting curves are discrete approximations because
the set of wet sample centres changes at the surface. Increase resolution
until draft, roll/pitch restoring moment, and damping change less than the
vehicle's calibration uncertainty. The practical default `(5, 2, 8)` creates
80 buoy/drag pairs.

The fluid field's surface smoothing must also be physically appropriate. A
very sharp boundary with sparse vertical samples produces force steps; an
excessively wide boundary smears the waterline and alters draft.

## Limits

This model covers low-speed displacement behavior. It does not model planing,
slamming, dynamic wave radiation, mesh collision, or CFD interaction between
multiple hulls. Add separate Manta parts for effects such as added mass; do not
fold vehicle control policy into the hull.

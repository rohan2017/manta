# Local ocean waves

For a local z-up world without a planet, `WaveOcean` extends `FlatOcean` with
deep-water sinusoidal waves. Each component is `(amplitude_m, wavelength_m,
direction_xy, phase_rad)`. Directions are normalized. The mean surface and
fluid properties use the same keyword arguments as `FlatOcean`.

```python
from manta import World
from manta.fields import FluidField, GravityField, UniformFluid, WaveOcean

fluid = FluidField()
fluid.add(UniformFluid(density=1.225, pressure=101325))
fluid.add(WaveOcean(
    components=((0.12, 12.0, (0.8, 0.6), 0.0),),
    surface_blend=0.2,
))
world = World().add_field(GravityField(g=(0, 0, -9.80665))).add_field(fluid)
```

The wet boundary follows `a cos(k·x − ωt + phase)`, with `k = 2π/λ` and
`ω = sqrt(gk)`. Fluid velocity includes the matching first-order orbital flow,
attenuated exponentially below mean sea level. Pressure combines the mean
hydrostatic column with the attenuated wave-pressure term. These are the same
linear wave laws used by the planetary `SeaWaves` path. An empty component
tuple reproduces `FlatOcean`.

Distributed `DisplacementHull` buoyancy and drag samples query this field at
their individual positions, producing restoring and wave-driven moments.
The model does not resolve slamming, wave radiation or inter-hull fluid
interaction. Use the planet path for Earth-anchored simulations. A renderer
must receive these same wave parameters and physical timestamps.

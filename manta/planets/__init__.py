"""Planets — `Planet` ABC + concrete presets (`Earth`, ...).

A `Planet` registered with a `World` via `world.add_planet(p)` contributes
its standing disturbances (gravity, ocean, atmosphere, magnetic dipole)
to the world's shared fields and provides:

  * Numpy + symbolic transforms between PlanetFrame and WorldFrame
    (so co-rotating disturbances can compute `ω × r` symbolically).
  * Factory methods for initial-state values expressed in PlanetFrame
    (`earth.position(x,y,z)`, `earth.velocity(vx,vy,vz)`), resolved at
    `Sim(world)` time to the WorldFrame seed the integrator
    actually uses.
  * Cartesian local `Scene` frames. A concrete planet may orient scene up
    from its physical surface geometry (`Earth` uses its oblate ellipsoid),
    but datum and latitude/longitude conversion remain outside Manta.

The base `Planet` is concrete enough to use directly when you just need
a rotating frame (no field contributions); subclass for concrete
preset bundles (see `Earth`).
"""

from .base import Planet
from .disturbances import Atmosphere, Ocean, PlanetFrameFluid
from .earth import Earth, SeaWaves
from .scene import Scene
from .state import PlanetState

__all__ = [
           "Atmosphere",
           "Earth",
           "Ocean",
           "Planet",
           "PlanetFrameFluid",
           "PlanetState",
           "Scene",
           "SeaWaves",
]

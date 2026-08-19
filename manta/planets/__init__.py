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
  * A reference spheroid (`Earth`: WGS-84) with geodetic lat/lon/alt
    (`ecef_from_geodetic`, `geodetic_from_ecef`) and local `Scene`
    frames (`scene_at`, `scene_at_geodetic`) whose up is the geodetic
    normal.

The base `Planet` is concrete enough to use directly when you just need
a rotating frame (no field contributions); subclass for concrete
preset bundles (see `Earth`).
"""

from .base import Planet
from .disturbances import Atmosphere, Ocean, PlanetFrameFluid
from .earth import Earth, SeaWaves
from .scene import Scene
from .state import PlanetState

__all__ = ["Planet", "Earth", "PlanetState", "Scene", "SeaWaves",
           "PlanetFrameFluid", "Atmosphere", "Ocean"]

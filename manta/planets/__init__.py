"""Planets — `Planet` ABC + concrete presets (`Earth`, ...).

A `Planet` registered with a `World` via `world.add_planet(p)` contributes
its standing disturbances (gravity, ocean, atmosphere, magnetic dipole)
to the world's shared fields and provides:

  * Numpy + symbolic transforms between PlanetFrame and WorldFrame
    (so co-rotating disturbances can compute `ω × r` symbolically).
  * Factory methods for initial-state values expressed in PlanetFrame
    (`earth.position(x,y,z)`, `earth.at_rest()`), resolved at
    `world.compile()` time to the WorldFrame seed the integrator
    actually uses.

The base `Planet` is concrete enough to use directly when you just need
a rotating frame (no field contributions); subclass for concrete
preset bundles (see `Earth`).
"""

from .base import Planet
from .earth import Earth
from .state import PlanetState

__all__ = ["Planet", "Earth", "PlanetState"]

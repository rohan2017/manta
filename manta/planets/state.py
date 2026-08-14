"""`PlanetState` — value wrapper marking an initial-state vector as
expressed in a planet's PlanetFrame.

The World resolves these wrappers at `compile()` time by calling
`planet.planet_to_world(p, v, t=0)` to convert PlanetFrame coords to
the WorldFrame seed the integrator actually uses. Plain tuples /
numpy arrays passed to `World.add_craft(position=..., velocity=...)`
continue to mean WorldFrame.

The user constructs these via factory methods on the planet:

    earth.position(0, 0, 100)     # PlanetFrame position 100m above origin
    earth.velocity(0, 0, 0)        # PlanetFrame velocity (at rest on earth)

For symmetry with the other Frame-tagged IR types, `PlanetState`
carries (a) the owning `Planet`, (b) the kind ('position' or
'velocity' — could grow to 'orientation' / 'angular_velocity' later),
and (c) the 3-tuple value in that planet's frame.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .base import Planet


class PlanetState:
    """Initial-state value carrying its PlanetFrame origin.

    Resolved by `World.add_craft` at compile time via
    `planet.planet_to_world(...)`.
    """

    __slots__ = ("planet", "kind", "value")

    def __init__(self,
                 planet: "Planet",
                 kind: str,
                 value: tuple[float, float, float]) -> None:
        if kind not in ("position", "velocity"):
            raise ValueError(
                f"PlanetState: kind must be 'position' or 'velocity', "
                f"got {kind!r}")
        self.planet = planet
        self.kind = kind
        self.value = tuple(float(x) for x in value)
        if len(self.value) != 3:
            raise ValueError(
                f"PlanetState: value must be length-3, got {value!r}")

    def __repr__(self) -> str:
        return (f"<PlanetState planet={self.planet.name!r} "
                f"kind={self.kind!r} value={self.value}>")

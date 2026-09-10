"""Craft-scoped binding to one planet-fixed Cartesian reference frame.

``PlanetBindingField`` is a compile-time context field, not a physical field
sampled over space.  It gives every part on one craft an unambiguous owning
``Planet`` in a world that may contain several planets.  The binding selects a
coordinate/reference body only; it does not gate gravity, fluid, magnetic, or
collision contributions from the world's physical fields.

Users normally create the field through ``World.add_craft(..., planet=...)``.
The compiler places it in that craft's ``TickContext`` and a dependent part
declares ``requires_fields = [PlanetBindingField]``.  ``ctx.field(...)`` then
returns the concrete binding while the CasADi graph is being traced, so the
planet transform is baked into the graph and no runtime lookup is introduced.
"""

from __future__ import annotations

from .base import Disturbance, Field


class PlanetBindingField(Field):
    """The exact planet whose body-fixed Cartesian frame a craft uses.

    This field is deliberately craft-scoped and carries no disturbances or
    sampled value.  Register it through ``World.add_craft(planet=...)`` rather
    than ``World.add_field``.
    """

    value_shape = type(None)

    def __init__(self, planet) -> None:
        from ..planets.base import Planet

        if not isinstance(planet, Planet):
            raise TypeError(
                "PlanetBindingField: planet must be a Planet, got "
                f"{type(planet).__name__}"
            )
        super().__init__()
        self.planet = planet

    def add(self, disturbance: Disturbance) -> Field:
        raise TypeError(
            "PlanetBindingField is a craft-scoped reference binding and "
            "does not accept disturbances"
        )

    def __repr__(self) -> str:
        return f"<PlanetBindingField planet={self.planet.name!r}>"


__all__ = ["PlanetBindingField"]

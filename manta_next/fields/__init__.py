"""Fields — additive superpositions of Disturbance objects.

A `Field` is a physical scalar/vector/tensor that pervades the world
(gravity, fluid density+velocity, magnetic field). It's the sum of
zero-or-more `Disturbance` contributions. The architecture is locked
per the Field redesign:

  * Single concrete class per physical kind. GravityField is one
    class; multiple gravity sources (uniform background, point masses,
    body-pulls) are different Disturbance subclasses added to the
    same Field instance.
  * `field.add(Disturbance)` — append a contribution.
  * `field.state_at_sym(point)` — return the symbolic MX value of the
    field at `point` (Vec3[AnchorFrame]). Equals the sum of every
    registered disturbance's contribution.

User-facing surface::

    from manta_next.fields import GravityField, UniformGravity, PointMassGravity

    g_field = GravityField()
    g_field.add(UniformGravity((0.0, 0.0, -9.81)))
    g_field.add(PointMassGravity(position=(1e8, 0, 0), GM=3.99e14))

    World().add_field(g_field).add_craft(...)
"""

from .base import Disturbance, Field
from .gravity import GravityField, PointMassGravity, UniformGravity
from .fluid   import CurrentFlow, FluidField, FluidState, UniformFluid
from .mag     import DipoleMag, MagField, UniformMag
from .collision import CollisionField, HalfSpace

__all__ = [
    "Disturbance", "Field",
    "GravityField", "UniformGravity", "PointMassGravity",
    "FluidField", "FluidState", "UniformFluid", "CurrentFlow",
    "MagField", "UniformMag", "DipoleMag",
    "CollisionField", "HalfSpace",
]

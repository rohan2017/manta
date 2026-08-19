"""Fields — superpositions of Disturbance objects.

A `Field` is a physical scalar/vector/tensor that pervades the world
(gravity, fluid density+pressure+temperature+velocity, magnetic field).
Most fields combine zero-or-more `Disturbance` contributions by linear
superposition (a plain sum). `FluidField` is richer: its disturbances
combine per a `combining` flag — `baseline` regime media (ocean, air)
that layer by spatial `membership`, plus `additive` perturbations and
`averaged` estimation overlays on top. `OpticalField` is the enumerated
exception: it carries discrete sources rather than a summable value. The
architecture is locked per the Field redesign:

  * Single concrete class per physical kind. GravityField is one
    class; multiple gravity sources (uniform background, point masses,
    body-pulls) are different Disturbance subclasses added to the
    same Field instance.
  * `field.add(Disturbance)` — append a contribution.
  * `field.value_at_sym(point, t)` — return the symbolic MX value of
    the field at `point` (Vec3[WorldFrame]) at time `t`, every
    registered disturbance combined (a plain sum, or `FluidField`'s
    baseline/averaged/additive blend).

User-facing surface::

    from manta.fields import GravityField, UniformGravity, PointMassGravity

    g_field = GravityField()
    g_field.add(UniformGravity((0.0, 0.0, -9.81)))
    g_field.add(PointMassGravity(position=(1e8, 0, 0), GM=3.99e14))

    World().add_field(g_field).add_craft(...)
"""

from .base import Disturbance, Field, SuperposedField
from .gravity import (
    BodyPointMassGravity, GravityField, J2Gravity, PointMassGravity,
    UniformGravity, gravity_at,
)
from .fluid   import (
    CurrentFlow, FlatOcean, FluidField, FluidState, UniformFluid,
    WeatherPatch,
    below_surface, within_sphere,
)
from .mag     import BodyDipoleMag, DipoleMag, MagField, UniformMag
from .collision import CollisionField, Ellipsoid, HalfSpace, Heightfield, Sphere
from .optical import (
    BodySemanticEllipsoid, OpticalField, SemanticEllipsoid,
)
from .wind_bubble import CraftWindBubble

__all__ = [
    "Disturbance", "Field", "SuperposedField",
    "GravityField", "UniformGravity", "PointMassGravity", "J2Gravity",
    "BodyPointMassGravity", "gravity_at",
    "FluidField", "FluidState", "FlatOcean", "UniformFluid", "CurrentFlow",
    "WeatherPatch",
    "below_surface", "within_sphere",
    "CraftWindBubble",
    "MagField", "UniformMag", "DipoleMag", "BodyDipoleMag",
    "CollisionField", "Ellipsoid", "HalfSpace", "Heightfield", "Sphere",
    "OpticalField", "SemanticEllipsoid", "BodySemanticEllipsoid",
]

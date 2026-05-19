"""Field + Disturbance base classes.

A `Field` is a single concrete physical-quantity object — there's one
GravityField class, one FluidField, one MagField. Per-source variation
is expressed by attaching different `Disturbance` subclasses to the
same Field instance: a uniform-g background, a planet's inverse-square
pull, a transient impulse — all GravityField disturbances.

`Field.state_at_sym(point)` returns the symbolic MX value of the field
at the queried anchor-frame point. Implementation is a fixed-shape sum
over every registered Disturbance's `contribute_at_sym(point)`.

Disturbances are pure-Python objects whose `contribute_at_sym` builds
a CasADi MX expression in terms of `point` (and any closed-over
parameters). They're attached at World-construction time and stay
fixed across compiled ticks. Time-varying behavior — gusts, transient
impulses — is captured downstream via Inputs or per-tick parameters,
not by mutating the field between ticks.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..ir.frames import AnchorFrame
from ..ir.types import Vec3


class Disturbance(ABC):
    """Base for one contribution to a Field.

    Subclass and implement `contribute_at_sym(point)`. The returned MX
    must have the Field's value shape (e.g. Vec3[AnchorFrame] for
    GravityField). Multiple disturbances on the same field are summed.

    Concrete subclasses live with their host Field — e.g.
    UniformGravity / PointMassGravity in manta_next.fields.gravity.
    """

    # Shape tag the Field uses to type-check disturbances added to it.
    # Subclasses MUST override.
    field_value_shape: type = type(None)

    @abstractmethod
    def contribute_at_sym(self, point: "Vec3"):
        """Return this disturbance's contribution at the given anchor-frame
        point. Output type matches the host Field's value type."""
        raise NotImplementedError


class Field(ABC):
    """Base for a typed physical field.

    Subclasses fix:
      * `value_shape` — the CasADi-MX type returned by `state_at_sym`
        (e.g. Vec3[AnchorFrame] for GravityField).
      * `_zero_value()` — the additive identity for the field at any
        point. Used as the seed of the disturbance sum and as the
        return value when no disturbances are registered.

    Concrete fields don't need to override `state_at_sym` itself — the
    base sum-over-disturbances implementation suffices.
    """

    #: The CasADi-wrapped type the field's value has. Subclasses set this.
    value_shape: type = type(None)

    def __init__(self) -> None:
        self._disturbances: list[Disturbance] = []

    @property
    def disturbances(self) -> tuple[Disturbance, ...]:
        return tuple(self._disturbances)

    def add(self, disturbance: Disturbance) -> "Field":
        """Register a disturbance with this field. Returns self for chaining."""
        if not isinstance(disturbance, Disturbance):
            raise TypeError(
                f"{type(self).__name__}.add: expected Disturbance, got "
                f"{type(disturbance).__name__}")
        if disturbance.field_value_shape is not self.value_shape:
            raise TypeError(
                f"{type(self).__name__}.add: disturbance "
                f"{type(disturbance).__name__} produces "
                f"{disturbance.field_value_shape!r}, not "
                f"{self.value_shape!r}")
        self._disturbances.append(disturbance)
        return self

    @abstractmethod
    def _zero_value(self):
        """Return the additive identity for the field as a value of
        `value_shape`. (E.g. Vec3[AnchorFrame].constant((0, 0, 0)) for
        GravityField.)"""
        raise NotImplementedError

    def state_at_sym(self, point: "Vec3"):
        """Sum every registered disturbance's contribution at `point`.

        Returns the field value as a CasADi-typed expression in the
        field's `value_shape`. If no disturbances are registered, returns
        the field's zero value (a constant).
        """
        out = self._zero_value()
        for d in self._disturbances:
            out = out + d.contribute_at_sym(point)
        return out

    def __repr__(self) -> str:
        return (f"<{type(self).__name__} "
                f"{len(self._disturbances)} disturbance(s)>")

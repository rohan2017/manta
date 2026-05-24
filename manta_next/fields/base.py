"""Field + Disturbance base classes.

A `Field` is a single concrete physical-quantity object — there's one
GravityField class, one FluidField, one MagField. Per-source variation
is expressed by attaching different `Disturbance` subclasses to the
same Field instance: a uniform-g background, a planet's inverse-square
pull, a transient impulse — all GravityField disturbances.

`Field.state_at_sym(point, t)` returns the symbolic MX value of the
field at the queried world-frame point at world-clock time `t`.
Implementation is a fixed-shape sum over every registered
Disturbance's `contribute_at_sym(point, t)`.

Disturbances may carry State / Noise declarations just like Parts —
this is how `WindBias` and friends become estimable: the framework
walks their declarations at compile time, rebinds their attributes
to symbolic graph inputs, and exposes the slots to the EKF as state
that gets estimated. See `manta_next.parts.base` for the
`State`/`Noise`/`Input`/`Parameter`/`Output` declaration sentinels —
Disturbance reuses the same machinery.
"""

from __future__ import annotations

import itertools
from abc import ABC, abstractmethod
from typing import Any

from ..ir.frames import WorldFrame
from ..ir.types import Vec3
from ..parts.base import (
    Input, Noise, Output, Parameter, State, _Declaration,
)


# Global counter for default disturbance names. Disturbances participate
# in the IR state vector by name, so unique names are required when more
# than one is registered. The user can pass `name="..."` explicitly; the
# default is `<ClassName>_<counter>`.
_DISTURBANCE_NAME_COUNTER = itertools.count()


class Disturbance(ABC):
    """Base for one contribution to a Field.

    Subclass and implement `contribute_at_sym(point, t)`. The returned
    MX must have the Field's value shape (e.g. Vec3[WorldFrame] for
    GravityField). Multiple disturbances on the same field combine
    according to their `combining` flag (see Phase-3 work).

    Disturbances may declare State / Noise channels at class scope.
    The framework picks them up at compile time exactly like it does
    for Parts:
      * Each State slot becomes a graph input + output named
        `<disturbance.name>.<slot>`; the disturbance's attribute is
        rebound to the symbolic input inside `contribute_at_sym`.
      * Each white Noise channel becomes a per-tick graph input.
      * Each RW Noise channel (sigma > 0) synthesizes a bias state +
        driver, evolving via `bias_next = bias + sqrt(dt)·driver`.

    Args:
        name — identifier used as the IR-slot prefix. Must be unique
               across the world's disturbances. Defaults to
               `<ClassName>_<counter>`.
    """

    field_value_shape: type = type(None)

    def __init__(self, name: str | None = None,
                 **overrides: Any) -> None:
        self.name = name if name is not None else (
            f"{type(self).__name__}_{next(_DISTURBANCE_NAME_COUNTER)}")
        self._apply_declarations(overrides)

    # --- Declaration walking ------------------------------------------

    def _apply_declarations(self, overrides: dict[str, Any]) -> None:
        decls = self._declarations()
        noise_sigma_keys = {
            f"{n}_sigma" for n, d in decls.items() if isinstance(d, Noise)
        }
        unknown = set(overrides) - set(decls) - noise_sigma_keys
        if unknown:
            raise TypeError(
                f"{type(self).__name__}({self.name!r}): unknown "
                f"parameter(s) {sorted(unknown)}. Declared: "
                f"{sorted(set(decls) | noise_sigma_keys)}")
        for attr_name, decl in decls.items():
            value = overrides.get(attr_name, decl.default)
            setattr(self, attr_name, value)
            if isinstance(decl, Noise):
                setattr(self, f"{attr_name}_sigma",
                        float(overrides.get(f"{attr_name}_sigma",
                                            decl.sigma)))

    @classmethod
    def _declarations(cls) -> dict[str, _Declaration]:
        decls: dict[str, _Declaration] = {}
        for klass in reversed(cls.__mro__):
            for nm, value in vars(klass).items():
                if isinstance(value, _Declaration):
                    decls[nm] = value
        return decls

    @classmethod
    def state_declarations(cls) -> dict[str, State]:
        return {n: d for n, d in cls._declarations().items()
                if isinstance(d, State)}

    @classmethod
    def noise_declarations(cls) -> dict[str, Noise]:
        return {n: d for n, d in cls._declarations().items()
                if isinstance(d, Noise)}

    @classmethod
    def input_declarations(cls) -> dict[str, Input]:
        return {n: d for n, d in cls._declarations().items()
                if isinstance(d, Input)}

    # --- Abstract contract --------------------------------------------

    @abstractmethod
    def contribute_at_sym(self, point: "Vec3", t):
        """Return this disturbance's contribution at the given world-frame
        point at world-clock time `t` (Scalar MX). Output type matches
        the host Field's value type. Static disturbances accept `t` and
        ignore it."""
        raise NotImplementedError


class Field(ABC):
    """Base for a typed physical field.

    Subclasses fix:
      * `value_shape` — the CasADi-MX type returned by `state_at_sym`
        (e.g. Vec3[WorldFrame] for GravityField).
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
        `value_shape`. (E.g. Vec3[WorldFrame].constant((0, 0, 0)) for
        GravityField.)"""
        raise NotImplementedError

    def state_at_sym(self, point: "Vec3", t):
        """Sum every registered disturbance's contribution at `point` at
        time `t`.

        Returns the field value as a CasADi-typed expression in the
        field's `value_shape`. If no disturbances are registered, returns
        the field's zero value (a constant).
        """
        out = self._zero_value()
        for d in self._disturbances:
            out = out + d.contribute_at_sym(point, t)
        return out

    def __repr__(self) -> str:
        return (f"<{type(self).__name__} "
                f"{len(self._disturbances)} disturbance(s)>")

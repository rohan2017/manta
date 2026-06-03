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
that gets estimated. See `manta.parts.base` for the
`State`/`Noise`/`Input`/`Parameter`/`Output` declaration sentinels —
Disturbance reuses the same machinery.
"""

from __future__ import annotations

import itertools
from abc import ABC, abstractmethod
from typing import Any

from ..ir.types import Vec3
from ..parts.base import DeclarationHost


# Global counter for default disturbance names. Disturbances participate
# in the IR state vector by name, so unique names are required when more
# than one is registered. The user can pass `name="..."` explicitly; the
# default is `<ClassName>_<counter>`.
_DISTURBANCE_NAME_COUNTER = itertools.count()


class Disturbance(DeclarationHost, ABC):
    """Base for one contribution to a Field.

    Subclass and implement `contribute_at_sym(point, t)`. The returned
    MX must have the Field's value shape (e.g. Vec3[WorldFrame] for
    GravityField). Multiple disturbances on the same field combine
    according to their `combining` flag (see `Field.state_at_sym`).

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
        name      — identifier used as the IR-slot prefix. Must be
                    unique across the world's disturbances. Defaults
                    to `<ClassName>_<counter>`.
        combining — how this disturbance's contribution composes with
                    others on the same field. See `Field.state_at_sym`
                    for the per-stage rule. One of:
                      "additive"  (default) — straight linear sum.
                      "averaged"  — arithmetic-mean stage; the running
                                    additive total + every averaged
                                    contribution are averaged together.
                      "projected" — only the component of this
                                    contribution that's orthogonal to
                                    (or extends) the running sum is
                                    kept. Order-dependent across
                                    multiple projected entries; the
                                    canonical order is insertion.
    """

    field_value_shape: type = type(None)

    # Subclasses may override the default combining mode. Each instance
    # can also override at construction via `combining=`.
    combining: str = "additive"

    def __init__(self, name: str | None = None, *,
                 combining: str | None = None,
                 **overrides: Any) -> None:
        self.name = name if name is not None else (
            f"{type(self).__name__}_{next(_DISTURBANCE_NAME_COUNTER)}")
        if combining is not None:
            if combining not in ("additive", "averaged", "projected"):
                raise ValueError(
                    f"Disturbance: combining must be 'additive', "
                    f"'averaged', or 'projected'; got {combining!r}")
            self.combining = combining
        self._apply_declarations(overrides)

    # --- Declaration walking: inherited from DeclarationHost ----------

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
        """Fold every registered disturbance's contribution at `point`
        at time `t` according to each disturbance's `combining` flag.

        Three stages, in order:

          1. **additive**: linear sum. The default; correct for
             gravity, B-field, fluid velocity superposition, collision
             penetration vectors.
          2. **averaged**: arithmetic mean. The running additive total
             plus every averaged contribution are averaged together.
             Used e.g. when several crafts each maintain a localized
             wind/current estimate and you want overlapping bubbles to
             agree.
          3. **projected**: Gram-Schmidt-style residual. Each
             contribution is added only by the component orthogonal to
             the running sum (or extending it along its direction).
             Vectors aligned-and-smaller than the running sum
             contribute zero; vectors at 90° pass through unchanged.
             Order-dependent across multiple projected entries —
             canonical order is insertion order.

        Returns the field value as a CasADi-typed expression in the
        field's `value_shape`. If no disturbances are registered,
        returns the field's zero value.
        """
        additive  = [d for d in self._disturbances if d.combining == "additive"]
        averaged  = [d for d in self._disturbances if d.combining == "averaged"]
        projected = [d for d in self._disturbances if d.combining == "projected"]

        out = self._zero_value()
        for d in additive:
            out = out + d.contribute_at_sym(point, t)
        if averaged:
            terms = [d.contribute_at_sym(point, t) for d in averaged]
            for term in terms:
                out = out + term
            out = self._scale(out, 1.0 / (1 + len(averaged)))
        for d in projected:
            out = self._project_combine(out, d.contribute_at_sym(point, t))
        return out

    # ---- Combining-stage hooks (subclasses can override) -------------

    def _scale(self, value, scalar: float):
        """Multiply a field value by a scalar (used by the `averaged`
        stage). Default for Vec3-valued fields uses CasADi's scalar
        broadcast. Compound fields (FluidState) override."""
        from ..ir.types import Vec3 as _Vec3
        if isinstance(value, _Vec3):
            return _Vec3[value._frame].from_mx(scalar * value._mx)
        return scalar * value

    def _project_combine(self, running, contribution):
        """Add `contribution` to `running` keeping only the component
        that's orthogonal to (or extending) the running sum. Default
        works on Vec3-valued fields. Compound fields override.
        """
        import casadi as ca
        from ..ir.types import Vec3 as _Vec3
        if not (isinstance(running, _Vec3) and isinstance(contribution, _Vec3)):
            return running + contribution
        r_mx = running._mx
        c_mx = contribution._mx
        denom = ca.dot(r_mx, r_mx) + 1e-12
        proj_scalar = ca.dot(c_mx, r_mx) / denom
        # Cap negative projections at zero so a contribution that
        # would shrink the running sum gets ignored along the run.
        proj_clip   = ca.fmax(0.0, proj_scalar)
        residual    = c_mx - proj_clip * r_mx
        return _Vec3[running._frame].from_mx(r_mx + residual)

    def __repr__(self) -> str:
        return (f"<{type(self).__name__} "
                f"{len(self._disturbances)} disturbance(s)>")

"""Field + Disturbance base classes.

A `Field` is a single concrete physical-quantity object — there's one
GravityField class, one FluidField, one MagField. Per-source variation
is expressed by attaching different `Disturbance` subclasses to the
same Field instance: a uniform-g background, a planet's inverse-square
pull, a transient impulse — all GravityField disturbances.

`Field.value_at_sym(point, t)` returns the symbolic MX value of the
field at the queried world-frame point at world-clock time `t`.
Implementation is a fixed-shape sum over every registered
Disturbance's `contribute_at_sym(point, t)`.

Disturbances may carry State / Noise declarations just like Parts —
this is how `WindBias` and friends become estimable: the framework
walks their declarations at compile time, rebinds their attributes
to symbolic graph inputs, and exposes the slots to the EKF as state
that gets estimated. See `manta.parts._declarations` for the
`State`/`Noise`/`Input`/`Parameter`/`Output` declaration sentinels —
Disturbance reuses the same machinery.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import casadi as ca

from ..ir.types import Vec3
from ..parts.base import DeclarationHost


def anchored_pose(craft, offset_body):
    """World pose of a body-fixed point during the tick trace.

    A disturbance emitted by a `FieldSource` part follows its carrying
    craft: its world position is ``craft_pos + R·offset`` and it inherits
    the craft's orientation. The craft's symbolic state is read from the
    active `TraceBindings` (the same channel `CraftWindBubble` uses), so
    nothing is stashed on the craft. Returns ``(center, quat)`` —
    ``center`` a Vec3[WorldFrame], ``quat`` the craft's Quat[WorldFrame,
    CraftFrame] (call ``quat.to_rotmat()`` for the rotation if you need
    it). Only callable inside a world-tick compile."""
    from ..parts._trace import active_trace
    from ..ir.frames import CraftFrame
    tr = active_trace()
    if tr is None:
        raise RuntimeError(
            "anchored_pose: no active trace — a field source's contribution "
            "is only callable during a world-tick compile.")
    st = tr.craft_sym_state(craft)
    pos, quat = st["position"], st["orientation"]
    off = Vec3[CraftFrame].constant(tuple(float(x) for x in offset_body))
    return pos + quat.apply(off), quat


# Disturbances participate in the IR state vector by name, so unique
# names are required when more than one is registered. The user can pass
# `name="..."` explicitly; otherwise the name is assigned at
# `Field.add()` time as `<ClassName>_<index within the field>` — a
# deterministic function of the model, not of construction order across
# the process (a global counter here once made two identical scripts
# produce different state keys).


class Disturbance(DeclarationHost, ABC):
    """Base for one contribution to a Field.

    Subclass and implement `contribute_at_sym(point, t)`. The returned
    MX must have the Field's value shape (e.g. Vec3[WorldFrame] for
    GravityField). Multiple disturbances on the same field combine
    according to their `combining` flag (see `Field.value_at_sym`).

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
                    others on the same field. One of:
                      "additive"  (default) — straight linear sum.
                      "baseline"  — a regime medium (e.g. ocean / air);
                                    baselines layer by membership rather
                                    than summing.
                      "averaged"  — a membership-weighted self-mean
                                    among the averaged disturbances (an
                                    estimation overlay, e.g. overlapping
                                    wind bubbles agreeing on a mean).
                    Only `FluidField` interprets "baseline"/"averaged"
                    (see `FluidField.value_at_sym`); for other superposed
                    fields (gravity, B-field) every disturbance is summed
                    additively regardless of this flag.
        membership — optional callable `(point, t) -> MX in [0, 1]` giving
                    this disturbance's spatial support. Defaults to 1
                    everywhere. Used by `FluidField` to bound regimes /
                    perturbations; ignored by purely-additive fields.
    """

    field_value_shape: type = type(None)

    # Subclasses may override the default combining mode. Each instance
    # can also override at construction via `combining=`.
    combining: str = "additive"

    def __init__(self, name: str | None = None, *,
                 combining: str | None = None,
                 membership=None,
                 **overrides: Any) -> None:
        from ..ir.module import check_name
        # `None` defers naming to `Field.add()`, which assigns a
        # deterministic per-field default (`<ClassName>_<index>`).
        self.name = (check_name(name, who=type(self).__name__)
                     if name is not None else None)
        if combining is not None:
            self.combining = combining
        # Validate the EFFECTIVE value — a subclass setting the class
        # attribute (the documented way to fix a mode) must be checked
        # too: a typo'd class-level `combining` would otherwise drop the
        # disturbance from every composition bucket silently (a fluid
        # regime vanishing → density 0, no error).
        if self.combining not in ("additive", "averaged", "baseline"):
            raise ValueError(
                f"{type(self).__name__}: combining must be 'additive', "
                f"'averaged', or 'baseline'; got {self.combining!r} "
                f"(set via the class attribute or the combining= kwarg)")
        if membership is not None:
            self._membership = membership
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

    def membership(self, point: "Vec3", t) -> "ca.MX":
        """Spatial support of this disturbance at `point`/`t`, an MX in
        [0, 1]. Defaults to 1 everywhere (global); a `membership=`
        callable passed at construction, or a subclass override, narrows
        it (a sea half-space, a bubble). `FluidField` weights each
        contribution by this; additive vector fields ignore it."""
        if getattr(self, "_membership", None) is not None:
            return self._membership(point, t)
        return ca.MX(1.0)


class Field:
    """Base for a typed physical field — a host that CARRIES disturbances.

    A Field fixes `value_shape` (the CasADi-MX type its disturbances
    produce) and registers compatible disturbances via `add()`. Two kinds
    of field extend it:

      * `SuperposedField` — the disturbances SUM into one
        `value_at_sym(point, t)` (gravity, B-field, fluid, collision).
      * `OpticalField` (an enumerated field) — the disturbances are kept
        SEPARATE and enumerated, never summed (a camera draws one box per
        ellipsoid). It has no `value_at_sym`.

    `Field` itself is the shared host (the registry + the value-shape
    contract); it is not meant to be instantiated directly.
    """

    #: The CasADi-wrapped type the field's value has. Subclasses set this.
    value_shape: type = type(None)

    #: Whether this field's composition rule evaluates
    #: `Disturbance.membership` at all. Only `FluidField`'s
    #: baseline/averaged/additive blend does; the plain additive fold
    #: (gravity, B-field, collision) never calls it, so `add()` refuses
    #: a disturbance carrying one rather than silently dropping the
    #: spatial support the user asked for.
    honors_membership: bool = False

    def __init__(self) -> None:
        self._disturbances: list[Disturbance] = []

    @property
    def disturbances(self) -> tuple[Disturbance, ...]:
        return tuple(self._disturbances)

    def add(self, disturbance: Disturbance) -> "Field":
        """Register a disturbance with this field. Returns self for chaining.

        A disturbance constructed without an explicit `name` is named
        HERE — `<ClassName>_<index among same-class disturbances of this
        field>` — so default names depend only on registration order
        within the field, never on how many disturbances the process
        happened to construct earlier (they become IR state-vector keys;
        two identical scripts must produce identical keys)."""
        from ..ir.module import check_name
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
        # A membership on a field whose fold never evaluates it would be
        # a silent no-op — the disturbance contributes at FULL strength
        # everywhere while the user believes it is spatially bounded.
        # That is physics loss, so it fails here at registration, not at
        # some downstream sample. Both channels count as "non-default":
        # a `membership=` callable stashed by the constructor and a
        # subclass overriding the `membership` method.
        if not self.honors_membership:
            has_custom_membership = (
                getattr(disturbance, "_membership", None) is not None
                or type(disturbance).membership is not Disturbance.membership)
            if has_custom_membership:
                raise ValueError(
                    f"{type(self).__name__}.add: disturbance "
                    f"{type(disturbance).__name__} carries a non-default "
                    f"membership, but {type(self).__name__} composes its "
                    f"disturbances by a plain additive fold and never "
                    f"evaluates membership — it would be silently ignored. "
                    f"Membership is only honored by combining-mode fluid "
                    f"fields (FluidField). Drop the membership, or bake the "
                    f"spatial support into contribute_at_sym.")
        if disturbance.name is None:
            n = sum(1 for d in self._disturbances
                    if type(d) is type(disturbance))
            disturbance.name = check_name(
                f"{type(disturbance).__name__}_{n}",
                who=type(disturbance).__name__)
        if any(d.name == disturbance.name for d in self._disturbances):
            raise ValueError(
                f"{type(self).__name__}.add: disturbance name "
                f"{disturbance.name!r} already exists")
        owner = getattr(disturbance, "_field", None)
        if owner is not None and owner is not self:
            raise ValueError(
                f"{type(self).__name__}.add: disturbance "
                f"{disturbance.name!r} already belongs to "
                f"{type(owner).__name__}")
        disturbance._field = self
        self._disturbances.append(disturbance)
        return self

    def __repr__(self) -> str:
        return (f"<{type(self).__name__} "
                f"{len(self._disturbances)} disturbance(s)>")


class SuperposedField(Field, ABC):
    """A `Field` whose disturbances SUM into one value at each point.

    Subclasses fix:
      * `value_shape` — the CasADi-MX type returned by `value_at_sym`
        (e.g. Vec3[WorldFrame] for GravityField).
      * `_zero_value()` — the additive identity for the field at any
        point. Used as the seed of the disturbance sum and as the
        return value when no disturbances are registered.

    Concrete fields don't override `value_at_sym` itself — the base
    sum-over-disturbances implementation suffices.
    """

    @abstractmethod
    def _zero_value(self):
        """Return the additive identity for the field as a value of
        `value_shape`. (E.g. Vec3[WorldFrame].constant((0, 0, 0)) for
        GravityField.)"""
        raise NotImplementedError

    def value_at_sym(self, point: "Vec3", t):
        """Linear superposition: the sum of every registered
        disturbance's contribution at `point` at time `t`.

        This base fold is purely additive — correct for gravity, the
        B-field, and collision penetration vectors, whose disturbances
        only ever combine additively. Fields with richer combining
        semantics (`FluidField`: layered regime baselines + perturbations
        + membership weighting) override this method. If no disturbances
        are registered, returns the field's zero value.
        """
        out = self._zero_value()
        for d in self._disturbances:
            out = out + d.contribute_at_sym(point, t)
        return out

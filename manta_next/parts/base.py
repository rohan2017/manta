"""Part base class + declaration sentinels.

A Part is a Python class that:

  * Declares its parameters at class scope using `Parameter(default)`,
    optionally typed via PEP-526 annotations (which are advisory only —
    runtime behavior comes from the assignment value, not the annotation).
  * Receives any per-tick external inputs via `Input(default)` declarations.
  * Implements `update(ctx) -> Wrench[CraftFrame]` to contribute a wrench
    each tick. (Future extensions: state declarations, multi-output, etc.)

Example::

    class Mass(Part):
        mass: ir.Scalar = Parameter(1.0)
        apply_gravity: bool = Parameter(True)

        def update(self, ctx):
            force = ctx.gravity * self.mass if self.apply_gravity else 0
            return Wrench.from_force_at(force, point_in_part=zero)

The `_declarations()` walk collects everything subclasses contribute,
including those inherited from parents. Construction-time overrides
(`Mass("body", mass=2.0)`) replace defaults. Parameters end up as plain
attributes on the instance — they're constants from the IR's perspective.

`Input`s also become attributes, but the tracer (in M2+) will rewrap them
as symbolic placeholders that the compiled tick function exposes as
named inputs. For M1 we don't have Inputs yet — only Parameters — so
the declaration class exists but isn't tracer-wired.
"""

from __future__ import annotations

from typing import Any, ClassVar


# ---------------------------------------------------------------------------
# Declaration sentinels
# ---------------------------------------------------------------------------

class _Declaration:
    """Base for class-attribute declarations. Stored on the class; replaced
    by the resolved value on each instance during `Part.__init__`."""

    __slots__ = ("default",)

    def __init__(self, default: Any) -> None:
        self.default = default


class Parameter(_Declaration):
    """Frozen-at-config-time value. Set when the user constructs a Part,
    used as a constant during graph tracing.

    Concrete attribute types are deduced from the default value at __init__
    time — a `Parameter(1.0)` becomes a Python float; a
    `Parameter((1.0, 0.0, 0.0))` stays a tuple until the part's update()
    promotes it to an IR vector via the Vec3.constant factory.
    """


class Input(_Declaration):
    """Per-tick external value. Becomes a named input on the compiled tick.
    M1 placeholder: not yet wired into the tracer.
    """


# ---------------------------------------------------------------------------
# Part base
# ---------------------------------------------------------------------------

class Part:
    """Base class for all parts.

    Subclasses declare their interface via class-attribute `Parameter`
    (and later `Input`/`State`) entries, then implement `update(ctx)` to
    contribute a `Wrench` per tick.

    Construction signature::

        class Mass(Part):
            mass: Scalar = Parameter(1.0)

        Mass("body")                  # uses defaults
        Mass("body", mass=2.0)        # overrides
    """

    # Subclasses MAY override these for static info / part-name dispatch.
    cpp_class: ClassVar[str] = ""           # filled in by future codegen backends

    def __init__(self, name: str, **overrides: Any) -> None:
        self.name = name
        self._apply_declarations(overrides)

    # --- Declaration resolution -------------------------------------------

    def _apply_declarations(self, overrides: dict[str, Any]) -> None:
        decls = self._declarations()
        unknown = set(overrides) - set(decls)
        if unknown:
            raise TypeError(
                f"{type(self).__name__}({self.name!r}): unknown parameter(s) "
                f"{sorted(unknown)}. Declared: {sorted(decls)}")
        for attr_name, decl in decls.items():
            value = overrides.get(attr_name, decl.default)
            # Just store as a plain attribute. Tracer reads attribute values
            # at update() time; Parameter values stay constant.
            setattr(self, attr_name, value)

    @classmethod
    def _declarations(cls) -> dict[str, _Declaration]:
        """Walk MRO (most-derived overrides parent) returning the union of
        `_Declaration` class attributes by attribute name."""
        decls: dict[str, _Declaration] = {}
        # Walk in reverse-MRO so subclass entries overwrite parents.
        for klass in reversed(cls.__mro__):
            for name, value in vars(klass).items():
                if isinstance(value, _Declaration):
                    decls[name] = value
        return decls

    # --- Required override ------------------------------------------------

    def update(self, ctx: "TickContext"):
        """Compute this part's wrench contribution for the current tick.
        Subclasses must override and return a `Wrench`."""
        raise NotImplementedError(
            f"{type(self).__name__}: must override update(self, ctx)")

    # --- Introspection ----------------------------------------------------

    def __repr__(self) -> str:
        params = ", ".join(f"{n}={getattr(self, n)!r}"
                           for n in self._declarations())
        return f"<{type(self).__name__}('{self.name}', {params})>"


# Forward declaration to avoid a circular import: TickContext lives in
# manta_next/craft.py and is bound when craft.py imports.
class TickContext:
    """Per-tick context passed to `Part.update`. The Craft populates it
    with the active gravity vector, current state queries, dt, etc. See
    `manta_next/craft.py` for the concrete fields."""
    pass

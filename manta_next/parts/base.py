"""Part base class + declaration sentinels.

A Part is a Python class that:

  * Declares its parameters at class scope using `Parameter(default)`.
  * Declares mutable per-tick state via `State(init=..., manifold=...)`.
  * Receives any per-tick external inputs via `Input(default)` (reserved
    for M4 wiring).
  * Implements `update(ctx) -> Wrench | PartUpdate` to contribute a wrench
    each tick (and optionally write new state).

Example::

    class FlywheelMotor(Part):
        I_axial:    float  = Parameter(0.01)
        torque_cmd: float  = Parameter(0.0)
        axis:       tuple  = Parameter((0.0, 0.0, 1.0))

        angle: Scalar = State(init=0.0)
        rate:  Scalar = State(init=0.0)

        def update(self, ctx):
            accel = self.torque_cmd / self.I_axial
            new_rate  = self.rate + accel * ctx.dt
            new_angle = self.angle + self.rate * ctx.dt
            # Reaction torque on parent.
            reaction = ir.Vec3[CraftFrame].constant(self.axis) * (-self.torque_cmd)
            return PartUpdate(
                wrench=Wrench(reaction, ...),
                new_state={"angle": new_angle, "rate": new_rate},
            )

Inside `update`, `self.angle` and `self.rate` read the *current* tick's
symbolic state nodes — the tracer rebinds them before calling update()
and restores the Python defaults afterward. State omitted from
`new_state` is passed through unchanged.

The `_declarations()` walk collects everything subclasses contribute,
including parents. Construction-time overrides (`Motor("m", I_axial=0.02)`)
replace defaults; State() declarations also accept an init override so
the initial value can vary per instance.
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
    """Per-tick external value.

    Declared at class scope on a Part. The framework:
      * Creates a graph input named "<part_name>.<input_name>" each compile.
      * Rebinds the part attribute to the symbolic node before calling
        `update()`, so `self.<input_name>` reads the current value.
      * Initial state from Craft.initial_state() includes the input slot
        seeded with the declaration's `default` (or the construction-time
        override if the user passed one).
      * Inputs pass through CompiledWorld.step's merge — they persist
        between steps until the user overrides. This makes per-tick
        commands ergonomic: set once, tick repeatedly, change when you
        want.

    Args:
        default — Python value used to seed the initial state. May be
                  overridden at construction (`Motor("m", torque_cmd=0.5)`)
                  in which case the override becomes the seed.

    The semantic distinction from `Parameter`: Parameter values are
    frozen into the compiled graph as constants; Input values are
    re-evaluated each tick from the state dict.
    """


class State(_Declaration):
    """Per-tick state slot.

    Declared at class scope. The framework:
      * Creates a graph input named "<part_name>.<state_name>" each compile.
      * Rebinds the part attribute to that input node before calling
        `update()`, so `self.<state_name>` reads the symbolic current value.
      * Reads the new value from `PartUpdate.new_state["<state_name>"]`
        and emits it as a graph output of the same name. Omitted states
        pass through unchanged.

    Args:
        init      Python value (default initial value across compiles).
        manifold  Tag describing how the state composes / what its tangent
                  space looks like. M3 supports 'R1' (Scalar). 'R3', 'SO3',
                  and 'RigidBody' will be added alongside the EKF
                  integration in a later milestone.
    """

    __slots__ = ("init", "manifold")

    def __init__(self, init, manifold: str = "R1") -> None:
        if manifold not in ("R1",):
            raise NotImplementedError(
                f"State.manifold={manifold!r} not yet supported. "
                f"M3 ships 'R1' only.")
        super().__init__(default=init)
        self.init = init
        self.manifold = manifold


# ---------------------------------------------------------------------------
# PartUpdate — return type for Part.update()
# ---------------------------------------------------------------------------

class PartUpdate:
    """Bundle returned by `Part.update(ctx)` describing this tick's
    contributions: a wrench (force + torque on parent in CraftFrame) plus
    new values for any declared `State` slots.

    Construction is positional or keyword::

        return PartUpdate(wrench, {"angle": new_angle})
        return PartUpdate(wrench=w, new_state={"angle": a, "rate": r})

    Stateless parts can return a bare `Wrench` instead — the framework
    wraps it as `PartUpdate(wrench=w, new_state={})` automatically.
    """

    __slots__ = ("wrench", "new_state")

    def __init__(self, wrench=None, new_state: dict | None = None) -> None:
        if wrench is None:
            raise TypeError("PartUpdate: wrench is required")
        self.wrench = wrench
        self.new_state = dict(new_state) if new_state else {}


# ---------------------------------------------------------------------------
# Part base
# ---------------------------------------------------------------------------

class Part:
    """Base class for all parts.

    Subclasses declare their interface via class-attribute `Parameter`
    (and later `Input`/`State`) entries, then implement `update(ctx)` to
    contribute a `Wrench` per tick.

    Every Part has a `transform` parameter — a static (x, y, z) position
    offset from its parent's frame. (Static orientation will land
    alongside articulated joints in M3; for M2 the static rotation is
    forced to identity.) The framework uses this transform to roll the
    part's wrench up into its parent's frame (force-at-offset → torque
    contribution at parent origin).

    Construction signature::

        class Mass(Part):
            mass: Scalar = Parameter(1.0)

        Mass("body")                            # at origin
        Mass("battery", mass=2.0,
             transform=(0.0, 0.0, -0.5))        # 0.5 m below origin
    """

    # Subclasses MAY override these for static info / part-name dispatch.
    cpp_class: ClassVar[str] = ""           # filled in by future codegen backends

    # Universal: every part has a static (parent → part) offset.
    transform: "tuple[float, float, float]" = Parameter((0.0, 0.0, 0.0))

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
            # Plain attribute. For State, this is the init value used both
            # as the seed in Craft.initial_state() and as the value the
            # attribute holds OUTSIDE of a trace. Inside a trace, the
            # framework rebinds it to the symbolic input node.
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

    @classmethod
    def state_declarations(cls) -> dict[str, "State"]:
        """Just the State entries (subset of _declarations)."""
        return {n: d for n, d in cls._declarations().items()
                if isinstance(d, State)}

    @classmethod
    def input_declarations(cls) -> dict[str, "Input"]:
        """Just the Input entries (subset of _declarations)."""
        return {n: d for n, d in cls._declarations().items()
                if isinstance(d, Input)}

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

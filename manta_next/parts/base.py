"""Part base class + declaration sentinels.

A Part is a Python class that:

  * Declares its parameters at class scope using `Parameter(default)`.
  * Declares mutable per-tick state via `State(init=..., manifold=...)`.
  * Receives any per-tick external inputs via `Input(default)`.
  * Emits per-tick observables via `Output(shape=...)`.
  * Implements `update(ctx) -> Wrench | PartUpdate` (and optionally
    `post_update(ctx_post) -> dict`) to contribute a wrench, write
    new state, and/or emit Output values each tick.

Example::

    from manta_next.parts import Part, Parameter, Mass, Wrench
    from manta_next.ir.frames import CraftFrame
    from manta_next.ir.types import Vec3

    class ConstantUpwardLift(Part):
        \"\"\"A toy Part: applies a fixed body-frame +z force.\"\"\"
        magnitude: float = Parameter(1.0)

        def update(self, ctx):
            f = Vec3[CraftFrame].constant((0.0, 0.0, self.magnitude))
            return Wrench(force=f,
                          torque=Vec3[CraftFrame].constant((0, 0, 0)))

Inside `update`, declared State / Input attributes read the *current*
tick's symbolic node — the tracer rebinds the attribute before calling
update() and restores the Python default after the tick is compiled.
State omitted from `PartUpdate.new_state` passes through unchanged.

`_declarations()` walks the MRO so subclasses inherit Parameter/State/
Input/Output entries from their parents. Construction-time overrides
(`MyPart("p", magnitude=0.5)`) replace declared defaults.
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


class Output(_Declaration):
    """Per-tick value produced by a part (sensor reading, derived quantity,
    telemetry signal).

    Declared at class scope. The part writes its computed value via
    `PartUpdate.outputs["<name>"] = <Vec3 | Scalar | …>`. The framework
    emits the value as a graph output named "<part_name>.<name>"; tick
    callers read it from the result dict (read-only, doesn't round-trip
    back as next-tick state).

    Args:
        shape    — "scalar" or "vec3" or "vec4" (= quaternion). Optional;
                   the framework picks up the actual shape from what the
                   part writes. Provided here so future codegen / EKF
                   plumbing can introspect output kinds without tracing.
    """

    __slots__ = ("shape",)

    def __init__(self, shape: str = "vec3") -> None:
        super().__init__(default=None)
        if shape not in ("scalar", "vec3", "vec4"):
            raise ValueError(
                f"Output: shape must be one of 'scalar', 'vec3', 'vec4'; "
                f"got {shape!r}")
        self.shape = shape


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
        manifold  Tag describing how the state composes / what its
                  tangent space looks like. Today only `'R1'` (scalar)
                  is wired through `Craft.compile_tick` for part-declared
                  State; the rigid-body slots use 'R3' and 'SO3'
                  internally but those manifolds aren't yet selectable
                  from a user Part. See `manta_next.math.manifold`.
    """

    __slots__ = ("init", "manifold")

    def __init__(self, init, manifold: str = "R1") -> None:
        if manifold not in ("R1",):
            raise NotImplementedError(
                f"State.manifold={manifold!r}: only 'R1' is currently "
                f"supported on user-declared Part state. 'R3'/'SO3'/"
                f"'RigidBody' are defined in manta_next.math.manifold "
                f"but not yet selectable here.")
        super().__init__(default=init)
        self.init = init
        self.manifold = manifold


# ---------------------------------------------------------------------------
# PartUpdate — return type for Part.update()
# ---------------------------------------------------------------------------

class PartUpdate:
    """Bundle returned by `Part.update(ctx)` describing this tick's
    contributions: a wrench (force + torque on parent in CraftFrame), new
    values for any declared State slots, and any declared Output values
    the part produces.

    Construction::

        return PartUpdate(wrench, {"angle": a})
        return PartUpdate(wrench=w, new_state={"angle": a, "rate": r})
        return PartUpdate(wrench=w, outputs={"gyro": gyro_vec})

    Stateless parts can return a bare `Wrench` instead — the framework
    wraps it as `PartUpdate(wrench=w)` automatically.
    """

    __slots__ = ("wrench", "new_state", "outputs")

    def __init__(self,
                 wrench=None,
                 new_state: dict | None = None,
                 outputs: dict | None = None) -> None:
        if wrench is None:
            raise TypeError("PartUpdate: wrench is required")
        self.wrench = wrench
        self.new_state = dict(new_state) if new_state else {}
        self.outputs   = dict(outputs)   if outputs   else {}


# ---------------------------------------------------------------------------
# Part base
# ---------------------------------------------------------------------------

class Part:
    """Base class for all parts.

    Subclasses declare their interface via class-attribute `Parameter`
    (and later `Input`/`State`) entries, then implement `update(ctx)` to
    contribute a `Wrench` per tick.

    Every Part has a `transform` parameter — a static (x, y, z) position
    offset from its parent's frame. Static orientation between part and
    parent is currently fixed at identity; non-identity static rotations
    would be a future extension. The framework uses this transform to
    roll the part's wrench up into its parent's frame (force-at-offset
    → torque contribution at parent origin).

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

    @classmethod
    def output_declarations(cls) -> dict[str, "Output"]:
        """Just the Output entries (subset of _declarations)."""
        return {n: d for n, d in cls._declarations().items()
                if isinstance(d, Output)}

    # --- Required + optional overrides ------------------------------------

    def update(self, ctx: "TickContext"):
        """Compute this part's wrench contribution for the current tick.
        Subclasses must override and return a `Wrench` or `PartUpdate`."""
        raise NotImplementedError(
            f"{type(self).__name__}: must override update(self, ctx)")

    def post_update(self, ctx_post: "PostUpdateContext") -> dict:
        """Optional second-phase hook called AFTER Newton-Euler runs.

        Override to emit `Output` values that depend on the just-computed
        body-frame acceleration / angular acceleration (e.g. an IMU
        accelerometer). Returns a dict of `{output_name: ir_value}`.
        Default: no extra outputs.

        The dict's keys are validated against the Part's declared
        `Output` slots, same as the main `update`. An Output slot may
        be filled by EITHER `update` OR `post_update`, but at most one.
        """
        return {}

    # --- Introspection ----------------------------------------------------

    def __repr__(self) -> str:
        params = ", ".join(f"{n}={getattr(self, n)!r}"
                           for n in self._declarations())
        return f"<{type(self).__name__}('{self.name}', {params})>"


# Forward declarations to avoid a circular import — both live in
# manta_next/craft.py and are bound when craft.py imports.
class TickContext:
    """Per-tick context passed to `Part.update`. See `manta_next/craft.py`
    for the concrete fields."""
    pass


class PostUpdateContext:
    """Post-Newton-Euler context passed to `Part.post_update`. Exposes
    the just-computed body-frame acceleration and angular acceleration
    alongside the standard kinematic state. See `manta_next/craft.py`."""
    pass

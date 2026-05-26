"""Part base class + declaration sentinels.

A Part is a Python class that:

  * Declares its parameters at class scope using `Parameter(default)`.
  * Declares mutable per-tick state via `State(init=..., manifold=...)`.
  * Receives any per-tick external inputs via `Input(default)`.
  * Emits per-tick observables via `Output(shape=...)`.
  * Implements `update(ctx) -> Wrench | PartUpdate` to contribute a
    wrench, write new state, and/or emit Output values each tick.

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


class Noise(_Declaration):
    """Per-tick noise channel declared at class scope.

    Two kinds:

      * `kind="white"` (default) — per-tick i.i.d. Gaussian.
        The framework creates a graph input named
        `<part>.<noise_name>`, rebinds the part attribute to that
        input, and the part adds it directly into its sensor reading
        (or process expression). σ is the per-tick measurement stddev.

      * `kind="rw"` — random-walk bias. The framework synthesizes:
          - A **state slot** `<part>.<noise_name>` holding the bias.
          - A driver noise input `<part>.<noise_name>_driver`.
          - A state update each tick:
                bias_next = bias + sqrt(dt) · driver,    driver ~ N(0, σ²).
        Inside `update()`, `self.<noise_name>` reads the bias state
        (the slowly-drifting current value) — typically added into
        the sensor signal alongside a separate white channel. σ has
        continuous σ/√Hz semantics; per-tick bias variance is dt·σ².

    Args:
        shape — "scalar" or "vec3". Default "vec3".
        kind  — "white" (default) or "rw".
        frame — Frame class for the vec3 form. Default `CraftFrame`.
                Ignored for scalar shape.
        sigma — 1-σ standard deviation, scalar (isotropic across axes).
                White: per-tick measurement stddev. RW: σ/√Hz drift
                density.
    """

    __slots__ = ("shape", "kind", "frame", "sigma")

    def __init__(self, shape: str = "vec3", kind: str = "white",
                 frame=None,
                 sigma: float = 0.0) -> None:
        super().__init__(default=None)
        if shape not in ("scalar", "vec3"):
            raise ValueError(
                f"Noise: shape must be 'scalar' or 'vec3'; got {shape!r}")
        if kind not in ("white", "rw"):
            raise ValueError(
                f"Noise: kind must be 'white' or 'rw'; got {kind!r}")
        self.shape = shape
        self.kind  = kind
        self.frame = frame
        self.sigma = float(sigma)


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
                  For R1 a float; for R3 a length-3 tuple / ndarray.
        manifold  Tag describing how the state composes:
                    'R1' (default) — scalar; tangent = same.
                    'R3'           — 3-vector in WorldFrame; tangent = 3.
                  'SO3' (quaternion) requires the dual-frame parametrization
                  and isn't yet user-selectable; rigid-body slots use it
                  internally.
        frame     Frame tag for R3 state. Default `CraftFrame`. Ignored
                  for R1.
    """

    __slots__ = ("init", "manifold", "frame")

    def __init__(self, init, manifold: str = "R1", frame=None) -> None:
        if manifold not in ("R1", "R3"):
            raise NotImplementedError(
                f"State.manifold={manifold!r}: only 'R1' (scalar) and "
                f"'R3' (vec3) are wired through compile_tick today. SO3 "
                f"requires the dual-frame parametrization (orientation "
                f"= Quat[from, to]) and isn't yet user-selectable.")
        if manifold == "R3":
            try:
                t = tuple(float(x) for x in init)
            except (TypeError, ValueError):
                raise ValueError(
                    f"State(manifold='R3'): init must be a 3-element "
                    f"sequence, got {init!r}")
            if len(t) != 3:
                raise ValueError(
                    f"State(manifold='R3'): init must be length-3, got "
                    f"{init!r}")
            init = t
        super().__init__(default=init)
        self.init = init
        self.manifold = manifold
        self.frame = frame


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
    offset from its parent's output frame. Static orientation between
    part and parent is currently fixed at identity; non-identity static
    rotations would be a future extension. The framework uses this
    transform to roll the part's wrench up into its parent's frame
    (force-at-offset → torque contribution at parent origin).

    Every Part also has a `parent` attribute — either another Part
    (typically a CompositePart like the craft's RootPart or a Joint)
    or `None` for the unattached state. Parents are set by
    `CompositePart.add(child)` when a child is attached. The craft's
    part tree is rooted at `Craft.root`.

    Construction signature::

        class Mass(Part):
            mass: Scalar = Parameter(1.0)

        Mass("body")                            # at origin of parent
        Mass("battery", mass=2.0,
             transform=(0.0, 0.0, -0.5))        # 0.5 m below parent origin
    """

    # Subclasses MAY override these for static info / part-name dispatch.
    cpp_class: ClassVar[str] = ""           # filled in by future codegen backends

    # Fields the part requires registered on the World. Verified at
    # `world.compile()`. Subclasses override with concrete Field subclasses
    # (e.g., `requires_fields = [GravityField]` for a Mass-like part).
    # Use Field BASE classes — the registered field may be any subclass.
    requires_fields: ClassVar[list[type]] = []

    # Planet subclass the part requires. None ⇒ no planet required.
    # Verified at `world.compile()`. Subclasses override; the matched
    # planet is then accessible via `ctx.planet(PlanetCls)`.
    requires_planet: ClassVar[type | None] = None

    # Universal: every part has a static (parent → part) offset.
    transform: "tuple[float, float, float]" = Parameter((0.0, 0.0, 0.0))

    def __init__(self, name: str, **overrides: Any) -> None:
        self.name = name
        self.parent: "Part | None" = None
        self._apply_declarations(overrides)

    # --- Declaration resolution -------------------------------------------

    def _apply_declarations(self, overrides: dict[str, Any]) -> None:
        decls = self._declarations()
        # Noise declarations expose a per-instance `<name>_sigma`
        # attribute. Recognize override keys of that form and route
        # them to the matching declaration without reporting "unknown".
        noise_sigma_keys = {
            f"{n}_sigma" for n, d in decls.items() if isinstance(d, Noise)
        }
        unknown = set(overrides) - set(decls) - noise_sigma_keys
        if unknown:
            raise TypeError(
                f"{type(self).__name__}({self.name!r}): unknown parameter(s) "
                f"{sorted(unknown)}. Declared: "
                f"{sorted(set(decls) | noise_sigma_keys)}")
        for attr_name, decl in decls.items():
            value = overrides.get(attr_name, decl.default)
            # Plain attribute. For State, this is the init value used both
            # as the seed in Craft.initial_state() and as the value the
            # attribute holds OUTSIDE of a trace. Inside a trace, the
            # framework rebinds it to the symbolic input node.
            setattr(self, attr_name, value)
            if isinstance(decl, Noise):
                sigma_key = f"{attr_name}_sigma"
                setattr(self, sigma_key,
                        float(overrides.get(sigma_key, decl.sigma)))

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

    @classmethod
    def noise_declarations(cls) -> dict[str, "Noise"]:
        """Just the Noise entries (subset of _declarations)."""
        return {n: d for n, d in cls._declarations().items()
                if isinstance(d, Noise)}

    def noise_R(self, name: str) -> Any:
        """Measurement-noise covariance for a declared Noise slot.

        Reads the per-instance `<name>_sigma` attribute (set at
        construction time, default from the declaration). Returns:
            * `σ²·I3` (np.ndarray, 3×3) for vec3 noise.
            * `σ²` (float) for scalar noise.

        Used by the EKF to size measurement updates without the user
        having to specify R separately:
            `ekf.update(h, z, R=imu.noise_R("gyro_noise"))`.
        """
        import numpy as np
        decls = self.noise_declarations()
        if name not in decls:
            raise KeyError(
                f"{type(self).__name__}('{self.name}'): no Noise slot "
                f"named {name!r}. Declared: {sorted(decls)}")
        decl = decls[name]
        sigma = float(getattr(self, f"{name}_sigma"))
        var = sigma ** 2
        if decl.shape == "scalar":
            return var
        return var * np.eye(3)

    # --- Required + optional overrides ------------------------------------

    def update(self, ctx: "TickContext"):
        """Compute this part's wrench contribution for the current tick.
        Subclasses must override and return a `Wrench` or `PartUpdate`."""
        raise NotImplementedError(
            f"{type(self).__name__}: must override update(self, ctx)")

    # --- Introspection ----------------------------------------------------

    def __repr__(self) -> str:
        params = ", ".join(f"{n}={getattr(self, n)!r}"
                           for n in self._declarations())
        return f"<{type(self).__name__}('{self.name}', {params})>"


# ---------------------------------------------------------------------------
# CompositePart — a Part that has children
# ---------------------------------------------------------------------------

class CompositePart(Part):
    """A Part that hosts other Parts as children.

    Children mount on this part's *output frame*. For a non-Joint
    CompositePart the output frame is identical to the part's own
    frame (translation only, via `transform`). A `Joint` overrides
    this — its output frame additionally rotates by the joint angle.

    `add(child)` appends a child Part, sets its `parent` to self, and
    returns the child (so chained construction reads naturally):

        gimbal = pan.add(Joint("tilt", axis=(0, 1, 0)))
        gimbal.add(Mass("camera", mass=0.05, transform=(0.1, 0, 0)))
    """

    def __init__(self, name: str, **overrides: Any) -> None:
        super().__init__(name, **overrides)
        self._children: list[Part] = []

    def add(self, child: Part) -> Part:
        if not isinstance(child, Part):
            raise TypeError(
                f"{type(self).__name__}('{self.name}').add: expected Part, "
                f"got {type(child).__name__}")
        if child.parent is not None:
            raise ValueError(
                f"{type(self).__name__}('{self.name}').add: child "
                f"'{child.name}' is already attached to "
                f"'{child.parent.name}'.")
        child.parent = self
        self._children.append(child)
        return child

    @property
    def children(self) -> tuple[Part, ...]:
        return tuple(self._children)

    def walk(self):
        """DFS over this part's subtree, yielding self then each descendant."""
        yield self
        for child in self._children:
            if isinstance(child, CompositePart):
                yield from child.walk()
            else:
                yield child

    def update(self, ctx):
        """CompositePart has no intrinsic wrench contribution by default —
        subclasses (RootPart, Joint, etc.) override if they need to."""
        from ..ir.wrench import Wrench
        from ..ir.frames import CraftFrame
        return Wrench.zero(CraftFrame)


# ---------------------------------------------------------------------------
# RootPart — the implicit root of every Craft's part tree
# ---------------------------------------------------------------------------

class RootPart(CompositePart):
    """The root of a Craft's part tree. Defines CraftFrame at the body
    origin; all other parts are descendants. RootPart itself has no
    mass or inertia — the body's mass distribution comes from explicit
    Mass children added under it.
    """

    # RootPart's transform is necessarily (0,0,0): it IS the craft frame
    # origin. Override the inherited Parameter to lock it.
    transform: "tuple[float, float, float]" = Parameter((0.0, 0.0, 0.0))

    def __init__(self, name: str) -> None:
        super().__init__(name)


# NOTE: documentation-only stub.
#
# The real `TickContext` class lives in `manta_next/craft.py`. We
# can't import it here at runtime because that would create a circular
# import (craft imports parts.base). The stub below exists purely as
# a forward-reference so user docstrings / IDE tooling have something
# to point at when they say "the Part's update receives a TickContext".
# Do NOT `isinstance(ctx, TickContext)` against this — that check
# would always pass against the placeholder regardless of what `ctx`
# actually is. The compile_tick loop in craft.py constructs and
# dispatches the concrete context directly.

class TickContext:   # noqa: D401  (docstring is the API doc)
    """Forward-reference stub. See `manta_next.craft.TickContext` for
    the concrete class with fields gravity, gravity_field, fluid_field,
    mag_field, collision_field, dt, position, orientation, velocity,
    angular_velocity, velocity_body, R_craft_from_input,
    acceleration_world, acceleration_body, angular_acceleration."""
    __slots__ = ()

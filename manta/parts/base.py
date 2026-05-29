"""Part base class + declaration sentinels.

A Part is a Python class that:

  * Declares its parameters at class scope using `Parameter(default)`.
  * Declares mutable per-tick state via `State(init=..., manifold=...)`.
  * Receives any per-tick external inputs via `Input(default)`.
  * Emits per-tick observables via `Output(shape=...)`.
  * Implements `update(ctx) -> Wrench | PartUpdate` to contribute a
    wrench, write new state, and/or emit Output values each tick.

Example::

    from manta.parts import Part, Parameter, Mass, Wrench
    from manta.ir.frames import CraftFrame
    from manta.ir.types import Vec3

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
      * Inputs pass through Sim.step's merge — they persist
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


class SynthesizedNoise:
    """Return value of `Noise.synthesize()` describing the IR plumbing
    one occurrence of a Noise channel contributes to a tick.

    Attrs:
        signal_sym    — IR symbol bound onto `part.<name>` so that user
                        code reading `self.<name>` inside `update()`
                        sees the symbolic current value.
        state_update  — Optional `(slot_full_name, next_value_sym)`; the
                        compiler appends this to the tick's state-output
                        list (the slot evolves over time).
    """

    __slots__ = ("signal_sym", "state_update")

    def __init__(self, signal_sym, state_update=None):
        self.signal_sym   = signal_sym
        self.state_update = state_update


class Noise(_Declaration):
    """Abstract base for noise-channel declarations.

    Subclasses set class-level metadata (`kind`, `contributes_state`)
    and implement `synthesize()` (the per-tick IR plumbing). Backends
    key on `signal_manifold.kind` via their own registry — no
    `isinstance(WhiteNoise)` dispatch anywhere in the codebase.

    Concrete subclasses:

      * `WhiteNoise` — per-tick i.i.d. Gaussian. The framework creates a
        graph input named `<part>.<noise_name>`, rebinds the part
        attribute to that input, and the part adds it directly into its
        sensor reading (or process expression). σ is the per-tick
        measurement stddev. `kind = "white"`.

      * `RandomWalkNoise` — random-walk bias. The framework synthesizes:
          - A **state slot** `<part>.<noise_name>` holding the bias.
          - A driver noise input `<part>.<noise_name>_driver`.
          - A state update each tick:
                bias_next = bias + sqrt(dt) · driver,    driver ~ N(0, σ²).
        Inside `update()`, `self.<noise_name>` reads the bias state
        (the slowly-drifting current value). σ has continuous σ/√Hz
        semantics; per-tick bias variance is dt·σ². `kind = "random_walk"`.

    Args:
        signal_manifold — `Manifold` instance OR shortcut string. The
                          manifold of the symbol user code reads as
                          `self.<name>`. Shortcuts: ``"scalar"``,
                          ``"vec3"`` (combine with ``frame=`` for vec3).
                          Default ``"vec3"``.
        frame           — Frame class, only consumed when
                          `signal_manifold` is a shortcut and resolves
                          to a vector-typed manifold. Ignored otherwise.
        sigma           — 1-σ standard deviation, scalar (isotropic
                          across axes). See subclass docstrings for
                          unit conventions.
    """

    # Class-level metadata. Subclasses MUST set both.
    kind:              str  = None    # type: ignore[assignment]
    contributes_state: bool = False

    __slots__ = ("signal_manifold", "sigma")

    def __init__(self, signal_manifold="vec3", *, frame=None,
                 sigma: float = 0.0) -> None:
        super().__init__(default=None)
        from ..ir.manifold import Manifold
        self.signal_manifold = _noise_manifold_from_shortcut(
            signal_manifold, frame=frame,
            owner_label=type(self).__name__,
        ) if not isinstance(signal_manifold, Manifold) else signal_manifold
        self.sigma = float(sigma)

    # ---- Manifolds (resolved at synthesis time for the late-bound
    # frame default; the canonical form just returns self.signal_manifold
    # and only the legacy R3Manifold(frame=None) case substitutes) -----

    def resolved_signal_manifold(self, *, default_frame=None):
        """Return `self.signal_manifold` with any unresolved frame
        substituted from `default_frame`. Used at IR synthesis time;
        the unresolved form keeps `R3Manifold(frame=None)` legal so
        a part can declare a noise without committing to a frame
        until the compiler knows which one it's in (CraftFrame for
        parts, WorldFrame for disturbances)."""
        from ..ir.manifold import R3Manifold
        if isinstance(self.signal_manifold, R3Manifold) \
                and self.signal_manifold.frame is None:
            return R3Manifold(frame=default_frame)
        return self.signal_manifold

    def state_manifold(self, *, default_frame=None):
        """Manifold of the synthesized state slot, or None. For RW
        the state lives in the same space as the per-tick signal."""
        if not self.contributes_state:
            return None
        return self.resolved_signal_manifold(default_frame=default_frame)

    # ---- Activity gate ------------------------------------------------

    def is_active(self, owner, name: str) -> bool:
        """Is this channel currently producing nonzero output? Reads
        the runtime `<name>_sigma` attribute on the owner."""
        return float(getattr(owner, f"{name}_sigma")) > 0.0

    # ---- Input-name classification (extract.py / ekf.py) --------------

    def driver_input_name(self, name: str) -> str:
        """The name of this channel's per-tick stochastic input. For
        White noise the signal IS the driver (same name); for RW the
        driver is a separate `<name>_driver` input distinct from the
        bias state name."""
        return name

    # ---- Initial state seeding ---------------------------------------

    def initial_state_entries(self, name: str, owner) -> dict[str, object]:
        """Names → zero values this channel contributes to the seed
        state dict (state_spec.unpack-compatible). Inert RW channels
        return an empty dict; everyone else seeds at least the signal
        slot."""
        return {name: self._zero_value()}

    # ---- IR synthesis (the world-tick compiler's entry point) --------

    def synthesize(self, *, base_name: str, name: str, dt, default_frame,
                   owner) -> SynthesizedNoise:
        """Build one tick's worth of IR plumbing for this channel.
        Subclasses implement; the world-tick compiler calls this once
        per noise declaration per owner."""
        raise NotImplementedError(
            f"{type(self).__name__}.synthesize must be implemented.")

    # ---- Helpers ------------------------------------------------------

    def _zero_value(self):
        """Numpy-side zero for seeding the input dict. Reads off the
        manifold so any shape (scalar, vec3, vec6, quat, …) Just Works."""
        return self.signal_manifold.default_value()


def _noise_manifold_from_shortcut(shortcut, *, frame, owner_label):
    """Resolve a Noise-side shortcut string to a Manifold instance.

    Vocabulary matches the prior `shape=` kwarg for ergonomic
    continuity: ``"scalar"`` → `ScalarManifold`, ``"vec3"`` →
    `R3Manifold(frame=frame)`. For richer manifolds (vec6, quat,
    custom), pass an explicit `Manifold` instance instead.
    """
    from ..ir.manifold import ScalarManifold, R3Manifold
    if shortcut == "scalar":
        return ScalarManifold()
    if shortcut == "vec3":
        return R3Manifold(frame=frame)
    raise ValueError(
        f"{owner_label}: signal_manifold shortcut must be "
        f"'scalar' or 'vec3'; got {shortcut!r}. For other manifolds, "
        f"pass an explicit Manifold instance.")


def _ir_input_for_manifold(manifold, name):
    """Build the IR input symbol for a Noise channel based on the
    manifold kind. Mirrors the per-State-kind dispatch in
    `world_tick.py`; the two dispatch sites should stay aligned (a
    new manifold kind needs a branch in both).
    """
    from .. import ir
    if manifold.kind == "scalar":
        return ir.Scalar.input(name)
    if manifold.kind == "vec":
        return ir.Vec3[manifold.frame].input(name)
    raise NotImplementedError(
        f"Noise: signal_manifold kind {manifold.kind!r} not yet wired "
        f"for IR input construction.")


def _ir_zero_for_manifold(manifold):
    import casadi as ca
    from .. import ir
    if manifold.kind == "scalar":
        return ir.Scalar.from_mx(ca.MX(0.0))
    if manifold.kind == "vec":
        return ir.Vec3[manifold.frame].from_mx(ca.MX.zeros(3, 1))
    raise NotImplementedError(
        f"Noise: signal_manifold kind {manifold.kind!r} not yet wired "
        f"for IR zero construction.")


def _ir_add_for_manifold(manifold, a_sym, b_mx):
    """Type-preserving sum used by RW bias_next = bias + √dt · driver."""
    from .. import ir
    if manifold.kind == "scalar":
        return ir.Scalar.from_mx(a_sym._mx + b_mx)
    if manifold.kind == "vec":
        return ir.Vec3[manifold.frame].from_mx(a_sym._mx + b_mx)
    raise NotImplementedError(
        f"Noise: signal_manifold kind {manifold.kind!r} not yet wired "
        f"for RW state update construction.")


class WhiteNoise(Noise):
    """Per-tick i.i.d. Gaussian noise channel. σ is the per-tick stddev."""

    kind              = "white"
    contributes_state = False

    __slots__ = ()

    def synthesize(self, *, base_name, name, dt, default_frame, owner):
        mfd = self.resolved_signal_manifold(default_frame=default_frame)
        return SynthesizedNoise(
            signal_sym=_ir_input_for_manifold(mfd, base_name),
        )


class RandomWalkNoise(Noise):
    """Random-walk bias channel. σ has σ/√Hz drift-density units."""

    kind              = "random_walk"
    contributes_state = True

    __slots__ = ()

    def driver_input_name(self, name: str) -> str:
        return f"{name}_driver"

    def initial_state_entries(self, name, owner):
        # Inert when σ=0: no slot, no driver, no seed.
        if not self.is_active(owner, name):
            return {}
        zero = self._zero_value()
        return {name: zero, f"{name}_driver": zero}

    def synthesize(self, *, base_name, name, dt, default_frame, owner):
        import casadi as ca
        mfd = self.resolved_signal_manifold(default_frame=default_frame)
        if not self.is_active(owner, name):
            # Inert: bind zero, no slot, no driver.
            return SynthesizedNoise(signal_sym=_ir_zero_for_manifold(mfd))
        # Active: bias-state input + driver input + state update.
        driver_name = f"{base_name}_driver"
        sqrt_dt    = ca.sqrt(dt._mx)
        bias_sym   = _ir_input_for_manifold(mfd, base_name)
        driver_sym = _ir_input_for_manifold(mfd, driver_name)
        bias_next  = _ir_add_for_manifold(mfd, bias_sym,
                                           sqrt_dt * driver_sym._mx)
        return SynthesizedNoise(
            signal_sym=bias_sym,
            state_update=(base_name, bias_next),
        )


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
        manifold  String shortcut ('R1', 'R3') or a `Manifold` instance.
                  'SO3' requires the dual-frame parametrization (used
                  internally for rigid-body slots; not user-selectable
                  via this declaration yet).
        frame     Frame tag for R3 state. Default `CraftFrame`. Ignored
                  for R1. Folded into the Manifold instance.

    `state.manifold` always reads back as a `Manifold` instance; the
    string form is normalized at construction.
    """

    __slots__ = ("init", "manifold", "frame")

    def __init__(self, init, manifold="R1", frame=None) -> None:
        from ..ir.manifold import (
            Manifold, ScalarManifold, R3Manifold, SO3Manifold,
            manifold_from_shortcut,
        )
        if isinstance(manifold, Manifold):
            mfd = manifold
        else:
            # String shortcut: SO(3) is intentionally not in the shortcut
            # table because it needs explicit from_frame/to_frame. Pass
            # `manifold=SO3Manifold(from_frame=..., to_frame=...)` instead.
            if manifold not in ("R1", "R3"):
                raise NotImplementedError(
                    f"State.manifold={manifold!r}: string shortcuts are "
                    f"only defined for 'R1' / 'R3'. For SO(3), pass an "
                    f"explicit SO3Manifold(from_frame=..., to_frame=...) "
                    f"instance to capture the dual-frame parametrization.")
            mfd = manifold_from_shortcut(manifold, frame=frame)
        if isinstance(mfd, R3Manifold):
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
        elif isinstance(mfd, SO3Manifold):
            if mfd.from_frame is None or mfd.to_frame is None:
                raise ValueError(
                    f"State(SO3Manifold): from_frame and to_frame must "
                    f"both be specified — got from_frame={mfd.from_frame!r}, "
                    f"to_frame={mfd.to_frame!r}.")
            try:
                t = tuple(float(x) for x in init)
            except (TypeError, ValueError):
                raise ValueError(
                    f"State(SO3Manifold): init must be a length-4 "
                    f"quaternion (w, x, y, z), got {init!r}")
            if len(t) != 4:
                raise ValueError(
                    f"State(SO3Manifold): init must be length-4, got "
                    f"{init!r}")
            init = t
        super().__init__(default=init)
        self.init     = init
        self.manifold = mfd
        # Keep the explicit `frame` attribute for read sites that
        # consult it directly; for R3 it mirrors the manifold's frame.
        # For SO3 the dual frames live on the manifold itself.
        self.frame = (mfd.frame if isinstance(mfd, R3Manifold) else frame)


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
    # `Sim(world)`. Subclasses override with concrete Field subclasses
    # (e.g., `requires_fields = [GravityField]` for a Mass-like part).
    # Use Field BASE classes — the registered field may be any subclass.
    requires_fields: ClassVar[list[type]] = []

    # Planet subclass the part requires. None ⇒ no planet required.
    # Verified at `Sim(world)`. Subclasses override; the matched
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
            * `σ²` (float) for scalar noise.
            * `σ²·I_d` (np.ndarray, d×d) for d-vector noise (sized
              off `signal_manifold.ambient_dim`).

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
        d = decl.signal_manifold.ambient_dim
        if d == 1:
            return var
        return var * np.eye(d)

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
# The real `TickContext` class lives in `manta/craft.py`. We
# can't import it here at runtime because that would create a circular
# import (craft imports parts.base). The stub below exists purely as
# a forward-reference so user docstrings / IDE tooling have something
# to point at when they say "the Part's update receives a TickContext".
# Do NOT `isinstance(ctx, TickContext)` against this — that check
# would always pass against the placeholder regardless of what `ctx`
# actually is. The world-tick loop in world_tick.py constructs and
# dispatches the concrete context directly.

class TickContext:   # noqa: D401  (docstring is the API doc)
    """Forward-reference stub. See `manta.craft.TickContext` for
    the concrete class with fields gravity, gravity_field, fluid_field,
    mag_field, collision_field, dt, position, orientation, velocity,
    angular_velocity, velocity_body, R_craft_from_input,
    acceleration_world, acceleration_body, angular_acceleration."""
    __slots__ = ()

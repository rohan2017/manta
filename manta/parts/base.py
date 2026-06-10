"""Part base class + declaration sentinels.

A Part is a Python class that:

  * Declares its parameters at class scope using `Parameter(default)`.
  * Declares mutable per-tick state via `State(init=..., manifold=...)`.
  * Receives any per-tick external inputs via `Input(default)`.
  * Emits per-tick observables via `Output()`.
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
tick's symbolic node — the compiler binds them in a thread-local
`TraceBindings` context that attribute reads consult while a trace is
active; the instance itself is never mutated. State omitted from
`PartUpdate.new_state` passes through unchanged.

`_declarations()` walks the MRO so subclasses inherit Parameter/State/
Input/Output entries from their parents. Construction-time overrides
(`MyPart("p", magnitude=0.5)`) replace declared defaults.
"""

from __future__ import annotations

import threading
from typing import Any, ClassVar


# ---------------------------------------------------------------------------
# Trace bindings — how `self.<state/input/noise>` reads the current tick's
# symbol inside update() WITHOUT the compiler ever mutating the instance.
# ---------------------------------------------------------------------------

_trace_local = threading.local()


class TraceBindings:
    """Per-compile attribute bindings (a thread-local context manager).

    While active, a `DeclarationHost` attribute read resolves through these
    bindings instead of the instance: `self.throttle` inside `update()`
    returns the current tick's symbolic node, and the part/disturbance
    instance is NEVER mutated. Exception-safe by construction — the
    bindings live here, not on the objects, so there is nothing to restore
    when a trace unwinds (normally or via an exception).

    Craft rigid-body state symbols live here too (`bind_craft_state` /
    `craft_sym_state`), so a disturbance anchored to a craft (e.g.
    CraftWindBubble) can read that craft's symbolic position mid-trace
    without the compiler ever stashing anything on the Craft instance.
    """

    __slots__ = ("_by_owner", "_craft_state")

    def __init__(self) -> None:
        self._by_owner: dict[int, dict[str, Any]] = {}
        self._craft_state: dict[int, dict[str, Any]] = {}

    def bind(self, owner, name: str, symbol) -> None:
        """Bind `owner.<name>` to `symbol` for the duration of the trace."""
        self._by_owner.setdefault(id(owner), {})[name] = symbol

    def bind_craft_state(self, craft, syms: dict) -> None:
        """Bind a craft's rigid-body state symbols (compile Pass 0a)."""
        self._craft_state[id(craft)] = syms

    def craft_sym_state(self, craft) -> dict:
        """The bound rigid-body state symbols of `craft` for this trace."""
        try:
            return self._craft_state[id(craft)]
        except KeyError:
            raise RuntimeError(
                f"TraceBindings: craft '{craft.name}' has no symbolic state "
                f"bound — it is not part of the world being compiled.")

    def __enter__(self) -> "TraceBindings":
        if getattr(_trace_local, "active", None) is not None:
            raise RuntimeError(
                "TraceBindings: a trace is already active on this thread.")
        _trace_local.active = self
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        _trace_local.active = None


def active_trace() -> "TraceBindings | None":
    """The trace active on this thread, if any. Compile-time hook for code
    that runs inside a trace without a `trace` handle in scope (e.g. a
    disturbance's `contribute_at_sym`)."""
    return getattr(_trace_local, "active", None)


def declared_attr(owner, name: str, default=None):
    """`getattr(owner, name, default)` that bypasses any active trace
    bindings — always the configured (numeric) instance value. For
    compile-time numeric snapshots that must not pick up a promoted
    parameter's symbol (rest-pose joint inertia, mass guards)."""
    try:
        return object.__getattribute__(owner, name)
    except AttributeError:
        return default


def unit_axis(value, *, who: str, what: str) -> tuple:
    """Validate a length-3, nonzero axis parameter; return it normalized.

    Construction-time guard for parts whose math assumes a unit axis
    (joint DOF axes, airfoil chord/normal): a zero axis is a config
    error and a non-unit one would silently scale the physics."""
    try:
        axis = tuple(float(x) for x in value)
    except (TypeError, ValueError):
        raise ValueError(
            f"{who}: {what} must be a length-3 vector, got {value!r}")
    if len(axis) != 3:
        raise ValueError(
            f"{who}: {what} must be length-3, got {value!r}")
    n = (axis[0] ** 2 + axis[1] ** 2 + axis[2] ** 2) ** 0.5
    if n == 0.0:
        raise ValueError(f"{who}: {what} must be nonzero.")
    return (axis[0] / n, axis[1] / n, axis[2] / n)


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
    promotes it to an IR vector (`Vec3[F].constant` / `Vec3[F].coerce`).

    Args:
        manifold — optional `Manifold` instance or shortcut string
                   (``"R1"``, ``"R3"``; same vocabulary as `Noise`).
                   Declaring it makes the parameter *promotable*: a
                   transform constructed with `parameters=[...]` (system
                   identification — see `manta.fit`) can promote it from
                   a baked graph constant to a live graph input named
                   `<craft>.<part>.<param>`. Inside `update()` a promoted
                   parameter reads as an IR value (the trace binds it),
                   so parts consume promotable parameters through the
                   `.coerce` factory, which accepts both forms.
                   `None` (default) — a plain Python config value.
        frame    — Frame tag, consumed when `manifold` is a shortcut
                   resolving to a vector manifold. The promoted input's
                   frame; must match what `update()` composes it with.
    """

    __slots__ = ("manifold",)

    def __init__(self, default: Any, *, manifold=None, frame=None) -> None:
        super().__init__(default)
        if manifold is None:
            self.manifold = None
        else:
            from ..ir.manifold import Manifold, manifold_from_shortcut
            self.manifold = (manifold if isinstance(manifold, Manifold)
                             else manifold_from_shortcut(manifold,
                                                         frame=frame))


class Output(_Declaration):
    """Per-tick value produced by a part (sensor reading, derived quantity,
    telemetry signal).

    Declared at class scope. The part writes its computed value via
    `PartUpdate.outputs["<name>"] = <Vec3 | Scalar | …>`. The framework
    emits the value as a graph output named "<part_name>.<name>"; tick
    callers read it from the result dict (read-only, doesn't round-trip
    back as next-tick state). The output's shape is whatever the part
    writes — nothing downstream needs it declared.
    """

    __slots__ = ()

    def __init__(self) -> None:
        super().__init__(default=None)


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
                          `self.<name>`. Shortcuts: ``"R1"`` (scalar),
                          ``"R3"`` (combine with ``frame=``). Default
                          ``"R3"``. Same vocabulary as `State(manifold=)`.
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

    def __init__(self, signal_manifold="R3", *, frame=None,
                 sigma: float = 0.0) -> None:
        super().__init__(default=None)
        from ..ir.manifold import manifold_from_shortcut
        self.signal_manifold = manifold_from_shortcut(
            signal_manifold, frame=frame)
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

    # ---- Input-name classification (tick_signature.py / ekf.py) -------

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


class WhiteNoise(Noise):
    """Per-tick i.i.d. Gaussian noise channel. σ is the per-tick stddev."""

    kind              = "white"
    contributes_state = False

    __slots__ = ()

    def synthesize(self, *, base_name, name, dt, default_frame, owner):
        mfd = self.resolved_signal_manifold(default_frame=default_frame)
        return SynthesizedNoise(signal_sym=mfd.ir_input(base_name))


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
            return SynthesizedNoise(signal_sym=mfd.ir_zero())
        # Active: bias-state input + driver input + state update.
        driver_name = f"{base_name}_driver"
        sqrt_dt    = ca.sqrt(dt._mx)
        bias_sym   = mfd.ir_input(base_name)
        driver_sym = mfd.ir_input(driver_name)
        bias_next  = mfd.ir_add(bias_sym, sqrt_dt * driver_sym._mx)
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
                  For R1 a float; for R3 a length-3 tuple / ndarray; for
                  SO(3) a length-4 quaternion (w, x, y, z).
        manifold  String shortcut ('R1', 'R3') or a `Manifold` instance.
                  SO(3) state is fully supported — pass an explicit
                  `SO3Manifold(from_frame=..., to_frame=...)` instance
                  (the string shortcut is intentionally disallowed because
                  SO(3) needs the dual-frame parametrization). The slot
                  then evolves on the manifold: the part integrates it
                  with `manifold.boxplus(q, ω·dt)`, the framework keeps it
                  unit-normalized, and the EKF/LQR linearization gives it
                  a 3-dim tangent automatically. See `tests/test_so3_state`.
        frame     Frame tag for R3 state. Default `CraftFrame`. Ignored
                  for R1 and SO(3) (the latter's frames live on the
                  manifold). Folded into the Manifold instance.

    `state.manifold` always reads back as a `Manifold` instance; the
    string form is normalized at construction.
    """

    __slots__ = ("init", "manifold", "frame")

    def __init__(self, init, manifold="R1", frame=None) -> None:
        from ..ir.manifold import (
            Manifold, R3Manifold, SO3Manifold, manifold_from_shortcut,
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
    values for any declared State slots, any declared Output values the
    part produces, and the rates its I/O runs at.

    Construction::

        return PartUpdate(wrench, {"angle": a})
        return PartUpdate(wrench=w, new_state={"angle": a, "rate": r})
        return PartUpdate(wrench=w, outputs={"gyro": gyro_vec},
                          rates={"gyro": self.rate})

    `rates` maps this part's Output slots and/or Input attribute names to
    a rate in Hz (`None` ⇒ every tick). It is metadata only — the
    compiled tick stays a pure function (no sample-and-hold state enters
    the kernel, so it never complicates autodiff or the EKF/LQR
    linearization). The runtimes gate the matching *port*: an Output is
    published once per 1/rate window and held in between; an Input
    command is latched (ZOH) once per window, so truth and the
    estimator's predict see the *same* held command.

    Stateless parts can return a bare `Wrench` instead — the framework
    wraps it as `PartUpdate(wrench=w)` automatically.
    """

    __slots__ = ("wrench", "new_state", "outputs", "rates")

    def __init__(self,
                 wrench=None,
                 new_state: dict | None = None,
                 outputs: dict | None = None,
                 rates: dict | None = None) -> None:
        if wrench is None:
            raise TypeError("PartUpdate: wrench is required")
        self.wrench = wrench
        self.new_state = dict(new_state) if new_state else {}
        self.outputs   = dict(outputs)   if outputs   else {}
        self.rates     = dict(rates)     if rates     else {}


# ---------------------------------------------------------------------------
# Part base
# ---------------------------------------------------------------------------

class DeclarationHost:
    """Mixin: class-scope `_Declaration` resolution.

    A *declaration host* is any object that declares Parameter / State /
    Input / Output / Noise channels as class attributes and wants them
    resolved onto each instance at construction (overrides applied, Noise
    `<name>_sigma` attributes seeded). Both `Part` and `Disturbance`
    (`manta.fields.base`) are hosts and share this one implementation — so
    a change to the declaration protocol (a new channel kind, the
    sigma-routing rule) lives in exactly one place.

    The host must set `self.name` before calling `_apply_declarations`.
    """

    def __getattribute__(self, name):
        # Inside an active trace, declared attributes resolve through the
        # trace's bindings (the current tick's symbols); everything else —
        # and everything outside a trace — reads the instance as normal.
        tr = getattr(_trace_local, "active", None)
        if tr is not None:
            bound = tr._by_owner.get(id(self))
            if bound is not None:
                sym = bound.get(name)
                if sym is not None:
                    return sym
        return object.__getattribute__(self, name)

    @classmethod
    def _declarations(cls) -> dict[str, "_Declaration"]:
        """Walk MRO (most-derived overrides parent) returning the union of
        `_Declaration` class attributes by attribute name."""
        decls: dict[str, _Declaration] = {}
        # Walk in reverse-MRO so subclass entries overwrite parents.
        for klass in reversed(cls.__mro__):
            for name, value in vars(klass).items():
                if isinstance(value, _Declaration):
                    decls[name] = value
        return decls

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
            # as the seed in initial_state() and as the value the attribute
            # holds OUTSIDE of a trace. Inside a trace, the framework
            # rebinds it to the symbolic input node.
            setattr(self, attr_name, value)
            if isinstance(decl, Noise):
                sigma_key = f"{attr_name}_sigma"
                setattr(self, sigma_key,
                        float(overrides.get(sigma_key, decl.sigma)))

    # The declaration accessors are INSTANCE methods: the default set is
    # class-scoped, but a host whose I/O is decided per instance (e.g. a
    # camera pointed at compile-discovered targets) overrides these with
    # the same instance-call convention. Every framework call site holds
    # an instance.

    def state_declarations(self) -> dict[str, "State"]:
        """Just the State entries (subset of _declarations)."""
        return {n: d for n, d in type(self)._declarations().items()
                if isinstance(d, State)}

    def input_declarations(self) -> dict[str, "Input"]:
        """Just the Input entries (subset of _declarations)."""
        return {n: d for n, d in type(self)._declarations().items()
                if isinstance(d, Input)}

    def output_declarations(self) -> dict[str, "Output"]:
        """Just the Output entries (subset of _declarations)."""
        return {n: d for n, d in type(self)._declarations().items()
                if isinstance(d, Output)}

    def noise_declarations(self) -> dict[str, "Noise"]:
        """Just the Noise entries (subset of _declarations)."""
        return {n: d for n, d in type(self)._declarations().items()
                if isinstance(d, Noise)}

    def promotable_parameter_declarations(self) -> dict[str, "Parameter"]:
        """The Parameter entries that declared a manifold — the ones a
        transform may promote to live graph inputs for system ID."""
        return {n: d for n, d in type(self)._declarations().items()
                if isinstance(d, Parameter) and d.manifold is not None}

    def declared_value(self, name: str):
        """The instance's configured (numeric) value for `name`, bypassing
        any active trace bindings — compile-time framework code uses this
        for numeric guards/snapshots while `name` is bound to a symbol."""
        return object.__getattribute__(self, name)


class Part(DeclarationHost):
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
    (typically a CompositePart like the craft's RootPart or a joint)
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
    # Promotable (manifold="R3") — a mount position is a classic system-ID
    # target (thruster torque arms, sensor lever arms). The bound symbol's
    # coords are the parent's OUTPUT frame, exactly like the constant; the
    # kinematic/inertia passes consume it raw.
    transform: "tuple[float, float, float]" = Parameter((0.0, 0.0, 0.0),
                                                        manifold="R3")

    def __init__(self, name: str, **overrides: Any) -> None:
        from ..ir.module import check_name
        self.name = check_name(name, who=type(self).__name__)
        self.parent: "Part | None" = None
        self._apply_declarations(overrides)

    # --- Declaration resolution: inherited from DeclarationHost ------------

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

    def update(self, ctx):
        """Compute this part's wrench contribution for the current tick.
        `ctx` is the `manta.craft.TickContext`. Subclasses must override
        and return a `Wrench` or `PartUpdate`."""
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

    Children mount on this part's *output frame*. For a non-joint
    CompositePart the output frame is identical to the part's own
    frame (translation only, via `transform`). An `ArticulatedJoint`
    overrides this — a `RevoluteJoint`'s output frame additionally
    rotates by the joint angle (a `PrismaticJoint`'s translates by its
    displacement).

    `add(child)` appends a child Part, sets its `parent` to self, and
    returns the child (so chained construction reads naturally):

        gimbal = pan.add(RevoluteJoint("tilt", axis=(0, 1, 0)))
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
        subclasses (RootPart, joints, etc.) override if they need to."""
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

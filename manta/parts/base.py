"""Part base classes — the part tree.

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
`TraceBindings` context (`manta.parts._trace`) that attribute reads
consult while a trace is active; the instance itself is never mutated.
State omitted from `PartUpdate.new_state` passes through unchanged.

The declaration sentinels themselves (Parameter / State / Input / Output /
Noise, plus PartUpdate and validators) live in `manta.parts._declarations`;
this module hosts the classes that resolve and own them: the
`DeclarationHost` mixin and the part-tree (`Part`, `CompositePart`,
`RootPart`).

`_declarations()` walks the MRO so subclasses inherit Parameter/State/
Input/Output entries from their parents. Construction-time overrides
(`MyPart("p", magnitude=0.5)`) replace declared defaults.
"""

from __future__ import annotations

from typing import Any, ClassVar

from ._declarations import (
    Input, Noise, Output, Parameter, State, _Declaration,
)
from ._trace import _trace_local


# ---------------------------------------------------------------------------
# DeclarationHost — class-scope declaration resolution
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
    # Verified at `Sim(world)` (`LinearizedSystem._validate_model`).
    requires_planet: ClassVar[type | None] = None

    # Does this part contribute inertial mass to the craft's rigid-body
    # dynamics? The inertia walks (numpy aggregation, symbolic rollup,
    # joint-space kinetic energy, COM-relative motion) enumerate ONLY
    # parts with this trait set — a `mass` attribute on its own is NOT
    # inertial (e.g. TrajectoryEndpoint's `mass` is a feedforward gain).
    # Genuinely inertial parts (Mass) set this True and must declare
    # `mass` (kg) and may declare `moi` (diagonal, about own origin).
    contributes_inertia: ClassVar[bool] = False

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

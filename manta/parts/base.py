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
    from manta.ir.frames import PartFrame
    from manta.ir.types import Vec3

    class ConstantUpwardLift(Part):
        \"\"\"A toy Part: applies a fixed +z force in its own frame.\"\"\"
        magnitude: float = Parameter(1.0)

        def update(self, ctx):
            # Parts emit wrenches in their OWN frame (PartFrame); the
            # framework rotates them up through the mount pose.
            f = Vec3[PartFrame].constant((0.0, 0.0, self.magnitude))
            return Wrench(force=f,
                          torque=Vec3[PartFrame].constant((0, 0, 0)))

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

from enum import Enum
from typing import Any, ClassVar

import numpy as np

from .._validation import require_finite, require_positive

from ._declarations import (
    Input, Noise, Output, Parameter, State, _Declaration,
)
from ._trace import _trace_local


# ---------------------------------------------------------------------------
# DeclarationHost — class-scope declaration resolution
# ---------------------------------------------------------------------------


class PartRole(str, Enum):
    """Semantic role used when deriving purpose-specific model profiles.

    This describes why a part must remain in a compiled model, independently
    of the declarations it happens to expose.  In particular, an ``Output``
    does not make a part a sensor and an ``Input`` does not necessarily make
    it a vehicle actuator.  Reducers can therefore select parts explicitly
    without guessing from implementation details.
    """

    DYNAMICS = "dynamics"
    ACTUATOR = "actuator"
    SENSOR = "sensor"


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
        # Noise declarations expose per-instance runtime attributes
        # (`<name>_sigma`, and `<name>_tau` for Gauss–Markov channels).
        # Recognize override keys of those forms and route them to the
        # matching declaration without reporting "unknown".
        noise_runtime_keys = {
            attr for n, d in decls.items() if isinstance(d, Noise)
            for attr in d.runtime_attributes(n)
        }
        unknown = set(overrides) - set(decls) - noise_runtime_keys
        if unknown:
            raise TypeError(
                f"{type(self).__name__}({self.name!r}): unknown parameter(s) "
                f"{sorted(unknown)}. Declared: "
                f"{sorted(set(decls) | noise_runtime_keys)}")
        for attr_name, decl in decls.items():
            value = overrides.get(attr_name, decl.default)
            if isinstance(decl.default, str):
                if not isinstance(value, str):
                    raise TypeError(
                        f"{type(self).__name__}({self.name!r}).{attr_name} "
                        f"must be a string, got {type(value).__name__}")
            elif value is None:
                # ``Output`` and explicitly optional Parameters use None as
                # absence, not as numeric data.
                pass
            elif isinstance(decl, Parameter) and not decl.numeric:
                # Typed object/callable configuration is validated by the
                # owning Part, which knows its protocol.  The opt-out is
                # explicit on the declaration; arbitrary objects never
                # silently bypass numeric validation.
                pass
            elif not (isinstance(decl, Parameter) and decl.allow_infinite):
                require_finite(
                    value,
                    name=f"{type(self).__name__}({self.name!r}).{attr_name}",
                )
            elif np.any(np.isnan(np.asarray(value, dtype=float))):
                raise ValueError(
                    f"{type(self).__name__}({self.name!r}).{attr_name} "
                    f"must not be NaN, got {value!r}")
            # Plain attribute. For State, this is the init value used both
            # as the seed in initial_state() and as the value the attribute
            # holds OUTSIDE of a trace. Inside a trace, the framework
            # rebinds it to the symbolic input node.
            setattr(self, attr_name, value)
            if isinstance(decl, Noise):
                for key, (default, allow_zero) in \
                        decl.runtime_attributes(attr_name).items():
                    setattr(self, key, require_positive(
                        overrides.get(key, default),
                        name=f"{type(self).__name__}({self.name!r}).{key}",
                        allow_zero=allow_zero,
                    ))

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
        mount pose may promote to live graph inputs for system ID."""
        return {n: d for n, d in type(self)._declarations().items()
                if isinstance(d, Parameter) and d.manifold is not None}

    def declared_value(self, name: str):
        """The instance's configured (numeric) value for `name`, bypassing
        any active trace bindings — compile-time framework code uses this
        for numeric guards/snapshots while `name` is bound to a symbol.
        The single home of the `object.__getattribute__` bypass;
        `_trace.declared_attr` is the defaulting wrapper around it."""
        return object.__getattribute__(self, name)


class Part(DeclarationHost):
    """Base class for all parts.

    Subclasses declare their interface via class-attribute `Parameter`
    (and later `Input`/`State`) entries, then implement `update(ctx)` to
    contribute a `Wrench` per tick.

    Every Part has a static mount pose relative to its parent's output
    frame: `mount_offset`, an (x, y, z) position, and
    `mount_orientation`, a wxyz quaternion rotating the part's OWN axes
    into the parent's output frame.

    Both carry the `mount_` prefix deliberately. It separates the
    STATIC installation from the DYNAMIC kinematics that share the same
    vocabulary — `ctx.position` and `ctx.orientation` are what the part
    is doing this tick, `self.mount_offset` and `self.mount_orientation`
    are where it was bolted — and it leaves `orientation` free for the
    thing parts actually output (an AHRS reports one). The framework uses the pose to express the part's
    kinematics and to roll its wrench up into the parent's frame
    (force-at-offset → torque contribution at parent origin).

    The two compose in the order you would bolt the thing on: the
    offset is measured in the PARENT's frame ("put it here"), then the
    rotation turns the part in place ("pointing that way"). So a
    thruster canted 30° outboard is one `mount_orientation`, not a hand-
    rotated thrust vector, and a sensor mounted on its side reports in
    a frame that is genuinely rotated rather than one the measurement
    function has to correct after the fact.

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
             mount_offset=(0.0, 0.0, -0.5))        # 0.5 m below parent origin
    """

    # Fields the part requires registered on the World. Verified at
    # `Sim(world)`. Subclasses override with concrete Field subclasses
    # (e.g., `requires_fields = [GravityField]` for a Mass-like part).
    # Use Field BASE classes — the registered field may be any subclass.
    requires_fields: ClassVar[list[type]] = []

    # Planet subclass the part requires. None ⇒ no planet required.
    # Verified at `Sim(world)` (`LinearizedSystem._validate_model`).
    # Forward provision: no stock part sets it yet — the validation hook
    # exists so a planet-coupled part can declare its need declaratively.
    requires_planet: ClassVar[type | None] = None

    # Purpose-specific transforms (control, estimation, simulation) select
    # retained device parts from this explicit role.  Ordinary structural and
    # hydrodynamic parts are dynamics by default; physical I/O parts override
    # it at their defining class so subclasses inherit the correct meaning.
    role: ClassVar[PartRole] = PartRole.DYNAMICS

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
    mount_offset: "tuple[float, float, float]" = Parameter((0.0, 0.0, 0.0),
                                                        manifold="R3")

    # Static mount rotation, wxyz, part axes → parent output axes.
    # Named `mount_orientation`, not `orientation`: sensors output the
    # latter, and a base-class Parameter would collide with it.
    # Promotable on SO3: a mount MISALIGNMENT is one of the most
    # valuable things a sysid fit can recover — an IMU a couple of
    # degrees off its nominal frame biases every attitude solution, and
    # is otherwise almost impossible to measure in situ.
    mount_orientation: "tuple[float, float, float, float]" = Parameter(
        (1.0, 0.0, 0.0, 0.0), manifold="SO3")

    def __init__(self, name: str, **overrides: Any) -> None:
        from ..ir.module import check_name
        self.name = check_name(name, who=type(self).__name__)
        self.parent: "Part | None" = None
        self._apply_declarations(overrides)
        q = np.asarray(self.declared_value("mount_orientation"), dtype=float)
        if q.shape != (4,):
            raise ValueError(
                f"{type(self).__name__}({name!r}): mount_orientation must be a "
                f"wxyz quaternion (4 values); got {q.shape}")
        norm = float(np.linalg.norm(q))
        if not np.isfinite(norm) or norm < 1e-12:
            raise ValueError(
                f"{type(self).__name__}({name!r}): mount_orientation must be a "
                f"finite non-zero quaternion; got {tuple(q)}")
        if abs(norm - 1.0) > 1e-9:
            # Normalize once here rather than per tick: a rotation that
            # drifts off the unit sphere silently rescales every vector
            # it touches.
            setattr(self, "mount_orientation",
                    tuple(float(v) for v in q / norm))

    @property
    def mounted_upright(self) -> bool:
        """True when this part's static mount rotation is identity —
        the common case, and the fast path the kinematic pass takes."""
        q = np.asarray(self.declared_value("mount_orientation"), dtype=float)
        return bool(abs(abs(float(q[0])) - 1.0) < 1e-12)

    # --- Declaration resolution: inherited from DeclarationHost ------------

    def noise_R(self, name: str) -> Any:
        """Measurement-noise covariance for a declared Noise slot.

        Reads the per-instance `<name>_sigma` attribute (set at
        construction time, default from the declaration). Returns:
            * `σ²` (float) for scalar noise.
            * `σ²·I_d` (np.ndarray, d×d) for d-vector noise (sized
              off `signal_manifold.ambient_dim`).

        User-facing convenience: the framework's own EKF wiring reads
        `<name>_sigma` directly, but hand-driven filter code can size a
        measurement update without restating σ²:
            `ekf.update(h, z, R=imu.noise_R("gyro_noise"))`.
        """
        decls = self.noise_declarations()
        if name not in decls:
            raise KeyError(
                f"{type(self).__name__}({self.name!r}): no Noise slot "
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

    def on_world_resolve(self, world, craft) -> None:
        """Compile-time resolution hook — called once per part by
        snapshot resolution, after planets and field sources have
        registered their disturbances, with the world and this part's
        craft. The generic slot for anything a part can only do against
        the *finished* model:

          * resolve cross-craft wiring (a camera collecting the optical
            ellipsoids it can see),
          * validate structural invariants that need the complete part
            tree (a thermal link's same-craft check, a root-mount
            requirement)

        so a user-authored part gets the same compile-time treatment as
        the stock ones — no `isinstance` ladder in World resolution.
        Default: no-op. Raise to reject a bad configuration at
        resolution time (before any tracing) rather than mid-trace."""

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
    frame — which is its parent's output frame displaced by
    `mount_offset` and turned by `mount_orientation`. An `ArticulatedJoint`
    overrides this — a `RevoluteJoint`'s output frame additionally
    rotates by the joint angle (a `PrismaticJoint`'s translates by its
    displacement).

    `add(child)` appends a child Part, sets its `parent` to self, and
    returns the child (so chained construction reads naturally):

        gimbal = pan.add(RevoluteJoint("tilt", axis=(0, 1, 0)))
        gimbal.add(Mass("camera", mass=0.05, mount_offset=(0.1, 0, 0)))
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
        # Names must be unique across the WHOLE part tree, not just among
        # siblings: every lookup (state keys, tick signature, tether
        # endpoints, parameter resolution) is by name and silently takes
        # the first match — a duplicate two branches apart would alias
        # state or pick the wrong part with no error. Check here, where
        # the mistake is made, not in the compile that trips over it.
        root = self
        while root.parent is not None:
            root = root.parent
        existing = {p.name for p in root.walk()}
        incoming = ([p.name for p in child.walk()]
                    if isinstance(child, CompositePart) else [child.name])
        dup = existing.intersection(incoming)
        if dup:
            raise ValueError(
                f"{type(self).__name__}('{self.name}').add: part name(s) "
                f"{sorted(dup)} already exist in this part tree. Part "
                f"names must be unique per craft — every name lookup "
                f"(state, sensors, parameters) would silently alias.")
        child.parent = self
        self._children.append(child)
        return child

    @property
    def children(self) -> tuple[Part, ...]:
        return tuple(self._children)

    def remove(self, child: Part | str) -> Part:
        """Detach one direct child by instance or name and return it."""
        match = next(
            (candidate for candidate in self._children
             if candidate is child
             or (isinstance(child, str) and candidate.name == child)),
            None,
        )
        if match is None:
            label = child if isinstance(child, str) else getattr(child, "name", child)
            raise KeyError(
                f"{type(self).__name__}('{self.name}').remove: no direct "
                f"child {label!r}")
        self._children.remove(match)
        match.parent = None
        return match

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
        subclasses (RootPart, joints, etc.) override if they need to.
        PartFrame, like every part's update: the tick rolls each wrench
        up from the part's own frame, so a CraftFrame zero here would
        FrameError the compile for any non-root composite."""
        from ..ir.wrench import Wrench
        from ..ir.frames import PartFrame
        return Wrench.zero(PartFrame)


# ---------------------------------------------------------------------------
# RootPart — the implicit root of every Craft's part tree
# ---------------------------------------------------------------------------

class RootPart(CompositePart):
    """The root of a Craft's part tree. Defines CraftFrame at the body
    origin; all other parts are descendants. RootPart itself has no
    mass or inertia — the body's mass distribution comes from explicit
    Mass children added under it.
    """

    # RootPart's mount pose is necessarily identity: it IS the craft
    # frame. Override the inherited Parameters to lock both halves —
    # a rotated root would silently redefine what "body frame" means
    # for every state, sensor and wrench on the craft.
    mount_offset: "tuple[float, float, float]" = Parameter((0.0, 0.0, 0.0))
    mount_orientation: "tuple[float, float, float, float]" = Parameter(
        (1.0, 0.0, 0.0, 0.0))

    def __init__(self, name: str) -> None:
        super().__init__(name)

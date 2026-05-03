"""Core descriptor types: Craft, PartDescriptor, FieldDescriptor, StaticTransform.

These are the *base* classes user descriptors and stock descriptors inherit from.
Stock part/field descriptors live in `manta_codegen.parts` and
`manta_codegen.fields` respectively.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar, Iterator

from ._format import cpp_float as _f
from .signal import Binding, BoundSignal, CRAFT_SENTINEL, Signal


# Craft-level signals exposed on every Craft instance. These read off the
# craft's scene_to_craft kinematic state — the emitter substitutes "craft"
# (not "craft.<part>()") for `{accessor}` because part_name is the
# CRAFT_SENTINEL. Part signals and craft signals share the same Binding /
# emitter machinery; only the accessor differs.
_CRAFT_SIGNALS: list[Signal] = [
    Signal(
        name="position",
        direction="out",
        n_floats=3,
        cpp_read_exprs=(
            "{accessor}.scene_to_craft().position().raw()(0)",
            "{accessor}.scene_to_craft().position().raw()(1)",
            "{accessor}.scene_to_craft().position().raw()(2)",
        ),
    ),
    Signal(
        name="orientation",
        direction="out",
        n_floats=4,           # (w, x, y, z) quaternion order
        cpp_read_exprs=(
            "{accessor}.scene_to_craft().orientation().raw().w()",
            "{accessor}.scene_to_craft().orientation().raw().x()",
            "{accessor}.scene_to_craft().orientation().raw().y()",
            "{accessor}.scene_to_craft().orientation().raw().z()",
        ),
    ),
    Signal(
        name="vel_linear",
        direction="out",
        n_floats=3,
        cpp_read_exprs=(
            "{accessor}.scene_to_craft().vel_linear().raw()(0)",
            "{accessor}.scene_to_craft().vel_linear().raw()(1)",
            "{accessor}.scene_to_craft().vel_linear().raw()(2)",
        ),
    ),
    Signal(
        name="vel_angular",
        direction="out",
        n_floats=3,
        cpp_read_exprs=(
            "{accessor}.scene_to_craft().vel_angular().raw()(0)",
            "{accessor}.scene_to_craft().vel_angular().raw()(1)",
            "{accessor}.scene_to_craft().vel_angular().raw()(2)",
        ),
    ),
    # World-clock time (seconds since sim start). Reads through the craft's
    # World handle — convenient single-field signal for telemetry timestamps.
    Signal(
        name="time",
        direction="out",
        n_floats=1,
        cpp_read_exprs=("{accessor}.world().clock().time()",),
    ),
]


# ---------------------------------------------------------------------------
# Static transform helper

@dataclass(frozen=True)
class StaticTransform:
    """Static (parent → part) transform. Position in parent frame; quaternion is
    (w, x, y, z) representing the part's orientation in the parent frame."""
    position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    quaternion: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)

    def emit_cpp(self, scalar: str = "manta::Real") -> str:
        """C++ literal for `manta::geom::StaticLink<ParentFrame, PartFrame, Scalar>`."""
        x, y, z = self.position
        w, qx, qy, qz = self.quaternion
        quat_type = (
            "Eigen::Quaternionf" if scalar == "manta::Real"
            else f"Eigen::Quaternion<{scalar}>"
        )
        # Cast quaternion components through `scalar(...)` so the call works
        # for any Scalar (Real, double, ceres::Jet, ...).
        qexpr = (f"{quat_type}{{{scalar}({_f(w)}), {scalar}({_f(qx)}), "
                 f"{scalar}({_f(qy)}), {scalar}({_f(qz)})}}")
        return (
            f"manta::geom::StaticLink<manta::ParentFrame, manta::PartFrame, {scalar}>{{"
            f"manta::geom::Vec3<manta::ParentFrame, {scalar}>{{"
            f"{scalar}({_f(x)}), {scalar}({_f(y)}), {scalar}({_f(z)})}}, "
            f"manta::geom::Ori<manta::ParentFrame, {scalar}>{{{qexpr}}}"
            "}"
        )


def tf(position: tuple[float, float, float] = (0.0, 0.0, 0.0),
       quaternion: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)) -> StaticTransform:
    """Convenience constructor: `tf((x,y,z))` reads more naturally than `StaticTransform(...)`."""
    return StaticTransform(position=position, quaternion=quaternion)


# ---------------------------------------------------------------------------
# Field descriptor base

class FieldDescriptor:
    """Base class for field descriptors. Subclasses set `cpp_class`, `cpp_header`,
    and `feature_macro` (the name codegen will `#define` in `craft_config.h`).

    `register_as` lists additional C++ base class names the same instance
    should be registered under (in addition to its concrete `cpp_class`).
    The codegen emits `w.register_field<Base>(var)` for each entry. Use this
    when a Part (e.g. Hull) queries via the abstract base — the typeid-keyed
    registry doesn't walk inheritance, so polymorphic lookup needs explicit
    multi-slot registration.
    """
    cpp_class: ClassVar[str] = ""           # "manta::fields::GravityField"
    cpp_header: ClassVar[str] = ""          # "manta/fields/gravity_field.hpp"
    feature_macro: ClassVar[str] = ""       # "MANTA_HAS_GRAVITY_FIELD"
    register_as: ClassVar[list[str]] = []   # additional base slots

    def __init__(self, **construction_args) -> None:
        self.construction_args = construction_args
        # Cross-process disturbance replication. When True, codegen wires a
        # Zenoh publisher + subscriber on `sync_topic` (defaults to
        # `manta/<world>/<field_var>/disturbance`) and binds the Field's
        # tx_hook + receive() so add()s on one process appear on all peers.
        # Only stock-tagged disturbances replicate; user-defined lambdas
        # (tag = USER) stay local. See manta/fields/gravity_field.hpp for
        # the wire layout.
        self.synchronized: bool = False
        self.sync_topic: str | None = None  # default derived in emit_main

    def emit_construction(self, var_name: str) -> str:
        """Returns a C++ statement that constructs the field instance into `var_name`.
        Subclasses override when their constructors take typed args.
        """
        return f"{self.cpp_class} {var_name}{{}};"

    def emit_extra_setup(self, var_name: str) -> list[str]:
        """Optional follow-up C++ statements after construction — e.g. add()
        calls for the new disturbance API. One statement per list entry; the
        emitter handles indentation. Default is none.
        """
        return []


# ---------------------------------------------------------------------------
# Planet descriptor base

class PlanetDescriptor:
    """Base class for planet descriptors. A Planet, when added to a craft via
    `craft.planets.append(...)`, gets emitted as `world.add_planet<CppClass>(...)`
    in the codegenerated main, and the world's first scene is anchored to it
    via `scene.set_planet(&planet)`.

    Subclasses set `cpp_class` and `cpp_header`, override `emit_constructor_args`
    if the planet takes typed arguments.
    """
    cpp_class:  ClassVar[str] = ""
    cpp_header: ClassVar[str] = ""

    def emit_constructor_args(self) -> str:
        """C++ argument list inside `world.add_planet<CppClass>(...)`. Default empty."""
        return ""


# ---------------------------------------------------------------------------
# Part descriptor base

class PartDescriptor:
    """Base for part descriptors. Subclass and set `cpp_class` + `cpp_header`,
    optionally declare `requires_fields`, and override the emit hooks.

    Subclasses are real Python classes — they may have factory methods, typed
    arguments, docstrings, computed properties. The codegen calls their methods
    to produce C++ snippets and (eventually) viewer rendering.
    """

    cpp_class:  ClassVar[str] = ""
    cpp_header: ClassVar[str] = ""
    # If non-empty, this is the C++ class TEMPLATE (e.g. `manta::parts::PointMassT`)
    # that the part should be instantiated through when generating a Scalar-
    # templated craft. The codegen emits `cpp_class_template<Scalar>` so the
    # template MUST take Scalar as its single parameter — parts that need
    # multiple variants (e.g. force-tensor count) should expose them as
    # separate descriptors backed by separate concrete C++ classes (e.g.
    # `Surface1T<Scalar>`, `Surface2T<Scalar>`), not as extra template args.
    # If empty, this part is not yet Scalar-templated and the codegen falls
    # back to the bare cpp_class (which forces Real). When ALL parts in a
    # craft have cpp_class_template set, the codegen emits the craft as a
    # class template; otherwise it emits the non-templated form.
    cpp_class_template: ClassVar[str] = ""
    # List of FieldDescriptor subclasses this part requires. Codegen verifies
    # all of these are registered with the craft and emits `static_assert`s on
    # the corresponding feature macros.
    requires_fields: ClassVar[list[type[FieldDescriptor]]] = []
    # The PlanetDescriptor subclass this part requires (if any). Codegen
    # validates that the craft has a planet of this type registered. The
    # part's update() can then call `craft().planet<CppPlanetClass>()`.
    requires_planet: ClassVar[type | None] = None
    # Bindable signals exposed by this part. Each Signal becomes a same-named
    # attribute on the instance after construction, returning a BoundSignal
    # the user passes to Craft.publish/subscribe. Subclasses override; default
    # is empty (the part exposes no publishable/subscribable values).
    signals: ClassVar[list[Signal]] = []

    def __init__(self,
                 name: str,
                 transform: StaticTransform | None = None,
                 static: bool = False) -> None:
        if not name.isidentifier():
            raise ValueError(f"Part name {name!r} must be a valid C++ identifier")
        self.name = name
        self.transform = transform or StaticTransform()
        self.static = static
        # Filled by Craft.add() / CompositePartRef.add()
        self._children: list[PartDescriptor] = []
        # Set by `_stamp_craft_ref` when the part is attached to a craft.
        # Used to propagate the craft reference to children added after
        # this part has been hooked up (e.g. `motor.add(flywheel)` after
        # `c.add(motor)`).
        self._craft: object = None
        # Attach a BoundSignal for each declared signal. Names are pre-validated
        # at signal-list-declaration time (Signal.__post_init__).
        for sig in type(self).signals:
            if hasattr(self, sig.name):
                raise TypeError(
                    f"Part {type(self).__name__}: signal {sig.name!r} collides "
                    f"with an existing attribute on the descriptor")
            object.__setattr__(self, sig.name, BoundSignal(part_name=name, signal=sig))

    # ---- subclass override hooks ----
    #
    # The `scalar` argument is the C++ type name to substitute for the part's
    # numeric scalar. Default `"manta::Real"` produces the original (Real-typed)
    # output; the templated codegen passes `"Scalar"` to generate code valid
    # inside a `template <class Scalar> class ...` body.

    def emit_constructor_args(self, scalar: str = "manta::Real") -> str:
        """C++ argument list for `parent.add<CppClass>(...)`. Subclasses override.
        The emitter wraps the result so the user only writes the args."""
        return f'"{self.name}"'

    def emit_post_construction(self, scalar: str = "manta::Real") -> list[str]:
        """C++ statements to run after the part is constructed and added.
        Variable `<name>_` (a pointer to the part) is in scope. Default: emit
        a `set_transform` call when the transform is not identity.
        Subclasses extending this should `return [*super().emit_post_construction(scalar), ...]`.
        """
        out: list[str] = []
        if (self.transform.position != (0.0, 0.0, 0.0)
                or self.transform.quaternion != (1.0, 0.0, 0.0, 0.0)):
            out.append(f"{self.name}_->set_transform({self.transform.emit_cpp(scalar)});")
        return out

    def telemetry_fields(self) -> list[tuple[str, str]]:
        """List of (member_name, cpp_type) the per-craft Telemetry struct should
        carry for this part. Default: empty (no telemetry)."""
        return []

    def emit_telemetry_reads(self) -> list[tuple[str, str]]:
        """Returns (member, cpp_expression) pairs that populate the telemetry
        sub-struct. Expressions should reference the part via `craft.<name>()`.

        Example for Thruster:
            [("throttle", f"craft.{self.name}().throttle()")]

        The codegen wraps each pair as `telem.<part>.<member> = <expr>;`.
        Default: empty list (matches `telemetry_fields()` returning empty)."""
        return []

    def render(self, telemetry: dict, path: str) -> None:
        """Log this part's geometry/state to rerun under `path`. Default: a small
        labeled coordinate axis. Subclasses override."""
        # Imported lazily so codegen-only environments don't need rerun installed.
        try:
            import rerun as rr
        except ImportError:
            return
        rr.log(path, rr.Transform3D(axis_length=0.1))

    # ---- tree helpers ----

    def add(self, child: "PartDescriptor") -> "PartDescriptor":
        """Attach a child part. Returns the child so chaining is convenient."""
        self._children.append(child)
        # Propagate this parent's craft ref to the child so World.publish /
        # subscribe can resolve the right craft accessor for nested signals
        # (e.g. a Mass attached under a Motor).
        if self._craft is not None:
            _stamp_craft_ref(child, self._craft)
        return child

    def walk(self) -> Iterator["PartDescriptor"]:
        """Pre-order traversal of this subtree (yields self first)."""
        yield self
        for c in self._children:
            yield from c.walk()


# ---------------------------------------------------------------------------
# Craft

class Craft:
    """Top-level craft spec. Construct one in your Python, build out a part tree
    via `craft.root.add(...)`, and pass it to `emit(...)`.

    Args:
        name: identifier, used as the C++ class name and Zenoh topic prefix.
        fields: list of FieldDescriptor instances registered with this craft.
        topic_prefix: prefix for autogenerated Zenoh topics.
                      Default: `manta/<name>`.
    """

    def __init__(self,
                 name: str,
                 topic_prefix: str | None = None) -> None:
        if not name.isidentifier():
            raise ValueError(f"Craft name {name!r} must be a valid C++ identifier")
        self.name = name
        self.topic_prefix = topic_prefix if topic_prefix is not None else f"manta/{name}"
        self.root: _RootProxy = _RootProxy(craft=self)
        # When True, codegen emits the craft as a class template
        # (`template <class Scalar = manta::Real> class FooCraftT : public
        # CraftT<Scalar>`) plus a `using FooCraft = FooCraftT<manta::Real>`
        # alias. Required for use with `manta::estimation::CraftEKF`. All
        # parts in the craft must have `cpp_class_template` set; codegen
        # raises if any part lacks it.
        self.scalar_templated: bool = False
        # Craft-level signals (pose + velocity) exposed as attributes for
        # symmetry with `imu.last_accel`-style part signal access. Each is a
        # BoundSignal pointing at the craft itself (part_name = CRAFT_SENTINEL);
        # the emitter recognizes the sentinel and substitutes plain `craft`
        # for the accessor. `craft_ref` is set to `self` so World-level
        # binding APIs can disambiguate which craft each signal belongs to.
        for sig in _CRAFT_SIGNALS:
            object.__setattr__(self, sig.name,
                               BoundSignal(part_name=CRAFT_SENTINEL, signal=sig,
                                           craft_ref=self))

    # Bindings live on the World now (since they can connect signals across
    # crafts and to user-declared world-level signal slots). Use
    # `world.publish(...)`, `world.subscribe(...)`, and `world.connect(...)`.

    def add(self, child: PartDescriptor) -> PartDescriptor:
        """Convenience shortcut for `c.root.add(child)`. Returns the child so
        chained-style still works, but the canonical pattern is
        construct-first-then-attach:

            imu = IMU("imu")
            c.add(imu)
            c.publish(imu.last_accel)

        Calls to `c.root.add(...)` continue to work for explicit composition
        when the user wants to nest below an articulated part — `motor.add(p)`
        for a child of a Motor's joint output, for example.
        """
        return self.root.add(child)

    # ---- iteration helpers used by emitters ----

    def all_parts(self) -> Iterator[PartDescriptor]:
        for p in self.root.children:
            yield from p.walk()


# ---------------------------------------------------------------------------
# World — top-level simulation container (phase 3 of the API redesign).

@dataclass
class _CraftEntry:
    craft: "Craft"
    on: "PlanetDescriptor | None" = None
    position:    tuple[float, float, float]                = (0.0, 0.0, 0.0)
    orientation: tuple[float, float, float, float]         = (1.0, 0.0, 0.0, 0.0)
    vel_linear:  tuple[float, float, float]                = (0.0, 0.0, 0.0)
    vel_angular: tuple[float, float, float]                = (0.0, 0.0, 0.0)


@dataclass
class Tether:
    """Spring-damper tether between two TetherEndpoints. Lives at world level
    (the tether spans crafts), so it's owned by World rather than by a Craft.
    Use World.add_tether() to register it; codegen emits the C++
    `manta::coupling::Tether` instance + the two endpoint hookups in main.cpp.
    """
    rest_length: float
    stiffness:   float
    damping:     float = 0.0


@dataclass
class _TetherEntry:
    tether: Tether
    # (craft, endpoint_part_name) for each end. The endpoint Part is added to
    # the craft AT MAIN-TIME (after construct), not in the per-Craft .cpp.
    endpoint_a: tuple["Craft", str]
    endpoint_b: tuple["Craft", str]


class World:
    """The simulation top-level. Holds fields, planets, crafts, and the sim
    loop config (dt, rate multiplier). Each craft is added with its own
    initial state in a chosen frame (`on=planet` for a planet-frame anchor,
    or default to WorldFrame).

    Typical usage:

        earth = Earth()
        drone = make_drone()        # a Craft

        w = World()
        w.add_planet(earth)
        w.add_craft(drone, on=earth, pos=(0, 0, 100.0))
        w.run(dt=0.001, sim_rate_mult=1.0)

    """

    def __init__(self, name: str | None = None) -> None:
        # World name. Defaults to the first craft's name at codegen time;
        # supplying it explicitly is required for multi-craft worlds (where
        # the binary's name shouldn't be tied to one of the crafts).
        self._explicit_name = name
        self.fields: list[FieldDescriptor] = []
        self.planets: list[PlanetDescriptor] = []
        self.crafts: list[_CraftEntry] = []
        self.tethers: list[_TetherEntry] = []
        # World-level pub/sub bindings. Each Binding holds a struct of
        # BoundSignal members (single-signal bindings are the degenerate
        # one-member case). Codegen iterates these to emit one Zenoh
        # publisher/subscriber per Binding. The BoundSignal's `craft_ref`
        # tells the emitter which craft owns each signal.
        self.bindings: list[Binding] = []
        self.dt: float = 0.001
        self.sim_rate_mult: float = 1.0

    @property
    def name(self) -> str:
        if self._explicit_name is not None:
            return self._explicit_name
        if not self.crafts:
            raise RuntimeError("World.name: no explicit name and no crafts registered")
        return self.crafts[0].craft.name

    def add_field(self, f: FieldDescriptor) -> "World":
        self.fields.append(f)
        return self

    # ---- pub/sub binding API ----

    def publish(self,
                what: BoundSignal | dict[str, BoundSignal],
                topic: str | None = None,
                protocol: str = "zenoh",
                encoding: str = "json") -> "World":
        """Register an outgoing binding from a craft signal (or bundled struct)
        to a topic. `what` is either a single BoundSignal or a
        `dict[str, BoundSignal]` for a bundled-struct payload. All members
        must have direction='out'."""
        members = self._normalize_binding_input(what, kind="publish")
        topic   = self._resolve_topic(topic, members, what)
        self.bindings.append(Binding(members=members, topic=topic,
                                     protocol=protocol, encoding=encoding))
        return self

    def subscribe(self,
                  what: BoundSignal | dict[str, BoundSignal],
                  topic: str | None = None,
                  protocol: str = "zenoh",
                  encoding: str = "json") -> "World":
        """Register an incoming binding from a topic to a craft signal.
        Symmetric to publish(); all members must have direction='in'."""
        members = self._normalize_binding_input(what, kind="subscribe")
        topic   = self._resolve_topic(topic, members, what)
        self.bindings.append(Binding(members=members, topic=topic,
                                     protocol=protocol, encoding=encoding))
        return self

    def _normalize_binding_input(self,
                                 what: BoundSignal | dict[str, BoundSignal],
                                 kind: str) -> dict[str, BoundSignal]:
        expect = "out" if kind == "publish" else "in"
        opposite = "subscribe" if kind == "publish" else "publish"
        if isinstance(what, BoundSignal):
            members = {what.name: what}
        elif isinstance(what, dict):
            if not what:
                raise ValueError(f"World.{kind}: empty struct dict")
            for k, v in what.items():
                if not isinstance(k, str) or not k.isidentifier():
                    raise ValueError(
                        f"World.{kind}: struct member name {k!r} must be a valid identifier")
                if not isinstance(v, BoundSignal):
                    raise TypeError(
                        f"World.{kind}: struct member {k!r} must be a BoundSignal, "
                        f"got {type(v).__name__}")
            members = dict(what)
        else:
            raise TypeError(
                f"World.{kind}: expected BoundSignal or dict[str, BoundSignal], "
                f"got {type(what).__name__}")
        bad = [(k, v) for k, v in members.items() if v.direction != expect]
        if bad:
            names = ", ".join(f"{k!r}" for k, _ in bad)
            raise ValueError(
                f"World.{kind}: member(s) {names} have direction!={expect!r}; "
                f"use World.{opposite} instead")
        # Every signal must point at a known craft.
        for k, v in members.items():
            if v.craft_ref is None:
                raise ValueError(
                    f"World.{kind}: signal {k!r} is not attached to any craft "
                    f"(did you forget to call `c.add(part)` before binding?)")
        return members

    def _resolve_topic(self,
                       topic: str | None,
                       members: dict[str, BoundSignal],
                       what: BoundSignal | dict[str, BoundSignal]) -> str:
        if topic is not None:
            return topic
        if isinstance(what, BoundSignal):
            craft = what.craft_ref
            prefix = craft.topic_prefix
            return f"{prefix}/{what.part_name}/{what.name}"
        raise ValueError(
            "World.publish/subscribe: a multi-member struct binding needs an "
            "explicit `topic` argument (no good default for bundled structs)")

    def add_planet(self, p: PlanetDescriptor) -> "World":
        self.planets.append(p)
        return self

    def add_craft(self,
                  craft: "Craft",
                  on: PlanetDescriptor | None = None,
                  pos:         tuple[float, float, float]                = (0.0, 0.0, 0.0),
                  ori:         tuple[float, float, float, float]         = (1.0, 0.0, 0.0, 0.0),
                  vel:         tuple[float, float, float]                = (0.0, 0.0, 0.0),
                  vel_angular: tuple[float, float, float]                = (0.0, 0.0, 0.0)) -> "World":
        """Attach a craft with its initial state in the chosen reference frame.

        `on=None` (default) → pos/vel are in WorldFrame (= scene frame when
        no planet anchor is set).
        `on=planet` → pos/vel are in the planet's PlanetFrame, and the
        scene anchors to that planet so the craft co-rotates with it.

        For now the implementation assumes one scene per World, with
        planet_to_scene = identity; multi-scene / scene-rebasing comes later.
        """
        if on is not None and on not in self.planets:
            raise ValueError(
                f"World.add_craft: planet {on!r} is not registered with this World; "
                f"call world.add_planet(planet) first")
        self.crafts.append(_CraftEntry(
            craft=craft, on=on,
            position=tuple(float(x) for x in pos),                 # type: ignore[arg-type]
            orientation=tuple(float(x) for x in ori),              # type: ignore[arg-type]
            vel_linear=tuple(float(x) for x in vel),               # type: ignore[arg-type]
            vel_angular=tuple(float(x) for x in vel_angular),      # type: ignore[arg-type]
        ))
        return self

    def add_tether(self,
                   tether: Tether,
                   endpoint_a: tuple["Craft", str],
                   endpoint_b: tuple["Craft", str]) -> "World":
        """Register a Tether between two crafts. Each endpoint is a
        (craft, name) tuple — codegen emits a TetherEndpoint Part with the
        given name onto each craft's root after construction, then wires
        the endpoints to the shared Tether instance.

        Same-craft tethers are allowed: pass the same Craft twice with
        distinct endpoint names.
        """
        for slot, ep in (("endpoint_a", endpoint_a), ("endpoint_b", endpoint_b)):
            if not (isinstance(ep, tuple) and len(ep) == 2):
                raise TypeError(
                    f"World.add_tether: {slot} must be (craft, name) tuple")
            craft, name = ep
            if not isinstance(craft, Craft):
                raise TypeError(
                    f"World.add_tether: {slot} craft must be a Craft")
            if not (isinstance(name, str) and name.isidentifier()):
                raise ValueError(
                    f"World.add_tether: {slot} name {name!r} must be a valid identifier")
            if not any(e.craft is craft for e in self.crafts):
                raise ValueError(
                    f"World.add_tether: {slot} craft is not registered with this World; "
                    f"call world.add_craft(c) first")
        self.tethers.append(_TetherEntry(tether, endpoint_a, endpoint_b))
        return self

    def run(self, dt: float = 0.001, sim_rate_mult: float = 1.0) -> "World":
        """Configure the sim loop:
            dt              : sim step (seconds)
            sim_rate_mult   : ratio of sim seconds to wall seconds (1.0 = realtime)
        """
        self.dt = float(dt)
        self.sim_rate_mult = float(sim_rate_mult)
        return self


class _RootProxy:
    """The `craft.root` handle. Holds direct children; doesn't itself emit C++
    (the C++ root is `Craft::root()`)."""

    def __init__(self, craft: Craft) -> None:
        self._craft = craft
        self.children: list[PartDescriptor] = []

    def add(self, child: PartDescriptor) -> PartDescriptor:
        # Stamp the craft reference on every BoundSignal hanging off this
        # part so World.publish/subscribe/connect can resolve which craft
        # the signal belongs to in multi-craft worlds.
        _stamp_craft_ref(child, self._craft)
        self.children.append(child)
        return child


def _stamp_craft_ref(part: PartDescriptor, craft: "Craft") -> None:
    """Walk a PartDescriptor's class-level signals table and set craft_ref
    on each BoundSignal attribute. Also stamps the part's own `_craft`
    field so any later `part.add(child)` call propagates the right ref.
    Recurses into already-attached children."""
    part._craft = craft
    for sig in getattr(part, "signals", []) or []:
        bs = getattr(part, sig.name, None)
        if isinstance(bs, BoundSignal):
            bs.craft_ref = craft
    for child in getattr(part, "_children", []) or []:
        _stamp_craft_ref(child, craft)

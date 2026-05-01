"""Core descriptor types: Craft, PartDescriptor, FieldDescriptor, StaticTransform.

These are the *base* classes user descriptors and stock descriptors inherit from.
Stock part/field descriptors live in `manta_codegen.parts` and
`manta_codegen.fields` respectively.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar, Iterator

from ._format import cpp_float as _f
from .signal import Binding, BoundSignal, Signal


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

    def emit_construction(self, var_name: str) -> str:
        """Returns a C++ statement that constructs the field instance into `var_name`.
        Subclasses override when their constructors take typed args.
        """
        return f"{self.cpp_class} {var_name}{{}};"


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
                 static: bool = False,
                 publish_state: bool = False,
                 state_topic: str | None = None,
                 subscribe_command: bool = False,
                 command_topic: str | None = None) -> None:
        if not name.isidentifier():
            raise ValueError(f"Part name {name!r} must be a valid C++ identifier")
        self.name = name
        self.transform = transform or StaticTransform()
        self.static = static
        # Legacy flag-based pub/sub. Phase 2 of the API redesign replaces these
        # with explicit Craft.publish(BoundSignal, topic) / subscribe calls;
        # both paths coexist until all examples migrate, then these go away.
        self.publish_state = publish_state
        self.state_topic = state_topic
        self.subscribe_command = subscribe_command
        self.command_topic = command_topic
        # Filled by Craft.add() / CompositePartRef.add()
        self._children: list[PartDescriptor] = []
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

    def emit_command_apply(self, part_accessor: str, payload_var: str) -> str:
        """C++ statements that apply a parsed command (`payload_var`, a
        `std::vector<float>` known to be non-empty) to this part. The part is
        accessed via `part_accessor` (e.g. `craft.motor_0()`). Default: empty."""
        return ""

    def emit_measurement_decode(self, part_accessor: str, payload_var: str) -> str:
        """C++ statements that parse `payload_var` (a `std::vector<float>`) as
        a measurement payload and call this part's `set_measurement(...)`.
        Used by the real_data workflow to emit Zenoh-driven estimator mains.

        Default: empty (non-sensor parts have no measurement input).
        Sensor part descriptors override; e.g. IMU emits a 6-float decode,
        DVL emits a 3-float decode.
        """
        return ""

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
                 fields: list[FieldDescriptor] | None = None,
                 planets: list[PlanetDescriptor] | None = None,
                 topic_prefix: str | None = None) -> None:
        if not name.isidentifier():
            raise ValueError(f"Craft name {name!r} must be a valid C++ identifier")
        self.name = name
        self.fields: list[FieldDescriptor] = fields or []
        # Planets registered on the World. The first scene the codegen
        # creates is anchored to the first planet here (if any).
        self.planets: list[PlanetDescriptor] = planets or []
        self.topic_prefix = topic_prefix if topic_prefix is not None else f"manta/{name}"
        self.root: _RootProxy = _RootProxy(craft=self)
        # Initial state is applied at scene-add time, not in the Craft's
        # constructor. Stored here so the codegen can emit it at the
        # `scene.add_craft(...)` call site. Default: identity (at origin, at rest).
        self._initial_state: dict[str, tuple] = {}
        # Sim-loop configuration used by the binary workflow's main emitter.
        self.dt: float = 0.001            # 1 kHz default
        self.sim_rate_mult: float = 1.0   # 1× wall-time
        # When True, codegen emits the craft as a class template
        # (`template <class Scalar = manta::Real> class FooCraftT : public
        # CraftT<Scalar>`) plus a `using FooCraft = FooCraftT<manta::Real>`
        # alias. Required for use with `manta::estimation::CraftEKF`. All
        # parts in the craft must have `cpp_class_template` set; codegen
        # raises if any part lacks it.
        self.scalar_templated: bool = False
        # Explicit pub/sub bindings (phase 2 of the API redesign). Each entry
        # is a Binding holding {member_name: BoundSignal}, a topic, a protocol,
        # and an encoding. Single-signal and bundled-struct bindings are the
        # same primitive; the single-signal case is just a one-member dict.
        # Codegen iterates these to emit one publisher / subscriber per
        # Binding; replaces the older publish_state / subscribe_command
        # flag-based path.
        self.bindings: list[Binding] = []

    def initial_state(self,
                      position:    tuple[float, float, float] = (0.0, 0.0, 0.0),
                      orientation: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
                      vel_linear:  tuple[float, float, float] = (0.0, 0.0, 0.0),
                      vel_angular: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> "Craft":
        """Set initial conditions applied when the craft joins a scene.

        Quaternion is (w, x, y, z). Returns self so calls chain.
        """
        self._initial_state = {
            "position":    tuple(float(x) for x in position),
            "orientation": tuple(float(x) for x in orientation),
            "vel_linear":  tuple(float(x) for x in vel_linear),
            "vel_angular": tuple(float(x) for x in vel_angular),
        }
        return self

    def has_initial_state(self) -> bool:
        return bool(self._initial_state)

    def sim_config(self, dt: float = 0.001, sim_rate_mult: float = 1.0) -> "Craft":
        """Configure the binary-workflow sim main:
            dt              : sim step (seconds) — passed to World::clock().set_dt(dt).
            sim_rate_mult   : ratio of sim seconds to wall seconds. 1.0 = realtime;
                              200.0 = 200x realtime (useful for orbit / long-horizon sims).
        Returns self so calls chain.
        """
        self.dt = float(dt)
        self.sim_rate_mult = float(sim_rate_mult)
        return self

    def emit_initial_state_cpp(self) -> str:
        """C++ literal for `manta::InitialState{...}`. Returns `manta::InitialState{}`
        when no initial state was set (the natural identity)."""
        s = self._initial_state
        if not s:
            return "manta::InitialState{}"
        px, py, pz = s["position"]
        ow, ox, oy, oz = s["orientation"]
        vx, vy, vz = s["vel_linear"]
        wx, wy, wz = s["vel_angular"]
        return (
            "manta::InitialState{"
            f"manta::geom::Vec3<manta::SceneFrame>{{{_f(px)}, {_f(py)}, {_f(pz)}}}, "
            f"manta::geom::Ori<manta::SceneFrame>{{Eigen::Quaternionf{{{_f(ow)}, {_f(ox)}, {_f(oy)}, {_f(oz)}}}}}, "
            f"manta::geom::Vec3<manta::SceneFrame>{{{_f(vx)}, {_f(vy)}, {_f(vz)}}}, "
            f"manta::geom::Vec3<manta::CraftFrame>{{{_f(wx)}, {_f(wy)}, {_f(wz)}}}"
            "}"
        )

    # ---- pub/sub binding API (phase 2) ----

    def publish(self,
                what: BoundSignal | dict[str, BoundSignal],
                topic: str | None = None,
                protocol: str = "zenoh",
                encoding: str = "json") -> "Craft":
        """Register an outgoing binding.

        `what` is either:
          * A single BoundSignal — wrapped to {signal.name: signal} and given
            a default topic of `f"{topic_prefix}/{part_name}/{signal_name}"`
            if `topic` is None.
          * A dict[str, BoundSignal] — the keys become struct member names in
            the published payload. `topic` is required; no good default.

        All members (one or many) must have direction='out'.
        """
        members = self._normalize_binding_input(what, kind="publish")
        topic   = self._resolve_topic(topic, members, what)
        self.bindings.append(Binding(members=members, topic=topic,
                                     protocol=protocol, encoding=encoding))
        return self

    def subscribe(self,
                  what: BoundSignal | dict[str, BoundSignal],
                  topic: str | None = None,
                  protocol: str = "zenoh",
                  encoding: str = "json") -> "Craft":
        """Register an incoming binding. Symmetric to publish(); all members
        must have direction='in'."""
        members = self._normalize_binding_input(what, kind="subscribe")
        topic   = self._resolve_topic(topic, members, what)
        self.bindings.append(Binding(members=members, topic=topic,
                                     protocol=protocol, encoding=encoding))
        return self

    def _normalize_binding_input(self,
                                 what: BoundSignal | dict[str, BoundSignal],
                                 kind: str) -> dict[str, BoundSignal]:
        expect = "out" if kind == "publish" else "in"
        opposite_method = "subscribe" if kind == "publish" else "publish"
        if isinstance(what, BoundSignal):
            members = {what.name: what}
        elif isinstance(what, dict):
            if not what:
                raise ValueError(f"Craft.{kind}: empty struct dict")
            for k, v in what.items():
                if not isinstance(k, str) or not k.isidentifier():
                    raise ValueError(
                        f"Craft.{kind}: struct member name {k!r} must be a valid identifier")
                if not isinstance(v, BoundSignal):
                    raise TypeError(
                        f"Craft.{kind}: struct member {k!r} must be a BoundSignal, got {type(v).__name__}")
            members = dict(what)
        else:
            raise TypeError(
                f"Craft.{kind}: expected BoundSignal or dict[str, BoundSignal], got {type(what).__name__}")
        bad = [(k, v) for k, v in members.items() if v.direction != expect]
        if bad:
            names = ", ".join(f"{k!r}" for k, _ in bad)
            raise ValueError(
                f"Craft.{kind}: member(s) {names} have direction!='{expect}'; "
                f"use Craft.{opposite_method} instead (or split the struct)")
        return members

    def _resolve_topic(self,
                       topic: str | None,
                       members: dict[str, BoundSignal],
                       what: BoundSignal | dict[str, BoundSignal]) -> str:
        if topic is not None:
            return topic
        # Default topic only makes sense for a single-signal binding — there's
        # no good default for a multi-member struct.
        if isinstance(what, BoundSignal):
            return f"{self.topic_prefix}/{what.part_name}/{what.name}"
        raise ValueError(
            "Craft.publish/subscribe: a multi-member struct binding needs an "
            "explicit `topic` argument (no good default for bundled structs)")

    # ---- iteration helpers used by emitters ----

    def all_parts(self) -> Iterator[PartDescriptor]:
        for p in self.root.children:
            yield from p.walk()

    def field_descriptors(self) -> list[FieldDescriptor]:
        return list(self.fields)


class _RootProxy:
    """The `craft.root` handle. Holds direct children; doesn't itself emit C++
    (the C++ root is `Craft::root()`)."""

    def __init__(self, craft: Craft) -> None:
        self._craft = craft
        self.children: list[PartDescriptor] = []

    def add(self, child: PartDescriptor) -> PartDescriptor:
        self.children.append(child)
        return child

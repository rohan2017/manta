"""World — top-level simulation container.

A World holds:
  * Crafts (each registered with optional initial state overrides).
  * Couplings (inter-craft constraints, e.g. `Tether`).
  * Planets (each contributing standing disturbances to the shared
    fields at compile time).
  * Fields (GravityField, FluidField, MagField, CollisionField). One
    instance per field kind; per-source variation is expressed as
    `Disturbance` subclasses added to that instance.

`Sim(world)` returns a `Sim` — the symbolic IR (a
single CasADi tick function covering every craft + every coupling).
Lower to a backend to actually run ticks::

    from manta import TargetNumpy

    w = World()
    w.add_field(GravityField().add_uniform((0, 0, -9.81)))

    drone = Craft("drone")
    drone.add(Mass("body", mass=1.0))
    w.add_craft(drone, position=(0, 0, 100))

    sim = TargetNumpy(Sim(w))       # native-Python runtime
    state = sim.initial_state()
    for _ in range(N):
        state = sim.step(state, dt=0.01, t=t)

State is nested by owner: `state["drone"]["position"]`, plus one
top-level key per state-bearing disturbance (e.g. a CraftWindBubble's
estimated wind appears at `state["drone_wind"]["wind"]`).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .craft import Craft
from .fields import (
    CollisionField, Field, FluidField, GravityField, MagField,
)


# Coupling is referenced below as a type and via isinstance; the ABC
# itself lives in couplings/base.py. External callers should import it
# from `manta.couplings` (or `manta`).
from .couplings.base import Coupling   # noqa: E402


# ---------------------------------------------------------------------------
# World
# ---------------------------------------------------------------------------

class World:
    """Top-level simulation container."""

    def __init__(self, name: str = "world") -> None:
        self.name = name
        # _crafts: list of dicts with craft, initial_state_overrides.
        self._crafts: list[dict[str, Any]] = []
        self._couplings: list[Coupling] = []
        # Fields keyed by exact subclass (one GravityField per world, one
        # FluidField, …). Concrete subclasses query by class.
        self._fields: dict[type, Field] = {}
        # Planets registered with this world. Each one contributes its
        # standing field disturbances at compile() time via
        # `planet.register_disturbances(world)`. Multi-planet supported;
        # disturbances superpose into the shared field instances.
        self._planets: list = []
        # Set to True once compile() has walked the planet list — guards
        # against double-registration if the user re-compiles.
        self._planets_registered = False

    # ---- Fields ----------------------------------------------------------

    def add_field(self, field: Field) -> "World":
        """Register a Field with this world. One instance per Field
        subclass is allowed — call `field.add(disturbance)` on the
        registered instance to attach more sources.

        Returns self for chaining.
        """
        if not isinstance(field, Field):
            raise TypeError(
                f"World.add_field: expected a Field, got "
                f"{type(field).__name__}")
        cls = type(field)
        if cls in self._fields:
            raise ValueError(
                f"World '{self.name}': field of type {cls.__name__} already "
                f"registered. Use `world.get_field({cls.__name__}).add(...)` "
                f"to attach additional disturbances to it.")
        self._fields[cls] = field
        return self

    def get_field(self, cls: type) -> Field | None:
        """Return the registered field of type `cls`, or None."""
        return self._fields.get(cls)

    def get_or_create_field(self, cls: type) -> Field:
        """Return the registered field of type `cls`, creating + adding
        a fresh empty instance if none is registered. Used by
        `Planet.register_disturbances` so a planet's contributions land
        on the same field instance whether or not the user pre-added
        one."""
        if cls in self._fields:
            return self._fields[cls]
        instance = cls()
        self.add_field(instance)
        return instance

    @property
    def fields(self) -> tuple[Field, ...]:
        return tuple(self._fields.values())

    @property
    def planets(self) -> tuple:
        return tuple(self._planets)

    def add_planet(self, planet) -> "World":
        """Register a planet with this world. The planet's
        `register_disturbances(world)` is called at `Sim(world)`
        time, attaching its standing contributions to the world's
        shared fields. Multi-planet worlds superpose contributions
        from every registered planet.
        """
        from .planets.base import Planet
        if not isinstance(planet, Planet):
            raise TypeError(
                f"World.add_planet: expected a Planet, got "
                f"{type(planet).__name__}")
        for existing in self._planets:
            if existing is planet:
                raise ValueError(
                    f"World '{self.name}': planet {planet.name!r} already added")
            if existing.name == planet.name:
                raise ValueError(
                    f"World '{self.name}': planet name {planet.name!r} "
                    f"collides with an existing planet")
        planet._world = self
        self._planets.append(planet)
        return self

    @property
    def crafts(self) -> tuple[Craft, ...]:
        return tuple(entry["craft"] for entry in self._crafts)

    @property
    def couplings(self) -> tuple[Coupling, ...]:
        return tuple(self._couplings)

    # ---- Adding things ---------------------------------------------------

    def add_craft(self,
                  craft: Craft,
                  *,
                  position=(0.0, 0.0, 0.0),
                  orientation=(1.0, 0.0, 0.0, 0.0),
                  velocity=(0.0, 0.0, 0.0),
                  angular_velocity=(0.0, 0.0, 0.0),
                  **extra_state: Any) -> Craft:
        """Add a craft to the world.

        Args:
            craft           — the Craft instance.
            position, ...   — rigid-body initial state in WorldFrame.
            **extra_state   — per-part state overrides
                              (e.g., `**{"wheel.angle": 0.5}`).
        """
        # Validate uniqueness.
        for entry in self._crafts:
            if entry["craft"] is craft:
                raise ValueError(
                    f"World '{self.name}': craft '{craft.name}' already added")
            if entry["craft"].name == craft.name:
                raise ValueError(
                    f"World '{self.name}': craft name '{craft.name}' collides "
                    f"with an existing craft")

        initial_state_overrides: dict[str, Any] = {
            "position":         position,
            "orientation":      orientation,
            "velocity":         velocity,
            "angular_velocity": angular_velocity,
            **extra_state,
        }
        self._crafts.append({
            "craft":  craft,
            "initial_state_overrides": initial_state_overrides,
        })
        return craft

    def add_coupling(self, coupling: Coupling) -> Coupling:
        """Add an inter-craft coupling. Both endpoint crafts must already
        be registered via `add_craft`. The coupling forces them into the
        same connected component at compile time → one shared compiled
        tick over both."""
        if not isinstance(coupling, Coupling):
            raise TypeError(
                f"World.add_coupling: expected Coupling, got "
                f"{type(coupling).__name__}")
        registered = {id(e["craft"]) for e in self._crafts}
        for c, label in ((coupling.craft_a, "craft_a"),
                          (coupling.craft_b, "craft_b")):
            if id(c) not in registered:
                raise ValueError(
                    f"World.add_coupling: coupling.{label} '{c.name}' is "
                    f"not registered with this World. Call add_craft first.")
        self._couplings.append(coupling)
        return coupling

    # ---- Compile --------------------------------------------------------

    def _initial_state_dict(self) -> dict[str, dict[str, Any]]:
        """Nested-by-owner initial state dict. Owners = registered
        crafts (with their `add_craft` overrides applied, including
        PlanetState resolution) + each Disturbance with State/Noise
        declarations. Used by both `compile()` and `EKF(self)`."""
        from .fields.base import Disturbance
        out: dict[str, dict[str, Any]] = {}
        for entry in self._crafts:
            craft = entry["craft"]
            out[craft.name] = craft.initial_state(
                **entry["initial_state_overrides"])
        for field in self.fields:
            for dist in field._disturbances:
                if not isinstance(dist, Disturbance):
                    continue
                sdecls = dist.state_declarations()
                ndecls = dist.noise_declarations()
                if not sdecls and not ndecls:
                    continue
                init: dict[str, Any] = {}
                for sname, sdecl in sdecls.items():
                    init[sname] = float(sdecl.init)
                for nname, ndecl in ndecls.items():
                    init.update(ndecl.initial_state_entries(nname, dist))
                out[dist.name] = init
        return out

    def _resolve_planet_state_overrides(self) -> None:
        """Walk every craft entry and replace any `PlanetState`-wrapped
        position / velocity with its WorldFrame equivalent at t=0,
        using the wrapping planet's `planet_to_world(...)` transform.

        Plain tuples / numpy arrays pass through unchanged.
        """
        from .planets.state import PlanetState
        for entry in self._crafts:
            overrides = entry["initial_state_overrides"]
            pos = overrides.get("position")
            vel = overrides.get("velocity")
            # Resolve as a pair so the planet sees both together (its
            # velocity transform depends on the position via ω × r).
            pos_planet = pos if isinstance(pos, PlanetState) else None
            vel_planet = vel if isinstance(vel, PlanetState) else None
            if pos_planet is None and vel_planet is None:
                continue
            # If one is PlanetState the other should be too (or default
            # plain (0,0,0) interpreted in the same frame).
            planet = (pos_planet or vel_planet).planet
            if pos_planet is not None and pos_planet.planet is not planet:
                raise ValueError(
                    f"World '{self.name}': craft "
                    f"{entry['craft'].name!r}: position and velocity "
                    f"reference different planets; pick one frame")
            if vel_planet is not None and vel_planet.planet is not planet:
                raise ValueError(
                    f"World '{self.name}': craft "
                    f"{entry['craft'].name!r}: position and velocity "
                    f"reference different planets; pick one frame")
            p_planet_val = (pos_planet.value
                            if pos_planet is not None
                            else tuple(pos))
            v_planet_val = (vel_planet.value
                            if vel_planet is not None
                            else tuple(vel))
            p_world, v_world = planet.planet_to_world(
                p_planet_val, v_planet_val, t=0.0)
            overrides["position"] = p_world
            overrides["velocity"] = v_world

    # ---- Repr -----------------------------------------------------------

    def __repr__(self) -> str:
        n_c = len(self._crafts)
        n_x = len(self._couplings)
        n_p = len(self._planets)
        return (f"<World '{self.name}': {n_c} craft(s), "
                f"{n_x} coupling(s), {n_p} planet(s)>")


# ---------------------------------------------------------------------------
# Sim — forward-dynamics IR
# ---------------------------------------------------------------------------

class Sim:
    """Forward-dynamics IR for a World — the compiled symbolic tick.

    `Sim(world)` is one analysis transform over the model, a sibling of
    `EKF(world)`: it compiles every craft + every coupling into a single
    CasADi tick function (covering field-mediated craft-to-craft coupling
    automatically — a wake or wind bubble is just a disturbance whose
    graph reads another craft's state in the same tick). The Sim itself
    only describes the symbolic graph; lower it to a backend to run::

        sim   = TargetNumpy(Sim(world))    # native-Python runtime
        state = sim.initial_state()
        state = sim.step(state, dt=0.01)

    (`Sim` is the IR; `TargetNumpy(Sim(...))` / `TargetCpp(Sim(...), ...)`
    are the runtime evaluators that consume it.)

    Construction prepares the world once: every registered Planet's
    `register_disturbances(world)` runs, PlanetState initial seeds resolve
    to WorldFrame, and per-part `requires_fields` / `requires_planet` are
    verified against the world's registry.

    State is nested by owner: `state["drone"]["position"]`, plus one
    top-level key per state-bearing disturbance.
    """

    def __init__(self, world: World) -> None:
        # Walk planets once, in registration order. A planet may create
        # field instances via `world.get_or_create_field(...)`.
        if not world._planets_registered:
            for planet in world._planets:
                planet.register_disturbances(world)
            world._planets_registered = True

        # Resolve any PlanetState-wrapped initial conditions to WorldFrame
        # seeds at t=0.
        world._resolve_planet_state_overrides()

        # Verify per-part `requires_fields` / `requires_planet` against the
        # world's registry. Stamp each craft with a back-pointer so parts
        # can introspect fields/planets via TickContext helpers.
        for entry in world._crafts:
            craft = entry["craft"]
            craft._world = world
            for part in craft.parts:
                for req_cls in getattr(type(part), "requires_fields", []):
                    if world.get_field(req_cls) is None and not any(
                            isinstance(f, req_cls) for f in world.fields):
                        raise ValueError(
                            f"World '{world.name}': part "
                            f"{type(part).__name__}('{part.name}') "
                            f"requires a registered {req_cls.__name__} but "
                            f"none is attached to this world.")
                req_planet = getattr(type(part), "requires_planet", None)
                if req_planet is not None and not any(
                        isinstance(p, req_planet) for p in world._planets):
                    raise ValueError(
                        f"World '{world.name}': part "
                        f"{type(part).__name__}('{part.name}') "
                        f"requires a {req_planet.__name__} planet but "
                        f"none is registered with this world.")

        if not world._crafts:
            raise ValueError(
                f"World '{world.name}': no crafts added; nothing to compile.")

        # One tick, all crafts. compile_world_tick handles N=1 cleanly —
        # the only difference is that slot names get prefixed with
        # `<craft.name>.` to disambiguate across crafts. `Sim.step`
        # (on the backend) translates between the user-facing nested state
        # dict and that flat-prefixed casadi input naming.
        from .world_tick import compile_world_tick
        all_crafts = [e["craft"] for e in world._crafts]
        tick = compile_world_tick(
            all_crafts, list(world._couplings),
            gravity_field=world._fields.get(GravityField),
            fluid_field=world._fields.get(FluidField),
            mag_field=world._fields.get(MagField),
            collision_field=world._fields.get(CollisionField))

        self._world   = world
        self._tick    = tick
        self._initial = world._initial_state_dict()
        self._crafts  = tuple(all_crafts)

    # ---- Accessors ------------------------------------------------------

    @property
    def world(self) -> World:
        return self._world

    @property
    def crafts(self) -> tuple[Craft, ...]:
        return self._crafts

    @property
    def tick(self):
        """The single CompiledGraph driving every craft in this world.

        Inputs are flat-prefixed (`<owner>.<slot>`, `dt`, `t`).
        Backends consume this directly; users typically work through
        a target wrapper (`TargetNumpy(Sim(world)).step(...)`).
        """
        return self._tick

    @property
    def initial(self) -> dict[str, dict[str, Any]]:
        """Raw nested initial-state dict (NOT a deep copy — backends
        copy as needed)."""
        return self._initial

    def __repr__(self) -> str:
        names = ", ".join(c.name for c in self._crafts)
        return f"<Sim crafts=[{names}]>"

"""World — top-level simulation container.

A World holds:
  * Crafts (each registered with optional initial state overrides).
  * Couplings (inter-craft constraints, e.g. `Tether`). Couplings force
    the two endpoint crafts into the same connected component → one
    shared compiled tick.
  * Fields (GravityField, FluidField, MagField, CollisionField). One
    instance per field kind; per-source variation is expressed as
    `Disturbance` subclasses added to that instance.

At `world.compile()` time the framework walks (craft, coupling) as an
undirected graph and computes connected components. Each component
becomes one IR tick graph: single-craft components use
`Craft.compile_tick`, multi-craft components route through
`compile_coupled_tick` in `manta_next.coupled_tick`.

User-facing API::

    w = World()
    w.add_field(GravityField().add_uniform((0, 0, -9.81)))

    drone = Craft("drone")
    drone.add(Mass("body", mass=1.0))
    w.add_craft(drone, position=(0, 0, 100))

    cw = w.compile()
    state = cw.initial_state()
    for _ in range(N):
        state = cw.step(state, dt=0.01)

For multi-craft worlds, `state` is a dict keyed by component id; each
value is the per-component state dict (slot names prefixed with the
craft name when the component contains more than one craft).
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np

from .craft import Craft
from .fields import (
    CollisionField, Field, FluidField, GravityField, MagField,
)


# Coupling is referenced below as a type and via isinstance; the ABC
# itself lives in couplings/base.py. External callers should import it
# from `manta_next.couplings` (or `manta_next`).
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
        `register_disturbances(world)` is called at `world.compile()`
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

    def compile(self) -> "CompiledWorld":
        """Walk the (craft, coupling) graph, compute connected components,
        and emit one compiled tick per component.

        Single-craft components compile via `Craft.compile_tick`;
        multi-craft components (joined by Couplings) route through
        `compile_coupled_tick` to share a single tick over their full
        connected state.

        Before tracing, every registered Planet's
        `register_disturbances(self)` runs once — attaching the
        planet's standing contributions to the world's shared fields.
        Re-compile is idempotent (planets register at most once per
        world). Then each craft's parts are checked against
        `requires_fields` / `requires_planet`.
        """
        # Walk planets once, in registration order. A planet may create
        # field instances via `world.get_or_create_field(...)`.
        if not self._planets_registered:
            for planet in self._planets:
                planet.register_disturbances(self)
            self._planets_registered = True

        # Resolve any PlanetState-wrapped initial conditions to
        # WorldFrame seeds at t=0.
        self._resolve_planet_state_overrides()

        # Verify per-part `requires_fields` / `requires_planet` against
        # the world's registry. Stamp each craft with a back-pointer so
        # parts can introspect fields/planets via TickContext helpers.
        for entry in self._crafts:
            craft = entry["craft"]
            craft._world = self
            for part in craft.parts:
                for req_cls in getattr(type(part), "requires_fields", []):
                    if self.get_field(req_cls) is None and not any(
                            isinstance(f, req_cls) for f in self.fields):
                        raise ValueError(
                            f"World '{self.name}': part "
                            f"{type(part).__name__}('{part.name}') "
                            f"requires a registered {req_cls.__name__} but "
                            f"none is attached to this world.")
                req_planet = getattr(type(part), "requires_planet", None)
                if req_planet is not None and not any(
                        isinstance(p, req_planet) for p in self._planets):
                    raise ValueError(
                        f"World '{self.name}': part "
                        f"{type(part).__name__}('{part.name}') "
                        f"requires a {req_planet.__name__} planet but "
                        f"none is registered with this world.")

        components = self._compute_components()
        compiled: dict[str, dict] = {}
        # Field lookups (shared across components).
        gravity_field   = self._fields.get(GravityField)
        fluid_field     = self._fields.get(FluidField)
        mag_field       = self._fields.get(MagField)
        collision_field = self._fields.get(CollisionField)

        for comp_id, comp_entries in components.items():
            comp_crafts = [e["craft"] for e in comp_entries]
            if len(comp_crafts) == 1:
                craft  = comp_crafts[0]
                tick = craft.compile_tick(
                    gravity_field=gravity_field,
                    fluid_field=fluid_field,
                    mag_field=mag_field,
                    collision_field=collision_field)
                init = craft.initial_state(
                    **comp_entries[0]["initial_state_overrides"])
                compiled[comp_id] = {
                    "crafts":  [craft],
                    "tick":    tick,
                    "initial": init,
                }
            else:
                # Multi-craft component — coupled tick over all of them.
                from .coupled_tick import compile_coupled_tick
                # Couplings restricted to this component.
                comp_craft_ids = {id(c) for c in comp_crafts}
                comp_couplings = [
                    cp for cp in self._couplings
                    if id(cp.craft_a) in comp_craft_ids
                    and id(cp.craft_b) in comp_craft_ids
                ]
                tick = compile_coupled_tick(
                    comp_crafts, comp_couplings,
                    gravity_field=gravity_field,
                    fluid_field=fluid_field,
                    mag_field=mag_field,
                    collision_field=collision_field)
                # Build the prefixed initial state.
                init: dict[str, Any] = {}
                for entry in comp_entries:
                    sub = entry["craft"].initial_state(
                        **entry["initial_state_overrides"])
                    for k, v in sub.items():
                        init[f"{entry['craft'].name}.{k}"] = v
                compiled[comp_id] = {
                    "crafts":  comp_crafts,
                    "tick":    tick,
                    "initial": init,
                }
        return CompiledWorld(compiled, self)

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

    def _compute_components(self) -> dict[str, list[dict]]:
        """Connected components of (craft, coupling) as an undirected
        graph. Returns a dict from component id (= first-craft name) to
        the list of entry dicts in that component. A craft with no
        couplings is its own singleton component."""
        # Build adjacency from couplings.
        craft_id_to_entry = {id(e["craft"]): e for e in self._crafts}
        adjacency: dict[int, list[int]] = {
            id(e["craft"]): [] for e in self._crafts}
        for c in self._couplings:
            a, b = id(c.craft_a), id(c.craft_b)
            if a not in adjacency or b not in adjacency:
                raise ValueError(
                    "World._compute_components: coupling references a craft "
                    "not registered in this world.")
            adjacency[a].append(b)
            adjacency[b].append(a)

        # DFS to find components.
        visited: set[int] = set()
        comps: dict[str, list[dict]] = {}
        for entry in self._crafts:
            craft = entry["craft"]
            if id(craft) in visited:
                continue
            # New component, BFS from this craft.
            comp: list[dict] = []
            stack = [id(craft)]
            while stack:
                node_id = stack.pop()
                if node_id in visited:
                    continue
                visited.add(node_id)
                comp.append(craft_id_to_entry[node_id])
                for n in adjacency[node_id]:
                    if n not in visited:
                        stack.append(n)
            comp_id = comp[0]["craft"].name
            comps[comp_id] = comp
        return comps

    # ---- Repr -----------------------------------------------------------

    def __repr__(self) -> str:
        n_c = len(self._crafts)
        n_x = len(self._couplings)
        n_p = len(self._planets)
        return (f"<World '{self.name}': {n_c} craft(s), "
                f"{n_x} coupling(s), {n_p} planet(s)>")


# ---------------------------------------------------------------------------
# CompiledWorld — runtime handle
# ---------------------------------------------------------------------------

class CompiledWorld:
    """Runtime wrapper around per-component compiled tick functions.

    State layout: `dict[component_id, dict[slot_name, value]]`. For a
    singleton (no-coupling) component, `component_id == craft.name` and
    the inner dict uses unprefixed slot names. For multi-craft coupled
    components, slot names are prefixed `<craft_name>.<slot>`.

    Stepping::

        cw = world.compile()
        state = cw.initial_state()
        for _ in range(N):
            state = cw.step(state, dt=0.01)
    """

    def __init__(self, components: dict[str, dict], world: World) -> None:
        self._components = components
        self._world = world

    # ---- Accessors ------------------------------------------------------

    @property
    def world(self) -> World:
        return self._world

    @property
    def components(self) -> tuple[str, ...]:
        return tuple(self._components.keys())

    def craft(self, component_id: str) -> Craft:
        crafts = self._components[component_id]["crafts"]
        if len(crafts) != 1:
            raise ValueError(
                f"CompiledWorld.craft: component '{component_id}' has "
                f"{len(crafts)} crafts (coupled). Use `.crafts(component_id)` "
                f"to get the list.")
        return crafts[0]

    def crafts(self, component_id: str) -> list:
        return list(self._components[component_id]["crafts"])

    def tick(self, component_id: str):
        """Return the CompiledGraph for a component (advanced use)."""
        return self._components[component_id]["tick"]

    # ---- State ----------------------------------------------------------

    def initial_state(self) -> dict[str, dict[str, Any]]:
        """Fresh copy of the initial state for every component."""
        return {
            comp_id: copy.deepcopy(comp["initial"])
            for comp_id, comp in self._components.items()
        }

    def step(self,
             state: dict[str, dict[str, Any]],
             dt: float,
             t: float = 0.0) -> dict[str, dict[str, Any]]:
        """Advance every component by `dt`. Returns a fresh state dict.

        `t` is the world-clock time at the start of this step. Most
        disturbances ignore it; planet-attached disturbances use it to
        compute the planet's current rotation. Defaults to 0 for sims
        that don't care about absolute time.
        """
        new_state: dict[str, dict[str, Any]] = {}
        for comp_id, comp in self._components.items():
            comp_state = state[comp_id]
            out = comp["tick"](t=t, dt=dt, **comp_state)
            # Output dict only contains slot keys the tick wrote; merge
            # over the input to preserve any extra metadata the user
            # may have stashed in there. (Defensive — for clean callers
            # this is a no-op.)
            new_state[comp_id] = {**comp_state, **out}
        return new_state

    def __repr__(self) -> str:
        ids = ", ".join(self._components)
        return f"<CompiledWorld components=[{ids}]>"

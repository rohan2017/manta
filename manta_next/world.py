"""World — top-level simulation container.

A World holds:
  * Crafts (each anchored to one Anchor).
  * Anchors (coordinate-frame roots; one per coupling component).
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
    w.add_uniform_gravity((0, 0, -9.81))

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
    UniformFluid, UniformGravity, UniformMag,
)


# ---------------------------------------------------------------------------
# Anchor
# ---------------------------------------------------------------------------

class Anchor:
    """A coordinate-frame root for one or more crafts.

    Anchors are floating origins: they're meant to drift with the
    cluster they hold so float32 stays precise. Today an Anchor is
    primarily a label — it has a name. Its physical meaning expands
    when planets land:
      * `parent=None` ⇒ rooted in the world's inertial frame.
      * `parent=<planet>` ⇒ rooted in a rotating planet frame.

    The Anchor type at IR level is `manta_next.ir.frames.AnchorFrame`;
    that's the type tag used in `Vec3[AnchorFrame]`. `Anchor` here is
    the runtime instance that carries metadata.

    Per-anchor field overrides aren't supported yet — every craft sees
    the world's registered fields. When planets land, an Anchor may
    carry a Planet pointer that contributes additional Disturbances to
    the world's fields.
    """

    def __init__(self, name: str) -> None:
        self.name = name

    def __repr__(self) -> str:
        return f"<Anchor '{self.name}'>"


# ---------------------------------------------------------------------------
# Coupling ABC — canonical home is couplings/base.py; re-exported here so
# the long-standing `from manta_next.world import Coupling` import path
# keeps working.
# ---------------------------------------------------------------------------

from .couplings.base import Coupling   # noqa: E402,F401


# ---------------------------------------------------------------------------
# World
# ---------------------------------------------------------------------------

class World:
    """Top-level simulation container."""

    def __init__(self, name: str = "world") -> None:
        self.name = name
        self._anchors: dict[str, Anchor] = {}
        # _crafts: list of dicts with craft, anchor, initial_state_overrides.
        self._crafts: list[dict[str, Any]] = []
        self._couplings: list[Coupling] = []
        # Fields keyed by exact subclass (one GravityField per world, one
        # FluidField, …). Concrete subclasses query by class.
        self._fields: dict[type, Field] = {}

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

    def add_uniform_gravity(self,
                            g_vec: tuple[float, float, float]) -> "World":
        """Shortcut: register a GravityField containing one UniformGravity
        disturbance. The common case for sims that don't care about
        position-dependent gravity. Returns self for chaining."""
        gf = GravityField()
        gf.add(UniformGravity(g_vec))
        return self.add_field(gf)

    def add_uniform_fluid(self,
                          density: float,
                          velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)
                          ) -> "World":
        """Shortcut: register a FluidField with one UniformFluid disturbance.

        Common values:
          * density=1.225, velocity=(0,0,0)  — standard sea-level air
          * density=1025                       — seawater
          * density=1000                       — fresh water

        Returns self for chaining.
        """
        ff = FluidField()
        ff.add(UniformFluid(density, velocity))
        return self.add_field(ff)

    def add_uniform_mag(self,
                        B_vec: tuple[float, float, float]) -> "World":
        """Shortcut: register a MagField with one UniformMag disturbance.

        Common Earth-field first-order approximations (Tesla):
          * Equator:   (3e-5, 0, 0)       — ~30 µT horizontal
          * Mid-lat:   (2e-5, 0, -4.5e-5) — ~50 µT inclined
          * Pole:      (0, 0, -6e-5)      — ~60 µT vertical

        Returns self for chaining.
        """
        mf = MagField()
        mf.add(UniformMag(B_vec))
        return self.add_field(mf)

    def get_field(self, cls: type) -> Field | None:
        """Return the registered field of type `cls`, or None."""
        return self._fields.get(cls)

    @property
    def fields(self) -> tuple[Field, ...]:
        return tuple(self._fields.values())

    @property
    def anchors(self) -> dict[str, Anchor]:
        return dict(self._anchors)

    @property
    def crafts(self) -> tuple[Craft, ...]:
        return tuple(entry["craft"] for entry in self._crafts)

    @property
    def couplings(self) -> tuple[Coupling, ...]:
        return tuple(self._couplings)

    # ---- Adding things ---------------------------------------------------

    def add_anchor(self, anchor: Anchor) -> Anchor:
        if anchor.name in self._anchors:
            raise ValueError(
                f"World '{self.name}': anchor '{anchor.name}' already exists")
        self._anchors[anchor.name] = anchor
        return anchor

    def add_craft(self,
                  craft: Craft,
                  *,
                  anchor: "Anchor | str | None" = None,
                  position=(0.0, 0.0, 0.0),
                  orientation=(1.0, 0.0, 0.0, 0.0),
                  velocity=(0.0, 0.0, 0.0),
                  angular_velocity=(0.0, 0.0, 0.0),
                  **extra_state: Any) -> Craft:
        """Add a craft to the world.

        Args:
            craft           — the Craft instance.
            anchor          — Anchor instance, anchor name, or None. If
                              None, the world auto-creates an anchor named
                              `<craft.name>_anchor`. If a str, looks up
                              an existing anchor (must be already registered).
            position, ...   — rigid-body initial state.
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

        # Resolve anchor.
        if anchor is None:
            anchor = Anchor(f"{craft.name}_anchor")
            self.add_anchor(anchor)
        elif isinstance(anchor, str):
            if anchor not in self._anchors:
                raise KeyError(
                    f"World '{self.name}': no anchor named '{anchor}' "
                    f"registered. Use add_anchor first or pass an Anchor "
                    f"instance.")
            anchor = self._anchors[anchor]
        elif isinstance(anchor, Anchor):
            if anchor.name not in self._anchors:
                self.add_anchor(anchor)
        else:
            raise TypeError(
                f"World.add_craft: anchor must be Anchor|str|None, "
                f"got {type(anchor).__name__}")

        initial_state_overrides: dict[str, Any] = {
            "position":         position,
            "orientation":      orientation,
            "velocity":         velocity,
            "angular_velocity": angular_velocity,
            **extra_state,
        }
        self._crafts.append({
            "craft":  craft,
            "anchor": anchor,
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
        """
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
                anchor = comp_entries[0]["anchor"]
                tick = craft.compile_tick(
                    gravity_field=gravity_field,
                    fluid_field=fluid_field,
                    mag_field=mag_field,
                    collision_field=collision_field)
                init = craft.initial_state(
                    **comp_entries[0]["initial_state_overrides"])
                compiled[comp_id] = {
                    "crafts":  [craft],
                    "anchor":  anchor,
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
                    mag_field=mag_field)
                # Build the prefixed initial state.
                init: dict[str, Any] = {}
                for entry in comp_entries:
                    sub = entry["craft"].initial_state(
                        **entry["initial_state_overrides"])
                    for k, v in sub.items():
                        init[f"{entry['craft'].name}.{k}"] = v
                compiled[comp_id] = {
                    "crafts":  comp_crafts,
                    "anchor":  comp_entries[0]["anchor"],
                    "tick":    tick,
                    "initial": init,
                }
        return CompiledWorld(compiled, self)

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
        n_a = len(self._anchors)
        n_x = len(self._couplings)
        return (f"<World '{self.name}': {n_c} craft(s), {n_a} anchor(s), "
                f"{n_x} coupling(s)>")


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
             dt: float) -> dict[str, dict[str, Any]]:
        """Advance every component by `dt`. Returns a fresh state dict."""
        new_state: dict[str, dict[str, Any]] = {}
        for comp_id, comp in self._components.items():
            comp_state = state[comp_id]
            out = comp["tick"](dt=dt, **comp_state)
            # Output dict only contains slot keys the tick wrote; merge
            # over the input to preserve any extra metadata the user
            # may have stashed in there. (Defensive — for clean callers
            # this is a no-op.)
            new_state[comp_id] = {**comp_state, **out}
        return new_state

    def __repr__(self) -> str:
        ids = ", ".join(self._components)
        return f"<CompiledWorld components=[{ids}]>"

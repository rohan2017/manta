"""Sim — forward-dynamics IR for a World.

`Sim(world)` is one analysis transform over the model, a sibling of
`EKF(world)` (`manta.estimation.ekf`): it compiles every craft + every
coupling into a single CasADi tick. Lower it to a backend to run::

    from manta import World, Sim, TargetNumpy

    w = World()
    w.add_field(GravityField().add_uniform((0, 0, -9.81)))
    drone = Craft("drone"); drone.add(Mass("body", mass=1.0))
    w.add_craft(drone, position=(0, 0, 100))

    sim   = TargetNumpy(Sim(w))          # native-Python runtime
    state = sim.initial_state()
    for _ in range(N):
        state = sim.step(state, dt=0.01, t=t)

`Sim` is the IR; `TargetNumpy(Sim(...))` / `TargetCpp(Sim(...), ...)` are
the runtime evaluators that consume it. State is nested by owner:
`state["drone"]["position"]`, plus one top-level key per state-bearing
disturbance (e.g. a CraftWindBubble's estimated wind at
`state["drone_wind"]["wind"]`).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .fields import CollisionField, FluidField, GravityField, MagField

if TYPE_CHECKING:
    from .craft import Craft
    from .world import World


class Sim:
    """Forward-dynamics IR for a World — the compiled symbolic tick.

    `Sim(world)` compiles every craft + every coupling into a single
    CasADi tick function (covering field-mediated craft-to-craft coupling
    automatically — a wake or wind bubble is just a disturbance whose
    graph reads another craft's state in the same tick). The Sim itself
    only describes the symbolic graph; lower it to a backend to run::

        sim   = TargetNumpy(Sim(world))    # native-Python runtime
        state = sim.initial_state()
        state = sim.step(state, dt=0.01)

    Construction prepares the world once: every registered Planet's
    `register_disturbances(world)` runs, PlanetState initial seeds resolve
    to WorldFrame, and per-part `requires_fields` / `requires_planet` are
    verified against the world's registry.
    """

    # Lowerable-block kind (see manta.codegen.block.KIND_SIM): a backend
    # lowers a Sim through its `lower_sim` handler.
    RUNTIME_KIND = "sim"

    def __init__(self, world: "World") -> None:
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
    def world(self) -> "World":
        return self._world

    @property
    def crafts(self) -> "tuple[Craft, ...]":
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

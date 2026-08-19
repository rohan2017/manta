"""Compile authoring objects into the model-independent IR state layout."""

from __future__ import annotations

from .ir.manifold import R3Manifold, SO3Manifold
from .ir.state_spec import StateSpec


def _add_craft_slots(craft, prefix: str, add) -> None:
    add(prefix + "position", R3Manifold())
    add(prefix + "orientation", SO3Manifold())
    add(prefix + "velocity", R3Manifold())
    add(prefix + "angular_velocity", R3Manifold())
    for part in craft.parts:
        for name, declaration in part.state_declarations().items():
            add(prefix + f"{part.name}.{name}", declaration.manifold)
        for name, declaration in part.noise_declarations().items():
            if (declaration.contributes_state
                    and declaration.is_active(part, name)):
                add(prefix + f"{part.name}.{name}",
                    declaration.state_manifold())


def state_spec_from_craft(craft) -> StateSpec:
    """Compile one Craft's declared state into a flat IR layout."""
    layout = []
    _add_craft_slots(craft, "", lambda name, mfd: layout.append((name, mfd)))
    return StateSpec.from_layout(layout)


def state_spec_from_world(world) -> StateSpec:
    """Compile a resolved World's craft and disturbance state layout."""
    from .fields.base import Disturbance

    layout = []
    add = lambda name, mfd: layout.append((name, mfd))
    for craft in world.crafts:
        _add_craft_slots(craft, f"{craft.name}.", add)
    for field in world.fields:
        for disturbance in field._disturbances:
            if not isinstance(disturbance, Disturbance):
                continue
            prefix = f"{disturbance.name}."
            for name, declaration in disturbance.state_declarations().items():
                add(prefix + name, declaration.manifold)
            for name, declaration in disturbance.noise_declarations().items():
                if (declaration.contributes_state
                        and declaration.is_active(disturbance, name)):
                    add(prefix + name, declaration.state_manifold())
    return StateSpec.from_layout(layout)

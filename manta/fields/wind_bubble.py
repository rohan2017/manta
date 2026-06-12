"""CraftWindBubble — a per-craft localized wind disturbance.

Each craft maintains its own estimable wind vector inside a spherical
bubble centered on its current position. When two crafts come close
enough that their bubbles overlap, the FluidField's "averaged"
combining flag means the overlap region sees the mean of the two
estimates — so a craft passing through another's bubble samples that
other craft's belief and can pull its own estimate toward it via the
EKF measurement updates.

The wind state is a RandomWalkNoise declaration on the Disturbance,
which the framework picks up as an R3 state slot named
`<bubble.name>.wind`. The EKF auto-Q assembles dt·σ² on that slot,
and the bubble's `contribute_at_sym` exposes the wind to any
DragSurface that queries the FluidField at a point inside the bubble.

Usage::

    drone = Craft("drone")
    drone.add(Mass("body", mass=1.0))
    drone.add(DragSurface("drag", ...))   # reads ctx.field(FluidField)

    w = World()
    w.add_craft(drone, position=(0, 0, 5))
    w.add_field(FluidField().add(CraftWindBubble(
        craft=drone, radius=20.0, sigma=1e-3,
        name=f"{drone.name}_wind")))
"""

from __future__ import annotations

import casadi as ca

from ..ir.frames import WorldFrame
from ..ir.types import Vec3
from ..parts._declarations import RandomWalkNoise
from ..parts._trace import active_trace, is_promoted
from .base import Disturbance
from .fluid import FluidState


_VEC3_W = Vec3[WorldFrame]


class CraftWindBubble(Disturbance):
    """A localized wind contribution anchored to a craft.

    Density contribution is zero (wind doesn't change density). The
    velocity contribution equals the estimated wind vector inside the
    bubble (where `|point - craft.position| < radius`) and zero
    outside, gated by a hard `ca.if_else`.

    The wind itself is a `RandomWalkNoise` channel — the framework
    synthesizes a state slot named `<bubble.name>.wind` and an RW
    driver, evolving the wind via `wind_next = wind + sqrt(dt)·driver`.

    Args:
        craft   — owning Craft. The bubble follows the craft's
                  WorldFrame position symbolically (read inside
                  `contribute_at_sym` from the active trace's bindings).
        radius  — bubble radius in meters. Outside this, the
                  contribution is exactly zero.
        sigma   — RW drift density σ/√Hz for the wind. Larger ⇒ EKF
                  expects more drift in the wind estimate.

    The default `combining="averaged"` matches the design intent:
    overlapping bubbles compromise on the mean of their wind
    estimates.
    """

    field_value_shape = FluidState
    # Overlapping bubbles average instead of summing.
    combining = "averaged"

    wind = RandomWalkNoise("R3", frame=WorldFrame, sigma=0.0)

    def __init__(self,
                 craft,
                 radius: float = 20.0,
                 sigma: float = 1e-3, *,
                 name: str | None = None,
                 combining: str | None = None) -> None:
        super().__init__(name=name or f"{craft.name}_wind",
                         combining=combining,
                         wind_sigma=sigma)
        self.craft  = craft
        self.radius = float(radius)

    def contribute_at_sym(self, point, t) -> FluidState:
        # Craft's symbolic position (bound on the active trace by
        # compile_world_tick Pass 0a). Bubbles must be added AFTER their
        # anchor craft was added to the world; Sim(world) then guarantees
        # the symbolic state is in scope.
        trace = active_trace()
        if trace is None:
            raise RuntimeError(
                "CraftWindBubble.contribute_at_sym: no active trace — this "
                "is only callable during a world-tick compile.")
        craft_pos = trace.craft_sym_state(self.craft)["position"]
        delta_mx  = point._mx - craft_pos._mx
        d_sq      = ca.dot(delta_mx, delta_mx)
        in_bubble = ca.if_else(d_sq < self.radius ** 2,
                               ca.MX(1.0), ca.MX(0.0))
        wind_mx = self.wind._mx if is_promoted(self.wind) else self.wind
        return FluidState(
            density  = ca.MX(0.0),
            velocity = _VEC3_W.from_mx(in_bubble * wind_mx),
        )

    def __repr__(self) -> str:
        return (f"<CraftWindBubble craft={self.craft.name!r} "
                f"radius={self.radius:.1f}>")

"""Mass — a lump of mass that contributes m·g to the craft's wrench when
gravity is registered.

M1 scope: gravity is provided directly on the TickContext. Field-based
gravity (and other field interactions) lands in M2+ alongside the World
abstraction.
"""

from __future__ import annotations

from ...ir.frames import CraftFrame
from ...ir.types import Vec3
from ..base import Part, Parameter
from ..wrench import Wrench


class Mass(Part):
    """A point mass with optional gravity contribution.

    Parameters:
        mass            (float)   — kilograms.
        apply_gravity   (bool)    — if True, applies `mass · gravity` to the
                                    body. Default True.
    """

    mass: float          = Parameter(1.0)
    apply_gravity: bool  = Parameter(True)

    def update(self, ctx) -> Wrench:
        if not self.apply_gravity:
            return Wrench.zero(CraftFrame)
        # ctx.gravity is in CraftFrame (M1: anchor frame == craft frame).
        # Scale by self.mass (a Python float at trace time → becomes a
        # constant scalar node in the graph).
        force = ctx.gravity * self.mass
        # Applied at the part origin in CraftFrame (zero torque about origin).
        return Wrench(
            force=force,
            torque=Vec3[CraftFrame].constant((0.0, 0.0, 0.0)),
        )

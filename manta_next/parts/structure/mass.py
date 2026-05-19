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
    """A lump of mass with diagonal inertia tensor and optional gravity.

    Parameters:
        mass            (float)              — kilograms.
        moi             (3-tuple)            — diagonal MOI tensor (Ixx, Iyy, Izz)
                                              about the part's own COM, in
                                              part frame. Defaults to zero
                                              (point mass).
        apply_gravity   (bool)               — if True, applies `mass · gravity`
                                              at the part's origin. Default True.

    The part's spatial location is set via its `transform` parameter
    (inherited from Part). Aggregation at the Craft level rolls these
    individual contributions into total mass, COM, and MOI about craft
    origin via parallel-axis lifts.
    """

    mass:          float                       = Parameter(1.0)
    moi:           "tuple[float, float, float]" = Parameter((0.0, 0.0, 0.0))
    apply_gravity: bool                        = Parameter(True)

    def update(self, ctx) -> Wrench:
        if not self.apply_gravity:
            return Wrench.zero(CraftFrame)
        # ctx.gravity is in CraftFrame (M2: orientation state allowed, the
        # ctx rotates the world-frame gravity into craft frame each tick).
        # The force is applied at the part's origin (= COM by convention
        # for a point mass; Mass.moi describes the tensor *about that
        # origin* in part frame). Wrench-at-offset bookkeeping lives in
        # Craft._aggregate_wrenches.
        force = ctx.gravity * self.mass
        return Wrench(
            force=force,
            torque=Vec3[CraftFrame].constant((0.0, 0.0, 0.0)),
        )

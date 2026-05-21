"""Mass — a lump of mass that contributes m·g to the craft's wrench
under the registered GravityField. Diagonal MOI feeds into the body's
inertia aggregation via parallel-axis lifts at Craft.compile_tick time.
"""

from __future__ import annotations

from ...ir.frames import CraftFrame
from ...ir.types import Vec3
from ..base import Part, Parameter
from ...math.wrench import Wrench


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
        # Query the GravityField at THIS part's anchor-frame position,
        # not at the craft origin (which is what ctx.gravity captures).
        # For a uniform field this folds to the same constant; for a
        # spatially-varying field (e.g. PointMassGravity) it picks up
        # the right local g for parts mounted far from the craft origin.
        offset_craft  = Vec3[CraftFrame].constant(tuple(self.transform))
        offset_anchor = ctx.orientation.apply(offset_craft)
        p_anchor      = ctx.position + offset_anchor
        g_anchor      = ctx.gravity_field.state_at_sym(p_anchor)
        g_local       = ctx.orientation.conjugate().apply(g_anchor)
        force = g_local * self.mass
        return Wrench(
            force=force,
            torque=Vec3[CraftFrame].constant((0.0, 0.0, 0.0)),
        )

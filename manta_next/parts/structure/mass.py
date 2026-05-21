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
        # `ctx.gravity` is the GravityField sampled at THIS part's anchor
        # position (computed by the kinematic pass), then rotated into
        # body-frame coords. For a flat craft with a uniform field this
        # folds to the constant body-frame gravity; for a spatially
        # varying field (e.g. PointMassGravity) it picks up the right
        # local g for parts mounted far from the craft origin.
        force = ctx.gravity * self.mass
        return Wrench(
            force=force,
            torque=Vec3[CraftFrame].constant((0.0, 0.0, 0.0)),
        )

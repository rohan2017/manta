"""Mass — a lump of mass that contributes m·g to the craft's wrench
when a GravityField is registered. Diagonal MOI feeds into the body's
inertia aggregation via parallel-axis lifts at world-compile time.
"""

from __future__ import annotations

from ...fields import GravityField
from ...ir.frames import CraftFrame
from ...ir.types import Vec3
from ..base import Part, Parameter
from ...ir.wrench import Wrench


class Mass(Part):
    """A lump of mass with diagonal inertia tensor.

    Parameters:
        mass — kilograms.
        moi  — 3-tuple, diagonal MOI tensor (Ixx, Iyy, Izz) about the
               part's own COM, in part frame. Defaults to zero (point
               mass).

    Gravity contribution is applied automatically whenever a
    `GravityField` is registered on the world: `F = m · g(p_world)`,
    sampled at the part's anchor position. With no `GravityField`
    registered, `ctx.field(GravityField)` returns an empty default and
    the contribution is identically zero — no special-case opt-out
    needed.

    The part's spatial location is set via its `transform` parameter
    (inherited from Part). Aggregation at the Craft level rolls these
    individual contributions into total mass, COM, and MOI about craft
    origin via parallel-axis lifts.
    """

    mass: float                              = Parameter(1.0)
    moi:  "tuple[float, float, float]"       = Parameter((0.0, 0.0, 0.0))

    def update(self, ctx) -> Wrench:
        # ctx.position is the part's mount-point in WorldFrame (chain-
        # composed by the kinematic pass). Querying the GravityField at
        # the part's actual position picks up non-uniform fields (e.g.
        # point-mass gravity) correctly when the part is mounted at a
        # non-zero transform on the craft.
        g_world = ctx.field(GravityField).state_at_sym(ctx.position, ctx.t)
        g_body  = ctx.orientation.conjugate().apply(g_world)
        return Wrench(
            force=g_body * self.mass,
            torque=Vec3[CraftFrame].constant((0.0, 0.0, 0.0)),
        )

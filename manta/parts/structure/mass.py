"""Mass — a lump of mass that contributes m·g to the craft's wrench
when a GravityField is registered. Diagonal MOI feeds into the body's
inertia aggregation via parallel-axis lifts at world-compile time.
"""

from __future__ import annotations

from ...fields import GravityField
from ...ir.frames import PartFrame, WorldFrame
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

    def __init__(self, name: str, **overrides) -> None:
        super().__init__(name, **overrides)
        # Zero is allowed per-part (the craft-level total-mass guard
        # catches an all-massless craft); negative mass/MOI is nonsense.
        if float(self.mass) < 0.0:
            raise ValueError(
                f"{type(self).__name__} {name!r}: mass must be >= 0, "
                f"got {self.mass!r}")
        moi = tuple(float(x) for x in self.moi)
        if len(moi) != 3 or any(x < 0.0 for x in moi):
            raise ValueError(
                f"{type(self).__name__} {name!r}: moi must be three "
                f"non-negative diagonal entries, got {self.moi!r}")

    def update(self, ctx) -> Wrench:
        # ctx.position[WorldFrame] is the part's mount-point (chain-composed
        # by the kinematic pass). Querying the GravityField there picks up
        # non-uniform fields (e.g. point-mass gravity) correctly for a part
        # mounted at a non-zero transform. Gravity in the part's own frame
        # via ctx.orientation; the framework rotates the wrench to body.
        g_world = ctx.field(GravityField).value_at_sym(
            ctx.position[WorldFrame], ctx.t)
        g_part  = ctx.orientation.conjugate().apply(g_world)
        return Wrench(
            force=g_part * self.mass,
            torque=Vec3[PartFrame].constant((0.0, 0.0, 0.0)),
        )

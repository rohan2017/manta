"""Mass — a lump of mass that contributes m·g to the craft's wrench
when a GravityField is registered. Diagonal MOI feeds into the body's
inertia aggregation via parallel-axis lifts at world-compile time.
"""

from __future__ import annotations

from ...fields import gravity_at
from ...ir.frames import PartFrame, WorldFrame
from ...ir.types import Scalar, Vec3
from .._declarations import Parameter
from ..base import Part
from ...ir.wrench import Wrench


class Mass(Part):
    """A lump of mass with diagonal inertia tensor.

    Parameters:
        mass — kilograms. Promotable (system-ID target).
        moi  — 3-tuple, diagonal MOI tensor (Ixx, Iyy, Izz) about the
               part's own COM, in part frame. Defaults to zero (point
               mass). Promotable, like `mass`: a tunable transform
               (`Sim(world, parameters=[...])` / `Fit`) promotes it to
               a live R3 input and the inertia rollup keeps it
               symbolic.

    Gravity contribution is applied automatically whenever a
    `GravityField` is registered on the world: `F = m · g(p_world)`,
    sampled at the part's anchor position. With no `GravityField`
    registered the contribution is explicitly zero (`gravity_at`
    branches on `ctx.has_field`) — a free-space world is legitimate,
    not a configuration error.

    The part's spatial location is set via its `mount_offset` parameter
    (inherited from Part). Aggregation at the Craft level rolls these
    individual contributions into total mass, COM, and MOI about craft
    origin via parallel-axis lifts.
    """

    # Genuinely inertial — the inertia walks enumerate this part.
    contributes_inertia = True

    mass: float                              = Parameter(1.0, manifold="R1")
    moi:  "tuple[float, float, float]"       = Parameter(
        (0.0, 0.0, 0.0), manifold="R3", frame=PartFrame)

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
        g_world = gravity_at(ctx, ctx.position[WorldFrame])
        g_part  = ctx.orientation.conjugate().apply(g_world)
        # `mass` is promotable — Scalar.coerce passes a promoted (tunable)
        # symbol through and bakes the plain float otherwise.
        return Wrench(
            force=g_part * Scalar.coerce(self.mass),
            torque=Vec3[PartFrame].constant((0.0, 0.0, 0.0)),
        )

"""PointBuoy — single-point buoyancy displacing a fixed volume.

A craft submerged in a fluid feels a buoyant force opposing gravity at
each sampled volume element. PointBuoy is the simplest such element: a
single sample with displacement volume V at the part's mount offset.

  F = ρ(p) · V · (a_fluid(p) - g(p))

evaluated at the buoy's world-frame position. This is the pressure resultant
implied by ``grad(P) = rho * (g - a_fluid)``. It reduces to ordinary
Archimedes buoyancy in a static fluid and supplies the centripetal force
required by a co-rotating ocean. ρ and the fluid material acceleration come
from the registered FluidField; g from the registered GravityField. A world with
no FluidField is a configuration error (`requires_fields`), rejected
when the first transform is built — an in-vacuum sanity world simply
omits the buoy. With Part.mount_offset set to a
non-zero offset, the framework's wrench-at-offset lift rolls the force
up into the parent's torque automatically.

``DisplacementHull`` is the stock multi-sample composition: it distributes
``PointBuoy`` and ``DragSurface`` children through a calibrated ellipsoidal
envelope. With enough samples and a smooth water/air boundary,
surface-crossing torques (the "righting moment" of a hull) emerge naturally.
"""

from __future__ import annotations

from typing import ClassVar

from ...fields import FluidField, GravityField
from ...ir.frames import PartFrame, WorldFrame
from ...ir.types import Vec3
from ...ir.wrench import Wrench
from .._declarations import Parameter, PartUpdate
from ..base import Part


class PointBuoy(Part):
    """Single-point buoyancy displacing a fixed volume.

    Parameters:
        volume — m³ displaced by the buoyancy element. Default 1e-3.

    Force = ρ(p_world) · V · (a_fluid - g) at the part's mount point,
    rotated from anchor to craft frame, applied at the offset (so the
    framework lifts force-at-offset → body-frame torque for tilt
    response).
    """

    requires_fields: ClassVar[list[type]] = [FluidField, GravityField]

    # Positive scalar and a system-identification target. Calibration should
    # normally tie distributed samples to one physical displacement scale.
    volume: float = Parameter(1e-3, manifold="R1")     # m³

    def __init__(self, name: str, **overrides) -> None:
        super().__init__(name, **overrides)
        if float(self.volume) < 0.0:
            raise ValueError(
                f"{type(self).__name__} {name!r}: volume must be >= 0, "
                f"got {self.volume!r}")

    def update(self, ctx) -> PartUpdate:
        # ctx.position is already the buoy's world-frame mount point (the
        # kinematic pass composed the transform + any joints). Field queries
        # there capture the correct local value for spatially varying fields;
        # for uniform fields it's the same as the craft origin.
        p_world = ctx.position[WorldFrame]
        fluid    = ctx.field(FluidField).value_at_sym(p_world, ctx.t)
        g_world  = ctx.field(GravityField).value_at_sym(p_world, ctx.t)

        # A static fluid has a_fluid=0 and recovers -rho*V*g. A rotating
        # fluid supplies the centripetal acceleration needed for an
        # Earth-fixed neutral body instead of making drag react after drift.
        scale = fluid.density * self.volume
        a_fluid = fluid.material_acceleration
        if a_fluid is None:
            a_fluid = Vec3[WorldFrame].constant((0.0, 0.0, 0.0))
        f_world = (a_fluid - g_world) * scale
        f_part  = ctx.orientation.conjugate().apply(f_world)

        zero_t = Vec3[PartFrame].constant((0.0, 0.0, 0.0))
        return PartUpdate(wrench=Wrench(force=f_part, torque=zero_t))

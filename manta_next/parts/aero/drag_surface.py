"""DragSurface — quadratic drag from fluid flow.

The standard high-Reynolds drag model used for submarine hulls, parachutes,
quadcopter bodies, and any non-streamlined body in a fluid:

    F_drag = -½ · ρ · A · Cd · |v_rel| · v_rel

where v_rel is the velocity of the surface point relative to the local
fluid. For a craft at velocity v with angular velocity ω, the velocity
at body offset r is v + R·(ω × r) in anchor frame; subtract the local
fluid velocity to get v_rel.

The drag force opposes the relative motion, scaled by ρ, area, Cd, and
the squared speed. It's applied at the part's mount offset — through
the framework's force-at-offset → body-frame torque lift, this gives
rotational damping for free.

What's deferred:
  * Lift (airfoils with angle of attack) — needs a Naca00xx-style
    CL/CD lookup. Submarine hulls and quadcopter bodies don't need it.
  * Direction-dependent drag (a flat plate has very different Cd along
    vs. across). For a v1 isotropic-Cd model the user can place multiple
    DragSurface parts with different Cd at different orientations to
    approximate anisotropy.
"""

from __future__ import annotations

import casadi as ca

from ...ir.frames import AnchorFrame, CraftFrame
from ...ir.types import Vec3
from ..base import Parameter, Part, PartUpdate
from ..wrench import Wrench


class DragSurface(Part):
    """Quadratic-drag surface in a fluid field.

    Parameters:
        area              — projected area in m². For a sphere, A = π·r².
                            For a submarine cylinder broadside, A = L·D.
        drag_coefficient  — dimensionless Cd. Common values:
                              * 0.04   streamlined fish/sub body
                              * 0.5    smooth sphere
                              * 1.0    short cylinder, broadside
                              * 1.2    flat plate normal to flow
                              * 1.5    parachute
    """

    area:              float = Parameter(0.01)
    drag_coefficient:  float = Parameter(1.0)

    def update(self, ctx) -> PartUpdate:
        # Anchor-frame position of the drag center.
        offset_craft = Vec3[CraftFrame].constant(tuple(self.transform))
        offset_anchor = ctx.orientation.apply(offset_craft)
        p_anchor      = ctx.position + offset_anchor

        # Velocity of the surface point in anchor frame:
        #   v_anchor(point) = v_origin + R · (ω × r_craft)
        # where ω is body angular velocity and r_craft is the body-frame
        # offset to this part.
        v_rotation_craft = ctx.angular_velocity.cross(offset_craft)
        v_rotation_anchor = ctx.orientation.apply(v_rotation_craft)
        v_surface_anchor = ctx.velocity + v_rotation_anchor

        # Fluid state at the surface point.
        fluid = ctx.fluid_field.state_at_sym(p_anchor)
        rho   = fluid.density
        v_fluid_anchor = fluid.velocity

        # Relative velocity: surface − fluid.
        v_rel = v_surface_anchor - v_fluid_anchor

        # Drag force: F = -½·ρ·A·Cd·|v_rel|·v_rel
        # Use a softened norm to keep the Jacobian regular at v_rel = 0.
        v_rel_mx = v_rel._mx
        speed_sq = ca.dot(v_rel_mx, v_rel_mx) + 1e-30
        speed    = ca.sqrt(speed_sq)
        coef     = -0.5 * rho * self.area * self.drag_coefficient * speed
        f_mx     = coef * v_rel_mx
        f_anchor = Vec3[AnchorFrame].from_mx(f_mx)
        f_craft  = ctx.orientation.conjugate().apply(f_anchor)

        zero_t = Vec3[CraftFrame].constant((0.0, 0.0, 0.0))
        return PartUpdate(wrench=Wrench(force=f_craft, torque=zero_t))

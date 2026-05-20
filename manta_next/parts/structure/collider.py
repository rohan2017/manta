"""Collider — point contact with the CollisionField.

Queries the registered CollisionField at the collider's anchor-frame
position. The field returns an outward-penetration vector (zero when
not in contact, magnitude=depth and direction=outward normal
otherwise). The Collider converts that to a contact wrench:

    F_contact = stiffness · pen_vec  +  F_damp

where the damper is gated by penetration magnitude so it doesn't pull
on free bodies:

    F_damp = pen_vec · (-damping · (v_point · pen_vec) / (|pen|² + ε))

At |pen|=0 the damping term goes smoothly to zero (because v·pen → 0
quadratically while denominator → ε). At |pen|>0 it reduces to the
expected −c·(v·n̂)·n̂ contact damping.
"""

from __future__ import annotations

import casadi as ca

from ...ir.frames import AnchorFrame, CraftFrame
from ...ir.types import Vec3
from ..base import Parameter, Part, PartUpdate
from ..wrench import Wrench


class Collider(Part):
    """Point contact element backed by the registered CollisionField.

    Parameters:
        stiffness — N/m. Spring constant of the contact normal-force.
                    Bigger = stiffer contact. Default 5e3.
        damping   — N·s/m. Damper coefficient for the relative velocity
                    along the outward normal direction. Bigger = more
                    energy dissipation per bounce. Default 50.0.
    """

    stiffness: float = Parameter(5e3)
    damping:   float = Parameter(50.0)

    def update(self, ctx) -> PartUpdate:
        offset_craft  = Vec3[CraftFrame].constant(tuple(self.transform))
        offset_anchor = ctx.orientation.apply(offset_craft)
        p_anchor      = ctx.position + offset_anchor

        # Velocity of the collider point in anchor frame (includes the
        # rotational lever arm).
        v_rot_craft = ctx.angular_velocity.cross(offset_craft)
        v_rot_anchor = ctx.orientation.apply(v_rot_craft)
        v_point_anchor = ctx.velocity + v_rot_anchor

        # Penetration vector from the CollisionField.
        pen = ctx.collision_field.state_at_sym(p_anchor)
        pen_mx = pen._mx
        v_mx   = v_point_anchor._mx
        pen_sq = ca.dot(pen_mx, pen_mx) + 1e-12

        # Spring: F = k · pen_vec (pushes outward when in contact, zero
        # when penetration is zero).
        F_spring_mx = self.stiffness * pen_mx
        # Damper: F = -c · (v · n̂) · n̂, smoothly gated so |pen|=0 → 0.
        v_dot_pen = ca.dot(v_mx, pen_mx)
        F_damp_mx = pen_mx * (-self.damping * v_dot_pen / pen_sq)

        F_anchor_mx = F_spring_mx + F_damp_mx
        F_anchor    = Vec3[AnchorFrame].from_mx(F_anchor_mx)
        F_craft     = ctx.orientation.conjugate().apply(F_anchor)

        zero_t = Vec3[CraftFrame].constant((0.0, 0.0, 0.0))
        return PartUpdate(wrench=Wrench(force=F_craft, torque=zero_t))

"""Collider — point contact with the CollisionField.

Queries the registered CollisionField at the collider's world-frame
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

from ...fields import CollisionField
from ...ir.frames import WorldFrame, PartFrame
from ...ir.types import Vec3
from ..base import Parameter, Part, PartUpdate
from ...ir.wrench import Wrench


class Collider(Part):
    """Point contact element backed by the registered CollisionField.

    Parameters:
        stiffness — N/m. Spring constant of the contact normal-force.
                    Bigger = stiffer contact. Default 5e3.
        damping   — N·s/m. Damper coefficient for the relative velocity
                    along the outward normal direction. Bigger = more
                    energy dissipation per bounce. Default 50.0.
        friction  — N·s/m. Viscous TANGENTIAL friction: opposes the
                    contact point's velocity perpendicular to the
                    outward normal, gated smoothly by penetration (a
                    smooth, EKF-friendly stand-in for Coulomb friction —
                    grips a resting contact against sliding). Default 0
                    (frictionless contact, the prior behaviour).
    """

    stiffness: float = Parameter(5e3)
    damping:   float = Parameter(50.0)
    friction:  float = Parameter(0.0)

    def update(self, ctx) -> PartUpdate:
        # ctx.position / ctx.velocity are already the collider point's
        # world-frame mount-point pose + velocity (kinematic pass composed
        # the transform and the rotational lever arm, including any joint
        # motion above the part).
        p_world        = ctx.position[WorldFrame]
        v_point_anchor = ctx.velocity[WorldFrame]

        # Penetration vector from the CollisionField.
        pen = ctx.field(CollisionField).value_at_sym(p_world, ctx.t)
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
        if self.friction != 0.0:
            # Tangential viscous friction: −c·v_⊥, gated by penetration so
            # it vanishes smoothly out of contact (pen_sq carries a +1e-12
            # regulariser, so gate → 0 at zero penetration).
            n_mx   = pen_mx / ca.sqrt(pen_sq)
            v_tan  = v_mx - ca.dot(v_mx, n_mx) * n_mx
            gate   = (pen_sq - 1e-12) / pen_sq
            F_anchor_mx = F_anchor_mx - self.friction * gate * v_tan
        F_anchor    = Vec3[WorldFrame].from_mx(F_anchor_mx)
        # Rotate into the collider's own frame; framework rotates back to
        # body and lifts force-at-offset → torque.
        F_part      = ctx.orientation.conjugate().apply(F_anchor)

        zero_t = Vec3[PartFrame].constant((0.0, 0.0, 0.0))
        return PartUpdate(wrench=Wrench(force=F_part, torque=zero_t))

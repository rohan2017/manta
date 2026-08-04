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
from .._declarations import Parameter, PartUpdate
from ..base import Part
from ...ir.wrench import Wrench


class Collider(Part):
    """Point contact element backed by the registered CollisionField.

    Parameters:
        stiffness — N/m. Spring constant of the contact normal-force.
                    Bigger = stiffer contact. Default 5e3.
        damping   — N·s/m. Damper coefficient for the relative velocity
                    along the outward normal direction. Bigger = more
                    energy dissipation per bounce. Default 50.0.
        friction  — N·s/m. Viscous TANGENTIAL friction: opposes the contact
                    point's slip in the contact plane, gated smoothly by
                    penetration (a smooth, EKF-friendly stand-in for Coulomb
                    friction — grips a resting contact against sliding).
                    ANISOTROPIC: a per-BODY-axis 3-vector of coefficients, so
                    a wheel can roll free along one axis and grip the others
                    (e.g. (0, c, 0) — free fore–aft, grips sideways). A scalar
                    is accepted as shorthand for isotropic (c, c, c). Default
                    0 → frictionless contact (the prior behaviour).
    """

    requires_fields = [CollisionField]

    stiffness: float = Parameter(5e3)
    damping:   float = Parameter(50.0)
    friction:  tuple = Parameter((0.0, 0.0, 0.0))

    def __init__(self, name: str, *, friction=0.0, **overrides) -> None:
        # `friction` is a per-body-axis 3-vector; a scalar broadcasts to the
        # isotropic (c, c, c).
        if isinstance(friction, (int, float)):
            friction = (float(friction),) * 3
        else:
            friction = tuple(float(c) for c in friction)
        super().__init__(name, friction=friction, **overrides)

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
        fric = tuple(float(c) for c in self.friction)
        if any(f != 0.0 for f in fric):
            # Anisotropic tangential viscous friction. Take the slip in the
            # contact plane, rotate it into the body frame, scale each axis by
            # its coefficient (so a wheel rolls free along one axis and grips
            # the others), rotate back, and project onto the contact plane so
            # the force is purely tangential — never a spurious normal force.
            # Penetration-gated (pen_sq carries a +1e-12 regulariser, so the
            # gate → 0 at zero penetration).
            n_mx   = pen_mx / ca.sqrt(pen_sq)
            v_tan  = v_mx - ca.dot(v_mx, n_mx) * n_mx
            gate   = (pen_sq - 1e-12) / pen_sq
            v_body = ctx.orientation.conjugate().apply(
                Vec3[WorldFrame].from_mx(v_tan))._mx
            F_fric = ctx.orientation.apply(
                Vec3[PartFrame].from_mx(ca.DM(fric) * v_body))._mx
            F_fric = F_fric - ca.dot(F_fric, n_mx) * n_mx
            F_anchor_mx = F_anchor_mx - gate * F_fric
        F_anchor    = Vec3[WorldFrame].from_mx(F_anchor_mx)
        # Rotate into the collider's own frame; framework rotates back to
        # body and lifts force-at-offset → torque.
        F_part      = ctx.orientation.conjugate().apply(F_anchor)

        zero_t = Vec3[PartFrame].constant((0.0, 0.0, 0.0))
        return PartUpdate(wrench=Wrench(force=F_part, torque=zero_t))

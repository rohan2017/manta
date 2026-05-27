"""Tether — spring-damper between two TetherEndpoint Parts.

  F = -k·(L − L_rest) · r̂   −   c · (v_rel · r̂) · r̂

where L is the instantaneous distance between endpoints in the world
frame, r̂ is the unit vector from A to B, v_rel is the relative anchor-
frame velocity of B's endpoint minus A's endpoint. The force is applied
to A in the direction r̂ (pulling A toward B when stretched and
extending v_rel · r̂ > 0), with equal-and-opposite F on B.

Each endpoint Part on the craft has a `Part.transform` giving the
attachment offset in body frame. The Tether reads both transforms,
computes the world-frame positions, evaluates the force, and emits a
pair of wrenches at-the-endpoint-offsets. The compile_world_tick
framework lifts these to craft-origin wrenches via the standard
force-at-offset → body-frame-torque mechanism, so off-axis attachments
produce the correct torques on each craft automatically.
"""

from __future__ import annotations

import casadi as ca

from ..ir.frames import WorldFrame, CraftFrame
from ..ir.types import Vec3
from ..ir.wrench import Wrench
from .base import Coupling


class Tether(Coupling):
    """Linear spring-damper tether between two TetherEndpoint Parts.

    Args:
        craft_a, craft_b       — the two coupled crafts (Craft instances).
        endpoint_a, endpoint_b — names of the TetherEndpoint Parts on
                                  craft_a / craft_b (strings).
        stiffness              — spring constant k, N/m.
        damping                — damper constant c, N·s/m.
        rest_length            — natural length L_rest, m. The spring
                                  exerts zero force at exactly this
                                  separation.

    Convention: tension positive — when L > rest_length the tether pulls
    A toward B (and B toward A). Compression (L < rest_length) pushes
    them apart, mimicking a rigid rod. For a string that goes slack,
    set stiffness=0 below rest_length using a custom subclass — the v1
    Tether models a rigid spring (no slack).
    """

    def __init__(self,
                 craft_a,
                 endpoint_a: str,
                 craft_b,
                 endpoint_b: str,
                 *,
                 stiffness: float,
                 damping: float = 0.0,
                 rest_length: float = 0.0) -> None:
        self._craft_a = craft_a
        self._craft_b = craft_b
        self.endpoint_a_name = str(endpoint_a)
        self.endpoint_b_name = str(endpoint_b)
        self.stiffness = float(stiffness)
        self.damping   = float(damping)
        self.rest_length = float(rest_length)

    @property
    def craft_a(self):
        return self._craft_a

    @property
    def craft_b(self):
        return self._craft_b

    def _find_endpoint(self, craft, name: str):
        for p in craft.parts:
            if p.name == name:
                return p
        raise ValueError(
            f"Tether: craft '{craft.name}' has no TetherEndpoint named "
            f"'{name}'")

    def compute_wrenches_sym(self, ctx_a, ctx_b) -> tuple[Wrench, Wrench]:
        """Return (wrench_on_a_at_craft_origin, wrench_on_b_at_craft_origin).

        Both wrenches are in their respective CraftFrame, lifted to the
        body origin (force-at-offset + lever-arm torque). The compile
        layer adds these directly to each craft's aggregate net wrench.
        """
        ep_a = self._find_endpoint(self._craft_a, self.endpoint_a_name)
        ep_b = self._find_endpoint(self._craft_b, self.endpoint_b_name)

        # Endpoint offsets in each body frame.
        off_a_craft = Vec3[CraftFrame].constant(tuple(ep_a.transform))
        off_b_craft = Vec3[CraftFrame].constant(tuple(ep_b.transform))

        # Endpoint positions in world frame.
        off_a_anchor = ctx_a.orientation.apply(off_a_craft)
        off_b_anchor = ctx_b.orientation.apply(off_b_craft)
        p_a_anchor = ctx_a.position + off_a_anchor
        p_b_anchor = ctx_b.position + off_b_anchor

        # Vector from A to B; instantaneous length with softened sqrt.
        r = p_b_anchor - p_a_anchor
        r_mx = r._mx
        r_sq = ca.dot(r_mx, r_mx) + 1e-30
        L    = ca.sqrt(r_sq)
        r_hat_mx = r_mx / L
        r_hat = Vec3[WorldFrame].from_mx(r_hat_mx)

        # Endpoint velocities in world frame:
        #   v_endpoint_anchor = v_origin + R · (ω × r_endpoint_body)
        v_rot_a_craft = ctx_a.angular_velocity.cross(off_a_craft)
        v_rot_b_craft = ctx_b.angular_velocity.cross(off_b_craft)
        v_a_anchor = ctx_a.velocity + ctx_a.orientation.apply(v_rot_a_craft)
        v_b_anchor = ctx_b.velocity + ctx_b.orientation.apply(v_rot_b_craft)
        v_rel = v_b_anchor - v_a_anchor

        # Tension positive when stretched; positive v_along means A and
        # B moving apart (damper resists that).
        stretch = L - self.rest_length
        v_along = v_rel.dot(r_hat)
        F_mag = self.stiffness * stretch + self.damping * v_along

        # Force on A is along +r̂ (toward B) when stretched/separating.
        F_on_a_anchor = r_hat * F_mag
        F_on_b_anchor = F_on_a_anchor * (-1.0)

        # Rotate into each craft's frame.
        F_on_a_craft = ctx_a.orientation.conjugate().apply(F_on_a_anchor)
        F_on_b_craft = ctx_b.orientation.conjugate().apply(F_on_b_anchor)

        # Lift force-at-offset to wrench-at-craft-origin:
        #   F at offset r yields F at origin plus torque r × F about origin.
        zero_t = Vec3[CraftFrame].constant((0.0, 0.0, 0.0))
        wrench_a = Wrench(
            force=F_on_a_craft,
            torque=off_a_craft.cross(F_on_a_craft) + zero_t,
        )
        wrench_b = Wrench(
            force=F_on_b_craft,
            torque=off_b_craft.cross(F_on_b_craft) + zero_t,
        )
        return wrench_a, wrench_b

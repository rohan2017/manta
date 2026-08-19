"""Tether — slack-capable spring-damper between two TetherEndpoint Parts.

  T_raw = k·(L − L_rest) + c·(v_rel · r̂)
  T     = w_taut(L − L_rest) · w_tension(T_raw) · T_raw        (T ≥ 0-ish)
  F_on_A = +T · r̂,   F_on_B = −T · r̂

where L is the instantaneous distance between endpoints in the world
frame, r̂ is the unit vector from A to B, and v_rel is the relative
anchor-frame velocity of B's endpoint minus A's endpoint. A tether is a
rope, not a rod: it transmits **tension only**. Two compact-support
gates (`hermite_blend`) enforce that:

  * `w_taut` — zero whenever the tether is slack by more than
    `slack_smoothing` metres. A slack rope exerts no force at all —
    neither spring push nor damper drag.
  * `w_tension` — zero whenever the raw spring+damper sum would be
    compressive (a strong damper during rapid approach cannot shove the
    crafts apart; the rope just goes momentarily slack).

Both gates have compact support, so outside their blend bands the force
is exactly zero — a slack tether contributes nothing to the graph's
value or Jacobian. Inside the bands the transition is C¹ (the same
`smoothstep` the fluid regime layering relies on), keeping EKF/LQR
Jacobians regular at the taut/slack boundary.

Each endpoint Part on the craft has a `Part.mount_offset` giving the
attachment offset in body frame. The Tether reads both transforms,
computes the world-frame positions, evaluates the force, and emits a
pair of wrenches at-the-endpoint-offsets. The compile_world_tick
framework lifts these to craft-origin wrenches via the standard
force-at-offset → body-frame-torque mechanism, so off-axis attachments
produce the correct torques on each craft automatically.
"""

from __future__ import annotations

import casadi as ca

from .._validation import require_positive
from ..ir.frames import WorldFrame, CraftFrame
from ..ir.types import Vec3
from ..ir.wrench import Wrench
from ..smoothing import hermite_blend, soft_norm
from .base import Coupling


class Tether(Coupling):
    """Slack-capable spring-damper tether between two TetherEndpoint Parts.

    Args:
        craft_a, craft_b       — the two coupled crafts (Craft instances).
        endpoint_a, endpoint_b — names of the TetherEndpoint Parts on
                                  craft_a / craft_b (strings).
        stiffness              — spring constant k, N/m (taut only).
        damping                — damper constant c, N·s/m (taut only).
        rest_length            — natural length L_rest, m. Slack below,
                                  taut above.
        slack_smoothing        — half-width, m, of the C¹ blend band
                                  around the taut/slack boundary (and,
                                  scaled by k, of the tension-only
                                  clamp). 0 ⇒ hard switches.

    Convention: tension only. When L > rest_length the tether pulls A
    toward B (and B toward A), the damper resisting length rate; the
    net force is clamped so it can never push. When L < rest_length the
    tether is slack and exerts exactly zero force — the crafts move
    freely until the rope tautens again.
    """

    def __init__(self,
                 craft_a,
                 endpoint_a: str,
                 craft_b,
                 endpoint_b: str,
                 *,
                 name: str | None = None,
                 stiffness: float,
                 damping: float = 0.0,
                 rest_length: float = 0.0,
                 slack_smoothing: float = 1e-3) -> None:
        if name is None:
            name = (f"{craft_a.name}_{endpoint_a}_to_"
                    f"{craft_b.name}_{endpoint_b}_tether")
        super().__init__(name)
        self._craft_a = craft_a
        self._craft_b = craft_b
        self.endpoint_a_name = str(endpoint_a)
        self.endpoint_b_name = str(endpoint_b)
        self.stiffness = require_positive(
            stiffness, name=f"Tether {self.name!r}.stiffness", allow_zero=True)
        self.damping = require_positive(
            damping, name=f"Tether {self.name!r}.damping", allow_zero=True)
        self.rest_length = require_positive(
            rest_length, name=f"Tether {self.name!r}.rest_length",
            allow_zero=True)
        self.slack_smoothing = require_positive(
            slack_smoothing, name=f"Tether {self.name!r}.slack_smoothing",
            allow_zero=True)
        # Resolve endpoints now — a bad name fails here, at the line that
        # wrote it, not at compile. Add endpoint Parts before the Tether.
        self.endpoint_a = self._find_endpoint(craft_a, self.endpoint_a_name)
        self.endpoint_b = self._find_endpoint(craft_b, self.endpoint_b_name)

    @property
    def craft_a(self):
        return self._craft_a

    @property
    def craft_b(self):
        return self._craft_b

    @staticmethod
    def _find_endpoint(craft, name: str):
        from ..parts.attachment.tether_endpoint import TetherEndpoint
        for p in craft.parts:
            if isinstance(p, TetherEndpoint) and p.name == name:
                # The wrench math below treats `endpoint.mount_offset` as a
                # craft-frame offset — only true for a root-mounted
                # endpoint. A nested endpoint (under a joint/composite)
                # would get a silently wrong attachment point.
                if p.parent is not craft.root:
                    raise ValueError(
                        f"Tether: TetherEndpoint '{name}' on craft "
                        f"'{craft.name}' is attached to "
                        f"'{p.parent.name}', not the craft root. Tether "
                        f"endpoints must hang directly off the root so "
                        f"their transform is a craft-frame offset.")
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
        # Endpoint offsets in each body frame. A promoted (tunable)
        # endpoint transform reads as a trace-bound IR value — keep the
        # symbol (the bound vector's tag is the generic PartFrame; an
        # endpoint hangs directly off the root, so its coords ARE craft
        # coords, same assumption the constant path makes).
        def _off(ep):
            from ..parts._trace import is_promoted
            tr = ep.mount_offset
            if is_promoted(tr):
                return Vec3[CraftFrame].from_mx(tr._mx)
            return Vec3[CraftFrame].constant(tuple(tr))

        off_a_craft = _off(self.endpoint_a)
        off_b_craft = _off(self.endpoint_b)

        # Endpoint positions in world frame. The coupling reads each craft's
        # root ctx (root frame = CraftFrame): orientation is
        # Quat[WorldFrame, CraftFrame] and position[WorldFrame] is the craft
        # origin in world.
        off_a_anchor = ctx_a.orientation.apply(off_a_craft)
        off_b_anchor = ctx_b.orientation.apply(off_b_craft)
        p_a_anchor = ctx_a.position[WorldFrame] + off_a_anchor
        p_b_anchor = ctx_b.position[WorldFrame] + off_b_anchor

        # Vector from A to B; instantaneous length with softened sqrt.
        r = p_b_anchor - p_a_anchor
        r_mx = r._mx
        L    = soft_norm(r_mx)
        r_hat_mx = r_mx / L
        r_hat = Vec3[WorldFrame].from_mx(r_hat_mx)

        # Endpoint velocities in world frame:
        #   v_endpoint_anchor = v_origin + R · (ω_body × r_endpoint_body)
        # ω_body (craft inertial ω, in craft coords) = R^T · ω[WorldFrame].
        omega_a_craft = ctx_a.orientation.conjugate().apply(
            ctx_a.angular_velocity[WorldFrame])
        omega_b_craft = ctx_b.orientation.conjugate().apply(
            ctx_b.angular_velocity[WorldFrame])
        v_rot_a_craft = omega_a_craft.cross(off_a_craft)
        v_rot_b_craft = omega_b_craft.cross(off_b_craft)
        v_a_anchor = (ctx_a.velocity[WorldFrame]
                      + ctx_a.orientation.apply(v_rot_a_craft))
        v_b_anchor = (ctx_b.velocity[WorldFrame]
                      + ctx_b.orientation.apply(v_rot_b_craft))
        v_rel = v_b_anchor - v_a_anchor

        # Tension positive when stretched; positive v_along means A and
        # B moving apart (damper resists that).
        stretch = L - self.rest_length
        v_along_mx = ca.dot(v_rel._mx, r_hat_mx)
        T_raw = self.stiffness * stretch + self.damping * v_along_mx

        # Rope, not rod: two compact-support gates. w_taut kills the
        # whole force when slack (a slack rope neither springs nor
        # damps); w_tension kills a compressive net sum (a hard damper
        # during rapid approach must not push — the rope goes slack
        # instead). Compact support ⇒ exactly zero outside the bands.
        w_taut = hermite_blend(stretch, self.slack_smoothing)
        w_tension = hermite_blend(
            T_raw, self.stiffness * self.slack_smoothing)
        F_mag = w_taut * w_tension * T_raw

        # Force on A is along +r̂ (toward B) when taut and stretched.
        F_on_a_anchor = r_hat * F_mag
        F_on_b_anchor = F_on_a_anchor * (-1.0)

        # Rotate into each craft's frame.
        F_on_a_craft = ctx_a.orientation.conjugate().apply(F_on_a_anchor)
        F_on_b_craft = ctx_b.orientation.conjugate().apply(F_on_b_anchor)

        # Lift force-at-offset to wrench-at-craft-origin:
        #   F at offset r yields F at origin plus torque r × F about origin.
        wrench_a = Wrench(
            force=F_on_a_craft,
            torque=off_a_craft.cross(F_on_a_craft),
        )
        wrench_b = Wrench(
            force=F_on_b_craft,
            torque=off_b_craft.cross(F_on_b_craft),
        )
        return wrench_a, wrench_b

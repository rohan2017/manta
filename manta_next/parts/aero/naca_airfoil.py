"""Naca00xx — symmetric airfoil with lift + drag.

Low-fidelity flat-plate / thin-airfoil model:

    α  = angle of attack, the signed angle between the relative wind
         and the chord direction in the airfoil's plane.
    CL = CL_max · sin(2α)                  (Hoerner / flat-plate form)
    CD = CD_0  + induced_k · sin²(α)       (parabolic drag polar)

Properties:
  * Smooth and branch-free — autodiff Jacobians stay finite.
  * Bounded: |CL| ≤ CL_max for all α.
  * Symmetric about α=0 (matches NACA 00xx symmetric profiles).
  * Linear regime: for small α, CL ≈ 2·CL_max·α (effective CL slope =
    2·CL_max). Tune CL_max to hit any desired thin-airfoil slope.

The airfoil orientation in body frame is fixed by two unit vectors:
    chord_axis   — leading edge → trailing edge (the chord line).
    normal_axis  — perpendicular to chord, in the airfoil plane. Positive
                   normal is the "lifting" side: positive α (wind coming
                   from below the chord) produces positive lift along
                   +normal_axis.

What's omitted vs. legacy NACA0012-with-tables:
  * Real CL(α) curve with stall/post-stall behavior (a smoothed table).
  * Reynolds-number CD scaling.
  * Spanwise variation (this is a strip-theory single-panel surface).
  * Control surface deflection (aileron, flap). A future variant could
    add a `flap_deflection` Input that shifts the zero-lift α.
"""

from __future__ import annotations

import casadi as ca

from ...ir.frames import CraftFrame
from ...ir.types import Vec3
from ..base import Parameter, Part, PartUpdate
from ...math.wrench import Wrench


class Naca00xx(Part):
    """Symmetric airfoil (NACA 00xx family).

    Parameters:
        area             — m². Planform area of the airfoil.
        chord_axis       — body-frame unit vector along the chord
                            (leading → trailing edge). Default (1, 0, 0).
        normal_axis      — body-frame unit vector normal to the chord,
                            in the airfoil's plane. Positive α (wind
                            from below) → positive lift along this axis.
                            Default (0, 0, 1).
        CL_max           — dimensionless. Peak lift coefficient
                            (achieved at α=45° in this model). For a
                            NACA 0012 a realistic stall-limited value
                            is ~1.0–1.3.
        CD_0             — dimensionless. Zero-lift drag coefficient
                            (skin friction + pressure drag at α=0).
                            Typical: 0.005–0.015 for clean airfoils.
        induced_k        — dimensionless. Induced/lift-dependent drag
                            coefficient. CD = CD_0 + induced_k·sin²(α).
                            Higher for lower aspect-ratio wings.
    """

    area:        float = Parameter(0.1)
    chord_axis:  tuple = Parameter((1.0, 0.0, 0.0))
    normal_axis: tuple = Parameter((0.0, 0.0, 1.0))
    CL_max:      float = Parameter(1.2)
    CD_0:        float = Parameter(0.01)
    induced_k:   float = Parameter(0.1)

    def update(self, ctx) -> PartUpdate:
        # --- relative wind at the airfoil's mount point (anchor frame) ---
        offset_craft = Vec3[CraftFrame].constant(tuple(self.transform))
        offset_anchor = ctx.orientation.apply(offset_craft)
        p_anchor = ctx.position + offset_anchor

        v_rot_craft = ctx.angular_velocity.cross(offset_craft)
        v_rot_anchor = ctx.orientation.apply(v_rot_craft)
        v_surface_anchor = ctx.velocity + v_rot_anchor

        fluid = ctx.fluid_field.state_at_sym(p_anchor)
        rho   = fluid.density
        v_rel = v_surface_anchor - fluid.velocity   # Vec3[AnchorFrame]

        # --- decompose v_rel into the airfoil's body-frame axes -----------
        # Rotate v_rel into CraftFrame.
        v_rel_craft = ctx.orientation.conjugate().apply(v_rel)

        chord_b  = Vec3[CraftFrame].constant(tuple(self.chord_axis))
        normal_b = Vec3[CraftFrame].constant(tuple(self.normal_axis))

        # Work in raw MX from here — CasADi's sqrt/div/etc. want MX, not
        # the typed Scalar/Vec3 wrappers.
        v_rel_mx = v_rel_craft._mx
        chord_mx = chord_b._mx
        normal_mx = normal_b._mx

        v_chord_mx  = ca.dot(v_rel_mx, chord_mx)
        v_normal_mx = ca.dot(v_rel_mx, normal_mx)

        # 2-D projected speed in the chord-normal plane. Softened sqrt
        # keeps Jacobians regular at v=0.
        v2d_sq = v_chord_mx**2 + v_normal_mx**2 + 1e-30
        v2d    = ca.sqrt(v2d_sq)

        # Angle of attack: positive when wind hits the +normal side of the
        # chord (wind from below in a "+normal = up" orientation). Wind
        # vector is `-v_rel`, so wind·normal = -v_normal_mx. AoA is the
        # angle from chord to wind in the chord-normal plane; positive
        # AoA means wind has positive +normal component, i.e. v_normal_mx
        # negative.
        #
        #   sin(α) = (-v_normal_mx) / v2d
        #   cos(α) =   v_chord_mx  / v2d
        #
        # So sin(2α) = 2·sin(α)·cos(α) = -2·v_normal_mx·v_chord_mx / v2d²
        # and sin²(α) = v_normal_mx² / v2d² (unchanged).
        sin_2alpha   = -2.0 * v_normal_mx * v_chord_mx / v2d_sq
        sin_sq_alpha = v_normal_mx**2 / v2d_sq

        CL = self.CL_max * sin_2alpha
        CD = self.CD_0 + self.induced_k * sin_sq_alpha

        # Dynamic pressure (rho is a CasADi MX from the FluidField).
        # Note: `rho` may be a python float (when no fluid field
        # registered → zero); coerce via multiplication.
        q_dyn = 0.5 * rho * v2d_sq

        L_mag = q_dyn * self.area * CL
        D_mag = q_dyn * self.area * CD

        # Drag direction in body frame: along -v_rel (opposing motion
        # through fluid), projected onto the chord-normal plane.
        d_hat_mx = -(chord_mx * (v_chord_mx / v2d)
                      + normal_mx * (v_normal_mx / v2d))
        # Lift direction: perpendicular to drag in the chord-normal plane,
        # oriented so positive AoA (wind from below the chord) produces
        # lift along +normal_axis.
        #
        # At α=0 (v_normal=0): drag = -chord, lift should be along +normal.
        # Lift = (-v_normal·chord + v_chord·normal) / v2d satisfies this:
        # at α=0, lift = (0 + v_chord·normal)/v2d = +normal. ✓
        # At positive AoA (v_normal > 0, v_chord > 0): lift·normal > 0. ✓
        l_hat_mx = (-chord_mx  * (v_normal_mx / v2d)
                     + normal_mx * (v_chord_mx  / v2d))

        F_body_mx = D_mag * d_hat_mx + L_mag * l_hat_mx
        F_craft   = Vec3[CraftFrame].from_mx(F_body_mx)

        zero_t = Vec3[CraftFrame].constant((0.0, 0.0, 0.0))
        return PartUpdate(wrench=Wrench(force=F_craft, torque=zero_t))

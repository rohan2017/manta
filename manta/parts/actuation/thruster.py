"""Thruster — linear + quadratic in throttle.

    F(t) = force · t + force_quad · t²
    τ(t) = torque · t + torque_quad · t²

`force` and `torque` cover the standard linear case (`F = throttle · v`,
common for an EDF or scaled-thrust model); `force_quad` and
`torque_quad` cover the rotor-blade case (`F = K_T · throttle²`, with
reaction torque `τ = K_Q · throttle²`).

    Thruster("edf",   force=(0, 0, 10))
    Thruster("blade", force_quad=(0, 0, K_T), torque_quad=(0, 0, K_Q))
    Thruster("prop",  force=(0, 0, 10), torque=(0, 0, 0.5))

The force is applied at the mount offset (Part.transform); the
framework lifts that to a body-origin wrench, so off-axis thrusters
produce correct body torques automatically.
"""

from __future__ import annotations

from ...ir.frames import CraftFrame
from ...ir.types import Vec3
from ..base import Input, Parameter, Part
from ...ir.wrench import Wrench


class Thruster(Part):
    """Polynomial-in-throttle thruster (linear + quadratic).

    Coefficients are 3-vectors in the thruster's own frame. For a thruster
    mounted directly on the craft root that frame is CraftFrame, so
    `Thruster("t", force=(0,0,1))` is a pure +z thrust in body coords.
    Mounted on a Joint's rotor, the thruster's frame spins with the rotor
    and the framework rotates the emitted wrench into body coords — a
    gimballed thruster's thrust direction tracks the joint angle
    automatically, with no frame handling here. Any unset coefficient
    defaults to zero.

    Input:
        throttle — scalar control input. Units depend on the scaling
                   of the coefficients.
    """

    force:       "tuple[float, float, float]" = Parameter((0.0, 0.0, 0.0))
    force_quad:  "tuple[float, float, float]" = Parameter((0.0, 0.0, 0.0))
    torque:      "tuple[float, float, float]" = Parameter((0.0, 0.0, 0.0))
    torque_quad: "tuple[float, float, float]" = Parameter((0.0, 0.0, 0.0))
    throttle: float = Input(default=0.0)

    def update(self, ctx):
        t  = self.throttle
        t2 = t * t

        c1F = Vec3[CraftFrame].constant(self.force)
        c2F = Vec3[CraftFrame].constant(self.force_quad)
        c1τ = Vec3[CraftFrame].constant(self.torque)
        c2τ = Vec3[CraftFrame].constant(self.torque_quad)

        return Wrench(
            force =c1F * t + c2F * t2,
            torque=c1τ * t + c2τ * t2,
        )

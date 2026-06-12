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

from ...ir.frames import PartFrame
from ...ir.types import Vec3
from .._declarations import Input, Parameter, WhiteNoise
from ..base import Part
from ...ir.wrench import Wrench


class Thruster(Part):
    """Polynomial-in-throttle thruster (linear + quadratic).

    Coefficients are 3-vectors in the thruster's own frame. For a thruster
    mounted directly on the craft root that frame is CraftFrame, so
    `Thruster("t", force=(0,0,1))` is a pure +z thrust in body coords.
    Mounted on a joint's rotor, the thruster's frame spins with the rotor
    and the framework rotates the emitted wrench into body coords — a
    gimballed thruster's thrust direction tracks the joint angle
    automatically, with no frame handling here. Any unset coefficient
    defaults to zero.

    Input:
        throttle — scalar control input. Units depend on the scaling
                   of the coefficients.

    Process-noise channels (the actuator analogue of a sensor's noise —
    set σ to engage, default 0 = a perfectly clean actuator):
        force_noise  : per-tick white force (N) added to the thrust, in
                       the thruster frame. Because it enters the *wrench*
                       (not an Output) it propagates through the dynamics
                       into the next state, so the EKF auto-builds `Q`
                       from it (just as σ on a sensor auto-builds `R`),
                       and a `NoiseDriver` jitters the truth thrust by it.
        torque_noise : per-tick white torque (N·m) added to the reaction
                       torque, same frame and same role for attitude.
    """

    # All four coefficients are promotable (system-ID targets): a tunable
    # transform constructed with `parameters=[...]` promotes them to live
    # graph inputs; `coerce` below consumes either form.
    force:       "tuple[float, float, float]" = Parameter(
        (0.0, 0.0, 0.0), manifold="R3", frame=PartFrame)
    force_quad:  "tuple[float, float, float]" = Parameter(
        (0.0, 0.0, 0.0), manifold="R3", frame=PartFrame)
    torque:      "tuple[float, float, float]" = Parameter(
        (0.0, 0.0, 0.0), manifold="R3", frame=PartFrame)
    torque_quad: "tuple[float, float, float]" = Parameter(
        (0.0, 0.0, 0.0), manifold="R3", frame=PartFrame)
    throttle: float = Input(default=0.0)

    force_noise  = WhiteNoise("R3", frame=PartFrame, sigma=0.0)
    torque_noise = WhiteNoise("R3", frame=PartFrame, sigma=0.0)

    def update(self, ctx):
        t  = self.throttle
        t2 = t * t

        c1F = Vec3[PartFrame].coerce(self.force)
        c2F = Vec3[PartFrame].coerce(self.force_quad)
        c1τ = Vec3[PartFrame].coerce(self.torque)
        c2τ = Vec3[PartFrame].coerce(self.torque_quad)

        return Wrench(
            force =c1F * t + c2F * t2 + self.force_noise,
            torque=c1τ * t + c2τ * t2 + self.torque_noise,
        )

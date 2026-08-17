"""ControlSurface — a wing section with a movable trailing-edge flap,
modelled as ONE coupled aerodynamic surface (not two hinged foils).

Why this exists. A trailing-edge control surface does not just add a small
local panel force: deflecting it changes the circulation over the *whole*
section — its effective camber, its zero-lift angle, and its pitching
moment. Modelling a wing + aileron as two independent `Aerofoil`s hinged
on a `RevoluteJoint` both misses that coupling AND pays for an articulated
multibody solve (a (3+K)×(3+K) factorisation) plus two state DOFs (joint
angle + rate) per surface — heavy for a panel that is a few percent of the
airframe mass.

`ControlSurface` instead extends `Aerofoil` with thin-airfoil **flap
theory**: a deflection δ shifts the effective zero-lift line by τ·δ and
adds a moment Cm_δ·δ, where the flap-effectiveness factor τ and Cm_δ come
from the flap-chord fraction. The combined section produces a single
wrench, the craft stays a flat (fast-path, SPD-solve) body, and the
surface costs ONE state instead of two.

Hinge dynamics (massless, quasi-static — one state δ). The servo commands
a deflection; a clamped first-order balance drives δ there:

    b·δ̇ = clamp( K_servo·(δ_cmd − δ), ±τ_stall ) + H_aero(α, δ, q)

with H_aero the (restoring) aerodynamic hinge moment. This gives, from one
equation: a first-order servo lag (bandwidth K_servo/b), torque saturation
at ±τ_stall, and **blowback** — in a fast dive H_aero ∝ q = ½ρV² can
exceed the stall torque, pushing the surface back toward float and bleeding
control authority, exactly as a real under-powered servo does. Because
H_aero scales with fluid density, the same surface blows back far more
readily underwater than in thin air, with no special-casing.
"""

from __future__ import annotations

import math

import casadi as ca

from ...ir.types import Scalar
from ...ir.wrench import Wrench
from .._declarations import Input, Parameter, PartUpdate, State
from .._trace import scalar_mx as _as_mx
from .aerofoil import Aerofoil
from ..base import PartRole


def _flap_effectiveness(E: float) -> tuple[float, float]:
    """Thin-airfoil flap derivatives for a trailing-edge flap of chord
    fraction ``E = c_flap / c``: the lift-effectiveness factor τ (so the
    zero-lift line shifts by τ·δ) and the quarter-chord moment slope Cm_δ.

    From the Glauert hinge angle ``cos θ_h = 2E − 1``:
        τ    = (π − θ_h + sin θ_h) / π          (≈0.6 for E=0.25)
        Cm_δ = sin θ_h · (cos θ_h − 1) / 2       (nose-down, < 0)
    """
    theta_h = math.acos(max(-1.0, min(1.0, 2.0 * E - 1.0)))
    tau = (math.pi - theta_h + math.sin(theta_h)) / math.pi
    cm_delta = math.sin(theta_h) * (math.cos(theta_h) - 1.0) / 2.0
    return tau, cm_delta


class ControlSurface(Aerofoil):
    """A wing section with a deflectable trailing-edge flap.

    Inherits all `Aerofoil` geometry/aero parameters (these describe the
    WING section — `area`, `chord`, `chord_axis`, `normal_axis`, `alpha_0`,
    `Cm_ac`, `CL_max`, …); the flap is parameterised by its chord fraction,
    not a separate foil. The deflection is a single state driven by a
    commanded angle through a saturating first-order servo.

    Flap + servo parameters:
        flap_chord_fraction — c_flap / c, in (0, 1). Sets the flap
                              effectiveness τ and moment slope Cm_δ.
        servo_gain          — K_servo, hinge restoring torque per rad of
                              command error (N·m/rad).
        stall_torque        — τ_stall, the servo's saturation torque (N·m).
                              Aero hinge moment beyond this blows the
                              surface back.
        hinge_damping       — b, hinge viscous damping (N·m·s/rad); with
                              servo_gain it sets the lag bandwidth K/b.
        max_deflection      — travel limit, rad (the state saturates here).
        Ch_alpha, Ch_delta  — hinge-moment coefficient slopes wrt angle of
                              attack and deflection (both restoring, < 0).
                              Engineering values; tune per surface.

    Input:
        deflection_cmd — commanded deflection, rad.
    State:
        deflection     — actual deflection δ, rad.
    """

    role = PartRole.ACTUATOR

    flap_chord_fraction: float = Parameter(0.25)
    servo_gain:          float = Parameter(20.0)
    stall_torque:        float = Parameter(5.0)
    hinge_damping:       float = Parameter(0.3)
    max_deflection:      float = Parameter(math.radians(25.0))
    Ch_alpha:            float = Parameter(-0.10)
    Ch_delta:            float = Parameter(-0.60)

    deflection_cmd: float = Input(default=0.0)
    deflection = State(init=0.0, manifold="R1")

    def __init__(self, name: str, **overrides) -> None:
        super().__init__(name, **overrides)
        who = f"{type(self).__name__} {name!r}"
        E = float(self.flap_chord_fraction)
        if not (0.0 < E < 1.0):
            raise ValueError(
                f"{who}: flap_chord_fraction must be in (0, 1), got {E!r}")
        if float(self.hinge_damping) <= 0.0:
            raise ValueError(
                f"{who}: hinge_damping must be > 0, got "
                f"{self.hinge_damping!r}")
        if float(self.stall_torque) < 0.0 or float(self.servo_gain) < 0.0:
            raise ValueError(
                f"{who}: stall_torque and servo_gain must be >= 0")
        # Build-time flap derivatives (Re-independent geometry).
        self._tau, self._cm_delta = _flap_effectiveness(E)

    def update(self, ctx) -> PartUpdate:
        delta = _as_mx(self.deflection)

        # Deflection's aero effect: shift the zero-lift line by τ·δ (a
        # positive δ — trailing edge down — adds lift, so it lowers the
        # zero-lift angle), and add the flap pitching moment Cm_δ·δ.
        d_alpha0 = -self._tau * delta
        d_Cm     = self._cm_delta * delta

        F_part, tau_part, alpha, q_dyn = self._aero_wrench(ctx, d_alpha0, d_Cm)

        # Quasi-static hinge balance → δ̇ (massless, so no δ̈ state).
        E = float(self.flap_chord_fraction)
        cmd = _as_mx(self.deflection_cmd)
        servo = self._servo_torque(cmd - delta)
        # Restoring aerodynamic hinge moment: H = q·S_f·c_f·(Chα·α + Chδ·δ),
        # with the flap area/chord a fraction E of the section's.
        H_aero = (q_dyn * (E * float(self.area)) * (E * float(self.chord))
                  * (float(self.Ch_alpha) * alpha
                     + float(self.Ch_delta) * delta))
        ddelta = (servo + H_aero) / float(self.hinge_damping)

        new_delta = delta + ddelta * _as_mx(ctx.dt)
        lim = float(self.max_deflection)
        new_delta = ca.fmin(ca.fmax(new_delta, -lim), lim)

        return PartUpdate(wrench=Wrench(force=F_part, torque=tau_part),
                          new_state={"deflection": Scalar(new_delta)})

    def _servo_torque(self, command_error):
        """Hinge torque produced by the servo before aerodynamic load."""
        stall = float(self.stall_torque)
        return ca.fmin(ca.fmax(float(self.servo_gain) * command_error,
                               -stall), stall)

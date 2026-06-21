"""Aerofoil — a cambered, Reynolds-aware thin-airfoil surface.

A smooth, branch-free lift/drag/moment model that generalises the old
symmetric `Naca00xx` along two axes the rest of `manta` now needs:

  * **Camber.** A non-zero zero-lift angle `alpha_0` and a constant
    moment-about-the-aerodynamic-centre `Cm_ac`, so a cambered foil lifts
    at zero geometric angle of attack and carries a pitching couple.

  * **Reynolds dependence.** The same airframe flies in air, water, and
    thin extraterrestrial atmospheres — three orders of magnitude in Re.
    The geometry-derived invariants (`alpha_0`, `CL_alpha`, `Cm_ac`) are
    ~Re-independent and come straight from thin-airfoil theory; the
    profile drag `CD_0` and the stall ceiling `CL_max` are NOT, and are
    rescaled from their reference values by the local Reynolds number
    (computed from the fluid's density, viscosity, and the foil chord).

Lift model (signed angle of attack α about the zero-lift line):

    α       = atan2(−v_normal, v_chord)                    (signed AoA)
    CLmax_e = CL_max · clmax_reynolds_factor(Re)           (Re-scaled cap)
    x       = (CL_alpha / CLmax_e) · (α − alpha_0)
    CL      = CLmax_e · sin(x)        # slope CL_alpha at α₀, peak CLmax_e
    CD      = CD_0 · cd0_reynolds_factor(Re) + induced_k · sin²(x)
    Cm      = Cm_ac                   # constant couple about the AC

Decoupling the lift-curve slope `CL_alpha` (≈2π, Re-independent) from the
stall ceiling `CL_max` (Re-dependent) is what the single `sin` cannot do
on its own; the `CL_alpha/CL_max` argument scaling buys it — and as a
bonus, a lower `CL_max` at low Re also pulls the stall angle in, which is
physically right. Past the (single) lift peak the `sin` keeps the model
smooth but is only a surrogate; deep post-stall (|α − α₀| beyond the
peak) is out of the model's fidelity, as it was for `Naca00xx`.

Orientation is fixed by the same `chord_axis` / `normal_axis` pair as
`Naca00xx`; the span (pitch) axis is `chord × normal`, about which the
`Cm_ac` couple acts.
"""

from __future__ import annotations

import math

import casadi as ca

from ...fields import FluidField
from ...ir.frames import PartFrame, WorldFrame
from ...ir.types import Vec3
from ...ir.wrench import Wrench
from ...smoothing import NORM_EPS_SQ, smoothstep
from .._declarations import Parameter, PartUpdate, unit_axis
from ..base import Part

# Reynolds-scaling reference points (engineering fits, not high-fidelity
# tables). All smooth in log10(Re) so Jacobians stay finite.
_RE_FLOOR: float = 1.0           # keep Re ≥ this inside divisions/powers
_MU_FLOOR: float = 1e-9          # Pa·s — guards a vacuum (μ=0) query
_RE_TRANSITION: float = 5.0e5    # laminar→turbulent skin-friction crossover
_RE_TRANS_HALFWIDTH: float = 0.6  # decades, half-width of the crossover blend
# Stall ceiling ramp: CL_max collapses toward a floor at low Re.
_CLMAX_RE_LO: float = 2.0e4      # below here, the stall floor dominates
_CLMAX_RE_HI: float = 1.0e6      # at/above here, the full reference CL_max
_CLMAX_FLOOR_FRAC: float = 0.5   # CL_max never falls below this fraction


def reynolds_number(density, speed, chord, viscosity):
    """``Re = ρ·V·c / μ`` (μ floored so a vacuum query stays finite)."""
    return density * speed * chord / ca.fmax(viscosity, _MU_FLOOR)


def _skin_friction(Re):
    """Flat-plate skin-friction coefficient, a C¹ blend of the laminar
    (Blasius ``1.328/√Re``) and turbulent (``0.074/Re^0.2``) branches
    across the transition Reynolds number."""
    Re = ca.fmax(Re, _RE_FLOOR)
    cf_lam = 1.328 / ca.sqrt(Re)
    cf_turb = 0.074 / Re ** 0.2
    # Blend on log10(Re): 0 (all-laminar) below the band, 1 above it.
    s = (ca.log10(Re) - math.log10(_RE_TRANSITION)) / _RE_TRANS_HALFWIDTH
    w = smoothstep(s)
    return (1.0 - w) * cf_lam + w * cf_turb


def cd0_reynolds_factor(Re, Re_ref: float = _RE_TRANSITION):
    """Multiplicative profile-drag scaling vs a reference Reynolds number:
    ``CD_0(Re) = CD_0_ref · cf(Re)/cf(Re_ref)`` — drag rises as Re drops."""
    return _skin_friction(Re) / float(ca.evalf(_skin_friction(ca.MX(Re_ref))))


def clmax_reynolds_factor(Re):
    """Stall-ceiling scaling in [floor, 1]: full CL_max at high Re, decaying
    smoothly to `_CLMAX_FLOOR_FRAC` of it below `_CLMAX_RE_LO`."""
    Re = ca.fmax(Re, _RE_FLOOR)
    s = (2.0 * (ca.log10(Re) - math.log10(_CLMAX_RE_LO))
         / (math.log10(_CLMAX_RE_HI) - math.log10(_CLMAX_RE_LO)) - 1.0)
    ramp = smoothstep(s)                       # 0 below Re_lo, 1 above Re_hi
    return _CLMAX_FLOOR_FRAC + (1.0 - _CLMAX_FLOOR_FRAC) * ramp


class Aerofoil(Part):
    """A cambered, Reynolds-aware lifting surface.

    Parameters:
        area         — m². Planform (reference) area.
        chord        — m. Mean aerodynamic chord; the Reynolds reference
                       length (and, for a `ControlSurface`, the wing chord
                       a flap is measured against).
        chord_axis   — body-frame unit vector along the chord (leading →
                       trailing edge). Default (1, 0, 0).
        normal_axis  — body-frame unit vector normal to the chord, in the
                       foil plane; the +lift side. Default (0, 0, 1).
        alpha_0      — zero-lift angle of attack, rad. 0 for a symmetric
                       foil; negative for positive camber. From the camber
                       line (Re-independent); `naca(...)` fills it.
        CL_alpha     — lift-curve slope, per rad. Thin-airfoil ≈ 2π.
        Cm_ac        — moment coefficient about the aerodynamic centre
                       (constant; nose-down negative for positive camber).
        CL_max       — reference (high-Re) stall lift coefficient. Scaled
                       DOWN at low Reynolds number.
        CD_0         — reference (at Re ≈ 5·10⁵) zero-lift drag coefficient.
                       Scaled by the local Reynolds number.
        induced_k    — induced/lift-dependent drag factor.
    """

    area:        float = Parameter(0.1)
    chord:       float = Parameter(0.2)
    chord_axis:  tuple = Parameter((1.0, 0.0, 0.0))
    normal_axis: tuple = Parameter((0.0, 0.0, 1.0))
    alpha_0:     float = Parameter(0.0)
    CL_alpha:    float = Parameter(2.0 * math.pi)
    Cm_ac:       float = Parameter(0.0)
    CL_max:      float = Parameter(1.3)
    CD_0:        float = Parameter(0.01)
    induced_k:   float = Parameter(0.1)

    def __init__(self, name: str, **overrides) -> None:
        super().__init__(name, **overrides)
        who = f"{type(self).__name__} {name!r}"
        if float(self.area) <= 0.0:
            raise ValueError(f"{who}: area must be > 0, got {self.area!r}")
        if float(self.chord) <= 0.0:
            raise ValueError(f"{who}: chord must be > 0, got {self.chord!r}")
        if float(self.CL_max) <= 0.0:
            raise ValueError(f"{who}: CL_max must be > 0, got {self.CL_max!r}")
        if float(self.CD_0) < 0.0 or float(self.induced_k) < 0.0:
            raise ValueError(
                f"{who}: CD_0 and induced_k must be >= 0, got "
                f"CD_0={self.CD_0!r}, induced_k={self.induced_k!r}")
        self.chord_axis  = unit_axis(self.chord_axis,
                                     who=who, what="chord_axis")
        self.normal_axis = unit_axis(self.normal_axis,
                                     who=who, what="normal_axis")
        dot = sum(c * n for c, n in zip(self.chord_axis, self.normal_axis))
        if abs(dot) > 1e-6:
            raise ValueError(
                f"{who}: chord_axis and normal_axis must be perpendicular "
                f"(unit-vector dot product is {dot:.3e}).")

    # The deflection-dependent aero shifts live here so a ControlSurface
    # subclass can add the flap contribution without re-deriving the wind.
    def _aero_shift(self, ctx):
        """(Δalpha_0, ΔCm) applied on top of the foil's own camber. Zero
        for a plain Aerofoil; a control surface overrides it."""
        return ca.MX(0.0), ca.MX(0.0)

    def update(self, ctx) -> PartUpdate:
        p_world          = ctx.position[WorldFrame]
        v_surface_anchor = ctx.velocity[WorldFrame]

        fluid = ctx.field(FluidField).value_at_sym(p_world, ctx.t)
        rho   = fluid.density
        mu    = fluid.viscosity
        v_rel = v_surface_anchor - fluid.velocity

        v_rel_craft = ctx.orientation.conjugate().apply(v_rel)
        chord_b  = Vec3[PartFrame].constant(tuple(self.chord_axis))
        normal_b = Vec3[PartFrame].constant(tuple(self.normal_axis))
        v_rel_mx = v_rel_craft._mx
        chord_mx = chord_b._mx
        normal_mx = normal_b._mx

        v_chord_mx  = ca.dot(v_rel_mx, chord_mx)
        v_normal_mx = ca.dot(v_rel_mx, normal_mx)
        v2d_sq = v_chord_mx**2 + v_normal_mx**2 + NORM_EPS_SQ
        v2d    = ca.sqrt(v2d_sq)

        # Signed angle of attack about the chord line (wind = −v_rel, so a
        # positive AoA — wind from the +normal side — has v_normal < 0).
        alpha = ca.atan2(-v_normal_mx, v_chord_mx)

        # Reynolds number from the local fluid + chord, and the Re-scaled
        # stall ceiling / profile drag.
        Re = reynolds_number(rho, v2d, float(self.chord), mu)
        CLmax_eff = float(self.CL_max) * clmax_reynolds_factor(Re)
        CD0_eff = float(self.CD_0) * cd0_reynolds_factor(Re)

        # Deflection shifts (flap): effective zero-lift line + extra moment.
        d_alpha0, d_Cm = self._aero_shift(ctx)
        alpha_eff = alpha - (float(self.alpha_0) + d_alpha0)

        x  = (float(self.CL_alpha) / CLmax_eff) * alpha_eff
        sx = ca.sin(x)
        CL = CLmax_eff * sx
        CD = CD0_eff + float(self.induced_k) * sx * sx
        Cm = float(self.Cm_ac) + d_Cm

        q_dyn = 0.5 * rho * v2d_sq
        L_mag = q_dyn * float(self.area) * CL
        D_mag = q_dyn * float(self.area) * CD

        d_hat_mx = -(chord_mx * (v_chord_mx / v2d)
                      + normal_mx * (v_normal_mx / v2d))
        l_hat_mx = (-chord_mx  * (v_normal_mx / v2d)
                     + normal_mx * (v_chord_mx  / v2d))
        F_body_mx = D_mag * d_hat_mx + L_mag * l_hat_mx
        F_part    = Vec3[PartFrame].from_mx(F_body_mx)

        # Pitching couple about the span axis (chord × normal): M = q·S·c·Cm.
        span_mx = ca.cross(chord_mx, normal_mx)
        M_mag   = q_dyn * float(self.area) * float(self.chord) * Cm
        tau_part = Vec3[PartFrame].from_mx(M_mag * span_mx)
        return PartUpdate(wrench=Wrench(force=F_part, torque=tau_part))


# ---------------------------------------------------------------------------
# NACA four-digit helper — derive the Re-independent invariants from the
# foil designation via thin-airfoil theory (numerically integrated once,
# at build time — these are plain Python floats, not per-step symbolics).
# ---------------------------------------------------------------------------

def _four_digit_invariants(m: float, p: float):
    """Thin-airfoil zero-lift angle (rad) and quarter-chord moment for a
    NACA 4-digit camber line with max camber `m` (fraction of chord) at
    chordwise position `p` (fraction). Symmetric (m=0) → (0, 0)."""
    if m == 0.0 or p == 0.0:
        return 0.0, 0.0
    import numpy as np
    theta = np.linspace(1e-6, math.pi - 1e-6, 4000)
    x = 0.5 * (1.0 - np.cos(theta))           # x/c via the Glauert map
    dyc_dx = np.where(
        x < p,
        (2.0 * m / p**2) * (p - x),
        (2.0 * m / (1.0 - p) ** 2) * (p - x),
    )

    def integ(integrand):
        return np.trapezoid(integrand, theta)

    alpha_L0 = -(1.0 / math.pi) * integ(dyc_dx * (np.cos(theta) - 1.0))
    A1 = (2.0 / math.pi) * integ(dyc_dx * np.cos(theta))
    A2 = (2.0 / math.pi) * integ(dyc_dx * np.cos(2.0 * theta))
    Cm_c4 = (math.pi / 4.0) * (A2 - A1)
    return float(alpha_L0), float(Cm_c4)


def naca(designation: str, name: str | None = None, **geom) -> Aerofoil:
    """Build an `Aerofoil` for a NACA 4-digit section (e.g. ``"2412"``,
    ``"0012"``).

    The first digit is max camber in % chord, the second its chordwise
    position in tenths, the last two the thickness in % chord. The
    Re-independent invariants — zero-lift angle `alpha_0` and moment
    `Cm_ac` — are derived from the camber line by thin-airfoil theory;
    the reference `CL_max` and `CD_0` are estimated from thickness (and
    are themselves rescaled by Reynolds number at run time). Any of these
    may be overridden, along with the geometry (`area`, `chord`,
    `chord_axis`, `normal_axis`, `induced_k`), via keyword.

    Example::

        a.add(naca("2412", "wing", area=0.72, chord=0.3))
    """
    d = designation.strip().upper().removeprefix("NACA").strip()
    if len(d) != 4 or not d.isdigit():
        raise ValueError(
            f"naca: expected a 4-digit designation like '2412', got "
            f"{designation!r}")
    m = int(d[0]) / 100.0          # max camber, fraction of chord
    p = int(d[1]) / 10.0           # position of max camber, fraction
    t = int(d[2:4]) / 100.0        # thickness, fraction of chord

    alpha_0, Cm_ac = _four_digit_invariants(m, p)
    # Thickness fits (engineering): form-factor profile drag and a stall
    # ceiling that grows mildly with thickness. Both are reference (high-Re)
    # values; the part rescales them by the local Reynolds number.
    CD_0 = 0.006 * (1.0 + 2.0 * t + 60.0 * t**4)
    CL_max = 1.2 + 1.5 * t

    params = dict(alpha_0=alpha_0, Cm_ac=Cm_ac, CL_alpha=2.0 * math.pi,
                  CL_max=CL_max, CD_0=CD_0)
    params.update(geom)            # caller overrides win
    return Aerofoil(name or f"naca{d}", **params)

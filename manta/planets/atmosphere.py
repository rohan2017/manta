"""Planet-independent fluid-column physics — ISA atmosphere + hydrostatic.

These are pure equations over symbolic (or numeric) scalars; they hold no
planet state, so any planet can wire them into an `Atmosphere` / `Ocean`
regime with its own constants (Earth's ISA values are the defaults here;
a Mars or Titan model supplies its own T0 / P0 / lapse / gas constant).

The atmosphere model is the standard ISA troposphere: a linear
temperature lapse with altitude, hydrostatic balance, and the ideal-gas
law — a mutually consistent `(T, P, ρ)` triple, so a barometer,
thermometer, and density-driven drag all read the same physical state
(no over-constraint). The ocean is treated as incompressible: constant
density with hydrostatic pressure growing with depth.
"""

from __future__ import annotations

import casadi as ca

# ISA sea-level reference (Earth defaults; override per planet).
T0_ISA: float = 288.15        # K   — sea-level temperature
P0_ISA: float = 101325.0      # Pa  — sea-level pressure
R_AIR: float = 287.05         # J/(kg·K) — specific gas constant, dry air
LAPSE_ISA: float = 6.5e-3     # K/m — troposphere temperature lapse rate

# Sutherland's law constants for dry air (the default gas species). A
# different gas (CO₂ on Mars, N₂ on Titan) supplies its own triple.
MU_REF_AIR: float = 1.716e-5  # Pa·s — reference dynamic viscosity
T_REF_SUTHERLAND: float = 273.15   # K — reference temperature for MU_REF
S_SUTHERLAND_AIR: float = 110.4    # K — Sutherland constant for air

# Sea-level density implied by the reference (P0/(R·T0) ≈ 1.225 kg/m³,
# the conventional ISA value) — handy as a default and a cross-check.
RHO0_ISA: float = P0_ISA / (R_AIR * T0_ISA)

# Keep temperature strictly positive inside powers / divisions so the
# Jacobian stays finite if a query strays above the troposphere (where
# the linear-lapse model is out of its validity anyway).
_T_FLOOR: float = 1.0         # K


def isa_temperature(altitude, T0: float = T0_ISA, lapse: float = LAPSE_ISA):
    """ISA troposphere temperature at `altitude` (m above the surface):
    ``T = T0 − lapse·altitude``."""
    return T0 - lapse * altitude


def isa_pressure(altitude,
                 P0: float = P0_ISA,
                 T0: float = T0_ISA,
                 lapse: float = LAPSE_ISA,
                 g: float = 9.80665,
                 R: float = R_AIR):
    """ISA troposphere pressure at `altitude` (m): the exact hydrostatic
    + ideal-gas solution under a linear lapse,
    ``P = P0·(T/T0)^(g/(R·lapse))`` with ``T = T0 − lapse·altitude``.

    The temperature is clamped at 0 *inside the power*, so above the
    troposphere — where the linear lapse predicts `T ≤ 0` (≈44 km for
    Earth ISA) — pressure goes smoothly to **zero** and stays there
    (density follows). The model is only physical up to the tropopause;
    this just keeps it sane (and vanishing in space) beyond."""
    T = ca.fmax(isa_temperature(altitude, T0, lapse), 0.0)
    exponent = g / (R * lapse)
    return P0 * (T / T0) ** exponent


def ideal_gas_density(pressure, temperature, R: float = R_AIR):
    """Ideal-gas density ``ρ = P / (R·T)`` (temperature floored positive)."""
    return pressure / (R * ca.fmax(temperature, _T_FLOOR))


def sutherland_viscosity(temperature,
                         mu_ref: float = MU_REF_AIR,
                         T_ref: float = T_REF_SUTHERLAND,
                         S: float = S_SUTHERLAND_AIR):
    """Gas dynamic viscosity μ(T) via Sutherland's law,
    ``μ = mu_ref·(T/T_ref)^1.5·(T_ref + S)/(T + S)`` (Pa·s).

    Viscosity of a dilute gas is essentially pressure-independent, so this
    keys on temperature alone. The defaults are dry air (μ ≈ 1.79e-5 Pa·s
    at the ISA sea-level 288.15 K); a different gas species passes its own
    ``mu_ref`` / ``T_ref`` / ``S``. Temperature is floored positive so the
    Jacobian stays finite above the troposphere (and at an unset T=0).
    Liquids do NOT follow this law — water sets its viscosity explicitly.
    """
    T = ca.fmax(temperature, _T_FLOOR)
    return mu_ref * (T / T_ref) ** 1.5 * (T_ref + S) / (T + S)


def hydrostatic_pressure(P_surface, rho, g, depth):
    """Incompressible hydrostatic pressure a `depth` (m, positive
    downward) below a surface at pressure `P_surface`:
    ``P = P_surface + ρ·g·depth``."""
    return P_surface + rho * g * depth

"""Aerofoil — cambered, Reynolds-aware lifting surface, and the naca()
four-digit helper."""

import math

import casadi as ca
import numpy as np
import pytest

from manta import Craft, Sim, TargetNumpy, World
from manta.fields import FluidField, GravityField
from manta.parts import Aerofoil, Mass, naca
from manta.parts.aero.aerofoil import (
    cd0_reynolds_factor,
    clmax_reynolds_factor,
    reynolds_number,
)


def _air(rho=1.225, **kw):
    return FluidField().add_uniform(density=rho, **kw)


def _fly(part, velocity, fluid=None, dt=0.001, steps=1, moi=(0.1, 0.1, 0.1)):
    w = (World()
         .add_field(GravityField().add_uniform((0, 0, 0)))
         .add_field(fluid if fluid is not None else _air()))
    c = Craft("foil")
    c.add(Mass("body", mass=1.0, moi=moi))
    c.add(part)
    w.add_craft(c, velocity=velocity)
    cw = TargetNumpy(Sim(w))
    state = None
    for _ in range(steps):
        state = cw.step(dt=dt)
    return state


# --- naca() four-digit invariants ------------------------------------------

def test_naca_symmetric_has_no_camber():
    f = naca("0012", area=1.0, chord=0.3)
    assert abs(f.alpha_0) < 1e-9
    assert abs(f.Cm_ac) < 1e-9


def test_naca_2412_matches_published_invariants():
    """Thin-airfoil theory for NACA 2412: zero-lift angle ≈ −2.1°,
    quarter-chord moment ≈ −0.05."""
    f = naca("2412", area=1.0, chord=0.3)
    np.testing.assert_allclose(math.degrees(f.alpha_0), -2.1, atol=0.3)
    np.testing.assert_allclose(f.Cm_ac, -0.05, atol=0.01)


def test_naca_more_camber_more_negative_alpha0():
    """4412 has twice the camber of 2412 → ~twice the zero-lift angle and
    moment magnitude."""
    f24 = naca("2412", area=1.0, chord=0.3)
    f44 = naca("4412", area=1.0, chord=0.3)
    assert f44.alpha_0 < f24.alpha_0 < 0.0
    assert f44.Cm_ac < f24.Cm_ac < 0.0


def test_naca_designation_validation():
    for bad in ("241", "24123", "24a2", "wing"):
        with pytest.raises(ValueError):
            naca(bad)
    # 'NACA' prefix is tolerated.
    assert naca("NACA2412", area=1.0).alpha_0 < 0.0


def test_naca_overrides_win():
    f = naca("2412", area=2.0, chord=0.5, CL_max=1.9, induced_k=0.2)
    assert f.area == 2.0 and f.chord == 0.5
    assert f.CL_max == 1.9 and f.induced_k == 0.2


# --- camber → lift at zero geometric AoA -----------------------------------

def test_cambered_foil_lifts_at_zero_geometric_aoa():
    """A cambered wing flown straight (α_geom = 0) still lifts, because its
    zero-lift line is below the chord; a symmetric foil does not."""
    cambered = _fly(naca("4412", "w", area=1.0, chord=0.3,
                         chord_axis=(1, 0, 0), normal_axis=(0, 0, 1)),
                    velocity=(10.0, 0.0, 0.0))
    symmetric = _fly(naca("0012", "w", area=1.0, chord=0.3,
                          chord_axis=(1, 0, 0), normal_axis=(0, 0, 1)),
                     velocity=(10.0, 0.0, 0.0))
    vz_cam = np.array(cambered["foil"]["velocity"]).ravel()[2]
    vz_sym = np.array(symmetric["foil"]["velocity"]).ravel()[2]
    assert vz_cam > 1e-4            # lifted upward
    assert abs(vz_sym) < 1e-9       # symmetric: no lift at α=0


def test_symmetric_aerofoil_zero_lift_only_drag_at_alpha_zero():
    """Symmetric foil, wind along the chord: pure drag, no lift, no
    side force (parity with the old Naca00xx behaviour)."""
    state = _fly(Aerofoil("w", area=1.0, chord=0.3, alpha_0=0.0, Cm_ac=0.0,
                          chord_axis=(1, 0, 0), normal_axis=(0, 0, 1)),
                 velocity=(10.0, 0.0, 0.0))
    v = np.array(state["foil"]["velocity"]).ravel()
    assert v[0] < 10.0
    assert abs(v[1]) < 1e-9
    assert abs(v[2]) < 1e-9


# --- pitching moment --------------------------------------------------------

def test_cambered_foil_produces_pitching_couple():
    """Mounted at the CG (no lever arm), a cambered foil still pitches the
    body via its Cm_ac couple; a symmetric one at α=0 does not."""
    cambered = _fly(naca("4412", "w", area=1.0, chord=0.3),
                    velocity=(10.0, 0.0, 0.0), moi=(0.05, 0.05, 0.05))
    symmetric = _fly(Aerofoil("w", area=1.0, chord=0.3),
                     velocity=(10.0, 0.0, 0.0), moi=(0.05, 0.05, 0.05))
    wy_cam = np.array(cambered["foil"]["angular_velocity"]).ravel()[1]
    wy_sym = np.array(symmetric["foil"]["angular_velocity"]).ravel()[1]
    assert abs(wy_cam) > 1e-5
    assert abs(wy_sym) < 1e-12


# --- Reynolds scaling -------------------------------------------------------

def test_cd0_factor_rises_as_reynolds_drops():
    los = float(ca.evalf(cd0_reynolds_factor(ca.MX(1.0e4))))
    hi = float(ca.evalf(cd0_reynolds_factor(ca.MX(5.0e5))))
    assert los > 2.0 * hi          # much draggier at low Re
    np.testing.assert_allclose(hi, 1.0, atol=1e-6)   # unit at the reference


def test_clmax_factor_decays_at_low_reynolds():
    lo = float(ca.evalf(clmax_reynolds_factor(ca.MX(1.0e3))))
    hi = float(ca.evalf(clmax_reynolds_factor(ca.MX(1.0e7))))
    assert lo < hi
    np.testing.assert_allclose(hi, 1.0, atol=1e-6)
    assert lo >= 0.5 - 1e-9        # never below the floor


def test_low_reynolds_increases_drag_end_to_end():
    """Same foil and speed, but a thick/viscous fluid (low Re) drags far
    harder than thin air (high Re)."""
    foil = lambda: Aerofoil("w", area=1.0, chord=0.3,
                            chord_axis=(1, 0, 0), normal_axis=(0, 0, 1))
    high_re = _fly(foil(), velocity=(10.0, 0.0, 0.0),
                   fluid=_air(rho=1.225))                    # Re ~ 2e5
    low_re = _fly(foil(), velocity=(10.0, 0.0, 0.0),
                  fluid=_air(rho=1.225, viscosity=0.1))      # Re ~ 37
    # Drag = 10 − v_x; bigger deceleration at low Re.
    drag_hi = 10.0 - np.array(high_re["foil"]["velocity"]).ravel()[0]
    drag_lo = 10.0 - np.array(low_re["foil"]["velocity"]).ravel()[0]
    assert drag_lo > drag_hi


def test_reynolds_number_formula():
    Re = float(ca.evalf(reynolds_number(ca.MX(1.225), ca.MX(10.0),
                                        0.3, ca.MX(1.8e-5))))
    np.testing.assert_allclose(Re, 1.225 * 10.0 * 0.3 / 1.8e-5, rtol=1e-9)


# --- robustness -------------------------------------------------------------

def test_no_nan_over_a_run_through_stall():
    """Sweeping a wide AoA (steep climb) keeps the integration finite —
    the smooth model has no singular Jacobian."""
    state = _fly(naca("2412", "w", area=0.5, chord=0.25),
                 velocity=(8.0, 0.0, 6.0), steps=200, dt=0.002)
    v = np.array(state["foil"]["velocity"]).ravel()
    assert np.all(np.isfinite(v))


def test_aerofoil_rejects_bad_geometry():
    with pytest.raises(ValueError):
        Aerofoil("w", area=-1.0)
    with pytest.raises(ValueError):
        Aerofoil("w", chord=0.0)
    with pytest.raises(ValueError):
        Aerofoil("w", chord_axis=(1, 0, 0), normal_axis=(1, 0, 0))

"""FlatOcean — the local z-up ocean, adopted rather than left dead.

`FlatOcean` shipped exported, documented, and never executed past its
constructor: no test, and the submarine examples hand-build their water
instead. These tests exercise the hydrostatic column, the smooth
surface blend that is the class's whole reason to exist, and the
layering contract that lets an atmosphere sit on top of it.
"""

import casadi as ca
import numpy as np
import pytest

from manta.fields import FlatOcean, FluidField, UniformFluid
from manta.ir.frames import WorldFrame
from manta.ir.types import Vec3

RHO = 1025.0
G = 9.80665
P0 = 101325.0


def _ocean(**kw):
    f = FluidField()
    f.add(FlatOcean(density=RHO, surface_z=0.0, surface_pressure=P0,
                    gravity=G, **kw))
    return f


def _at(field, z):
    p = Vec3[WorldFrame].constant((0.0, 0.0, float(z)))
    st = field.value_at_sym(p, 0.0)
    return {
        "density": float(ca.evalf(st.density)),
        "pressure": float(ca.evalf(st.pressure)),
        "temperature": float(ca.evalf(st.temperature)),
        "velocity": np.asarray(ca.evalf(st.velocity._mx)).ravel(),
    }


# ---------------------------------------------------------------------------
# The hydrostatic column
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("depth", [1.0, 10.0, 100.0, 1000.0])
def test_pressure_is_the_incompressible_column(depth):
    """P = P_surface + ρ·g·depth, well below the blend band.

    The depth is smooth-floored — `smooth_max0(d, blend²)` — so the
    column carries a small extra head of `½(√(d²+blend²) − d)`, which
    the expectation includes rather than papers over with a loose
    tolerance: it is 0.6 mm of water at 1 m down under the default 5 cm
    blend, shrinking as 1/d, and it is what keeps the derivative regular
    at the waterline.
    """
    blend = 0.05                                     # the constructor default
    softened = 0.5 * (np.hypot(depth, blend) - depth)
    got = _at(_ocean(), -depth)
    assert got["pressure"] == pytest.approx(
        P0 + RHO * G * (depth + softened), rel=1e-9)


def test_density_is_constant_with_depth():
    deep = _at(_ocean(), -1000.0)
    shallow = _at(_ocean(), -1.0)
    assert deep["density"] == pytest.approx(shallow["density"])
    assert deep["density"] == pytest.approx(RHO)


def test_a_current_rides_the_regime():
    got = _at(_ocean(velocity=(0.5, -0.25, 0.0)), -20.0)
    np.testing.assert_allclose(got["velocity"], (0.5, -0.25, 0.0), atol=1e-12)


# ---------------------------------------------------------------------------
# The surface blend — the reason `surface_blend` is not optional
# ---------------------------------------------------------------------------

def test_the_regime_fades_out_above_the_surface():
    """Water below, nothing above, and a graded band between. A broaching
    hull loses buoyancy instead of floating with full lift."""
    blend = 2.0
    f = _ocean(surface_blend=blend)
    assert _at(f, -10 * blend)["density"] == pytest.approx(RHO, rel=1e-6)
    assert _at(f, +10 * blend)["density"] == pytest.approx(0.0, abs=1e-9)
    mid = _at(f, 0.0)["density"]
    assert 0.0 < mid < RHO


def test_density_through_the_surface_is_monotone_and_smooth():
    """The point of the blend: no step, and a BOUNDED slope. A hard
    cutoff would put an unbounded derivative here and ring every
    integrator downstream."""
    blend = 0.5
    f = _ocean(surface_blend=blend)
    zs = np.linspace(-4 * blend, 4 * blend, 401)
    rho = np.array([_at(f, z)["density"] for z in zs])
    assert np.all(np.diff(rho) <= 1e-9)                   # non-increasing
    slope = np.abs(np.diff(rho) / np.diff(zs))
    # A smooth ramp over ~blend metres cannot be steeper than a few
    # multiples of rho/blend; a step would be orders of magnitude worse.
    assert slope.max() < 10.0 * RHO / blend


def test_the_column_never_dips_below_the_surface_pressure():
    """The membership scales the WHOLE FluidState (a fading regime fades
    its pressure with its density), so the raw pressure drops to zero
    above the waterline. What must not happen is the *column itself*
    going negative inside the blend band — the depth is smooth-floored
    at zero for exactly that reason. Divide the membership back out to
    see it."""
    f = _ocean(surface_blend=1.0)
    for z in np.linspace(-3.0, 3.0, 61):
        got = _at(f, z)
        w = got["density"] / RHO
        if w < 1e-6:
            continue                      # dry: nothing to check
        assert got["pressure"] / w >= P0 - 1e-6


def test_a_wider_blend_spreads_the_same_transition():
    narrow = _ocean(surface_blend=0.1)
    wide = _ocean(surface_blend=5.0)
    # One metre up: still substantially wet under a wide blend, dry under
    # a narrow one.
    assert _at(narrow, 1.0)["density"] < 1.0
    assert _at(wide, 1.0)["density"] > 0.1 * RHO


# ---------------------------------------------------------------------------
# Layering — "a world that wants air as well overlays an Atmosphere"
# ---------------------------------------------------------------------------

def test_air_overlays_the_ocean_by_membership():
    """Both are `baseline` media: added first, the ocean is overridden
    above the surface by the unbounded air and survives below it."""
    f = FluidField()
    f.add(FlatOcean(density=RHO, surface_z=0.0, surface_pressure=P0,
                    gravity=G, surface_blend=0.05))
    f.add(UniformFluid(density=1.225, pressure=P0, temperature=288.15))
    assert _at(f, +50.0)["density"] == pytest.approx(1.225, rel=1e-9)
    # Below the surface the air (added later, membership 1 everywhere)
    # still wins — which is exactly why the air belongs FIRST in a world
    # that wants both. Pin the ordering contract, not an accident.
    assert _at(f, -50.0)["density"] == pytest.approx(1.225, rel=1e-9)

    ordered = FluidField()
    ordered.add(UniformFluid(density=1.225, pressure=P0, temperature=288.15))
    ordered.add(FlatOcean(density=RHO, surface_z=0.0, surface_pressure=P0,
                          gravity=G, surface_blend=0.05))
    assert _at(ordered, +50.0)["density"] == pytest.approx(1.225, rel=1e-9)
    assert _at(ordered, -50.0)["density"] == pytest.approx(RHO, rel=1e-9)


def test_surface_z_moves_the_waterline():
    f = FluidField()
    f.add(FlatOcean(density=RHO, surface_z=-30.0, surface_pressure=P0,
                    gravity=G))
    assert _at(f, -10.0)["density"] == pytest.approx(0.0, abs=1e-9)
    assert _at(f, -40.0)["pressure"] == pytest.approx(P0 + RHO * G * 10.0,
                                                      rel=1e-5)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kwargs, match", [
    (dict(density=0.0), "density must be > 0"),
    (dict(gravity=-1.0), "gravity must be >= 0"),
    (dict(surface_blend=0.0), "surface_blend must be > 0"),
    (dict(velocity=(1.0, 2.0)), "velocity must be length-3"),
])
def test_constructor_validation(kwargs, match):
    with pytest.raises(ValueError, match=match):
        FlatOcean(**kwargs)


def test_a_hard_cutoff_is_refused_with_its_reason():
    """The error explains itself — this is the policy `Earth`'s hard
    surface cutoff is measured against."""
    with pytest.raises(ValueError, match="rings every integrator"):
        FlatOcean(surface_blend=0.0)

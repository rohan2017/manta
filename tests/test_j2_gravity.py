"""J2Gravity — the oblateness perturbation, against an independent
closed form and against physics.

`J2Gravity` shipped with no test and no consumer that turns it on
(`Earth(include_j2=True)` is opt-in and nothing in the repo opts in), so
a sign slip in the bracket would have shipped silently. The reference
below is written the textbook per-axis way, NOT as a copy of the
vectorized implementation, so the two disagree if either is wrong.
"""

import casadi as ca
import numpy as np
import pytest

from manta.fields import GravityField, J2Gravity
from manta.ir.frames import WorldFrame
from manta.ir.types import Vec3

GM = 3.986004418e14        # Earth, m³/s²
J2 = 1.0826267e-3
R_EQ = 6.378137e6


def _eval_at(field, point_xyz):
    p = Vec3[WorldFrame].constant(point_xyz)
    return np.asarray(ca.evalf(field.value_at_sym(p, 0.0)._mx)).ravel()


def _reference_j2(p):
    """Textbook per-axis J2 acceleration for a z polar axis:

        a_x = -(3/2) J2 (GM/r²) (R_eq/r)² (1 − 5 z²/r²) (x/r)
        a_y = same, with y
        a_z = -(3/2) J2 (GM/r²) (R_eq/r)² (3 − 5 z²/r²) (z/r)
    """
    x, y, z = (float(v) for v in p)
    r = np.sqrt(x * x + y * y + z * z)
    k = -1.5 * J2 * (GM / r ** 2) * (R_EQ / r) ** 2
    zr2 = (z / r) ** 2
    return np.array([k * (1.0 - 5.0 * zr2) * (x / r),
                     k * (1.0 - 5.0 * zr2) * (y / r),
                     k * (3.0 - 5.0 * zr2) * (z / r)])


def _field(**kw):
    gf = GravityField()
    gf.add(J2Gravity(position=(0.0, 0.0, 0.0), GM=GM, J2=J2,
                     eq_radius=R_EQ, eps=0.0, **kw))
    return gf


@pytest.mark.parametrize("p", [
    (R_EQ + 400e3, 0.0, 0.0),                  # equatorial
    (0.0, 0.0, R_EQ + 400e3),                  # polar
    (4.6e6, 0.0, 4.6e6),                       # 45° latitude
    (3.0e6, -4.0e6, 5.0e6),                    # generic
    (2.0e7, 1.0e7, -1.5e7),                    # far out, southern
])
def test_matches_the_closed_form(p):
    np.testing.assert_allclose(_eval_at(_field(), p), _reference_j2(p),
                               rtol=1e-10, atol=0.0)


def test_equatorial_perturbation_points_inward():
    """On the equator the bulge mass is nearer than a point mass would
    put it, so J2 ADDS attraction toward the centre. A sign slip here
    would raise low-inclination orbits instead of lowering them."""
    p = np.array([R_EQ + 400e3, 0.0, 0.0])
    g = _eval_at(_field(), tuple(p))
    assert g[0] < 0.0                      # toward the centre
    assert np.allclose(g[1:], 0.0, atol=1e-20)


def test_polar_perturbation_points_outward():
    """Over a pole the bulge is farther away, so gravity is WEAKER than
    the point-mass value — the perturbation points away from the
    centre, the opposite sign to the equatorial case."""
    g = _eval_at(_field(), (0.0, 0.0, R_EQ + 400e3))
    assert g[2] > 0.0
    assert np.allclose(g[:2], 0.0, atol=1e-20)


def test_magnitude_is_the_expected_fraction_of_point_mass_gravity():
    """At the surface J2's perturbation is ~1e-3 of g — the size that
    makes it matter for orbit propagation and nothing else. This pins
    the J2·R_eq² scaling: a factor-of-two slip would show up here."""
    r = R_EQ
    g_point = GM / r ** 2
    g_j2 = np.linalg.norm(_eval_at(_field(), (r, 0.0, 0.0)))
    assert 5e-4 < g_j2 / g_point < 3e-3


def test_falls_off_faster_than_the_point_mass_term():
    """J2 goes as 1/r⁴, the point mass as 1/r²: doubling r drops the
    perturbation by ~16, not ~4."""
    near = np.linalg.norm(_eval_at(_field(), (1.0e7, 0.0, 0.0)))
    far = np.linalg.norm(_eval_at(_field(), (2.0e7, 0.0, 0.0)))
    assert near / far == pytest.approx(16.0, rel=1e-9)


def test_polar_axis_rotates_the_whole_pattern():
    """With ẑ along x, the equatorial/polar roles swap: the point that
    was 'over the pole' is now on the equator and vice versa."""
    z_axis = _field()
    x_axis = _field(polar_axis=(1.0, 0.0, 0.0))
    p_on_x = (R_EQ + 400e3, 0.0, 0.0)
    p_on_z = (0.0, 0.0, R_EQ + 400e3)
    # x-axis field at a point on x == z-axis field at a point on z, rotated.
    np.testing.assert_allclose(_eval_at(x_axis, p_on_x)[0],
                               _eval_at(z_axis, p_on_z)[2], rtol=1e-12)
    np.testing.assert_allclose(_eval_at(x_axis, p_on_z)[2],
                               _eval_at(z_axis, p_on_x)[0], rtol=1e-12)


def test_superposes_onto_a_point_mass():
    """J2Gravity is the perturbation ALONE — the docstring's contract.
    Added beside a PointMassGravity it must simply sum."""
    from manta.fields import PointMassGravity
    p = (4.6e6, 0.0, 4.6e6)
    both = GravityField()
    both.add(PointMassGravity(position=(0.0, 0.0, 0.0), GM=GM, eps=0.0))
    both.add(J2Gravity(position=(0.0, 0.0, 0.0), GM=GM, J2=J2,
                       eq_radius=R_EQ, eps=0.0))
    point_only = GravityField()
    point_only.add(PointMassGravity(position=(0.0, 0.0, 0.0), GM=GM, eps=0.0))
    np.testing.assert_allclose(
        _eval_at(both, p) - _eval_at(point_only, p), _reference_j2(p),
        rtol=1e-8, atol=1e-14)


@pytest.mark.parametrize("kwargs, match", [
    ({"GM": -1.0}, "GM must be > 0"),
    ({"eq_radius": 0.0}, "eq_radius must be > 0"),
    ({"eps": -1.0}, "eps must be >= 0"),
    ({"polar_axis": (0.0, 0.0, 0.0)}, "polar_axis must be nonzero"),
])
def test_constructor_validation(kwargs, match):
    args = {"position": (0.0, 0.0, 0.0), "GM": GM, "J2": J2, "eq_radius": R_EQ}
    args.update(kwargs)
    with pytest.raises(ValueError, match=match):
        J2Gravity(**args)

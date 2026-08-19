"""The oblate planet — geodetic up, WGS-84 Earth, and why J2 comes with it.

Contracts pinned here:

  * `Planet.local_tangent_basis` returns the GEODETIC normal of the
    reference spheroid (a plumb line, a GNSS "up"), reducing exactly to
    the radial for a sphere.
  * `ecef_from_geodetic` / `geodetic_from_ecef` round-trip and hit the
    textbook WGS-84 points; the numpy and CasADi (collision `Ellipsoid`)
    formulations agree.
  * Earth's sea surface is the ellipsoid: geodetic altitude is 0 on it at
    every latitude and +h at height h; the ocean/atmosphere switch there.
  * The physical reason for all of it: point mass + J2 + the spinning
    frame's centrifugal term leave a craft at rest on the ellipsoid with
    no tangential acceleration to O(f²) — gravity is normal to the sea.
"""

import casadi as ca
import numpy as np
import pytest

from manta import Craft, Planet, Sim, TargetNumpy, World
from manta.fields import CollisionField, Ellipsoid, FluidField, GravityField
from manta.ir.frames import WorldFrame
from manta.ir.types import Vec3
from manta.parts import Mass
from manta.planets import Earth
from manta.planets.base import geodetic_from_cylindrical


A = Earth.R_EQ
F = Earth.FLATTENING
B = A * (1.0 - F)


def _geodetic_normal(lat_deg, lon_deg=0.0):
    lat, lon = np.radians(lat_deg), np.radians(lon_deg)
    return np.array([np.cos(lat) * np.cos(lon),
                     np.cos(lat) * np.sin(lon),
                     np.sin(lat)])


# ---------------------------------------------------------------------------
# local_tangent_basis
# ---------------------------------------------------------------------------

def test_tangent_basis_up_is_geodetic_normal_on_the_ellipsoid():
    """At 45° N on the WGS-84 surface the outward normal is the geodetic
    (cos φ, 0, sin φ) — ~0.19° poleward of the radial. North/East follow."""
    earth = Earth()
    p = earth.ecef_from_geodetic(45.0, 0.0, 0.0)
    east, north, up = earth.local_tangent_basis(tuple(p))
    np.testing.assert_allclose(up, _geodetic_normal(45.0), atol=1e-12)
    radial = p / np.linalg.norm(p)
    assert np.degrees(np.arccos(radial @ up)) == pytest.approx(0.192, abs=0.005)
    np.testing.assert_allclose(north, (-np.sin(np.radians(45)), 0,
                                       np.cos(np.radians(45))), atol=1e-12)
    np.testing.assert_allclose(east, (0, 1, 0), atol=1e-12)


def test_tangent_basis_geodetic_up_holds_at_altitude_and_off_meridian():
    """The normal through a point 10 km up at (−33°, 151°) is the same
    geodetic normal — the basis is a function of geodetic lat/lon only."""
    earth = Earth()
    p = earth.ecef_from_geodetic(-33.0, 151.0, 10_000.0)
    _, _, up = earth.local_tangent_basis(tuple(p))
    np.testing.assert_allclose(up, _geodetic_normal(-33.0, 151.0), atol=1e-12)


def test_tangent_basis_flattening_zero_is_the_radial():
    """A spherical planet keeps the radial up exactly (bit-for-bit the old
    behaviour), including for a tilted spin axis and off-centre planet."""
    lat = np.radians(45.0)
    planet = Planet(position=(0, 0, -6.371e6),
                    rotation_axis=(np.cos(lat), 0.0, np.sin(lat)),
                    omega=7.272e-5)
    for point in ((0.0, 0.0, -0.2), (1000.0, -2000.0, 300.0)):
        _, _, up = planet.local_tangent_basis(point)
        r = np.asarray(point) - planet.center
        np.testing.assert_allclose(up, r / np.linalg.norm(r), atol=0.0)
    sphere = Earth(flattening=0.0)
    p = A * _geodetic_normal(45.0)
    _, _, up = sphere.local_tangent_basis(tuple(p))
    np.testing.assert_allclose(up, _geodetic_normal(45.0), atol=1e-15)


def test_oblate_planet_requires_a_size():
    with pytest.raises(ValueError, match="equatorial_radius"):
        Planet(flattening=0.001)
    with pytest.raises(ValueError, match="flattening"):
        Planet(equatorial_radius=1.0, flattening=1.0)


# ---------------------------------------------------------------------------
# Geodetic ↔ Cartesian
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("lat, lon, alt, expected", [
    (0.0, 0.0, 0.0, (A, 0.0, 0.0)),
    (0.0, 90.0, 0.0, (0.0, A, 0.0)),
    (90.0, 0.0, 0.0, (0.0, 0.0, B)),
    (-90.0, 0.0, 100.0, (0.0, 0.0, -B - 100.0)),
    (45.0, 0.0, 0.0, (A / np.sqrt(1.0 + (1.0 - F) ** 2), 0.0,
                      A * (1.0 - F) ** 2 / np.sqrt(1.0 + (1.0 - F) ** 2))),
])
def test_ecef_from_geodetic_known_points(lat, lon, alt, expected):
    """Textbook WGS-84 points; the 45° case is the closed form
    ρ = a cos β, z = b sin β with tan β = (1−f) tan φ."""
    p = Earth().ecef_from_geodetic(lat, lon, alt)
    np.testing.assert_allclose(p, expected, atol=1e-6)


@pytest.mark.parametrize("lat, lon, alt", [
    (0.0, 0.0, 0.0), (32.7, -117.2, 5.0), (-45.0, 170.0, -3000.0),
    (89.9, 12.0, 0.0), (90.0, 0.0, 10.0), (-90.0, 0.0, 0.0),
    (60.0, -179.9, 400e3), (1e-9, 0.0, -1.0),
])
def test_geodetic_round_trip(lat, lon, alt):
    earth = Earth()
    p = earth.ecef_from_geodetic(lat, lon, alt)
    lat2, lon2, alt2 = earth.geodetic_from_ecef(p)
    assert lat2 == pytest.approx(lat, abs=1e-9)
    if abs(lat) < 90.0:                              # lon undefined at the pole
        assert lon2 == pytest.approx(lon, abs=1e-9)
    assert alt2 == pytest.approx(alt, abs=1e-6)


def test_geodetic_conversion_needs_a_reference_shape():
    with pytest.raises(ValueError, match="equatorial_radius"):
        Planet().ecef_from_geodetic(0.0, 0.0)
    with pytest.raises(ValueError, match="lat must be"):
        Earth().ecef_from_geodetic(91.0, 0.0)


def test_planet_frame_axes_are_ecef_for_the_default_spin_axis():
    """axis +z, prime meridian along +x, 90° E along +y — the WGS-84 ECEF
    convention, so a consumer's ECEF vector IS Earth's PlanetFrame vector."""
    earth = Earth()
    np.testing.assert_allclose(earth.ecef_from_geodetic(0, 0), (A, 0, 0), atol=1e-9)
    np.testing.assert_allclose(earth.ecef_from_geodetic(0, 90), (0, A, 0), atol=1e-9)
    np.testing.assert_allclose(earth.ecef_from_geodetic(90, 0), (0, 0, B), atol=1e-9)


def test_numpy_and_symbolic_geodesy_agree():
    """`geodetic_from_cylindrical` (numpy, Planet) and
    `Ellipsoid.signed_height_sym` (CasADi, fields) are the same formulas —
    the surface the collider feels and the altitude the planet reports
    must be one geometry."""
    surface = Ellipsoid(center=(0, 0, 0), equatorial_radius=A, flattening=F)
    x = ca.MX.sym("x", 3)
    h_sym, up_sym = surface.signed_height_sym(x)
    fn = ca.Function("geo", [x], [h_sym, up_sym])
    earth = Earth()
    for lat, lon, alt in ((0, 0, 0), (32.7, -117.2, 5.0), (-45, 170, -3000),
                          (89.999, 0, 0), (90, 0, 10), (60, 20, 400e3)):
        p = earth.ecef_from_geodetic(lat, lon, alt)
        h, up = fn(p)
        rho = float(np.hypot(p[0], p[1]))
        lat_np, h_np = geodetic_from_cylindrical(rho, float(p[2]), A, F)
        assert float(h) == pytest.approx(h_np, abs=1e-6)
        assert float(h) == pytest.approx(alt, abs=1e-6)
        np.testing.assert_allclose(np.asarray(up).ravel(),
                                   _geodetic_normal(lat, lon), atol=1e-9)
        assert np.degrees(lat_np) == pytest.approx(lat, abs=1e-9)


def test_scene_at_geodetic_anchors_on_the_ellipsoid_with_geodetic_up():
    earth = Earth()
    scene = earth.scene_at_geodetic(32.0, -117.0, 0.0)
    np.testing.assert_allclose(scene.anchor_planet,
                               earth.ecef_from_geodetic(32.0, -117.0), atol=0)
    np.testing.assert_allclose(scene.R_planet_from_scene[:, 2],
                               _geodetic_normal(32.0, -117.0), atol=1e-12)
    ks = scene.at_rest((0.0, 0.0, -10.0))
    assert earth.geodetic_from_ecef(ks["position"])[2] == pytest.approx(-10.0,
                                                                        abs=1e-6)


# ---------------------------------------------------------------------------
# The sea surface is the ellipsoid
# ---------------------------------------------------------------------------

def _sample(world, point, t=0.0):
    p = Vec3[WorldFrame].constant(point)
    return world.get_field(FluidField).value_at_sym(p, ca.MX(float(t)))


def _fluid_world(**kw):
    earth = Earth(rotation_rate=0.0, **kw)
    w = World(); w.add_planet(earth)
    c = Craft("probe"); c.add(Mass("body", mass=1.0))
    w.add_craft(c, position=(A, 0, 0))
    return earth, Sim(w).world


@pytest.mark.parametrize("lat", [0.0, 32.0, 45.0, 60.0, 89.0, -70.0])
def test_sea_surface_is_the_ellipsoid_at_every_latitude(lat):
    """Just above the WGS-84 surface is air, just below is water — the
    old R_EQ sphere would have put the 45° surface 10.7 km up in the air."""
    earth, w = _fluid_world()
    for alt, rho in ((0.05, earth.air_density), (-0.05, earth.water_density)):
        p = earth.ecef_from_geodetic(lat, 30.0, alt)
        got = float(ca.evalf(_sample(w, tuple(p)).density))
        assert got == pytest.approx(rho, rel=1e-3), f"lat={lat} alt={alt}"


def test_hydrostatic_and_isa_columns_use_geodetic_altitude():
    """Pressure at geodetic depth d is P0 + ρ g d and the ISA pressure at
    geodetic height h matches the sea-level-referenced profile, at 45°."""
    from manta.fields.fluid_props import R_AIR, isa_pressure
    earth, w = _fluid_world()
    g0 = earth.gravity_mu / earth.planet_radius ** 2
    P0 = earth.air_density * R_AIR * earth.sea_level_temperature
    p_wet = earth.ecef_from_geodetic(45.0, 10.0, -100.0)
    P_wet = float(ca.evalf(_sample(w, tuple(p_wet)).pressure))
    assert P_wet == pytest.approx(P0 + earth.water_density * g0 * 100.0, rel=1e-9)
    p_dry = earth.ecef_from_geodetic(45.0, 10.0, 1500.0)
    P_dry = float(ca.evalf(_sample(w, tuple(p_dry)).pressure))
    assert P_dry == pytest.approx(
        isa_pressure(1500.0, P0, earth.sea_level_temperature,
                     earth.lapse_rate, g0, R_AIR), rel=1e-9)


def test_sea_level_offset_raises_the_ellipsoid():
    earth, w = _fluid_world(sea_level=2.0)
    p_air = earth.ecef_from_geodetic(45.0, 0.0, 2.05)
    p_wet = earth.ecef_from_geodetic(45.0, 0.0, 1.95)
    assert float(ca.evalf(_sample(w, tuple(p_air)).density)) == pytest.approx(
        earth.air_density, rel=1e-3)
    assert float(ca.evalf(_sample(w, tuple(p_wet)).density)) == pytest.approx(
        earth.water_density, rel=1e-3)


def test_surface_collision_is_the_ellipsoid_with_geodetic_normal():
    """The registered obstacle pushes along the geodetic normal by the
    geodetic penetration depth — the same surface the water switches on."""
    earth = Earth(rotation_rate=0.0)
    w = World(); w.add_planet(earth)
    c = Craft("probe"); c.add(Mass("body", mass=1.0))
    w.add_craft(c, position=(A, 0, 0))
    w = Sim(w).world
    cf = w.get_field(CollisionField)
    p_in = earth.ecef_from_geodetic(45.0, -20.0, -0.3)
    pen = np.asarray(ca.evalf(cf.value_at_sym(
        Vec3[WorldFrame].constant(tuple(p_in)), ca.MX(0.0))._mx)).ravel()
    np.testing.assert_allclose(pen, 0.3 * _geodetic_normal(45.0, -20.0), atol=1e-6)
    p_out = earth.ecef_from_geodetic(45.0, -20.0, 0.3)
    pen = np.asarray(ca.evalf(cf.value_at_sym(
        Vec3[WorldFrame].constant(tuple(p_out)), ca.MX(0.0))._mx)).ravel()
    np.testing.assert_allclose(pen, 0.0, atol=1e-6)


# ---------------------------------------------------------------------------
# Gravity is normal to the ellipsoid — the reason the truth Earth is oblate
# ---------------------------------------------------------------------------

def test_earth_defaults_to_j2_and_honours_an_explicit_off():
    assert Earth().include_j2 is True
    assert Earth(flattening=0.0).include_j2 is False
    assert Earth(include_j2=False).include_j2 is False
    assert Earth(flattening=0.0, include_j2=True).include_j2 is True


@pytest.mark.parametrize("lat", [15.0, 32.0, 45.0, 60.0, 80.0])
def test_effective_gravity_is_normal_to_the_ellipsoid(lat):
    """Point mass + J2 + centrifugal at a point on the WGS-84 surface has a
    tangential (north) component below 1e-4 m/s² — the O(f²) residual of
    a second-degree field; a point mass alone leaves ~1.7e-2 m/s²·sin 2φ
    pulling toward the equator, which would slide every resting hull."""
    earth = Earth()
    w = World(); w.add_planet(earth)
    c = Craft("probe"); c.add(Mass("body", mass=1.0))
    w.add_craft(c, position=(A, 0, 0))
    w = Sim(w).world
    p = earth.ecef_from_geodetic(lat, 0.0, 0.0)
    g = np.asarray(ca.evalf(w.get_field(GravityField).value_at_sym(
        Vec3[WorldFrame].constant(tuple(p)), ca.MX(0.0))._mx)).ravel()
    omega = earth.omega_vec_world()
    centrifugal = -np.cross(omega, np.cross(omega, p))
    east, north, up = earth.local_tangent_basis(tuple(p))
    g_eff = g + centrifugal
    assert abs(g_eff @ north) < 1e-4
    assert abs(g_eff @ east) < 1e-12
    # Magnitude: Somigliana's WGS-84 normal gravity to ~1e-4 (the J4 term).
    s2 = np.sin(np.radians(lat)) ** 2
    somigliana = 9.7803253359 * (1 + 0.00193185265241 * s2) / np.sqrt(
        1 - 0.00669437999013 * s2)
    assert g_eff @ up == pytest.approx(-somigliana, abs=2e-4)
    # Without J2 the same point is far from balanced.
    bare = Earth(include_j2=False)
    w2 = World(); w2.add_planet(bare)
    c2 = Craft("probe"); c2.add(Mass("body", mass=1.0))
    w2.add_craft(c2, position=(A, 0, 0))
    w2 = Sim(w2).world
    g_bare = np.asarray(ca.evalf(w2.get_field(GravityField).value_at_sym(
        Vec3[WorldFrame].constant(tuple(p)), ca.MX(0.0))._mx)).ravel()
    assert abs((g_bare + centrifugal) @ north) > 5e-3


def test_craft_at_rest_on_the_ellipsoid_feels_no_tangential_acceleration():
    """The dynamics-level version at 32° N: one small step of a free mass
    placed `at_rest` on the WGS-84 sea surface changes its ground-relative
    velocity only along up (free fall) — the tangent-plane specific force
    is below 1e-4 m/s²."""
    earth = Earth(surface_collision=False)
    w = World(); w.add_planet(earth)
    c = Craft("probe"); c.add(Mass("body", mass=1.0, moi=(0.1, 0.1, 0.1)))
    scene = earth.scene_at_geodetic(32.0, -117.0, 0.0)
    w.add_craft(c, **scene.at_rest())
    sim = TargetNumpy(Sim(w))
    dt = 1e-3
    sim.step(dt)
    rel = scene.relative(dict(sim.state["probe"]), t=dt)
    v = np.asarray(rel["velocity"])
    accel = v / dt
    assert abs(accel[0]) < 1e-4 and abs(accel[1]) < 1e-4
    assert accel[2] == pytest.approx(-9.79, abs=0.03)

"""Planet — transform helper for PlanetFrame ↔ WorldFrame.

These tests verify:
  * `planet_to_world` and `world_to_planet` are exact inverses.
  * A point at PlanetFrame rest gets a tangential WorldFrame velocity
    (`ω × p_world`).
  * Coriolis emerges from running an inertial WorldFrame simulation and
    viewing the trajectory in PlanetFrame — the framework adds no
    pseudo-forces; they fall out of the transform.
"""

import numpy as np

from manta import Craft, Planet, World
from manta.fields import GravityField
from manta.parts import Mass


# ---------------------------------------------------------------------------
# Pure-Python sanity checks on the transform
# ---------------------------------------------------------------------------

def test_rotation_matrix_at_t_zero_is_identity():
    planet = Planet(rotation_axis=(0, 0, 1), omega=0.5)
    R = planet.R_world_from_planet(0.0)
    np.testing.assert_allclose(R, np.eye(3), atol=1e-15)


def test_rotation_matrix_at_quarter_period_is_90_degrees_about_z():
    planet = Planet(rotation_axis=(0, 0, 1), omega=1.0)
    R = planet.R_world_from_planet(np.pi / 2)
    expected = np.array([[0.0, -1.0, 0.0],
                         [1.0,  0.0, 0.0],
                         [0.0,  0.0, 1.0]])
    np.testing.assert_allclose(R, expected, atol=1e-12)


def test_planet_to_world_and_back_is_identity():
    planet = Planet(rotation_axis=(0.3, 0.4, np.sqrt(0.75)), omega=0.7)
    p0 = (1.5, -2.0, 3.0)
    v0 = (0.1, 0.2, -0.05)
    for t in (0.0, 0.123, 1.0, 5.5):
        p_w, v_w = planet.planet_to_world(p0, v0, t)
        p_p, v_p = planet.world_to_planet(p_w, v_w, t)
        np.testing.assert_allclose(p_p, p0, atol=1e-12)
        np.testing.assert_allclose(v_p, v0, atol=1e-12)


def test_planet_rest_picks_up_tangential_world_velocity():
    """A point at PlanetFrame rest has WorldFrame velocity = ω × p_world.
    For axis +z at angular rate ω, a point at PlanetFrame (R, 0, 0) gets
    a velocity of (0, ω·R, 0) — eastward tangential at t=0."""
    R_earth = 6.371e6
    omega = 7.272e-5
    planet = Planet(rotation_axis=(0, 0, 1), omega=omega)
    p_w, v_w = planet.planet_to_world((R_earth, 0, 0), (0, 0, 0), t=0.0)
    np.testing.assert_allclose(p_w, (R_earth, 0, 0), atol=1e-9)
    np.testing.assert_allclose(v_w, (0, omega * R_earth, 0), atol=1e-9)


# ---------------------------------------------------------------------------
# Coriolis emerges from inertial WorldFrame integration
# ---------------------------------------------------------------------------

def test_inertial_straight_line_curves_in_planet_frame():
    """A craft moving inertially in WorldFrame (no forces) appears to
    curve in PlanetFrame. Closed-form check: for a body starting on the
    rotation axis with PlanetFrame velocity (v, 0, 0) — equivalently
    WorldFrame velocity (v, 0, 0) since starting at the axis cancels
    the ω × p term — its WorldFrame trajectory is the straight line
    p_world(t) = (v·t, 0, 0). Viewed in PlanetFrame, that traces a
    spiral whose y-component at small t is -v·ω·t² (Coriolis
    deflection)."""
    omega = 0.5
    planet = Planet(rotation_axis=(0, 0, 1), omega=omega)
    v0 = 2.0

    c = Craft("ball")
    c.add(Mass("body", mass=1.0))
    tick = c.compile_tick(gravity_field=GravityField(g=(0.0, 0.0, 0.0)))

    # Start on the rotation axis with v_planet = (v0, 0, 0).
    p_w0, v_w0 = planet.planet_to_world((0, 0, 0), (v0, 0, 0), t=0.0)
    state = c.initial_state(position=tuple(p_w0), velocity=tuple(v_w0))

    dt = 0.001
    n_steps = 1000  # 1 s
    for _ in range(n_steps):
        out = tick(dt=dt, **state)
        state = {**state, **out}

    t = n_steps * dt
    p_planet, _ = planet.world_to_planet(
        state["position"], state["velocity"], t)

    # Closed form: p_world(t) = (v0·t, 0, 0); rotate back by -ω·t:
    expected_x = v0 * t * np.cos(omega * t)
    expected_y = -v0 * t * np.sin(omega * t)
    np.testing.assert_allclose(p_planet[0], expected_x, atol=1e-4)
    np.testing.assert_allclose(p_planet[1], expected_y, atol=1e-4)
    # And y is significantly nonzero — Coriolis is observable.
    assert abs(p_planet[1]) > 0.1

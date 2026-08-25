"""CollisionField + Collider tests."""

import casadi as ca
import numpy as np

from manta import Craft, Sim, TargetNumpy, World
from manta.fields import CollisionField, GravityField, HalfSpace
from manta.ir.frames import WorldFrame
from manta.ir.types import Vec3
from manta.parts import Collider, Mass


def _eval_pen_at(field, point_xyz):
    p = Vec3[WorldFrame].constant(point_xyz)
    val = field.value_at_sym(p, 0.0)
    return np.asarray(ca.evalf(val._mx)).ravel()


# ---------------------------------------------------------------------------
# CollisionField + HalfSpace
# ---------------------------------------------------------------------------

def test_empty_collision_field_is_zero():
    cf = CollisionField()
    np.testing.assert_allclose(_eval_pen_at(cf, (0, 0, -10)), np.zeros(3),
                               atol=1e-12)


def test_halfspace_outside_returns_zero():
    """Points above the plane have zero penetration (modulo smoothing)."""
    cf = CollisionField()
    cf.add(HalfSpace(origin=(0, 0, 0), normal=(0, 0, 1)))   # ground z=0
    pen = _eval_pen_at(cf, (1, 2, 5))   # 5 m above
    np.testing.assert_allclose(pen, np.zeros(3), atol=1e-5)


def test_halfspace_inside_returns_outward_vector():
    cf = CollisionField()
    cf.add(HalfSpace(origin=(0, 0, 0), normal=(0, 0, 1)))   # ground z=0
    # Point 0.5 m below the plane → 0.5 m penetration along +z outward.
    pen = _eval_pen_at(cf, (3, 0, -0.5))
    np.testing.assert_allclose(pen, (0.0, 0.0, 0.5), atol=1e-6)


def test_halfspace_depth_scales_linearly_with_penetration():
    cf = CollisionField()
    cf.add(HalfSpace(origin=(0, 0, 0), normal=(0, 0, 1)))
    for d in [0.01, 0.1, 1.0, 5.0]:
        pen = _eval_pen_at(cf, (0, 0, -d))
        np.testing.assert_allclose(pen, (0, 0, d), rtol=1e-5)


def test_inclined_halfspace_returns_correct_normal():
    """Plane at 45° in the (x, z) plane: normal = (1, 0, 1)/√2."""
    cf = CollisionField()
    n = (1.0 / np.sqrt(2), 0.0, 1.0 / np.sqrt(2))
    cf.add(HalfSpace(origin=(0, 0, 0), normal=n))
    # Point on the -normal side at depth 1.0: (-sqrt(2)/2, 0, -sqrt(2)/2)?
    p_in = tuple(-d * n[i] for i, d in enumerate([1.0, 1.0, 1.0]))
    pen = _eval_pen_at(cf, p_in)
    np.testing.assert_allclose(pen, tuple(1.0 * n[i] for i in range(3)),
                               rtol=1e-5)


def test_multiple_obstacles_superpose_at_corner():
    """Floor + wall meeting at a corner. A point inside both contributes
    the sum of outward normals."""
    cf = CollisionField()
    cf.add(HalfSpace(origin=(0, 0, 0), normal=(0, 0, 1)))     # ground
    cf.add(HalfSpace(origin=(0, 0, 0), normal=(1, 0, 0)))     # wall at x=0
    # Point at (-0.3, 0, -0.4): inside both. Expected pen = (0.3, 0, 0.4).
    pen = _eval_pen_at(cf, (-0.3, 0, -0.4))
    np.testing.assert_allclose(pen, (0.3, 0.0, 0.4), atol=1e-5)


# ---------------------------------------------------------------------------
# Collider Part dynamics
# ---------------------------------------------------------------------------

def test_collider_requires_collision_field():
    """A Collider with no CollisionField registered is a configuration
    error, caught at transform build — not a silently inert part."""
    import pytest
    g = (0.0, 0.0, -9.81)
    w = World().add_field(GravityField().add_uniform(g))
    c = Craft("ball")
    c.add(Mass("body", mass=1.0, moi=(0.01, 0.01, 0.01)))
    c.add(Collider("contact", stiffness=1e5, damping=10.0))
    w.add_craft(c, position=(0, 0, 5))
    with pytest.raises(ValueError, match="CollisionField"):
        Sim(w)


def test_ball_rests_on_ground_at_compression_equilibrium():
    """A ball resting on a stiff ground compresses by m·g/k. Verify the
    equilibrium depth is approximately that value."""
    m = 1.0
    g = 9.81
    k = 1e5     # stiff
    c_damp = 100.0

    w = (World()
         .add_field(GravityField().add_uniform((0, 0, -g)))
         .add_field(CollisionField().add(HalfSpace(normal=(0, 0, 1)))))
    cr = Craft("ball")
    cr.add(Mass("body", mass=m, moi=(0.01, 0.01, 0.01)))
    cr.add(Collider("contact", stiffness=k, damping=c_damp))
    w.add_craft(cr, position=(0, 0, 1.0))     # start above ground
    cw = TargetNumpy(Sim(w))

    for _ in range(5000):
        cw.step(0.001)

    z_final = float(cw.state["ball"]["position"][2])
    expected_depth = m * g / k       # equilibrium compression
    assert np.isclose(z_final, -expected_depth, atol=2e-4), (
        f"z_final={z_final} expected≈{-expected_depth}")
    # Velocity should have decayed to ~0.
    assert abs(cw.state["ball"]["velocity"][2]) < 1e-3


def test_dropped_ball_bounces():
    """A ball dropped above the ground bounces, then settles with damping."""
    g = 9.81
    w = (World()
         .add_field(GravityField().add_uniform((0, 0, -g)))
         .add_field(CollisionField().add(HalfSpace(normal=(0, 0, 1)))))
    cr = Craft("ball")
    cr.add(Mass("body", mass=1.0, moi=(0.01, 0.01, 0.01)))
    cr.add(Collider("contact", stiffness=2e4, damping=20.0))
    w.add_craft(cr, position=(0, 0, 1.0))
    cw = TargetNumpy(Sim(w))

    # Track the maximum +z height attained AFTER the first bounce.
    z_history = []
    for _ in range(5000):
        cw.step(0.001)
        z_history.append(float(cw.state["ball"]["position"][2]))

    z = np.array(z_history)
    # After the first bounce (~ t=0.45s) the ball should reach a positive
    # peak that's less than the initial 1.0 m (energy was dissipated).
    peak_after_bounce = z[500:].max()
    assert 0.0 < peak_after_bounce < 1.0


def test_offset_collider_produces_tip_over_torque():
    """A body resting on the ground via an off-axis collider produces a
    body-frame torque about the perpendicular axis — the classic 'tip
    over' moment when the support isn't under the COM."""
    g = 9.81
    w = (World()
         .add_field(GravityField().add_uniform((0, 0, -g)))
         .add_field(CollisionField().add(HalfSpace(normal=(0, 0, 1)))))
    cr = Craft("tower")
    cr.add(Mass("body", mass=1.0, moi=(0.05, 0.05, 0.05)))
    # Collider mounted at +x = 0.5 m from body origin.
    cr.add(Collider("foot", stiffness=1e4, damping=20.0,
                    mount_offset=(0.5, 0.0, 0.0)))
    # Start just above the ground so the offset foot starts penetrating
    # immediately.
    w.add_craft(cr, position=(0, 0, 0.01))
    cw = TargetNumpy(Sim(w))
    # Long enough to develop body-frame torque about y as body tilts.
    for _ in range(500):
        cw.step(0.001)
    omega = np.array(cw.state["tower"]["angular_velocity"]).ravel()
    # x and z stay zero; y develops as body tips.
    assert abs(omega[1]) > 1e-6
    assert abs(omega[0]) < 1e-9
    assert abs(omega[2]) < 1e-9


# ---------------------------------------------------------------------------
# Collision in a coupled (multi-craft) component
# ---------------------------------------------------------------------------

def test_collision_field_reaches_world_tick():
    """A Collider on one craft of a tethered pair should see the
    ground plane the World registers. Before the world_tick fix this
    silently dropped the CollisionField and the body free-fell through
    the floor."""
    from manta.couplings import Tether
    from manta.parts import TetherEndpoint

    g  = 9.81
    cf = CollisionField()
    cf.add(HalfSpace(origin=(0, 0, 0), normal=(0, 0, 1)))
    w = (World()
         .add_field(GravityField().add_uniform((0.0, 0.0, -g)))
         .add_field(cf))

    a = Craft("a")
    a.add(Mass("body", mass=1.0, moi=(0.01, 0.01, 0.01)))
    a.add(TetherEndpoint("hook"))
    a.add(Collider("foot", stiffness=1e5, damping=100.0))

    b = Craft("b")
    b.add(Mass("body", mass=1.0, moi=(0.01, 0.01, 0.01)))
    b.add(TetherEndpoint("hook"))

    w.add_craft(a, position=(0.0, 0.0, 1.0))
    w.add_craft(b, position=(2.0, 0.0, 1.0))
    w.add_coupling(Tether(a, "hook", b, "hook",
                          stiffness=10.0, damping=0.5, rest_length=2.0))

    cw = TargetNumpy(Sim(w))

    for _ in range(3000):
        cw.step(0.001)

    # Craft `a` rests at the ground-compression equilibrium z = -m·g/k.
    z_a = float(cw.state["a"]["position"][2])
    expected = -1.0 * g / 1e5
    assert np.isclose(z_a, expected, atol=2e-4), (
        f"a.z={z_a}, expected≈{expected}; the world_tick path is "
        f"probably dropping the CollisionField again.")


# ---------------------------------------------------------------------------
# Sphere (planet-surface) obstacle
# ---------------------------------------------------------------------------

def test_sphere_outside_returns_zero():
    from manta.fields import Sphere
    cf = CollisionField()
    cf.add(Sphere(center=(0, 0, 0), radius=10.0))
    np.testing.assert_allclose(_eval_pen_at(cf, (0, 0, 12.0)), np.zeros(3),
                               atol=1e-5)


def test_sphere_inside_outward_radial():
    from manta.fields import Sphere
    cf = CollisionField()
    cf.add(Sphere(center=(1, 0, 0), radius=10.0))
    # 0.5 m below the surface along +z from the centre: outward = +z.
    pen = _eval_pen_at(cf, (1, 0, 9.5))
    np.testing.assert_allclose(pen, (0.0, 0.0, 0.5), atol=1e-6)
    # And along an arbitrary radial the outward direction follows it.
    d = np.array([0.0, 3.0, 4.0]) / 5.0
    pen = _eval_pen_at(cf, tuple(np.array([1, 0, 0]) + 9.0 * d))
    np.testing.assert_allclose(pen, d * 1.0, atol=1e-6)


def test_earth_registers_surface_collision():
    """Earth(surface_collision=True) lets a Collider craft rest anywhere
    on the sphere — here the equator, with no per-site half-space."""
    from manta.planets import Earth
    earth = Earth(rotation_rate=0.0)
    w = World(); w.add_planet(earth)
    c = Craft("lander")
    c.add(Mass("body", mass=1.0, moi=(0.01, 0.01, 0.01)))
    c.add(Collider("foot", stiffness=1e4, damping=100.0))
    w.add_craft(c, position=(earth.R_EQ + 0.001, 0.0, 0.0))
    sim = TargetNumpy(Sim(w))
    for _ in range(3000):
        sim.step(0.001)
    p = np.asarray(sim.state["lander"]["position"]).ravel()
    # Settled onto the surface: radius ≈ R_EQ minus the static sag.
    r = np.linalg.norm(p)
    sag = 1.0 * 9.81 / 1e4
    np.testing.assert_allclose(r, earth.R_EQ - sag, atol=2e-4)
    # And it sits on the +x side of the planet (no sideways slide).
    assert p[0] > earth.R_EQ - 1.0


def test_collider_friction_damps_tangential_slide():
    """friction > 0 grips a sliding contact; the default stays
    frictionless (no horizontal force from a normal-only contact)."""
    def slide(friction):
        c = Craft("box")
        c.add(Mass("m", mass=1.0, moi=(0.01, 0.01, 0.01)))
        c.add(Collider("foot", stiffness=2000.0, damping=30.0,
                       friction=friction))
        w = (World()
             .add_field(GravityField().add_uniform((0.0, 0.0, -9.81)))
             .add_field(CollisionField().add_half_space()))
        w.add_craft(c, position=(0, 0, 0.0), velocity=(1.0, 0.0, 0.0))
        sim = TargetNumpy(Sim(w))
        for _ in range(1000):
            sim.step(0.001)
        return float(np.asarray(sim.state["box"]["velocity"]).ravel()[0])

    assert slide(0.0) > 0.99            # frictionless: keeps sliding
    assert abs(slide(5.0)) < 0.05       # viscous grip: ~stopped in 1 s


def test_halfspace_normalizes_normal():
    """A non-unit `normal` is normalized at construction, so the
    penetration response doesn't scale with |normal|²."""
    hs = HalfSpace(origin=(0, 0, 0), normal=(0, 0, 2.0))
    np.testing.assert_allclose(hs.normal, (0.0, 0.0, 1.0))
    cf = CollisionField().add(hs)
    pen = _eval_pen_at(cf, (0, 0, -0.5))
    np.testing.assert_allclose(pen, (0.0, 0.0, 0.5), atol=1e-6)


def test_halfspace_rejects_zero_normal():
    import pytest
    with pytest.raises(ValueError, match="normal"):
        HalfSpace(origin=(0, 0, 0), normal=(0, 0, 0))


# ---------------------------------------------------------------------------
# Heightfield terrain
# ---------------------------------------------------------------------------

def test_heightfield_penetration_on_inclined_plane():
    """A plane encoded as a heightfield is reproduced EXACTLY by the
    B-spline (splines reproduce polynomials up to their degree), so the
    contact math has closed-form expectations.

    h(x, y) = 0.1·x. At p = (10, 0, 0.5): surface height 1.0, vertical
    excess e = 0.5. Normal direction (−0.1, 0, 1)/√1.01; perpendicular
    depth = e/√1.01 = 0.49752; outward vector = depth·n̂ =
    (−0.049505, 0, 0.495050).
    """
    from manta.fields import Heightfield

    x = np.arange(21, dtype=float)          # 0..20, dx = 1
    y = np.arange(11, dtype=float)
    H = 0.1 * x[:, None] + 0.0 * y[None, :]
    hf = Heightfield(H, x0=0.0, y0=0.0, dx=1.0, dy=1.0)

    p = Vec3[WorldFrame].constant((10.0, 0.0, 0.5))
    out = np.asarray(ca.evalf(hf.contribute_at_sym(p, ca.MX(0.0))._mx)).ravel()
    np.testing.assert_allclose(out, [-0.0495050, 0.0, 0.4950495],
                               atol=1e-6)

    # Above the surface: zero.
    p_up = Vec3[WorldFrame].constant((10.0, 0.0, 2.0))
    out_up = np.asarray(
        ca.evalf(hf.contribute_at_sym(p_up, ca.MX(0.0))._mx)).ravel()
    np.testing.assert_allclose(out_up, 0.0, atol=1e-9)

    # Surface queries agree with the encoded plane.
    assert np.isclose(hf.height_at(7.3, 4.2), 0.73, atol=1e-9)
    alt = float(ca.evalf(hf.altitude_of_sym(
        Vec3[WorldFrame].constant((7.3, 4.2, 5.0))._mx)))
    assert np.isclose(alt, 5.0 - 0.73, atol=1e-9)


def test_ball_rests_on_heightfield_terrain():
    """A ball dropped onto terrain settles at the local surface height
    minus the m·g/k compression. The terrain is non-trivial (a ramp
    region raises part of the grid) but flat where the ball lands, so
    the equilibrium is exact — slope contact math is pinned separately
    by the inclined-plane worked example (viscous tangential friction
    never fully stops creep on a slope, so a sloped resting site would
    test the friction model, not the terrain)."""
    from manta.fields import CollisionField

    m, g, k = 1.0, 9.81, 1e5
    x = np.arange(21, dtype=float)
    y = np.arange(21, dtype=float)
    H = 2.0 + 0.2 * np.clip(x[:, None] - 14.0, 0.0, None) \
        + 0.0 * y[None, :]                            # flat, ramp at x>14

    w = World().add_field(GravityField().add_uniform((0, 0, -g)))
    cf = CollisionField()
    hf = cf.add_heightfield(H, x0=0.0, y0=0.0, dx=1.0, dy=1.0)
    w.add_field(cf)

    cr = Craft("ball")
    cr.add(Mass("body", mass=m, moi=(0.01, 0.01, 0.01)))
    # damping high enough (ζ ≈ 0.5) that the bounce train dies well
    # inside the 5 s run — the equilibrium itself is damping-blind.
    cr.add(Collider("contact", stiffness=k, damping=300.0))
    w.add_craft(cr, position=(7.0, 10.0, 2.5))        # over the flat part
    cw = TargetNumpy(Sim(w))
    for _ in range(5000):
        cw.step(0.001)

    xf, yf, zf = (float(v) for v in cw.state["ball"]["position"])
    surface = hf.height_at(xf, yf)                    # 2.0 on the flat
    assert np.isclose(surface, 2.0, atol=1e-6)
    assert np.isclose(zf, surface - m * g / k, atol=2e-4), (
        f"settled z={zf}, surface={surface}")
    assert abs(cw.state["ball"]["velocity"][2]) < 1e-3


def test_heightfield_validation():
    import pytest

    from manta.fields import Heightfield

    with pytest.raises(ValueError, match="4 samples"):
        Heightfield(np.zeros((3, 8)))
    with pytest.raises(ValueError, match="spacing"):
        Heightfield(np.zeros((8, 8)), dx=0.0)
    with pytest.raises(ValueError, match="non-finite"):
        Heightfield(np.full((8, 8), np.nan))

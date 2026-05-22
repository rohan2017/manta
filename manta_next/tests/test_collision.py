"""CollisionField + Collider tests."""

import casadi as ca
import numpy as np

from manta_next import Craft, World
from manta_next.fields import CollisionField, HalfSpace
from manta_next.ir.frames import WorldFrame
from manta_next.ir.types import Vec3
from manta_next.parts import Collider, Mass


def _eval_pen_at(field, point_xyz):
    p = Vec3[WorldFrame].constant(point_xyz)
    val = field.state_at_sym(p)
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

def test_no_collision_field_means_no_contact_force():
    """Without a CollisionField registered, a Collider on a free-falling
    body has no effect — the body just free-falls."""
    g = (0.0, 0.0, -9.81)
    w = World().add_uniform_gravity(g)
    c = Craft("ball")
    c.add(Mass("body", mass=1.0, moi=(0.01, 0.01, 0.01)))
    c.add(Collider("contact", stiffness=1e5, damping=10.0))
    w.add_craft(c, position=(0, 0, 5))
    cw = w.compile()
    state = cw.initial_state()
    for _ in range(500):
        state = cw.step(state, dt=0.001)
    # Free-fall from z=5 for 0.5s: z = 5 - 0.5·g·t² = 5 - 1.22625 = 3.774.
    assert np.isclose(state["ball"]["position"][2], 3.774, atol=1e-3)


def test_ball_rests_on_ground_at_compression_equilibrium():
    """A ball resting on a stiff ground compresses by m·g/k. Verify the
    equilibrium depth is approximately that value."""
    m = 1.0
    g = 9.81
    k = 1e5     # stiff
    c_damp = 100.0

    w = (World()
         .add_uniform_gravity((0, 0, -g))
         .add_field(CollisionField().add(HalfSpace(normal=(0, 0, 1)))))
    cr = Craft("ball")
    cr.add(Mass("body", mass=m, moi=(0.01, 0.01, 0.01)))
    cr.add(Collider("contact", stiffness=k, damping=c_damp))
    w.add_craft(cr, position=(0, 0, 1.0))     # start above ground
    cw = w.compile()
    state = cw.initial_state()

    for _ in range(5000):
        state = cw.step(state, dt=0.001)

    z_final = float(state["ball"]["position"][2])
    expected_depth = m * g / k       # equilibrium compression
    assert np.isclose(z_final, -expected_depth, atol=2e-4), (
        f"z_final={z_final} expected≈{-expected_depth}")
    # Velocity should have decayed to ~0.
    assert abs(state["ball"]["velocity"][2]) < 1e-3


def test_dropped_ball_bounces():
    """A ball dropped above the ground bounces, then settles with damping."""
    g = 9.81
    w = (World()
         .add_uniform_gravity((0, 0, -g))
         .add_field(CollisionField().add(HalfSpace(normal=(0, 0, 1)))))
    cr = Craft("ball")
    cr.add(Mass("body", mass=1.0, moi=(0.01, 0.01, 0.01)))
    cr.add(Collider("contact", stiffness=2e4, damping=20.0))
    w.add_craft(cr, position=(0, 0, 1.0))
    cw = w.compile()
    state = cw.initial_state()

    # Track the maximum +z height attained AFTER the first bounce.
    z_history = []
    for _ in range(5000):
        state = cw.step(state, dt=0.001)
        z_history.append(float(state["ball"]["position"][2]))

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
         .add_uniform_gravity((0, 0, -g))
         .add_field(CollisionField().add(HalfSpace(normal=(0, 0, 1)))))
    cr = Craft("tower")
    cr.add(Mass("body", mass=1.0, moi=(0.05, 0.05, 0.05)))
    # Collider mounted at +x = 0.5 m from body origin.
    cr.add(Collider("foot", stiffness=1e4, damping=20.0,
                    transform=(0.5, 0.0, 0.0)))
    # Start just above the ground so the offset foot starts penetrating
    # immediately.
    w.add_craft(cr, position=(0, 0, 0.01))
    cw = w.compile()
    state = cw.initial_state()
    # Long enough to develop body-frame torque about y as body tilts.
    for _ in range(500):
        state = cw.step(state, dt=0.001)
    omega = np.array(state["tower"]["angular_velocity"]).ravel()
    # x and z stay zero; y develops as body tips.
    assert abs(omega[1]) > 1e-6
    assert abs(omega[0]) < 1e-9
    assert abs(omega[2]) < 1e-9


# ---------------------------------------------------------------------------
# Collision in a coupled (multi-craft) component
# ---------------------------------------------------------------------------

def test_collision_field_reaches_coupled_tick():
    """A Collider on one craft of a tethered pair should see the
    ground plane the World registers. Before the coupled_tick fix this
    silently dropped the CollisionField and the body free-fell through
    the floor."""
    from manta_next.couplings import Tether
    from manta_next.parts import TetherEndpoint

    g  = 9.81
    cf = CollisionField()
    cf.add(HalfSpace(origin=(0, 0, 0), normal=(0, 0, 1)))
    w = (World()
         .add_uniform_gravity((0.0, 0.0, -g))
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

    cw = w.compile()
    state = cw.initial_state()
    comp = list(cw.components)[0]

    for _ in range(3000):
        state = cw.step(state, dt=0.001)

    # Craft `a` rests at the ground-compression equilibrium z = -m·g/k.
    z_a = float(state[comp]["a.position"][2])
    expected = -1.0 * g / 1e5
    assert np.isclose(z_a, expected, atol=2e-4), (
        f"a.z={z_a}, expected≈{expected}; the coupled_tick path is "
        f"probably dropping the CollisionField again.")

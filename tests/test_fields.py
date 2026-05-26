"""Fields-as-superposition: GravityField with Uniform/PointMass disturbances.

The locked design from [[project-field-redesign-0501]] (legacy C++):
  - One concrete class per physical kind. Per-source variation is
    expressed by attaching different Disturbance subclasses.
  - field.state_at_sym(point) returns the symbolic sum of all
    disturbances' contributions at that point.

This file pins the architecture by exercising:
  1. Uniform field reproduces the old constant-gravity behavior.
  2. PointMassGravity gives correct inverse-square pull at sample points.
  3. Two disturbances superpose additively.
  4. Mass auto-applies field gravity when registered through World.
  5. Type checking rejects mismatched-shape disturbances.
"""

import numpy as np
import pytest

from manta import Craft, World, TargetNumpy
from manta.fields import (
    Disturbance, Field, GravityField, PointMassGravity, UniformGravity,
)
from manta.ir.frames import WorldFrame
from manta.ir.types import Vec3
from manta.parts import Mass


# ---------------------------------------------------------------------------
# Field plumbing
# ---------------------------------------------------------------------------

def _eval_at(field, point_xyz):
    """Evaluate a Field's state_at at a concrete point, returning np array."""
    import casadi as ca
    p = Vec3[WorldFrame].constant(point_xyz)
    val = field.state_at_sym(p, 0.0)
    return np.asarray(ca.evalf(val._mx)).ravel()


def test_empty_field_state_at_is_zero():
    """A GravityField with no disturbances returns (0, 0, 0)."""
    gf = GravityField()
    np.testing.assert_allclose(_eval_at(gf, (1.0, 2.0, 3.0)),
                               np.zeros(3), atol=1e-12)


def test_uniform_disturbance_is_position_independent():
    gf = GravityField()
    gf.add(UniformGravity((0.0, 0.0, -9.81)))
    for query in [(0, 0, 0), (10, -5, 100), (-1e6, 1e6, 1e6)]:
        np.testing.assert_allclose(_eval_at(gf, query), (0, 0, -9.81),
                                   atol=1e-12)


def test_point_mass_gravity_inverse_square():
    """Point mass at origin: g(p) = -GM·p̂ / |p|²."""
    GM  = 3.986e14            # Earth's gravitational parameter
    gf  = GravityField()
    gf.add(PointMassGravity(position=(0.0, 0.0, 0.0), GM=GM, eps=0.0))

    r = 6.371e6   # ≈ Earth surface radius
    for direction in [(1, 0, 0), (0, 1, 0), (0, 0, 1)]:
        p_xyz = tuple(r * x for x in direction)
        out   = _eval_at(gf, p_xyz)
        mag_expected = GM / (r * r)
        np.testing.assert_allclose(
            out, [-mag_expected * d for d in direction], rtol=1e-9)


def test_disturbance_superposition_adds():
    """Two disturbances combine additively."""
    gf = GravityField()
    gf.add(UniformGravity((1.0, 0.0, 0.0)))
    gf.add(UniformGravity((0.0, 2.0, 0.0)))
    np.testing.assert_allclose(_eval_at(gf, (0, 0, 0)), (1, 2, 0), atol=1e-12)


# ---------------------------------------------------------------------------
# Type checking
# ---------------------------------------------------------------------------

def test_field_rejects_mismatched_disturbance():
    """A disturbance with the wrong value_shape can't be added."""
    class WrongShape(Disturbance):
        field_value_shape = float        # not Vec3[WorldFrame]
        def contribute_at_sym(self, point, t): return 0.0

    gf = GravityField()
    with pytest.raises(TypeError, match="not"):
        gf.add(WrongShape())


# ---------------------------------------------------------------------------
# World integration
# ---------------------------------------------------------------------------

def test_world_add_field_keys_by_class():
    """Each Field subclass appears at most once in the World registry."""
    w = World()
    w.add_field(GravityField().add(UniformGravity((0, 0, -9.81))))
    with pytest.raises(ValueError, match="already registered"):
        w.add_field(GravityField())
    assert w.get_field(GravityField) is not None


def test_world_uniform_gravity_matches_direct_compile():
    """Bit-equal behavior between the World/Field path and the
    direct gravity_world escape hatch."""
    c1 = Craft("solo_a")
    c1.add(Mass("body", mass=1.0))
    direct = c1.compile_tick(gravity_field=GravityField(g=(0.0, 0.0, -9.81)))

    c2 = Craft("solo_b")
    c2.add(Mass("body", mass=1.0))
    w = World().add_field(GravityField().add_uniform((0.0, 0.0, -9.81)))
    w.add_craft(c2, position=(0.0, 0.0, 50.0))
    cw = TargetNumpy(w.compile())

    state_d = c1.initial_state(position=(0.0, 0.0, 50.0))
    state_w = cw.initial_state()
    for _ in range(200):
        out = direct(dt=0.005, **state_d)
        state_d = {**state_d, **out}
        state_w = cw.step(state_w, dt=0.005)

    np.testing.assert_allclose(state_w["solo_b"]["position"],
                               state_d["position"], atol=1e-12)


def test_world_point_mass_gravity_orbital_acceleration():
    """A craft far from a planet feels the right magnitude pull."""
    GM = 3.986e14
    earth_r = 6.371e6
    w = World()
    gf = GravityField()
    gf.add(PointMassGravity(position=(0.0, 0.0, 0.0), GM=GM))
    w.add_field(gf)

    c = Craft("orbiter")
    c.add(Mass("body", mass=1.0))
    w.add_craft(c, position=(earth_r, 0.0, 0.0))     # on +x surface
    cw = TargetNumpy(w.compile())

    state = cw.initial_state()
    # Single step: v_x ← -GM/r² · dt.
    state = cw.step(state, dt=1e-3)
    expected_vx = -GM / (earth_r ** 2) * 1e-3
    np.testing.assert_allclose(state["orbiter"]["velocity"][0],
                               expected_vx, rtol=1e-6)
    np.testing.assert_allclose(state["orbiter"]["velocity"][1:], 0.0, atol=1e-9)


def test_world_no_field_is_in_vacuum():
    """A World with no GravityField → craft is in free space (no
    acceleration). Useful for orbital dynamics where gravity is provided
    by per-craft Disturbances elsewhere."""
    w = World()      # no add_uniform_gravity
    c = Craft("floater")
    c.add(Mass("body", mass=1.0))
    w.add_craft(c, position=(0, 0, 100), velocity=(1.0, 0, 0))
    cw = TargetNumpy(w.compile())

    state = cw.initial_state()
    for _ in range(100):
        state = cw.step(state, dt=0.01)
    # No gravity → x advances at 1 m/s; z stays at 100.
    assert np.isclose(state["floater"]["position"][0], 1.0,   atol=1e-6)
    assert np.isclose(state["floater"]["position"][2], 100.0, atol=1e-9)


# ---------------------------------------------------------------------------
# Mass queries gravity at its own mount offset (audit E)
# ---------------------------------------------------------------------------

def test_offset_mass_under_point_mass_gravity_feels_local_g():
    """A Mass with a large body-frame transform under a PointMassGravity
    field should feel g at its actual location, not at the craft origin.

    Setup: craft origin at distance r0 from the point mass; a Mass
    mounted at offset r0 (along the same axis) in the body so its
    world-frame position is 2·r0 from the source. The Mass should
    feel g(2·r0) = -GM/(2·r0)² = ¼ · g(r0), not g(r0).
    """
    GM = 1.0e10
    r0 = 1.0e3
    gf = GravityField()
    gf.add(PointMassGravity(position=(0, 0, 0), GM=GM, eps=0.0))

    c = Craft("offset_mass")
    # Use apply_gravity=False on a placeholder body Mass so we isolate
    # the offset Mass's contribution. The placeholder still provides
    # mass for N-E (need m > 0 for the integrator).
    c.add(Mass("hub", mass=1.0, apply_gravity=False,
               moi=(0.0, 0.0, 0.0)))
    # The Mass whose gravity behaviour we measure: m=1 kg, mounted at
    # craft-frame offset (+r0, 0, 0).
    c.add(Mass("probe", mass=1.0, apply_gravity=True,
               transform=(r0, 0.0, 0.0)))

    w = World().add_field(gf)
    # Craft origin at anchor (r0, 0, 0) → "probe" sits at anchor (2·r0, 0, 0).
    w.add_craft(c, position=(r0, 0.0, 0.0))
    cw = TargetNumpy(w.compile())
    state = cw.initial_state()

    # One short tick — extract the net body force from the velocity
    # change. m_total = 2 kg, so a = F_net / 2.
    dt = 1e-4
    state = cw.step(state, dt=dt)

    # Expected: ONLY "probe" contributes gravity, at its actual location
    # (anchor 2·r0). g_probe = -GM/(2·r0)² along the -x direction at that
    # position; the body experiences F = m_probe · g_probe applied at
    # the offset. Per-axis force-and-torque arithmetic in Craft does the
    # right rolling, so the net body acceleration in the -x direction
    # equals (1 kg · g_probe) / 2 kg.
    g_at_probe = -GM / (2 * r0) ** 2
    expected_ax = (1.0 * g_at_probe) / 2.0
    vx = float(np.array(state["offset_mass"]["velocity"]).ravel()[0])
    expected_vx = expected_ax * dt
    np.testing.assert_allclose(vx, expected_vx, rtol=1e-3)


def test_uniform_gravity_unchanged_after_offset_fix():
    """Sanity: with a uniform GravityField, an offset Mass feels the
    same g as one at the craft origin — bit-identical to pre-fix behavior."""
    g = (0.0, 0.0, -9.81)

    # Setup A: Mass at origin.
    cA = Craft("a")
    cA.add(Mass("body", mass=1.0))
    wA = World().add_field(GravityField().add_uniform(g))
    wA.add_craft(cA, position=(0, 0, 10))
    cwA = TargetNumpy(wA.compile())
    sA = cwA.initial_state()

    # Setup B: Mass at non-zero transform on a zero-mass hub.
    cB = Craft("b")
    cB.add(Mass("hub", mass=1e-12, apply_gravity=False))
    cB.add(Mass("body", mass=1.0, transform=(5.0, -3.0, 2.0)))
    wB = World().add_field(GravityField().add_uniform(g))
    wB.add_craft(cB, position=(0, 0, 10))
    cwB = TargetNumpy(wB.compile())
    sB = cwB.initial_state()

    for _ in range(100):
        sA = cwA.step(sA, dt=0.001)
        sB = cwB.step(sB, dt=0.001)

    # Both crafts should fall identically in z (same g, same total mass).
    np.testing.assert_allclose(sA["a"]["velocity"][2],
                               sB["b"]["velocity"][2], rtol=1e-9)

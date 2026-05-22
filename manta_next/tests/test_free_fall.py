"""End-to-end M1/M2 demo: a Mass part defined in Python, traced into a
graph, compiled via CasADi, and run in pure Python.

M1 cases (linear free-fall) test that x(t) = x₀ + v₀·t + ½·g·t² is exact.
M2 cases ride the new 6-DOF tick — orientation just stays at identity for
these straight-line tests; the rotational dynamics get their own coverage
in test_rigid_body.py.
"""

import numpy as np
import pytest

from manta_next import ir
from manta_next.fields import GravityField
from manta_next.craft import Craft
from manta_next.parts import Mass


def _initial_at_rest(c: Craft, z=0.0):
    return c.initial_state(position=(0.0, 0.0, z))


# ---------------------------------------------------------------------------
# Free-fall (linear dynamics; no orientation change expected)
# ---------------------------------------------------------------------------

def test_single_mass_free_fall_numerics():
    """1 kg craft at z=100, no initial velocity, gravity = (0,0,-9.81)."""
    c = Craft("free_fall")
    c.add(Mass("body", mass=1.0))

    tick = c.compile_tick(gravity_field=GravityField(g=(0.0, 0.0, -9.81)))

    state = _initial_at_rest(c, z=100.0)
    dt = 0.001
    n_steps = 1000   # 1 second

    for _ in range(n_steps):
        out = tick(dt=dt, **state)
        state = {**state, **out}

    expected_z  = 100.0 - 0.5 * 9.81 * 1.0      # 95.095
    expected_vz = -9.81

    assert np.isclose(state["position"][2], expected_z,  atol=1e-5)
    assert np.isclose(state["velocity"][2], expected_vz, atol=1e-5)
    assert np.allclose(state["position"][:2],         [0.0, 0.0], atol=1e-9)
    assert np.allclose(state["velocity"][:2],         [0.0, 0.0], atol=1e-9)
    # No external torques, no initial angular velocity, identity initial q —
    # the body should not rotate.
    assert np.allclose(state["orientation"],          [1.0, 0.0, 0.0, 0.0], atol=1e-9)
    assert np.allclose(state["angular_velocity"],     [0.0, 0.0, 0.0],      atol=1e-9)


def test_multi_mass_aggregates_correctly():
    """Three masses at offsets, all apply gravity. Acceleration = g
    regardless of geometry."""
    c = Craft("multi_mass")
    c.add(Mass("heavy", mass=2.0))                                 # at origin
    c.add(Mass("light", mass=1.0, transform=(0.5, 0.0, 0.0)))      # offset
    c.add(Mass("third", mass=0.5, transform=(0.0, 0.5, 0.0)))
    assert c.total_mass == 3.5

    tick = c.compile_tick(gravity_field=GravityField(g=(0.0, 0.0, -9.81)))
    state = _initial_at_rest(c)
    for _ in range(100):
        out = tick(dt=0.01, **state)
        state = {**state, **out}

    # After 1 s: vz = -9.81, body origin z depends on offsets — but the
    # COM falls by exactly 0.5·g·t² regardless of COM location.
    inertials = c.aggregate_inertials()
    com_init = np.array([0.0, 0.0, 0.0]) + inertials["com"]
    com_after = state["position"] + inertials["com"]
    expected_com_z = com_init[2] - 0.5 * 9.81 * 1.0   # -4.905
    assert np.isclose(com_after[2], expected_com_z, atol=1e-5)
    assert np.isclose(state["velocity"][2], -9.81,    atol=1e-5)


def test_apply_gravity_false_makes_part_inert():
    c = Craft("partially_inert")
    c.add(Mass("ballast", mass=10.0, apply_gravity=False))
    c.add(Mass("active",  mass=1.0))

    tick = c.compile_tick(gravity_field=GravityField(g=(0.0, 0.0, -9.81)))

    state = _initial_at_rest(c)
    out = tick(dt=0.1, **state)
    # F_total = 1·g (only the active mass contributes).
    # a = F / m_total = -9.81 / 11.
    # v(0.1) = a·dt.
    expected_vz = -9.81 / 11.0 * 0.1
    assert np.isclose(out["velocity"][2], expected_vz, atol=1e-9)


def test_horizontal_gravity():
    """Gravity along +x produces +x acceleration of the origin."""
    c = Craft("sideways")
    c.add(Mass("body", mass=1.0))
    tick = c.compile_tick(gravity_field=GravityField(g=(2.0, 0.0, 0.0)))

    state = _initial_at_rest(c)
    for _ in range(100):
        out = tick(dt=0.01, **state)
        state = {**state, **out}

    assert np.isclose(state["velocity"][0], 2.0, atol=1e-5)
    assert np.isclose(state["position"][0], 1.0, atol=1e-5)


# ---------------------------------------------------------------------------
# API / construction errors
# ---------------------------------------------------------------------------

def test_unknown_parameter_raises():
    with pytest.raises(TypeError, match="unknown parameter"):
        Mass("body", mass=1.0, gravity=9.81)


def test_empty_craft_compile_raises():
    c = Craft("empty")
    with pytest.raises(ValueError, match="no parts"):
        c.compile_tick()


def test_zero_mass_craft_raises():
    c = Craft("massless")
    c.add(Mass("body", mass=0.0))
    with pytest.raises(ValueError, match="total mass"):
        c.compile_tick()


def test_parameter_introspection():
    m = Mass("body", mass=2.5, apply_gravity=False, transform=(0.1, 0.2, 0.3))
    assert m.mass == 2.5
    assert m.apply_gravity is False
    assert m.transform == (0.1, 0.2, 0.3)
    r = repr(m)
    assert "mass=2.5" in r

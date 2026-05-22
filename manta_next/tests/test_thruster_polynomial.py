"""Polynomial Thruster tests — exercises orders 0..4 and the
torque-coefficient pattern used by quadcopter rotors."""

import numpy as np
import pytest

from manta_next import Craft
from manta_next.parts import Mass, Thruster


def test_zeroth_order_constant_force():
    """F_0 alone: constant force regardless of throttle (a passive lift,
    bias, etc.)."""
    c = Craft("c")
    c.add(Mass("body", mass=1.0, moi=(0.1, 0.1, 0.1)))
    c.add(Thruster("t", forces=((0.0, 0.0, 5.0),)))   # F_0 only
    tick = c.compile_tick(gravity_world=(0, 0, 0))
    state = c.initial_state()
    state["t.throttle"] = 0.0
    for _ in range(100):
        out = tick(dt=0.001, **state)
        state = {**state, **out}
    # After 0.1s with 5 N constant force on 1 kg: v_z = 5·0.1 = 0.5 m/s.
    assert np.isclose(state["velocity"][2], 0.5, atol=1e-3)


def test_first_order_linear_in_throttle():
    """Standard linear thruster: F = throttle · F_1, built via the
    one-vector `force=…` shortcut."""
    c = Craft("c")
    c.add(Mass("body", mass=1.0, moi=(0.1, 0.1, 0.1)))
    c.add(Thruster("t", force=(0.0, 0.0, 10.0)))
    tick = c.compile_tick(gravity_world=(0, 0, 0))
    state = c.initial_state()
    state["t.throttle"] = 0.5    # half throttle = 5 N
    for _ in range(100):
        out = tick(dt=0.001, **state)
        state = {**state, **out}
    # v_z = 5·0.1 = 0.5 m/s.
    assert np.isclose(state["velocity"][2], 0.5, atol=1e-3)


def test_second_order_quadratic_in_throttle():
    """F = F_2 · throttle² — RPM²-style propeller thrust."""
    c = Craft("c")
    c.add(Mass("body", mass=1.0, moi=(0.1, 0.1, 0.1)))
    # F_0 = 0, F_1 = 0, F_2 = (0, 0, 4) → at throttle=0.5 → F = 4·0.25 = 1.0 N
    c.add(Thruster("t", forces=((0,0,0), (0,0,0), (0,0,4.0))))
    tick = c.compile_tick(gravity_world=(0, 0, 0))
    state = c.initial_state()
    state["t.throttle"] = 0.5
    for _ in range(100):
        out = tick(dt=0.001, **state)
        state = {**state, **out}
    # v_z = 1.0 · 0.1 = 0.1 m/s
    assert np.isclose(state["velocity"][2], 0.1, atol=1e-3)


def test_fourth_order_polynomial():
    """All four orders combined. Validates the loop unrolls correctly."""
    c = Craft("c")
    c.add(Mass("body", mass=1.0, moi=(0.1, 0.1, 0.1)))
    # F(t) = (0, 0, 1 + t + t² + t³ + t⁴)
    c.add(Thruster("t",
                   forces=((0,0,1), (0,0,1), (0,0,1), (0,0,1), (0,0,1))))
    tick = c.compile_tick(gravity_world=(0, 0, 0))
    state = c.initial_state()
    state["t.throttle"] = 2.0
    out = tick(dt=0.001, **state)
    # F_z = 1 + 2 + 4 + 8 + 16 = 31 N → a = 31 m/s² → v_z after 1 ms = 0.031
    assert np.isclose(out["velocity"][2], 0.031, atol=1e-6)


def test_torque_coefficient_yaw_reaction():
    """Quadcopter prop: linear thrust along +z with linear yaw torque about z.
    Both come from the one-vector shortcut."""
    c = Craft("c")
    c.add(Mass("body", mass=1.0, moi=(0.1, 0.1, 0.1)))
    c.add(Thruster("prop",
                   force=(0.0, 0.0, 10.0),
                   torque=(0.0, 0.0, 0.1)))
    tick = c.compile_tick(gravity_world=(0, 0, 0))
    state = c.initial_state()
    state["prop.throttle"] = 1.0
    out = tick(dt=0.001, **state)
    # 0.1 N·m / 0.1 kg·m² → α = 1 rad/s². After 1 ms ω_z ≈ 0.001 rad/s.
    assert np.isclose(out["angular_velocity"][2], 0.001, atol=1e-6)


def test_polynomial_order_cap_enforced():
    with pytest.raises(ValueError, match="exceeds the cap"):
        Thruster("t", forces=tuple([(0,0,0)] * 6))   # 5th-order → too high


def test_invalid_coefficient_shape_raises():
    with pytest.raises(ValueError, match="length-3"):
        Thruster("t", forces=((0, 0),))


def test_one_vector_shortcut_matches_explicit_polynomial():
    """The `force=(x,y,z)` shortcut produces the same forces tuple as
    explicit `forces=[(0,0,0), (x,y,z)]`."""
    t1 = Thruster("a",
                   force=(0.0, 0.0, 10.0),
                   torque=(0.0, 0.0, 0.05))
    t2 = Thruster("a",
                   forces=((0,0,0), (0,0,10.0)),
                   torques=((0,0,0), (0,0,0.05)))
    assert t1.forces == t2.forces
    assert t1.torques == t2.torques


def test_force_and_forces_mutually_exclusive():
    """Passing both `force` and `forces` is a construction error."""
    with pytest.raises(ValueError, match="not both"):
        Thruster("a", force=(0, 0, 1), forces=[(0, 0, 0), (0, 0, 1)])

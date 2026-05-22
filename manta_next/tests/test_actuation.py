"""Thruster actuation part — commanded force along a fixed body axis."""

import numpy as np

from manta_next import Craft, World
from manta_next.parts import Mass, Thruster


# ---------------------------------------------------------------------------
# Thrust = (m·|g|) cancels gravity → craft hovers
# ---------------------------------------------------------------------------

def test_hover_thrust_cancels_gravity():
    g_world = (0.0, 0.0, -9.81)
    m = 2.0
    c = Craft("hover")
    c.add(Mass("body", mass=m, moi=(0.1, 0.1, 0.1)))
    c.add(Thruster("t", force=(0.0, 0.0, 1.0)))

    tick = c.compile_tick(gravity_world=g_world)
    state = c.initial_state()
    state["t.throttle"] = m * 9.81           # exactly counters gravity

    for _ in range(1000):
        out = tick(dt=0.001, **state)
        state = {**state, **out}

    np.testing.assert_allclose(np.array(state["position"]).ravel(),
                               np.zeros(3), atol=1e-6)
    np.testing.assert_allclose(np.array(state["velocity"]).ravel(),
                               np.zeros(3), atol=1e-6)


# ---------------------------------------------------------------------------
# Thrust > m·|g| → craft accelerates upward
# ---------------------------------------------------------------------------

def test_excess_thrust_accelerates_up():
    g_world = (0.0, 0.0, -9.81)
    m = 1.0
    c = Craft("ascent")
    c.add(Mass("body", mass=m, moi=(0.1, 0.1, 0.1)))
    c.add(Thruster("t", force=(0.0, 0.0, 1.0)))

    tick = c.compile_tick(gravity_world=g_world)
    state = c.initial_state()
    state["t.throttle"] = m * 9.81 + 1.0     # net = +1 N upward → a = 1 m/s²

    for _ in range(1000):
        out = tick(dt=0.001, **state)
        state = {**state, **out}

    # v(1s) = 1 m/s; z(1s) = ½·1·1² = 0.5 m. Symplectic-Euler has tiny
    # bias but well within tolerance.
    assert np.isclose(np.array(state["velocity"]).ravel()[2], 1.0, atol=1e-3)
    assert np.isclose(np.array(state["position"]).ravel()[2], 0.5, atol=1e-3)


# ---------------------------------------------------------------------------
# Thrust at an offset → torque
# ---------------------------------------------------------------------------

def test_offset_thruster_produces_torque():
    """Thruster mounted offset from body z-axis → roll moment about y."""
    c = Craft("rolling")
    c.add(Mass("body", mass=1.0, moi=(0.1, 0.1, 0.1)))
    # Thruster mounted at (+1, 0, 0), pointing +z → torque about body y.
    c.add(Thruster("t", force=(0.0, 0.0, 1.0), transform=(1.0, 0.0, 0.0)))

    tick = c.compile_tick(gravity_world=(0.0, 0.0, 0.0))
    state = c.initial_state()
    state["t.throttle"] = 1.0     # 1 N at 1 m → τ = +1 N·m about body y.

    for _ in range(100):
        out = tick(dt=0.001, **state)
        state = {**state, **out}

    # α_y = τ / I_y = 1 / 0.1 = 10 rad/s². After 0.1 s → ω_y ≈ -1.0
    # (negative because r × F = (1,0,0) × (0,0,1) = (0, +1, 0)? wait
    # let me check: cross product (1,0,0)×(0,0,1) = (0·1-0·0, 0·0-1·1, 1·0-0·0) = (0, -1, 0).
    # So torque is -y direction → ω_y becomes negative.
    assert np.isclose(np.array(state["angular_velocity"]).ravel()[1],
                      -1.0, atol=1e-3)

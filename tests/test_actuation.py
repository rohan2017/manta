"""Thruster actuation part — commanded force along a fixed body axis."""

import numpy as np

from manta import Craft, Sim, TargetNumpy, World
from manta.fields import GravityField
from manta.parts import Mass, Thruster


# ---------------------------------------------------------------------------
# Thrust = (m·|g|) cancels gravity → craft hovers
# ---------------------------------------------------------------------------

def test_hover_thrust_cancels_gravity():
    g_world = (0.0, 0.0, -9.81)
    m = 2.0
    c = Craft("hover")
    c.add(Mass("body", mass=m, moi=(0.1, 0.1, 0.1)))
    c.add(Thruster("t", force=(0.0, 0.0, 1.0)))

    w = World().add_field(GravityField(g=g_world))
    w.add_craft(c, **{"t.throttle": m * 9.81})   # exactly counters gravity
    sim = TargetNumpy(Sim(w))

    for _ in range(1000):
        sim.step(0.001)

    np.testing.assert_allclose(np.array(sim.state["hover"]["position"]).ravel(),
                               np.zeros(3), atol=1e-6)
    np.testing.assert_allclose(np.array(sim.state["hover"]["velocity"]).ravel(),
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

    w = World().add_field(GravityField(g=g_world))
    w.add_craft(c, **{"t.throttle": m * 9.81 + 1.0})   # net = +1 N up → a = 1 m/s²
    sim = TargetNumpy(Sim(w))

    for _ in range(1000):
        sim.step(0.001)

    # v(1s) = 1 m/s; z(1s) = ½·1·1² = 0.5 m. Symplectic-Euler has tiny
    # bias but well within tolerance.
    assert np.isclose(np.array(sim.state["ascent"]["velocity"]).ravel()[2], 1.0, atol=1e-3)
    assert np.isclose(np.array(sim.state["ascent"]["position"]).ravel()[2], 0.5, atol=1e-3)


# ---------------------------------------------------------------------------
# Thrust at an offset → torque
# ---------------------------------------------------------------------------

def test_offset_thruster_produces_torque():
    """Thruster mounted offset from body z-axis → roll moment about y."""
    c = Craft("rolling")
    c.add(Mass("body", mass=1.0, moi=(0.1, 0.1, 0.1)))
    # Thruster mounted at (+1, 0, 0), pointing +z → torque about body y.
    c.add(Thruster("t", force=(0.0, 0.0, 1.0), mount_offset=(1.0, 0.0, 0.0)))

    w = World().add_field(GravityField(g=(0.0, 0.0, 0.0)))
    w.add_craft(c, **{"t.throttle": 1.0})     # 1 N at 1 m → τ = +1 N·m about body y.
    sim = TargetNumpy(Sim(w))

    for _ in range(100):
        sim.step(0.001)

    # α_y = τ / I_y = 1 / 0.1 = 10 rad/s². After 0.1 s → ω_y ≈ -1.0
    # (cross product (1,0,0)×(0,0,1) = (0, -1, 0) → torque is -y direction).
    assert np.isclose(np.array(sim.state["rolling"]["angular_velocity"]).ravel()[1],
                      -1.0, atol=1e-3)

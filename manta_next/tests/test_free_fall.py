"""End-to-end M1 demo: a Mass part defined in Python, traced into a
graph, compiled via CasADi, and run in pure Python. The numerics must
match analytic free-fall: x(t) = x₀ + v₀·t + ½·g·t².
"""

import numpy as np
import pytest

from manta_next import ir
from manta_next.craft import Craft
from manta_next.parts import Mass


# ---------------------------------------------------------------------------
# Demo: free-fall craft
# ---------------------------------------------------------------------------

def test_single_mass_free_fall_numerics():
    """1 kg craft at z=100, no initial velocity, gravity = (0,0,-9.81)."""
    c = Craft("free_fall")
    c.add(Mass("body", mass=1.0))

    tick = c.compile_tick(gravity_anchor=(0.0, 0.0, -9.81))

    state = {
        "position": np.array([0.0, 0.0, 100.0]),
        "velocity": np.array([0.0, 0.0, 0.0]),
    }
    dt = 0.001
    n_steps = 1000   # 1 second of sim

    for _ in range(n_steps):
        out = tick(dt=dt, **state)
        state = {"position": out["position"], "velocity": out["velocity"]}

    # Analytic: x(1) = 100 + 0 - 0.5·9.81·1² = 95.095 ; v(1) = -9.81.
    expected_z = 100.0 - 0.5 * 9.81 * 1.0
    expected_vz = -9.81

    # Symplectic-Euler is exact for constant accel (the position update
    # uses v·dt + ½·a·dt² which is the analytic increment per tick).
    assert np.isclose(state["position"][2], expected_z, atol=1e-6)
    assert np.isclose(state["velocity"][2], expected_vz, atol=1e-6)
    assert np.allclose(state["position"][:2], [0.0, 0.0])


def test_multi_mass_aggregates_correctly():
    """Two masses, total m = 3 kg, both apply gravity. Acceleration = g
    regardless of how the mass is divided across parts."""
    c = Craft("multi_mass")
    c.add(Mass("heavy", mass=2.0))
    c.add(Mass("light", mass=1.0))
    assert c.total_mass == 3.0

    tick = c.compile_tick(gravity_anchor=(0.0, 0.0, -9.81))

    state = {
        "position": np.array([0.0, 0.0, 0.0]),
        "velocity": np.array([0.0, 0.0, 0.0]),
    }
    dt = 0.01
    for _ in range(100):
        out = tick(dt=dt, **state)
        state = {"position": out["position"], "velocity": out["velocity"]}

    # After 1s of free-fall: vz = -9.81, z = -0.5·9.81 = -4.905.
    assert np.isclose(state["velocity"][2], -9.81, atol=1e-6)
    assert np.isclose(state["position"][2], -4.905, atol=1e-6)


def test_apply_gravity_false_makes_part_inert():
    c = Craft("partially_inert")
    c.add(Mass("ballast", mass=10.0, apply_gravity=False))
    c.add(Mass("active",  mass=1.0))

    tick = c.compile_tick(gravity_anchor=(0.0, 0.0, -9.81))

    state = {
        "position": np.array([0.0, 0.0, 0.0]),
        "velocity": np.array([0.0, 0.0, 0.0]),
    }
    dt = 0.1
    out = tick(dt=dt, **state)
    # F_total = 1·g (only the active mass contributes).
    # a = F / m_total = 1·g / 11 = -9.81/11.
    # v(0.1) = a·dt = -9.81/11 · 0.1.
    expected_vz = -9.81 / 11.0 * 0.1
    assert np.isclose(out["velocity"][2], expected_vz, atol=1e-9)


def test_horizontal_gravity():
    """Sanity: gravity along +x produces +x acceleration."""
    c = Craft("sideways")
    c.add(Mass("body", mass=1.0))
    tick = c.compile_tick(gravity_anchor=(2.0, 0.0, 0.0))   # 2 m/s² in +x

    state = {
        "position": np.array([0.0, 0.0, 0.0]),
        "velocity": np.array([0.0, 0.0, 0.0]),
    }
    for _ in range(100):
        out = tick(dt=0.01, **state)
        state = {"position": out["position"], "velocity": out["velocity"]}

    # After 1s: vx = 2, x = 1, others zero.
    assert np.isclose(state["velocity"][0],  2.0,  atol=1e-6)
    assert np.isclose(state["position"][0],  1.0,  atol=1e-6)
    assert np.allclose(state["velocity"][1:], [0.0, 0.0])
    assert np.allclose(state["position"][1:], [0.0, 0.0])


# ---------------------------------------------------------------------------
# Misc Craft / Part API tests
# ---------------------------------------------------------------------------

def test_unknown_parameter_raises():
    with pytest.raises(TypeError, match="unknown parameter"):
        Mass("body", mass=1.0, gravity=9.81)   # 'gravity' isn't declared


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
    m = Mass("body", mass=2.5, apply_gravity=False)
    assert m.mass == 2.5
    assert m.apply_gravity is False
    # repr includes parameter values.
    r = repr(m)
    assert "mass=2.5" in r
    assert "apply_gravity=False" in r


def test_jacobian_of_tick_wrt_velocity():
    """For a constant-mass craft, d(position_next)/d(velocity) = dt · I.
    Free for us via CasADi's symbolic differentiation through the
    compiled graph."""
    c = Craft("jacobian_demo")
    c.add(Mass("body", mass=1.0))

    # We can't go through compile_tick() directly to get the Jacobian
    # because that already calls .compile(). Replicate just enough to
    # get the Graph; this exercises the same trace path.
    from manta_next.ir.frames import AnchorFrame, CraftFrame
    from manta_next.parts.wrench import Wrench

    with ir.Graph(name="tick_jac") as g:
        pos = ir.Vec3[AnchorFrame].input("position")
        vel = ir.Vec3[AnchorFrame].input("velocity")
        dt  = ir.Scalar.input("dt")
        ctx_gravity = ir.Vec3[CraftFrame].constant((0.0, 0.0, -9.81))
        from manta_next.craft import TickContext
        ctx = TickContext(gravity=ctx_gravity, dt=dt)
        net = Wrench.zero(CraftFrame)
        for p in c.parts:
            net = net + p.update(ctx)
        m_total = ir.Scalar.constant(c.total_mass)
        f_a = ir.Vec3(net.force._mx, frame=AnchorFrame)
        accel = f_a / m_total
        new_pos = pos + vel * dt + accel * 0.5 * dt * dt
        g.output(new_pos, "position")

    J = g.jacobian(of="position", wrt="velocity")
    out = J(position=[0, 0, 0], velocity=[0, 0, 0], dt=0.05)
    assert np.allclose(out["jacobian"], 0.05 * np.eye(3))

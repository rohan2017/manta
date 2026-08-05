"""Thruster tests — linear + quadratic in throttle, plus reaction torque."""

import numpy as np
import pytest

from manta import Craft, Sim, TargetNumpy, World
from manta.fields import GravityField
from manta.parts import Mass, Thruster


def test_linear_in_throttle():
    """F = throttle · force_vec — the standard linear case."""
    c = Craft("c")
    c.add(Mass("body", mass=1.0, moi=(0.1, 0.1, 0.1)))
    c.add(Thruster("t", force=(0.0, 0.0, 10.0)))
    w = World().add_field(GravityField(g=(0, 0, 0)))
    w.add_craft(c, **{"t.throttle": 0.5})    # half throttle ⇒ 5 N
    sim = TargetNumpy(Sim(w))
    for _ in range(100):
        sim.step(0.001)
    assert np.isclose(sim.state["c"]["velocity"][2], 0.5, atol=1e-3)


def test_quadratic_in_throttle():
    """F = force_quad · throttle² — rotor-blade form."""
    c = Craft("c")
    c.add(Mass("body", mass=1.0, moi=(0.1, 0.1, 0.1)))
    # K_T = 4 N/throttle². At throttle=0.5 → F = 4·0.25 = 1.0 N.
    c.add(Thruster("t", force_quad=(0.0, 0.0, 4.0)))
    w = World().add_field(GravityField(g=(0, 0, 0)))
    w.add_craft(c, **{"t.throttle": 0.5})
    sim = TargetNumpy(Sim(w))
    for _ in range(100):
        sim.step(0.001)
    assert np.isclose(sim.state["c"]["velocity"][2], 0.1, atol=1e-3)


def test_linear_plus_quadratic_combine():
    """F = force·t + force_quad·t². At throttle=2, force=(0,0,3),
    force_quad=(0,0,1): F_z = 3·2 + 1·4 = 10 N. After 1 ms on 1 kg:
    v_z = 0.01 m/s."""
    c = Craft("c")
    c.add(Mass("body", mass=1.0, moi=(0.1, 0.1, 0.1)))
    c.add(Thruster("t", force=(0, 0, 3.0), force_quad=(0, 0, 1.0)))
    w = World().add_field(GravityField(g=(0, 0, 0)))
    w.add_craft(c, **{"t.throttle": 2.0})
    sim = TargetNumpy(Sim(w))
    sim.step(0.001)
    assert np.isclose(sim.state["c"]["velocity"][2], 0.01, atol=1e-6)


def test_linear_reaction_torque():
    """Prop reaction torque: τ = throttle · torque_vec along z."""
    c = Craft("c")
    c.add(Mass("body", mass=1.0, moi=(0.1, 0.1, 0.1)))
    c.add(Thruster("prop",
                   force=(0.0, 0.0, 10.0),
                   torque=(0.0, 0.0, 0.1)))
    w = World().add_field(GravityField(g=(0, 0, 0)))
    w.add_craft(c, **{"prop.throttle": 1.0})
    sim = TargetNumpy(Sim(w))
    sim.step(0.001)
    # 0.1 N·m / 0.1 kg·m² → α = 1 rad/s². After 1 ms ω_z ≈ 0.001 rad/s.
    assert np.isclose(sim.state["c"]["angular_velocity"][2], 0.001, atol=1e-6)


def test_quadratic_reaction_torque():
    """Rotor reaction torque proportional to throttle² — the standard
    K_Q model alongside K_T thrust."""
    c = Craft("c")
    c.add(Mass("body", mass=1.0, moi=(0.1, 0.1, 0.1)))
    c.add(Thruster("rotor",
                   force_quad=(0, 0, 10.0),
                   torque_quad=(0, 0, 0.4)))
    w = World().add_field(GravityField(g=(0, 0, 0)))
    w.add_craft(c, **{"rotor.throttle": 0.5})      # τ = 0.4 · 0.25 = 0.1 N·m
    sim = TargetNumpy(Sim(w))
    sim.step(0.001)
    assert np.isclose(sim.state["c"]["angular_velocity"][2], 0.001, atol=1e-6)


def test_quadratic_term_preserves_direction():
    """`force_quad` uses `t·|t|`, not `t²`.

    Plain `t²` is positive on both sides of zero, so a reversible
    thruster commanded to full reverse would PUSH FORWARD. On a rotor
    that only spins one way the two forms are identical (t >= 0), which
    is exactly why the bug hid: every quadrotor in the examples is
    blind to it, and the first marine thruster is not.

    Full reverse must mirror full forward, and half reverse must be a
    quarter of it — the quadratic shape, with the sign kept.
    """
    def accel(throttle: float) -> float:
        w = World().add_field(GravityField().add_uniform((0.0, 0.0, 0.0)))
        c = Craft("boat")
        c.add(Mass("body", mass=10.0, moi=(1.0, 1.0, 1.0)))
        c.add(Thruster("t", force_quad=(20.0, 0.0, 0.0)))
        w.add_craft(c)
        sim = TargetNumpy(Sim(w))
        sim.state["boat"]["t.throttle"] = throttle
        sim.step(0.01)
        return float(sim.state["boat"]["velocity"][0]) / 0.01

    fwd = accel(1.0)
    rev = accel(-1.0)
    assert fwd == pytest.approx(20.0 / 10.0, rel=1e-6)
    assert rev == pytest.approx(-fwd, rel=1e-6), (
        f"full reverse gave {rev:+.3f} m/s^2 against {fwd:+.3f} forward — "
        f"a t^2 term pushes the wrong way")
    # Quadratic, not linear: half command is a quarter of the force.
    assert accel(-0.5) == pytest.approx(0.25 * rev, rel=1e-6)

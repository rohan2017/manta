"""State declarations on parts — read inside `update()`, written via
PartUpdate.new_state, integrated into the compiled tick."""

import math

import numpy as np
import pytest

from manta_next.craft import Craft
from manta_next.parts import (
    FlywheelMotor,
    Mass,
    PartUpdate,
    SpinningRotor,
    State,
)


# ---------------------------------------------------------------------------
# SpinningRotor — pure kinematic state
# ---------------------------------------------------------------------------

def test_spinning_rotor_advances_angle_at_constant_rate():
    """1 rad/s for 1 s ⇒ angle = 1.0."""
    c = Craft("rotor_demo")
    c.add(Mass("body", mass=1.0))                        # gives the craft mass
    c.add(SpinningRotor("wheel", spin_rate=1.0))

    tick = c.compile_tick(gravity_anchor=(0.0, 0.0, 0.0))

    state = c.initial_state()
    assert state["wheel.angle"] == 0.0   # init default

    for _ in range(1000):
        out = tick(dt=0.001, **state)
        state = {**state, **out}

    assert np.isclose(state["wheel.angle"], 1.0, atol=1e-9)


def test_spinning_rotor_initial_angle_override():
    """User can set the initial angle via initial_state()."""
    c = Craft("rotor_demo")
    c.add(Mass("body", mass=1.0))
    c.add(SpinningRotor("wheel", spin_rate=2.0))

    tick = c.compile_tick(gravity_anchor=(0.0, 0.0, 0.0))
    state = c.initial_state(**{"wheel.angle": 0.5})
    assert state["wheel.angle"] == 0.5

    for _ in range(500):
        out = tick(dt=0.001, **state)
        state = {**state, **out}

    # 0.5 + 2.0 * 0.5 s = 1.5
    assert np.isclose(state["wheel.angle"], 1.5, atol=1e-9)


def test_multiple_rotors_have_independent_state():
    """Two rotors with different spin rates evolve independently."""
    c = Craft("twin_rotors")
    c.add(Mass("body", mass=1.0))
    c.add(SpinningRotor("fast", spin_rate=5.0))
    c.add(SpinningRotor("slow", spin_rate=1.0))

    tick = c.compile_tick(gravity_anchor=(0.0, 0.0, 0.0))
    state = c.initial_state()
    for _ in range(1000):
        out = tick(dt=0.001, **state)
        state = {**state, **out}

    assert np.isclose(state["fast.angle"], 5.0, atol=1e-9)
    assert np.isclose(state["slow.angle"], 1.0, atol=1e-9)


# ---------------------------------------------------------------------------
# FlywheelMotor — state + reaction wrench
# ---------------------------------------------------------------------------

def test_flywheel_under_torque_accelerates():
    """A free flywheel (heavy body, no other dynamics) accelerates under
    its commanded torque: α = τ / I_axial."""
    c = Craft("flywheel_alone")
    # Heavy body with isotropic MOI so the craft barely rotates.
    c.add(Mass("body", mass=100.0, moi=(1000.0, 1000.0, 1000.0)))
    c.add(FlywheelMotor("wheel", I_axial=0.05, torque_cmd=0.5))

    tick = c.compile_tick(gravity_anchor=(0.0, 0.0, 0.0))
    state = c.initial_state()
    for _ in range(1000):
        out = tick(dt=0.001, **state)
        state = {**state, **out}

    # α = 0.5 / 0.05 = 10 rad/s² → ω(1s) = 10, θ(1s) = 5.
    assert np.isclose(state["wheel.rate"],  10.0, atol=1e-6)
    assert np.isclose(state["wheel.angle"],  5.0, atol=1e-6)


def test_flywheel_reaction_spins_body_counter():
    """Conservation of angular momentum: starting from rest, spinning up
    the flywheel about +z should counter-rotate the body about -z, so
    that L_body + L_flywheel stays at zero.

    L_flywheel = I_axial · θ̇ = 0.05 · 10 = 0.5 kg·m²/s (after 1 s under
    τ=0.5 N·m). The body's I_zz = 0.1 kg·m². So body ω_z = -0.5 / 0.1 = -5 rad/s.

    Note: in M3 the reaction torque is the commanded -τ along axis, so this
    integration uses Newton's-3rd reaction directly — the body picks up
    angular momentum at the same rate the flywheel does, opposite sign.
    """
    c = Craft("conservation")
    c.add(Mass("body", mass=1.0, moi=(0.1, 0.1, 0.1)))
    c.add(FlywheelMotor("wheel", I_axial=0.05, torque_cmd=0.5))

    tick = c.compile_tick(gravity_anchor=(0.0, 0.0, 0.0))
    state = c.initial_state()
    for _ in range(1000):
        out = tick(dt=0.001, **state)
        state = {**state, **out}

    # Flywheel: ω_wheel = 0.5/0.05 · 1 = 10.0 rad/s.
    assert np.isclose(state["wheel.rate"], 10.0, atol=1e-6)
    # Body: τ_react = -0.5 along +z, I_body_zz = 0.1 → α_body_z = -5 rad/s².
    assert np.isclose(state["angular_velocity"][2], -5.0, atol=1e-3)


# ---------------------------------------------------------------------------
# Mixed wrench + state contributions
# ---------------------------------------------------------------------------

def test_rotor_doesnt_break_free_fall():
    """Adding a SpinningRotor (zero wrench) to a free-fall craft must not
    perturb the linear dynamics."""
    c = Craft("fall_with_rotor")
    c.add(Mass("body", mass=1.0))
    c.add(SpinningRotor("wheel", spin_rate=10.0))

    tick = c.compile_tick(gravity_anchor=(0.0, 0.0, -9.81))
    state = c.initial_state(position=(0.0, 0.0, 100.0))
    for _ in range(1000):
        out = tick(dt=0.001, **state)
        state = {**state, **out}

    assert np.isclose(state["position"][2],   95.095, atol=1e-5)
    assert np.isclose(state["velocity"][2],   -9.81,  atol=1e-5)
    assert np.isclose(state["wheel.angle"],   10.0,   atol=1e-9)


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------

def test_unknown_state_slot_raises():
    """Returning new_state with a key not declared as State should error."""
    from manta_next.parts.base import Part, Parameter

    class BadPart(Part):
        a = State(init=0.0)

        def update(self, ctx):
            return PartUpdate(
                wrench=__import__("manta_next").parts.Wrench.zero(
                    __import__("manta_next").ir.frames.CraftFrame),
                new_state={"a": 1.0, "b": 2.0},   # 'b' not declared
            )

    c = Craft("bad")
    c.add(Mass("m", mass=1.0))
    c.add(BadPart("bad"))
    with pytest.raises(KeyError, match="unknown state slot"):
        c.compile_tick()


def test_initial_state_unknown_slot_raises():
    c = Craft("ok")
    c.add(Mass("body", mass=1.0))
    c.add(SpinningRotor("wheel", spin_rate=1.0))
    with pytest.raises(KeyError, match="unknown slot"):
        c.initial_state(**{"wheel.nonexistent": 0.0})


def test_state_declarations_introspection():
    c = Craft("intro")
    c.add(SpinningRotor("wheel", spin_rate=2.0))
    motor = FlywheelMotor("motor", I_axial=0.02)
    c.add(motor)

    rotor_decls = c.parts[0].state_declarations()
    assert set(rotor_decls.keys()) == {"angle"}

    motor_decls = motor.state_declarations()
    assert set(motor_decls.keys()) == {"angle", "rate"}

"""State declarations on parts — read inside `update()`, written via
PartUpdate.new_state, integrated into the compiled tick.

Exercises the `Joint` part (revolute joint with Mass rotor child) — the
unified replacement for the v1 FlywheelMotor + SpinningRotor combo.
"""

import numpy as np
import pytest

from manta_next.craft import Craft
from manta_next.parts import (
    Joint,
    Mass,
    PartUpdate,
    State,
)


# ---------------------------------------------------------------------------
# Joint(passive) — kinematic spin via initial rate
# ---------------------------------------------------------------------------

def test_passive_joint_spins_at_initial_rate():
    """Passive joint with initial rate=1 rad/s and a rotor child runs
    free — angle accumulates at 1 rad/s indefinitely (no friction)."""
    c = Craft("passive_demo")
    c.add(Mass("body", mass=1.0))
    j = Joint("wheel", mode="passive")
    j.add(Mass("rotor", mass=0.1, moi=(0.01, 0.01, 0.05)))
    c.add(j)

    tick = c.compile_tick(gravity_anchor=(0.0, 0.0, 0.0))
    state = c.initial_state(**{"wheel.rate": 1.0})
    assert state["wheel.angle"] == 0.0
    assert state["wheel.rate"]  == 1.0

    for _ in range(1000):
        out = tick(dt=0.001, **state)
        state = {**state, **out}

    assert np.isclose(state["wheel.angle"], 1.0, atol=1e-9)
    assert np.isclose(state["wheel.rate"],  1.0, atol=1e-9)


def test_multiple_joints_have_independent_state():
    c = Craft("twin")
    c.add(Mass("body", mass=1.0))
    fast = Joint("fast", mode="passive")
    fast.add(Mass("fast_disk", mass=0.1, moi=(0.001, 0.001, 0.01)))
    slow = Joint("slow", mode="passive")
    slow.add(Mass("slow_disk", mass=0.1, moi=(0.001, 0.001, 0.01)))
    c.add(fast)
    c.add(slow)

    tick = c.compile_tick(gravity_anchor=(0.0, 0.0, 0.0))
    state = c.initial_state(**{"fast.rate": 5.0, "slow.rate": 1.0})
    for _ in range(1000):
        out = tick(dt=0.001, **state)
        state = {**state, **out}

    assert np.isclose(state["fast.angle"], 5.0, atol=1e-9)
    assert np.isclose(state["slow.angle"], 1.0, atol=1e-9)


# ---------------------------------------------------------------------------
# Joint(saturating) — commanded torque drives rotor + reacts on body
# ---------------------------------------------------------------------------

def test_saturating_joint_below_stall_accelerates_rotor():
    """τ_cmd = 0.5 N·m < stall = 1.0 N·m: rotor accelerates at τ/I_axial."""
    c = Craft("flywheel")
    c.add(Mass("body", mass=100.0, moi=(1000.0, 1000.0, 1000.0)))  # heavy
    j = Joint("wheel", mode="saturating", stall_torque=1.0)
    j.add(Mass("rotor", mass=0.5, moi=(0.025, 0.025, 0.05)))
    c.add(j)

    tick = c.compile_tick(gravity_anchor=(0.0, 0.0, 0.0))
    state = c.initial_state()
    state["wheel.torque_cmd"] = 0.5
    for _ in range(1000):
        out = tick(dt=0.001, **state)
        state = {**state, **out}

    # I_axial = 0.05 (rotor's I_zz, axis defaults to +z).
    # α = 0.5 / 0.05 = 10 rad/s² over 1 s → ω = 10.
    assert np.isclose(state["wheel.rate"], 10.0, atol=1e-6)


def test_saturating_joint_clamps_at_stall():
    """τ_cmd above stall → torque clipped to ±stall, rotor accelerates
    at the clipped rate."""
    c = Craft("clamped")
    c.add(Mass("body", mass=100.0, moi=(1000.0, 1000.0, 1000.0)))
    j = Joint("wheel", mode="saturating", stall_torque=0.2)
    j.add(Mass("rotor", mass=0.5, moi=(0.025, 0.025, 0.05)))
    c.add(j)

    tick = c.compile_tick(gravity_anchor=(0.0, 0.0, 0.0))
    state = c.initial_state()
    state["wheel.torque_cmd"] = 5.0   # way above stall=0.2
    for _ in range(1000):
        out = tick(dt=0.001, **state)
        state = {**state, **out}

    # Effective τ = 0.2; α = 0.2 / 0.05 = 4 rad/s² → ω after 1s = 4.
    assert np.isclose(state["wheel.rate"], 4.0, atol=1e-6)


def test_saturating_joint_reaction_spins_body_counter():
    """Reaction torque on body: τ_react = -τ_clamped along axis. The
    body's MOI aggregate INCLUDES the rotor child's MOI (via Joint's
    `mass`/`moi` properties), so the effective inertia about the spin
    axis is I_body + I_rotor = 0.1 + 0.05 = 0.15. With τ = 0.5 N·m:
    α_body_z = -0.5/0.15 ≈ -3.33 rad/s²."""
    c = Craft("conservation")
    c.add(Mass("body", mass=1.0, moi=(0.1, 0.1, 0.1)))
    j = Joint("wheel", mode="saturating", stall_torque=1.0)
    j.add(Mass("rotor", mass=0.1, moi=(0.005, 0.005, 0.05)))
    c.add(j)

    tick = c.compile_tick(gravity_anchor=(0.0, 0.0, 0.0))
    state = c.initial_state()
    state["wheel.torque_cmd"] = 0.5
    for _ in range(1000):
        out = tick(dt=0.001, **state)
        state = {**state, **out}

    # ω after 1 s under constant α = -0.5/0.15. Loose tolerance because
    # the gyroscopic coupling (rotor spin × body ω) feeds back as ω grows.
    assert np.isclose(state["angular_velocity"][2], -0.5 / 0.15, atol=0.2)


# ---------------------------------------------------------------------------
# Joint contributes mass + MOI to the body
# ---------------------------------------------------------------------------

def test_joint_child_mass_appears_in_body_aggregate():
    """A 1 kg rotor child in a Joint contributes its mass to the body
    aggregate (so apparent inertia + gravity loading reflect it)."""
    c = Craft("weighted")
    body_mass = 1.0
    rotor_mass = 0.5
    c.add(Mass("body", mass=body_mass, moi=(0.1, 0.1, 0.1)))
    j = Joint("wheel", mode="passive")
    j.add(Mass("rotor", mass=rotor_mass, moi=(0.01, 0.01, 0.05)))
    c.add(j)

    inertials = c.aggregate_inertials()
    assert np.isclose(inertials["m_total"], body_mass + rotor_mass)


def test_joint_rejects_child_with_nonzero_transform():
    j = Joint("axle", mode="passive")
    with pytest.raises(ValueError, match="nonzero transform"):
        j.add(Mass("offset_rotor", mass=0.1, transform=(0.0, 0.0, 0.1)))


def test_joint_rejects_non_mass_child():
    from manta_next.parts import IMU
    j = Joint("axle", mode="passive")
    with pytest.raises(TypeError, match="Mass children"):
        j.add(IMU("g"))


def test_joint_unknown_mode_raises():
    with pytest.raises(ValueError, match="mode must be"):
        Joint("bad", mode="nonsense")


def test_joint_doesnt_break_free_fall():
    """A passive joint with a rotor doesn't perturb linear free-fall."""
    c = Craft("fall_with_joint")
    c.add(Mass("body", mass=1.0))
    j = Joint("wheel", mode="passive")
    j.add(Mass("rotor", mass=0.1, moi=(0.001, 0.001, 0.01)))
    c.add(j)

    tick = c.compile_tick(gravity_anchor=(0.0, 0.0, -9.81))
    state = c.initial_state(position=(0.0, 0.0, 100.0),
                             **{"wheel.rate": 10.0})
    for _ in range(1000):
        out = tick(dt=0.001, **state)
        state = {**state, **out}

    # ½·g·t² = 4.905; z = 100 - 4.905 = 95.095.
    assert np.isclose(state["position"][2],   95.095, atol=1e-5)
    assert np.isclose(state["velocity"][2],   -9.81,  atol=1e-5)
    assert np.isclose(state["wheel.angle"],   10.0,   atol=1e-9)


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------

def test_unknown_state_slot_raises():
    """Returning new_state with a key not declared as State should error."""
    from manta_next.parts.base import Part

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
    j = Joint("wheel", mode="passive")
    j.add(Mass("rotor", mass=0.1, moi=(0.001, 0.001, 0.01)))
    c.add(j)
    with pytest.raises(KeyError, match="unknown slot"):
        c.initial_state(**{"wheel.nonexistent": 0.0})


def test_joint_state_declarations_introspection():
    j = Joint("wheel", mode="passive")
    decls = j.state_declarations()
    assert set(decls.keys()) == {"angle", "rate"}


# ---------------------------------------------------------------------------
# Nested-Joint guard (correctness fix #6)
# ---------------------------------------------------------------------------

def test_joint_add_rejects_nested_joint_child():
    """A Joint can't host another Joint yet (v1 of the new hierarchy);
    raised at construction time. Lifted in M20.2 when the symbolic
    kinematic pass lands."""
    outer = Joint("outer", mode="passive")
    inner = Joint("inner", mode="passive")
    with pytest.raises(TypeError, match="nested Joints not supported yet"):
        outer.add(inner)


def test_craft_compile_rejects_nested_joint_via_back_door():
    """Defense-in-depth: even if a Joint subclass somehow bypasses
    `add`'s type check and ends up with a Joint child, the
    Craft.compile_tick guard catches it."""
    outer = Joint("outer", mode="passive")
    inner = Joint("inner", mode="passive")
    inner.add(Mass("inner_rotor", mass=0.1, moi=(0.001, 0.001, 0.001)))
    # Bypass the public API and shove a nested Joint into the children
    # list directly; this is what a future subclass with a custom add()
    # might do.
    outer._children.append(inner)
    outer._children.append(Mass("outer_rotor", mass=0.1,
                                 moi=(0.001, 0.001, 0.001)))

    c = Craft("with_nested_joint")
    c.add(Mass("body", mass=1.0))
    c.add(outer)

    with pytest.raises(TypeError, match="nested Joint"):
        c.compile_tick(gravity_anchor=(0.0, 0.0, 0.0))

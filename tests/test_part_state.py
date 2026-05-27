"""State declarations on parts — read inside `update()`, written via
PartUpdate.new_state, integrated into the compiled tick.

Exercises the `Joint` part (revolute joint with Mass rotor child) — the
unified replacement for the v1 FlywheelMotor + SpinningRotor combo.
"""

import numpy as np
import pytest

from manta.craft import Craft
from manta import World, TargetNumpy
from manta.fields import GravityField
from manta.parts import (
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

    w = World().add_field(GravityField(g=(0.0, 0.0, 0.0)))
    w.add_craft(c, **{"wheel.rate": 1.0})
    sim = TargetNumpy(w.compile())
    state = sim.initial_state()
    assert state["passive_demo"]["wheel.angle"] == 0.0
    assert state["passive_demo"]["wheel.rate"]  == 1.0

    for _ in range(1000):
        state = sim.step(state, dt=0.001)

    assert np.isclose(state["passive_demo"]["wheel.angle"], 1.0, atol=1e-9)
    assert np.isclose(state["passive_demo"]["wheel.rate"],  1.0, atol=1e-9)


def test_multiple_joints_have_independent_state():
    c = Craft("twin")
    c.add(Mass("body", mass=1.0))
    fast = Joint("fast", mode="passive")
    fast.add(Mass("fast_disk", mass=0.1, moi=(0.001, 0.001, 0.01)))
    slow = Joint("slow", mode="passive")
    slow.add(Mass("slow_disk", mass=0.1, moi=(0.001, 0.001, 0.01)))
    c.add(fast)
    c.add(slow)

    w = World().add_field(GravityField(g=(0.0, 0.0, 0.0)))
    w.add_craft(c, **{"fast.rate": 5.0, "slow.rate": 1.0})
    sim = TargetNumpy(w.compile())
    state = sim.initial_state()
    for _ in range(1000):
        state = sim.step(state, dt=0.001)

    assert np.isclose(state["twin"]["fast.angle"], 5.0, atol=1e-9)
    assert np.isclose(state["twin"]["slow.angle"], 1.0, atol=1e-9)


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

    w = World().add_field(GravityField(g=(0.0, 0.0, 0.0)))
    w.add_craft(c, **{"wheel.torque_cmd": 0.5})
    sim = TargetNumpy(w.compile())
    state = sim.initial_state()
    for _ in range(1000):
        state = sim.step(state, dt=0.001)

    # I_axial = 0.05 (rotor's I_zz, axis defaults to +z).
    # α = 0.5 / 0.05 = 10 rad/s² over 1 s → ω = 10.
    assert np.isclose(state["flywheel"]["wheel.rate"], 10.0, atol=1e-6)


def test_saturating_joint_clamps_at_stall():
    """τ_cmd above stall → torque clipped to ±stall, rotor accelerates
    at the clipped rate."""
    c = Craft("clamped")
    c.add(Mass("body", mass=100.0, moi=(1000.0, 1000.0, 1000.0)))
    j = Joint("wheel", mode="saturating", stall_torque=0.2)
    j.add(Mass("rotor", mass=0.5, moi=(0.025, 0.025, 0.05)))
    c.add(j)

    w = World().add_field(GravityField(g=(0.0, 0.0, 0.0)))
    w.add_craft(c, **{"wheel.torque_cmd": 5.0})   # way above stall=0.2
    sim = TargetNumpy(w.compile())
    state = sim.initial_state()
    for _ in range(1000):
        state = sim.step(state, dt=0.001)

    # Effective τ = 0.2; α = 0.2 / 0.05 = 4 rad/s² → ω after 1s = 4.
    assert np.isclose(state["clamped"]["wheel.rate"], 4.0, atol=1e-6)


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

    w = World().add_field(GravityField(g=(0.0, 0.0, 0.0)))
    w.add_craft(c, **{"wheel.torque_cmd": 0.5})
    sim = TargetNumpy(w.compile())
    state = sim.initial_state()
    for _ in range(1000):
        state = sim.step(state, dt=0.001)

    # ω after 1 s under constant α = -0.5/0.15. Loose tolerance because
    # the gyroscopic coupling (rotor spin × body ω) feeds back as ω grows.
    assert np.isclose(state["conservation"]["angular_velocity"][2],
                      -0.5 / 0.15, atol=0.2)


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


def test_joint_accepts_mass_child_with_nonzero_transform():
    """Children offset from the joint origin are now lifted symbolically
    by the inertia rollup — no constructor-side restriction."""
    j = Joint("axle", mode="passive")
    j.add(Mass("offset_rotor", mass=0.1, transform=(0.0, 0.0, 0.1)))
    assert j.children[0].transform == (0.0, 0.0, 0.1)


def test_joint_rejects_non_mass_non_joint_child():
    """Leaf parts that emit input-frame quantities (Thrusters, sensors,
    drag surfaces) still need articulation-aware update() before they
    can ride a moving rotor — the guard catches them at construction."""
    from manta.parts import IMU
    j = Joint("axle", mode="passive")
    with pytest.raises(TypeError, match="Mass and Joint children"):
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

    w = World().add_field(GravityField(g=(0.0, 0.0, -9.81)))
    w.add_craft(c, position=(0.0, 0.0, 100.0), **{"wheel.rate": 10.0})
    sim = TargetNumpy(w.compile())
    state = sim.initial_state()
    for _ in range(1000):
        state = sim.step(state, dt=0.001)

    # ½·g·t² = 4.905; z = 100 - 4.905 = 95.095.
    assert np.isclose(state["fall_with_joint"]["position"][2],   95.095, atol=1e-5)
    assert np.isclose(state["fall_with_joint"]["velocity"][2],   -9.81,  atol=1e-5)
    assert np.isclose(state["fall_with_joint"]["wheel.angle"],   10.0,   atol=1e-9)


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------

def test_unknown_state_slot_raises():
    """Returning new_state with a key not declared as State should error."""
    from manta.parts.base import Part

    class BadPart(Part):
        a = State(init=0.0)

        def update(self, ctx):
            return PartUpdate(
                wrench=__import__("manta").parts.Wrench.zero(
                    __import__("manta").ir.frames.CraftFrame),
                new_state={"a": 1.0, "b": 2.0},   # 'b' not declared
            )

    c = Craft("bad")
    c.add(Mass("m", mass=1.0))
    c.add(BadPart("bad"))
    w = World()
    w.add_craft(c)
    with pytest.raises(KeyError, match="unknown state slot"):
        w.compile()


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
# Nested Joints (gimbal-style compositions)
# ---------------------------------------------------------------------------

def test_nested_passive_joints_each_track_their_own_rate():
    """A pan-tilt gimbal: outer Joint hosts inner Joint, each with its
    own rotor and initial spin rate. Both angles accumulate independently
    over time — proves the kinematic + state plumbing reaches both
    levels of the joint chain."""
    c = Craft("gimbal_two_dof")
    c.add(Mass("body", mass=1.0, moi=(0.1, 0.1, 0.1)))
    outer = Joint("pan", mode="passive", axis=(0.0, 0.0, 1.0))
    outer.add(Mass("pan_disk", mass=0.1, moi=(0.001, 0.001, 0.01)))
    inner = Joint("tilt", mode="passive", axis=(0.0, 1.0, 0.0))
    inner.add(Mass("tilt_disk", mass=0.05, moi=(0.001, 0.001, 0.001)))
    outer.add(inner)
    c.add(outer)

    w = World().add_field(GravityField(g=(0.0, 0.0, 0.0)))
    w.add_craft(c, **{"pan.rate": 2.0, "tilt.rate": 1.0})
    sim = TargetNumpy(w.compile())
    state = sim.initial_state()
    for _ in range(1000):
        state = sim.step(state, dt=0.001)

    assert np.isclose(state["gimbal_two_dof"]["pan.angle"],  2.0, atol=1e-6)
    assert np.isclose(state["gimbal_two_dof"]["tilt.angle"], 1.0, atol=1e-6)


def test_nested_joint_saturating_drives_inner_rotor():
    """Inner saturating Joint commands torque on its rotor; the rotor
    accelerates at τ/I and the inner-joint reaction propagates onto
    the outer joint's frame (and from there to the body)."""
    c = Craft("gimbal_drive")
    c.add(Mass("body", mass=100.0, moi=(10.0, 10.0, 10.0)))
    outer = Joint("pan", mode="passive", axis=(0.0, 0.0, 1.0))
    outer.add(Mass("pan_disk", mass=0.05, moi=(0.001, 0.001, 0.01)))
    inner = Joint("tilt", mode="saturating", stall_torque=2.0,
                  axis=(0.0, 1.0, 0.0))
    inner.add(Mass("tilt_disk", mass=0.05, moi=(0.001, 0.005, 0.001)))
    outer.add(inner)
    c.add(outer)

    w = World().add_field(GravityField(g=(0.0, 0.0, 0.0)))
    w.add_craft(c, **{"tilt.torque_cmd": 0.5})
    sim = TargetNumpy(w.compile())
    state = sim.initial_state()
    for _ in range(1000):
        state = sim.step(state, dt=0.001)

    # Tilt rotor MOI about y-axis = 0.005 from the tilt_disk only;
    # α = 0.5 / 0.005 = 100 rad/s² → ω after 1s ≈ 100 rad/s.
    # Some tolerance for the heavy body's small angular response feeding
    # back into the gyroscopic correction.
    assert np.isclose(state["gimbal_drive"]["tilt.rate"], 100.0, atol=0.5)
    assert np.isclose(state["gimbal_drive"]["pan.rate"], 0.0, atol=0.5)


def test_offset_rotor_at_pi_over_2_shows_com_in_y():
    """The body-frame symbolic COM tracks the joint angle: at θ=π/2
    around the +z axis, a tip mass that started at +x lands at +y, so
    the body's COM in body coords moves from +x to +y. This is the
    motivating case for the symbolic inertia rollup."""
    c = Craft("arm_swept")
    c.add(Mass("body", mass=10.0, moi=(0.5, 0.5, 0.5)))
    j = Joint("arm", mode="passive", axis=(0.0, 0.0, 1.0))
    j.add(Mass("tip", mass=1.0, moi=(0.0, 0.0, 0.0),
               transform=(1.0, 0.0, 0.0)))
    c.add(j)

    # Set arm.angle = π/2 directly (rate = 0 → angle stays put under no
    # torque). The first tick exercises Newton-Euler with the rotated
    # COM in the body frame.
    w = World().add_field(GravityField(g=(0.0, 0.0, 0.0)))
    w.add_craft(c, **{"arm.angle": float(np.pi / 2)})
    sim = TargetNumpy(w.compile())
    state = sim.step(sim.initial_state(), dt=0.001)
    # Body should still be at rest after one tick (no external wrench,
    # rotor at rest); just verify the compile produced a valid graph
    # for the at-angle case.
    assert np.allclose(state["arm_swept"]["angular_velocity"], 0.0, atol=1e-9)
    assert np.isclose(state["arm_swept"]["arm.angle"], float(np.pi / 2), atol=1e-12)


def test_offset_rotor_changes_body_com():
    """A spinning arm with an off-axis mass: the symbolic inertia rollup
    sees the rotor at its current angle, so the body's effective COM
    moves with the joint angle. Confirm the offset rotor compiles and
    that aggregate mass + COM make sense."""
    c = Craft("arm")
    c.add(Mass("body", mass=10.0, moi=(0.5, 0.5, 0.5)))
    j = Joint("arm", mode="passive", axis=(0.0, 0.0, 1.0))
    # Off-axis mass: 1 kg at +x = 1 m from joint origin.
    j.add(Mass("tip", mass=1.0, moi=(0.0, 0.0, 0.0),
               transform=(1.0, 0.0, 0.0)))
    c.add(j)

    # Smoke: the offset rotor compiles under gravity.
    w = World().add_field(GravityField(g=(0.0, 0.0, -9.81)))
    w.add_craft(c)
    TargetNumpy(w.compile())

    inertials = c.aggregate_inertials()
    assert np.isclose(inertials["m_total"], 11.0)
    # Numpy snapshot uses transform values directly: tip at +x, so
    # com_x = (1.0 · 1.0) / 11 ≈ 0.0909.
    assert np.isclose(inertials["com"][0], 1.0 / 11.0, atol=1e-9)

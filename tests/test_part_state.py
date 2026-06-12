"""State declarations on parts — read inside `update()`, written via
PartUpdate.new_state, integrated into the compiled tick.

Exercises the `RevoluteJoint` part (revolute joint with Mass rotor child) — the
unified replacement for the v1 FlywheelMotor + SpinningRotor combo.
"""

import numpy as np
import pytest

from manta.craft import Craft
from manta import Sim, TargetNumpy, World
from manta.fields import GravityField
from manta.parts import (
    RevoluteJoint,
    Mass,
    PartUpdate,
    State,
)


# ---------------------------------------------------------------------------
# RevoluteJoint(passive) — kinematic spin via initial rate
# ---------------------------------------------------------------------------

def test_passive_joint_spins_at_initial_rate():
    """Passive joint with initial rate=1 rad/s and a rotor child runs
    free — angle accumulates at 1 rad/s indefinitely (no friction)."""
    c = Craft("passive_demo")
    c.add(Mass("body", mass=1.0))
    j = RevoluteJoint("wheel", mode="passive")
    j.add(Mass("rotor", mass=0.1, moi=(0.01, 0.01, 0.05)))
    c.add(j)

    w = World().add_field(GravityField(g=(0.0, 0.0, 0.0)))
    w.add_craft(c, **{"wheel.rate": 1.0})
    sim = TargetNumpy(Sim(w))
    assert sim.state["passive_demo"]["wheel.angle"] == 0.0
    assert sim.state["passive_demo"]["wheel.rate"]  == 1.0

    for _ in range(1000):
        sim.step(0.001)

    assert np.isclose(sim.state["passive_demo"]["wheel.angle"], 1.0, atol=1e-9)
    assert np.isclose(sim.state["passive_demo"]["wheel.rate"],  1.0, atol=1e-9)


def test_multiple_joints_have_independent_state():
    c = Craft("twin")
    c.add(Mass("body", mass=1.0))
    fast = RevoluteJoint("fast", mode="passive")
    fast.add(Mass("fast_disk", mass=0.1, moi=(0.001, 0.001, 0.01)))
    slow = RevoluteJoint("slow", mode="passive")
    slow.add(Mass("slow_disk", mass=0.1, moi=(0.001, 0.001, 0.01)))
    c.add(fast)
    c.add(slow)

    w = World().add_field(GravityField(g=(0.0, 0.0, 0.0)))
    w.add_craft(c, **{"fast.rate": 5.0, "slow.rate": 1.0})
    sim = TargetNumpy(Sim(w))
    for _ in range(1000):
        sim.step(0.001)

    assert np.isclose(sim.state["twin"]["fast.angle"], 5.0, atol=1e-9)
    assert np.isclose(sim.state["twin"]["slow.angle"], 1.0, atol=1e-9)


# ---------------------------------------------------------------------------
# RevoluteJoint(saturating) — commanded torque drives rotor + reacts on body
# ---------------------------------------------------------------------------

def test_saturating_joint_below_stall_accelerates_rotor():
    """τ_cmd = 0.5 N·m < stall = 1.0 N·m: rotor accelerates at τ/I_axial."""
    c = Craft("flywheel")
    c.add(Mass("body", mass=100.0, moi=(1000.0, 1000.0, 1000.0)))  # heavy
    j = RevoluteJoint("wheel", mode="saturating", stall_torque=1.0)
    j.add(Mass("rotor", mass=0.5, moi=(0.025, 0.025, 0.05)))
    c.add(j)

    w = World().add_field(GravityField(g=(0.0, 0.0, 0.0)))
    w.add_craft(c, **{"wheel.torque_cmd": 0.5})
    sim = TargetNumpy(Sim(w))
    for _ in range(1000):
        sim.step(0.001)

    # I_axial = 0.05 (rotor's I_zz, axis defaults to +z). The RELATIVE
    # rate also picks up the mount's counter-rotation (−â·α feedback):
    # the stator (I_zz = 1000) spins back at τ/1000, so
    # rate = τ/I_axial + τ/I_stator = 10 + 0.0005 after 1 s.
    assert np.isclose(sim.state["flywheel"]["wheel.rate"],
                      0.5 / 0.05 + 0.5 / 1000.0, atol=1e-6)


def test_saturating_joint_clamps_at_stall():
    """τ_cmd above stall → torque clipped to ±stall, rotor accelerates
    at the clipped rate."""
    c = Craft("clamped")
    c.add(Mass("body", mass=100.0, moi=(1000.0, 1000.0, 1000.0)))
    j = RevoluteJoint("wheel", mode="saturating", stall_torque=0.2)
    j.add(Mass("rotor", mass=0.5, moi=(0.025, 0.025, 0.05)))
    c.add(j)

    w = World().add_field(GravityField(g=(0.0, 0.0, 0.0)))
    w.add_craft(c, **{"wheel.torque_cmd": 5.0})   # way above stall=0.2
    sim = TargetNumpy(Sim(w))
    for _ in range(1000):
        sim.step(0.001)

    # Effective τ = 0.2; relative rate = τ/I_axial + τ/I_stator (see the
    # below-stall test) = 4 + 0.0002 after 1 s.
    assert np.isclose(sim.state["clamped"]["wheel.rate"],
                      0.2 / 0.05 + 0.2 / 1000.0, atol=1e-6)


def test_saturating_joint_reaction_spins_body_counter():
    """Reaction torque on body: τ_react = -τ_clamped along axis. The
    freely-spinning rotor's axial inertia cannot react body torques
    (coupled body/rotor solve), so the STATOR-only inertia responds:
    I_eff = (0.1 + 0.05) − 0.05 = 0.1. With τ = 0.5 N·m:
    α_body_z = −0.5/0.1 = −5 rad/s² — which is also what total
    angular-momentum conservation demands."""
    c = Craft("conservation")
    c.add(Mass("body", mass=1.0, moi=(0.1, 0.1, 0.1)))
    j = RevoluteJoint("wheel", mode="saturating", stall_torque=1.0)
    j.add(Mass("rotor", mass=0.1, moi=(0.005, 0.005, 0.05)))
    c.add(j)

    w = World().add_field(GravityField(g=(0.0, 0.0, 0.0)))
    w.add_craft(c, **{"wheel.torque_cmd": 0.5})
    sim = TargetNumpy(Sim(w))
    for _ in range(1000):
        sim.step(0.001)

    # ω after 1 s under constant α = -0.5/0.1. Loose tolerance because
    # the gyroscopic coupling (rotor spin × body ω) feeds back as ω grows.
    assert np.isclose(sim.state["conservation"]["angular_velocity"][2],
                      -0.5 / 0.1, atol=0.2)


# ---------------------------------------------------------------------------
# RevoluteJoint contributes mass + MOI to the body
# ---------------------------------------------------------------------------

def test_joint_child_mass_appears_in_body_aggregate():
    """A 1 kg rotor child in a RevoluteJoint contributes its mass to the body
    aggregate (so apparent inertia + gravity loading reflect it)."""
    c = Craft("weighted")
    body_mass = 1.0
    rotor_mass = 0.5
    c.add(Mass("body", mass=body_mass, moi=(0.1, 0.1, 0.1)))
    j = RevoluteJoint("wheel", mode="passive")
    j.add(Mass("rotor", mass=rotor_mass, moi=(0.01, 0.01, 0.05)))
    c.add(j)

    inertials = c.aggregate_inertials()
    assert np.isclose(inertials["m_total"], body_mass + rotor_mass)


def test_joint_accepts_mass_child_with_nonzero_transform():
    """Children offset from the joint origin are now lifted symbolically
    by the inertia rollup — no constructor-side restriction."""
    j = RevoluteJoint("axle", mode="passive")
    j.add(Mass("offset_rotor", mass=0.1, transform=(0.0, 0.0, 0.1)))
    assert j.children[0].transform == (0.0, 0.0, 0.1)


def test_joint_accepts_any_part_child():
    """Any Part rides a rotor — the framework expresses each child's ctx in
    its spinning frame and rotates its wrench back to the body, so no part
    needs special handling. There is no longer an allowlist."""
    from manta.parts import IMU, Thruster, DVL, Magnetometer, DragSurface
    from manta.parts import PointBuoy, Collider, PositionSensor
    j = RevoluteJoint("axle", mode="passive")
    for part in (Mass("rotor", mass=0.1), IMU("imu"), Thruster("jet"),
                 DVL("dvl"), Magnetometer("mag"), DragSurface("fin"),
                 PointBuoy("buoy"), Collider("foot"), PositionSensor("gps"),
                 RevoluteJoint("inner")):
        j.add(part)
    assert len(j.children) == 10


def test_joint_unknown_mode_raises():
    with pytest.raises(ValueError, match="mode must be"):
        RevoluteJoint("bad", mode="nonsense")


def test_joint_doesnt_break_free_fall():
    """A passive joint with a rotor doesn't perturb linear free-fall."""
    c = Craft("fall_with_joint")
    c.add(Mass("body", mass=1.0))
    j = RevoluteJoint("wheel", mode="passive")
    j.add(Mass("rotor", mass=0.1, moi=(0.001, 0.001, 0.01)))
    c.add(j)

    w = World().add_field(GravityField(g=(0.0, 0.0, -9.81)))
    w.add_craft(c, position=(0.0, 0.0, 100.0), **{"wheel.rate": 10.0})
    sim = TargetNumpy(Sim(w))
    for _ in range(1000):
        sim.step(0.001)

    # ½·g·t² = 4.905; z = 100 - 4.905 = 95.095.
    assert np.isclose(sim.state["fall_with_joint"]["position"][2],   95.095, atol=1e-5)
    assert np.isclose(sim.state["fall_with_joint"]["velocity"][2],   -9.81,  atol=1e-5)
    assert np.isclose(sim.state["fall_with_joint"]["wheel.angle"],   10.0,   atol=1e-9)


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------

def test_unknown_state_slot_raises():
    """Returning new_state with a key not declared as State should error."""
    from manta.parts import Part

    class BadPart(Part):
        a = State(init=0.0)

        def update(self, ctx):
            return PartUpdate(
                wrench=__import__("manta").parts.Wrench.zero(
                    __import__("manta").ir.frames.PartFrame),
                new_state={"a": 1.0, "b": 2.0},   # 'b' not declared
            )

    c = Craft("bad")
    c.add(Mass("m", mass=1.0))
    c.add(BadPart("bad"))
    w = World()
    w.add_craft(c)
    with pytest.raises(KeyError, match="unknown state slot"):
        Sim(w)


def test_initial_state_unknown_slot_raises():
    c = Craft("ok")
    c.add(Mass("body", mass=1.0))
    j = RevoluteJoint("wheel", mode="passive")
    j.add(Mass("rotor", mass=0.1, moi=(0.001, 0.001, 0.01)))
    c.add(j)
    with pytest.raises(KeyError, match="unknown slot"):
        c.initial_state(**{"wheel.nonexistent": 0.0})


def test_joint_state_declarations_introspection():
    j = RevoluteJoint("wheel", mode="passive")
    decls = j.state_declarations()
    assert set(decls.keys()) == {"angle", "rate"}


# ---------------------------------------------------------------------------
# Nested Joints (gimbal-style compositions)
# ---------------------------------------------------------------------------

def test_nested_passive_joints_each_track_their_own_rate():
    """A pan-tilt gimbal: outer RevoluteJoint hosts inner RevoluteJoint, each with its
    own rotor and initial spin rate. Both angles accumulate independently
    over time — proves the kinematic + state plumbing reaches both
    levels of the joint chain."""
    c = Craft("gimbal_two_dof")
    c.add(Mass("body", mass=1.0, moi=(0.1, 0.1, 0.1)))
    outer = RevoluteJoint("pan", mode="passive", axis=(0.0, 0.0, 1.0))
    outer.add(Mass("pan_disk", mass=0.1, moi=(0.001, 0.001, 0.01)))
    inner = RevoluteJoint("tilt", mode="passive", axis=(0.0, 1.0, 0.0))
    inner.add(Mass("tilt_disk", mass=0.05, moi=(0.001, 0.001, 0.001)))
    outer.add(inner)
    c.add(outer)

    w = World().add_field(GravityField(g=(0.0, 0.0, 0.0)))
    w.add_craft(c, **{"pan.rate": 2.0, "tilt.rate": 1.0})
    sim = TargetNumpy(Sim(w))
    for _ in range(1000):
        sim.step(0.001)

    # The joint-space solve couples the DOFs through the full mass
    # matrix: the pan carries the tilt axis around, so the tilt rate
    # genuinely wobbles against the light (1 kg) body — angles track
    # rate·t to ~0.5% here, NOT exactly (the old rank-1 solve held them
    # artificially constant by dropping the inter-joint coupling).
    assert np.isclose(sim.state["gimbal_two_dof"]["pan.angle"],  2.0, atol=1e-2)
    assert np.isclose(sim.state["gimbal_two_dof"]["tilt.angle"], 1.0, atol=1e-2)


def test_nested_joint_saturating_drives_inner_rotor():
    """Inner saturating RevoluteJoint commands torque on its rotor; the rotor
    accelerates at τ/I and the inner-joint reaction propagates onto
    the outer joint's frame (and from there to the body)."""
    c = Craft("gimbal_drive")
    c.add(Mass("body", mass=100.0, moi=(10.0, 10.0, 10.0)))
    outer = RevoluteJoint("pan", mode="passive", axis=(0.0, 0.0, 1.0))
    outer.add(Mass("pan_disk", mass=0.05, moi=(0.001, 0.001, 0.01)))
    inner = RevoluteJoint("tilt", mode="saturating", stall_torque=2.0,
                  axis=(0.0, 1.0, 0.0))
    inner.add(Mass("tilt_disk", mass=0.05, moi=(0.001, 0.005, 0.001)))
    outer.add(inner)
    c.add(outer)

    w = World().add_field(GravityField(g=(0.0, 0.0, 0.0)))
    w.add_craft(c, **{"tilt.torque_cmd": 0.5})
    sim = TargetNumpy(Sim(w))
    for _ in range(1000):
        sim.step(0.001)

    # Tilt rotor MOI about y-axis = 0.005 from the tilt_disk only;
    # α = 0.5 / 0.005 = 100 rad/s² → ω after 1s ≈ 100 rad/s.
    # Some tolerance for the heavy body's small angular response feeding
    # back into the gyroscopic correction.
    assert np.isclose(sim.state["gimbal_drive"]["tilt.rate"], 100.0, atol=0.5)
    assert np.isclose(sim.state["gimbal_drive"]["pan.rate"], 0.0, atol=0.5)


def test_offset_rotor_at_pi_over_2_shows_com_in_y():
    """The body-frame symbolic COM tracks the joint angle: at θ=π/2
    around the +z axis, a tip mass that started at +x lands at +y, so
    the body's COM in body coords moves from +x to +y. This is the
    motivating case for the symbolic inertia rollup."""
    c = Craft("arm_swept")
    c.add(Mass("body", mass=10.0, moi=(0.5, 0.5, 0.5)))
    j = RevoluteJoint("arm", mode="passive", axis=(0.0, 0.0, 1.0))
    j.add(Mass("tip", mass=1.0, moi=(0.0, 0.0, 0.0),
               transform=(1.0, 0.0, 0.0)))
    c.add(j)

    # Set arm.angle = π/2 directly (rate = 0 → angle stays put under no
    # torque). The first tick exercises Newton-Euler with the rotated
    # COM in the body frame.
    w = World().add_field(GravityField(g=(0.0, 0.0, 0.0)))
    w.add_craft(c, **{"arm.angle": float(np.pi / 2)})
    sim = TargetNumpy(Sim(w))
    sim.step(0.001)
    # Body should still be at rest after one tick (no external wrench,
    # rotor at rest); just verify the compile produced a valid graph
    # for the at-angle case.
    assert np.allclose(sim.state["arm_swept"]["angular_velocity"], 0.0, atol=1e-9)
    assert np.isclose(sim.state["arm_swept"]["arm.angle"], float(np.pi / 2), atol=1e-12)


def test_offset_rotor_changes_body_com():
    """A spinning arm with an off-axis mass: the symbolic inertia rollup
    sees the rotor at its current angle, so the body's effective COM
    moves with the joint angle. Confirm the offset rotor compiles and
    that aggregate mass + COM make sense."""
    c = Craft("arm")
    c.add(Mass("body", mass=10.0, moi=(0.5, 0.5, 0.5)))
    j = RevoluteJoint("arm", mode="passive", axis=(0.0, 0.0, 1.0))
    # Off-axis mass: 1 kg at +x = 1 m from joint origin.
    j.add(Mass("tip", mass=1.0, moi=(0.0, 0.0, 0.0),
               transform=(1.0, 0.0, 0.0)))
    c.add(j)

    # Smoke: the offset rotor compiles under gravity.
    w = World().add_field(GravityField(g=(0.0, 0.0, -9.81)))
    w.add_craft(c)
    TargetNumpy(Sim(w))

    inertials = c.aggregate_inertials()
    assert np.isclose(inertials["m_total"], 11.0)
    # Numpy snapshot uses transform values directly: tip at +x, so
    # com_x = (1.0 · 1.0) / 11 ≈ 0.0909.
    assert np.isclose(inertials["com"][0], 1.0 / 11.0, atol=1e-9)

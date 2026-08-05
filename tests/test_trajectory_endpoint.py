"""TrajectoryEndpoint — SE(3) spring-damper that slews a craft along a
prescribed pose trajectory (the "playback" actuator)."""

import numpy as np
import pytest

from manta import Craft, Sim, TargetNumpy, World
from manta.fields import GravityField
from manta.ir.frames import PartFrame, WorldFrame
from manta.ir.types import Quat, Vec3
from manta.parts import (
    LinearTrajectory, Mass, TrajectoryEndpoint, TrajectorySample, hover,
)


def _run(craft, world, *, dt=0.01, n=400):
    sim = TargetNumpy(Sim(world))
    for _ in range(n):
        sim.step(dt)
    return sim


def _ghost(traj, *, mass=1.0, **gains):
    c = Craft("ghost")
    c.add(Mass("body", mass=mass, moi=(0.2, 0.2, 0.2)))
    c.add(TrajectoryEndpoint("slew", trajectory=traj, mass=mass, **gains))
    return c


# ---------------------------------------------------------------------------
# Construction guards
# ---------------------------------------------------------------------------

def test_requires_callable_trajectory():
    with pytest.raises(ValueError, match="trajectory"):
        TrajectoryEndpoint("slew", trajectory=None)


def test_rejects_negative_mass():
    with pytest.raises(ValueError, match="mass"):
        TrajectoryEndpoint("slew", trajectory=hover((0, 0, 0)), mass=-1.0)


# ---------------------------------------------------------------------------
# Position tracking
# ---------------------------------------------------------------------------

def test_constant_velocity_tracks_exactly_under_gravity():
    # Gravity feedforward (mass set) ⇒ the spring only mops up residual,
    # so a constant-velocity reference is tracked to machine-ish precision.
    M = 1.0
    traj = LinearTrajectory(start=(0, 0, 10), velocity=(2.0, -1.0, 0.0),
                            orientation=(1, 0, 0, 0))
    c = _ghost(traj, mass=M, kp_pos=60, kd_pos=16, kp_att=12, kd_att=6)
    w = World().add_field(GravityField(g=(0, 0, -9.81)))
    w.add_craft(c, position=(0, 0, 10))
    sim = _run(c, w, dt=0.01, n=400)

    t = 4.0
    p = np.asarray(sim.state["ghost"]["position"]).ravel()
    ref = np.array([2.0 * t, -1.0 * t, 10.0])
    assert np.linalg.norm(p - ref) < 1e-3


def test_hover_holds_position():
    M = 2.0
    c = _ghost(hover((1.0, -2.0, 5.0)), mass=M,
               kp_pos=80, kd_pos=24)
    w = World().add_field(GravityField(g=(0, 0, -9.81)))
    # Start displaced; the spring should pull it to the hover point.
    w.add_craft(c, position=(0, 0, 0))
    sim = _run(c, w, dt=0.01, n=800)
    p = np.asarray(sim.state["ghost"]["position"]).ravel()
    assert np.linalg.norm(p - np.array([1.0, -2.0, 5.0])) < 1e-2


def test_pure_spring_droops_under_gravity_without_feedforward():
    # mass=0 ⇒ no gravity feedforward ⇒ steady-state droop F = k·Δz = m·g.
    M = 1.0
    c = Craft("ghost")
    c.add(Mass("body", mass=M))
    c.add(TrajectoryEndpoint("slew", trajectory=hover((0, 0, 10)),
                             mass=0.0, kp_pos=50.0, kd_pos=20.0))
    w = World().add_field(GravityField(g=(0, 0, -9.81)))
    w.add_craft(c, position=(0, 0, 10))
    sim = _run(c, w, dt=0.005, n=2000)
    p = np.asarray(sim.state["ghost"]["position"]).ravel()
    droop = 10.0 - p[2]
    # F = k·droop balances m·g ⇒ droop ≈ m·g/k = 9.81/50 ≈ 0.196 m.
    assert droop == pytest.approx(9.81 / 50.0, rel=0.1)


# ---------------------------------------------------------------------------
# Attitude tracking
# ---------------------------------------------------------------------------

def test_slews_to_reference_attitude():
    # Reference attitude: 90° about world z. Start at identity; the
    # attitude spring should drive the body there.
    ang = np.pi / 2
    q_ref = (np.cos(ang / 2), 0.0, 0.0, np.sin(ang / 2))

    def traj(t):
        return TrajectorySample(
            position=Vec3[WorldFrame].constant((0.0, 0.0, 0.0)),
            orientation=Quat[WorldFrame, PartFrame].constant(q_ref))

    M = 1.0
    c = _ghost(traj, mass=M, kp_pos=40, kd_pos=14, kp_att=20, kd_att=9)
    w = World().add_field(GravityField(g=(0, 0, 0)))
    w.add_craft(c, position=(0, 0, 0))
    sim = _run(c, w, dt=0.005, n=1200)
    q = np.asarray(sim.state["ghost"]["orientation"]).ravel()
    # Align sign (q and -q are the same rotation) and compare.
    if q[0] < 0:
        q = -q
    assert np.allclose(q, np.array(q_ref), atol=1e-2)


# ---------------------------------------------------------------------------
# `mass` is a feedforward GAIN, not inertial mass
# ---------------------------------------------------------------------------

def test_endpoint_mass_does_not_double_count_inertia():
    # Documented usage: Mass("body", mass=M) + TrajectoryEndpoint(mass=M).
    # The endpoint's `mass` is a feedforward gain — it must NOT join the
    # inertia walks (m_total = 2M would halve every acceleration).
    M = 2.0
    c = _ghost(hover((0, 0, 0)), mass=M)
    assert c.total_mass == M
    assert c.aggregate_inertials()["m_total"] == M


def test_endpoint_spring_acceleration_uses_true_mass():
    # Zero gravity, pure position spring: a = kp·err / m_total. With the
    # endpoint's gain double-counted (m_total = 2M) the first-step
    # velocity would come out at half this.
    M, kp, dt = 2.0, 10.0, 1e-3
    c = Craft("ghost")
    c.add(Mass("body", mass=M))
    c.add(TrajectoryEndpoint("slew", trajectory=hover((0.0, 0.0, 0.0)),
                             mass=M, kp_pos=kp, kd_pos=0.0,
                             kp_att=0.0, kd_att=0.0))
    w = World()
    w.add_craft(c, position=(1.0, 0.0, 0.0))
    sim = TargetNumpy(Sim(w))
    sim.step(dt)
    v = np.asarray(sim.state["ghost"]["velocity"]).ravel()
    # err = (0,0,0) − (1,0,0); a_x = −kp/M; one Euler step ⇒ v_x = a_x·dt.
    assert v[0] == pytest.approx(-kp / M * dt, rel=1e-9)


def test_nested_endpoint_rejected_at_compile():
    from manta.parts import RevoluteJoint
    c = Craft("ghost")
    c.add(Mass("body", mass=1.0, moi=(0.1, 0.1, 0.1)))
    j = c.add(RevoluteJoint("gim", axis=(0.0, 0.0, 1.0)))
    j.add(Mass("rotor", mass=0.1, moi=(0.01, 0.01, 0.01),
               mount_offset=(0.1, 0.0, 0.0)))
    j.add(TrajectoryEndpoint("slew", trajectory=hover((0.0, 0.0, 0.0))))
    w = World()
    w.add_craft(c)
    with pytest.raises(ValueError, match="craft root"):
        TargetNumpy(Sim(w))

"""Wrench-cascade equivalence gate.

The bottom-up cascade must (a) reduce to the old rotate-lift-sum for a
craft with no joints, and (b) for an articulated craft, absorb the axial
component of the subtree torque into the joint DOF rather than leaking it
to the body. These guard the cascade before pendulum/Foucault work.
"""

import numpy as np

from manta import Sim, TargetNumpy, World
from manta.craft import Craft
from manta.fields import GravityField
from manta.parts import Mass, Joint, Thruster


# ---------------------------------------------------------------------------
# Flat craft: an offset force produces the expected body torque (the cascade
# lift reduces to r × F when there are no joints).
# ---------------------------------------------------------------------------

def test_flat_craft_offset_thruster_torque():
    # Thruster mounted at +x offset, firing along +z → torque about −y.
    c = Craft("flat")
    c.add(Mass("body", mass=1.0, moi=(1.0, 1.0, 1.0)))
    c.add(Thruster("t", force=(0.0, 0.0, 1.0), transform=(0.5, 0.0, 0.0)))
    w = World().add_field(GravityField(g=(0.0, 0.0, 0.0)))
    w.add_craft(c)
    rt = TargetNumpy(Sim(w))
    rt.state["flat"]["t.throttle"] = 10.0           # 10 N at x=0.5 → τ_y = −5 N·m
    dt = 1e-4
    rt.step(dt)
    omega = np.asarray(rt.state["flat"]["angular_velocity"])
    # I_yy = 1 → α_y = τ_y/I = −5 rad/s²; ω_y after dt ≈ −5·dt.
    assert np.isclose(omega[1], -5.0 * dt, rtol=1e-3)
    assert abs(omega[0]) < 1e-9 and abs(omega[2]) < 1e-9


# ---------------------------------------------------------------------------
# Articulated craft: the swing torque drives the DOF, not the body.
# ---------------------------------------------------------------------------

def test_no_spurious_wrench_on_deflected_passive_joint():
    # With no gravity and no motion, a deflected passive joint feels zero
    # axial torque, so the DOF must not accelerate and the body must not
    # rotate. Any drift would mean the cascade injected a spurious wrench.
    L = 0.5
    c = Craft("pend")
    c.add(Mass("body", mass=10.0, moi=(1.0, 1.0, 1.0)))
    h = Joint("hinge", mode="passive", axis=(0.0, 1.0, 0.0))
    h.add(Mass("bob", mass=1.0, moi=(1e-9, 1e-9, 1e-9), transform=(0.0, 0.0, -L)))
    c.add(h)
    w = World().add_field(GravityField(g=(0.0, 0.0, 0.0)))   # NO gravity
    w.add_craft(c)
    rt = TargetNumpy(Sim(w))
    rt.state["pend"]["hinge.angle"] = 0.6
    for _ in range(2000):
        rt.step(1e-3)
    assert np.isclose(rt.state["pend"]["hinge.angle"], 0.6, atol=1e-9)
    assert np.isclose(rt.state["pend"]["hinge.rate"], 0.0, atol=1e-9)
    assert np.allclose(np.asarray(rt.state["pend"]["angular_velocity"]), 0.0,
                       atol=1e-9)


def test_free_body_recoils_from_swinging_bob():
    # A light free body (only 10× the bob) is NOT a fixed pivot: as the bob
    # swings, the body must recoil so the system's angular momentum about
    # its COM evolves only under gravity. Here we just confirm the recoil
    # exists and the DOF still swings — i.e. the cascade couples the bob and
    # body both ways (this is physical, not a leak).
    L = 0.5
    c = Craft("pend")
    c.add(Mass("body", mass=10.0, moi=(1.0, 1.0, 1.0)))
    h = Joint("hinge", mode="passive", axis=(0.0, 1.0, 0.0))
    h.add(Mass("bob", mass=1.0, moi=(1e-9, 1e-9, 1e-9), transform=(0.0, 0.0, -L)))
    c.add(h)
    w = World().add_field(GravityField(g=(0.0, 0.0, -9.81)))
    w.add_craft(c)
    rt = TargetNumpy(Sim(w))
    rt.state["pend"]["hinge.angle"] = 0.6
    swung = body_recoil = 0.0
    for _ in range(1000):
        rt.step(1e-3)
        swung = max(swung, abs(rt.state["pend"]["hinge.angle"] - 0.6))
        body_recoil = max(body_recoil,
                          abs(np.asarray(rt.state["pend"]["angular_velocity"])[1]))
    assert swung > 0.3            # the DOF swings
    assert body_recoil > 1e-3     # the free body recoils about the joint axis

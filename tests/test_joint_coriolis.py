"""Coriolis joint torque — base rotation coupling through the rotor inertia.

A RevoluteJoint's `rate` dynamics pick up τ_corio = −[ω_rotor × (I_joint·ω_rotor)]·axis
with ω_rotor = ω_mount + rate·axis. It is identically zero for an axisymmetric
rotor whose COM lies on the joint axis, and nonzero when a tumbling base drives
an asymmetric rotor. These tests pin both, with an analytic first-step value.
"""

import numpy as np

from manta import Craft, Sim, TargetNumpy, World
from manta.fields import GravityField
from manta.parts import Mass, RevoluteJoint


def _single_step_rate(moi, omega0, *, dt=0.01) -> float:
    """One world step of a heavy tumbling hub carrying a passive z-axis
    RevoluteJoint with one rotor Mass. Returns the joint rate after the
    step. The hub pins the base motion (ω ≈ const over the step) so the
    analytic fixed-base Coriolis value is exact to O(I_rotor/I_hub) —
    without it the stator is massless and its frame orientation is a
    gauge, not physics."""
    c = Craft("tumbler")
    c.add(Mass("hub", mass=1e6, moi=(1e6, 1e6, 1e6)))
    j = RevoluteJoint("rotor", mode="passive", axis=(0.0, 0.0, 1.0))
    j.add(Mass("disk", mass=1.0, moi=moi))
    c.add(j)
    w = World().add_field(GravityField().add_uniform((0.0, 0.0, 0.0)))
    w.add_craft(c, angular_velocity=omega0)
    sim = TargetNumpy(Sim(w))
    state = sim.step(dt=dt)
    return float(np.asarray(state["tumbler"]["rotor.rate"]).ravel()[0])


def test_coriolis_zero_for_axisymmetric_rotor():
    """Ix == Iy ⇒ the axial component of ω × (I·ω) vanishes for any base
    tumble, so the passive joint stays put."""
    rate = _single_step_rate(moi=(2.0, 2.0, 1.0), omega0=(1.0, 1.0, 0.0))
    assert abs(rate) < 1e-14


def test_coriolis_drives_asymmetric_rotor():
    """axis = ẑ, I_joint = diag(1, 3, 2), ω_mount = (1, 1, 0), rate₀ = 0:

        ω × (I·ω) = (1,1,0) × (1,3,0) = (0, 0, 2)  →  ·ẑ = 2
        τ_corio   = −2
        accel     = τ_corio / I_axial(=I_zz=2) = −1
        rate(dt)  = rate₀ + accel·dt = −dt
    """
    dt = 0.01
    rate = _single_step_rate(moi=(1.0, 3.0, 2.0), omega0=(1.0, 1.0, 0.0), dt=dt)
    assert np.isclose(rate, -dt, atol=1e-9)


def test_coriolis_sign_flips_with_inertia_asymmetry():
    """Swapping Ix and Iy flips the sign of the axial Coriolis torque."""
    dt = 0.01
    pos = _single_step_rate(moi=(1.0, 3.0, 2.0), omega0=(1.0, 1.0, 0.0), dt=dt)
    neg = _single_step_rate(moi=(3.0, 1.0, 2.0), omega0=(1.0, 1.0, 0.0), dt=dt)
    assert np.isclose(pos, -neg, atol=1e-12)
    assert pos < 0.0 < neg

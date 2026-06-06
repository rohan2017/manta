"""Passive revolute joint as a gravity-driven pendulum.

The wrench cascade feeds the axial component of the subtree's external
torque into the joint DOF, so a passive joint now swings under gravity:
θ̈ = (r × m g)·axis / I_axial = −(g/L)·sinθ for a point bob of length L.
"""

import numpy as np

from manta import Sim, TargetNumpy, World
from manta.craft import Craft
from manta.fields import GravityField
from manta.parts import Mass, RevoluteJoint


G = 9.81


def _pendulum(L: float, m: float = 1.0, damping: float = 0.0) -> Craft:
    """Heavy fixed anchor + a passive hinge about +y with a point bob
    hanging at (0, 0, −L). At angle 0 the bob hangs straight down; a
    positive angle swings it toward −x. The anchor is heavy enough that
    the body is effectively an inertial pivot (so I_axial = m L²)."""
    c = Craft("pend")
    c.add(Mass("anchor", mass=1.0e6, moi=(1.0e6, 1.0e6, 1.0e6)))
    h = RevoluteJoint("hinge", mode="passive", axis=(0.0, 1.0, 0.0), damping=damping)
    h.add(Mass("bob", mass=m, moi=(1e-9, 1e-9, 1e-9), transform=(0.0, 0.0, -L)))
    c.add(h)
    return c


def _sim(craft, **init):
    w = World().add_field(GravityField(g=(0.0, 0.0, -G)))
    w.add_craft(craft)
    rt = TargetNumpy(Sim(w))
    for k, v in init.items():
        rt.state["pend"][k] = v
    return rt


# ---------------------------------------------------------------------------
# The DOF acceleration is the analytic pendulum torque / I_axial
# ---------------------------------------------------------------------------

def test_initial_acceleration_matches_g_over_L_sin_theta():
    L = 0.8
    theta0 = 0.4
    rt = _sim(_pendulum(L), **{"hinge.angle": theta0, "hinge.rate": 0.0})
    dt = 1e-5
    rt.step(dt)
    # rate started at 0, so θ̈ ≈ rate_after / dt.
    theta_ddot = rt.state["pend"]["hinge.rate"] / dt
    expected = -(G / L) * np.sin(theta0)
    assert np.isclose(theta_ddot, expected, rtol=1e-3), (
        f"θ̈={theta_ddot:.5f}, expected {expected:.5f}")


# ---------------------------------------------------------------------------
# Small-angle period ≈ 2π√(L/g)
# ---------------------------------------------------------------------------

def test_small_angle_period():
    L = 0.8
    rt = _sim(_pendulum(L), **{"hinge.angle": 0.05})
    dt = 1e-3
    angles, ts = [], []
    t = 0.0
    for _ in range(6000):          # 6 s, several periods
        rt.step(dt); t += dt
        angles.append(rt.state["pend"]["hinge.angle"]); ts.append(t)
    angles = np.array(angles)
    # Period from successive upward zero-crossings.
    sign = np.sign(angles)
    ups = [i for i in range(1, len(sign)) if sign[i - 1] < 0 <= sign[i]]
    periods = np.diff(np.array(ups) * dt)
    measured = float(np.mean(periods))
    expected = 2 * np.pi * np.sqrt(L / G)
    assert np.isclose(measured, expected, rtol=2e-2), (
        f"period {measured:.4f}s vs {expected:.4f}s")


# ---------------------------------------------------------------------------
# Frictionless swing conserves energy (bounded, symplectic)
# ---------------------------------------------------------------------------

def test_frictionless_energy_converges_with_dt():
    # The joint uses the same "symplectic-flavored" explicit integrator as
    # the body, so a frictionless swing has a small first-order energy
    # drift that vanishes as dt → 0 (no spurious energy source in the
    # model). Confirm the drift is small at a fine dt and roughly halves
    # when dt halves.
    L = 0.8; m = 1.0
    I_ax = m * L * L
    theta0 = 0.6
    E0 = m * G * L * (1.0 - np.cos(theta0))

    def drift(dt: float, T: float = 10.0) -> float:
        rt = _sim(_pendulum(L, m), **{"hinge.angle": theta0})
        for _ in range(int(T / dt)):
            rt.step(dt)
        th = rt.state["pend"]["hinge.angle"]; rate = rt.state["pend"]["hinge.rate"]
        E = 0.5 * I_ax * rate * rate + m * G * L * (1.0 - np.cos(th))
        return abs(E - E0) / E0

    d_coarse = drift(2e-4)
    d_fine   = drift(1e-4)
    assert d_fine < 1e-2, f"energy drift {d_fine:.2e} too large at dt=1e-4"
    # First-order convergence: halving dt roughly halves the drift.
    assert 1.6 < d_coarse / d_fine < 2.4, (
        f"non-linear convergence: {d_coarse:.2e} vs {d_fine:.2e}")


# ---------------------------------------------------------------------------
# Viscous joint damping decays the amplitude
# ---------------------------------------------------------------------------

def test_damping_decays_amplitude():
    L = 0.8; m = 1.0
    rt = _sim(_pendulum(L, m, damping=0.5), **{"hinge.angle": 0.5})
    dt = 1e-3
    amp_early, amp_late = 0.0, 0.0
    for i in range(8000):          # 8 s
        rt.step(dt)
        a = abs(rt.state["pend"]["hinge.angle"])
        if i < 2000:
            amp_early = max(amp_early, a)
        if i > 6000:
            amp_late = max(amp_late, a)
    assert amp_late < 0.5 * amp_early, (
        f"amplitude did not decay: early={amp_early:.3f}, late={amp_late:.3f}")

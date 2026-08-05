"""Double pendulum — the headline exactness test for the joint-space solve.

Two nested passive revolute joints under gravity, against an
INDEPENDENT textbook reference: the classic point-mass double-pendulum
equations of motion (absolute angles, m1/m2/L1/L2) integrated with a
hand-rolled RK4. The trajectories only agree if the off-diagonal
joint-space mass-matrix coupling M₁₂(q) is right — a per-joint solve
(each DOF dividing its own axial torque by its own inertia) diverges
within a fraction of a swing.

The manta craft hangs from a heavy anchor hovering on a support
thruster (free fall would, correctly, freeze the pendulum), so the
fixed-pivot reference applies to O(m_bob/m_anchor).
"""

import numpy as np

from manta import Craft, Sim, TargetNumpy, World
from manta.fields import GravityField
from manta.parts import Mass, RevoluteJoint, Thruster


G = 9.81
ANCHOR_M = 1.0e6


def _double_pendulum(L1, L2, m1, m2):
    c = Craft("dp")
    c.add(Mass("anchor", mass=ANCHOR_M, moi=(1e6, 1e6, 1e6)))
    c.add(Thruster("hold",
                   force=(0.0, 0.0, (ANCHOR_M + m1 + m2) * G)))
    h1 = RevoluteJoint("h1", mode="passive", axis=(0.0, 1.0, 0.0))
    h1.add(Mass("bob1", mass=m1, moi=(1e-9, 1e-9, 1e-9),
                mount_offset=(0.0, 0.0, -L1)))
    h2 = RevoluteJoint("h2", mode="passive", axis=(0.0, 1.0, 0.0),
                       mount_offset=(0.0, 0.0, -L1))
    h2.add(Mass("bob2", mass=m2, moi=(1e-9, 1e-9, 1e-9),
                mount_offset=(0.0, 0.0, -L2)))
    h1.add(h2)
    c.add(h1)
    return c


def _reference_rk4(phi1, phi2, w1, w2, *, L1, L2, m1, m2, dt, n):
    """Canonical double-pendulum EoM in ABSOLUTE angles φ (from
    vertical; the standard myphysicslab form), RK4-integrated. Returns
    the (φ1, φ2) trajectory array."""

    def deriv(y):
        t1, t2, om1, om2 = y
        d = t1 - t2
        den = 2 * m1 + m2 - m2 * np.cos(2 * t1 - 2 * t2)
        a1 = (-G * (2 * m1 + m2) * np.sin(t1)
              - m2 * G * np.sin(t1 - 2 * t2)
              - 2 * np.sin(d) * m2 * (om2 * om2 * L2
                                      + om1 * om1 * L1 * np.cos(d))
              ) / (L1 * den)
        a2 = (2 * np.sin(d) * (om1 * om1 * L1 * (m1 + m2)
                               + G * (m1 + m2) * np.cos(t1)
                               + om2 * om2 * L2 * m2 * np.cos(d))
              ) / (L2 * den)
        return np.array([om1, om2, a1, a2])

    y = np.array([phi1, phi2, w1, w2], dtype=float)
    out = np.empty((n, 2))
    for i in range(n):
        k1 = deriv(y)
        k2 = deriv(y + 0.5 * dt * k1)
        k3 = deriv(y + 0.5 * dt * k2)
        k4 = deriv(y + dt * k3)
        y = y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        out[i] = y[0:2]
    return out


def test_double_pendulum_first_step_matches_textbook():
    """The instantaneous joint accelerations at a deflected pose match
    the textbook EoM to the m_bob/m_anchor ≈ 1e-6 fixed-pivot tier —
    this is the direct probe of the off-diagonal mass-matrix coupling
    (a per-joint solve is off by O(10%) here)."""
    L1, L2, m1, m2 = 0.7, 0.5, 1.0, 0.6
    th1_0, th2_0 = 0.5, 0.3
    c = _double_pendulum(L1, L2, m1, m2)
    w = World().add_field(GravityField(g=(0.0, 0.0, -G)))
    w.add_craft(c)
    rt = TargetNumpy(Sim(w))
    rt.state["dp"]["hold.throttle"] = 1.0
    rt.state["dp"]["h1.angle"] = th1_0
    rt.state["dp"]["h2.angle"] = th2_0
    dt = 1e-6
    rt.step(dt)
    a1 = rt.state["dp"]["h1.rate"] / dt
    a2 = rt.state["dp"]["h2.rate"] / dt

    p1, p2 = th1_0, th1_0 + th2_0
    d = p2 - p1
    sd, cd = np.sin(d), np.cos(d)
    den = m1 + m2 * sd * sd
    A1 = (m2 * G * np.sin(p2) * cd
          - (m1 + m2) * G * np.sin(p1)) / (L1 * den)
    A2 = ((m1 + m2) * (G * np.sin(p1) * cd
                       - G * np.sin(p2))) / (L2 * den)
    np.testing.assert_allclose(a1, A1, rtol=1e-5)
    np.testing.assert_allclose(a2, A2 - A1, rtol=1e-5)


def test_double_pendulum_converges_to_textbook_rk4():
    """Trajectory vs the RK4 reference: manta's integrator is first
    order, so the gap must shrink ~linearly with dt — which can only
    happen if the underlying continuous dynamics are the same. (A
    coupling error would leave an O(1) dt-independent residual.)"""
    L1, L2, m1, m2 = 0.7, 0.5, 1.0, 0.6
    th1_0, th2_0 = 0.4, 0.2
    T = 1.0

    def manta_traj(dt):
        c = _double_pendulum(L1, L2, m1, m2)
        w = World().add_field(GravityField(g=(0.0, 0.0, -G)))
        w.add_craft(c)
        rt = TargetNumpy(Sim(w))
        rt.state["dp"]["hold.throttle"] = 1.0
        rt.state["dp"]["h1.angle"] = th1_0
        rt.state["dp"]["h2.angle"] = th2_0
        n = int(T / dt)
        for _ in range(n):
            rt.step(dt)
        return np.array([rt.state["dp"]["h1.angle"],
                         rt.state["dp"]["h2.angle"]])

    def err(dt):
        ref = _reference_rk4(th1_0, th1_0 + th2_0, 0.0, 0.0,
                             L1=L1, L2=L2, m1=m1, m2=m2,
                             dt=dt, n=int(T / dt))
        ref_end = np.array([ref[-1, 0], ref[-1, 1] - ref[-1, 0]])
        return np.abs(manta_traj(dt) - ref_end).max()

    e_coarse = err(2e-4)
    e_fine   = err(1e-4)
    assert e_fine < 2e-2, f"endpoint error {e_fine:.2e} rad at dt=1e-4"
    assert 1.6 < e_coarse / e_fine < 2.4, (
        f"non-linear dt-convergence: {e_coarse:.2e} vs {e_fine:.2e} — "
        f"suggests a dynamics (not integrator) discrepancy")


def test_double_pendulum_conserves_energy_undamped():
    """Frictionless double pendulum: total mechanical energy drifts only
    at the integrator's O(dt) — small at dt = 1e-4 over 2 s."""
    L1, L2, m1, m2 = 0.7, 0.5, 1.0, 0.6
    c = _double_pendulum(L1, L2, m1, m2)
    w = World().add_field(GravityField(g=(0.0, 0.0, -G)))
    w.add_craft(c)
    rt = TargetNumpy(Sim(w))
    rt.state["dp"]["hold.throttle"] = 1.0
    rt.state["dp"]["h1.angle"] = 0.8
    rt.state["dp"]["h2.angle"] = -0.4

    def energy():
        th1 = float(rt.state["dp"]["h1.angle"])
        th2 = float(rt.state["dp"]["h2.angle"])
        w1 = float(rt.state["dp"]["h1.rate"])
        w2 = float(rt.state["dp"]["h2.rate"])
        p1, p2 = th1, th1 + th2          # absolute angles
        om1, om2 = w1, w1 + w2
        v1 = L1 * om1
        v2x = L1 * om1 * np.cos(p1) + L2 * om2 * np.cos(p2)
        v2z = L1 * om1 * np.sin(p1) + L2 * om2 * np.sin(p2)
        ke = 0.5 * m1 * v1 * v1 + 0.5 * m2 * (v2x * v2x + v2z * v2z)
        pe = (-m1 * G * L1 * np.cos(p1)
              - m2 * G * (L1 * np.cos(p1) + L2 * np.cos(p2)))
        return ke + pe

    E0 = energy()
    dt = 1e-4
    for _ in range(int(2.0 / dt)):
        rt.step(dt)
    drift = abs(energy() - E0) / (m1 + m2) / G / (L1 + L2)
    assert drift < 2e-3, f"energy drift {drift:.2e}"

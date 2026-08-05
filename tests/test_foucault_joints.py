"""Foucault precession through a stacked 2-axis gimbal — the coupling
the old per-joint solve fundamentally could not represent.

A bob hangs from two stacked revolute joints (x-axis then y-axis — a
ball-pivot built from 1-DOF joints) on a heavy hub that hovers on a
thruster while spinning at Ω about the vertical. Released deflected, the
bob swings in an inertially-fixed plane; in the rotating body frame that
plane precesses at −Ω (the pole-mounted Foucault pendulum). The
precession is driven entirely by the inter-joint / body Coriolis
coupling in the joint-space mass matrix — the legacy rank-1 solve
dropped it (which is why the Foucault demo needed a Tether).
"""

import numpy as np

from manta import Craft, Sim, TargetNumpy, World
from manta.fields import GravityField
from manta.parts import Mass, RevoluteJoint, Thruster


G = 9.81


def test_gimbal_pendulum_precesses_at_minus_omega():
    L = 0.5
    OMEGA = 0.3          # hub spin about +z (the "planet")
    HUB_M = 1.0e6
    bob_m = 0.2

    c = Craft("rig")
    c.add(Mass("hub", mass=HUB_M, moi=(1e6, 1e6, 1e6)))
    c.add(Thruster("hold", force=(0.0, 0.0, (HUB_M + bob_m) * G)))
    j1 = RevoluteJoint("swing_x", mode="passive", axis=(1.0, 0.0, 0.0))
    j2 = RevoluteJoint("swing_y", mode="passive", axis=(0.0, 1.0, 0.0))
    j2.add(Mass("bob", mass=bob_m, moi=(1e-9, 1e-9, 1e-9),
                mount_offset=(0.0, 0.0, -L)))
    j1.add(j2)
    c.add(j1)

    w = World().add_field(GravityField(g=(0.0, 0.0, -G)))
    w.add_craft(c, angular_velocity=(0.0, 0.0, OMEGA))
    rt = TargetNumpy(Sim(w))
    rt.state["rig"]["hold.throttle"] = 1.0
    rt.state["rig"]["swing_y.angle"] = 0.05      # released deflected
    # Co-rotating release: the bob starts at rest in the SPINNING frame
    # (a real Foucault bob is released at rest on the planet), which the
    # joint state already encodes (joint rates are relative to the hub).

    dt = 2e-4
    T = 6.0
    xs, ys = [], []
    for i in range(int(T / dt)):
        rt.step(dt)
        if i % 10 == 9:
            th1 = float(rt.state["rig"]["swing_x.angle"])
            th2 = float(rt.state["rig"]["swing_y.angle"])
            # Small-angle bob offset in the body frame ≈ L·(−θ2, θ1).
            xs.append(-th2)
            ys.append(th1)
    xs = np.array(xs)
    ys = np.array(ys)

    # Swing-plane azimuth per window via PCA: the trajectory is a slowly
    # precessing near-line (ellipticity ~ Ω/ω_p), so the principal axis
    # of each ~half-swing-period window IS the plane. Unwrap mod π and
    # fit the drift.
    n_win = 16
    win = len(xs) // n_win
    t_mid, phi = [], []
    for k in range(n_win):
        sl = slice(k * win, (k + 1) * win)
        cov = np.cov(np.vstack([xs[sl], ys[sl]]))
        evals, evecs = np.linalg.eigh(cov)
        v = evecs[:, -1]                      # dominant axis
        phi.append(np.arctan2(v[1], v[0]))
        t_mid.append((k + 0.5) * win * 10 * dt)
    # Unwrap mod π (a plane is a line: φ and φ+π are the same plane).
    phi = np.array(phi)
    for k in range(1, len(phi)):
        while phi[k] - phi[k - 1] > np.pi / 2:
            phi[k] -= np.pi
        while phi[k] - phi[k - 1] < -np.pi / 2:
            phi[k] += np.pi
    slope = np.polyfit(np.array(t_mid), phi, 1)[0]
    assert np.isclose(slope, -OMEGA, rtol=0.05), (
        f"precession {slope:.4f} rad/s, expected {-OMEGA:.4f}")


def test_no_precession_without_hub_spin():
    """Control: Ω = 0 keeps the swing plane fixed."""
    L = 0.5
    HUB_M = 1.0e6
    bob_m = 0.2
    c = Craft("rig")
    c.add(Mass("hub", mass=HUB_M, moi=(1e6, 1e6, 1e6)))
    c.add(Thruster("hold", force=(0.0, 0.0, (HUB_M + bob_m) * G)))
    j1 = RevoluteJoint("swing_x", mode="passive", axis=(1.0, 0.0, 0.0))
    j2 = RevoluteJoint("swing_y", mode="passive", axis=(0.0, 1.0, 0.0))
    j2.add(Mass("bob", mass=bob_m, moi=(1e-9, 1e-9, 1e-9),
                mount_offset=(0.0, 0.0, -L)))
    j1.add(j2)
    c.add(j1)
    w = World().add_field(GravityField(g=(0.0, 0.0, -G)))
    w.add_craft(c)
    rt = TargetNumpy(Sim(w))
    rt.state["rig"]["hold.throttle"] = 1.0
    rt.state["rig"]["swing_y.angle"] = 0.05
    worst = 0.0
    for i in range(int(2.0 / 2e-4)):
        rt.step(2e-4)
        worst = max(worst, abs(rt.state["rig"]["swing_x.angle"]))
    assert worst < 1e-6, f"swing leaked into the x-joint: {worst:.2e}"

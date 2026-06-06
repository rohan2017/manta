"""Free-floating 2-axis gimbal — energy + momentum conservation.

A torque-free craft carrying a stacked pan(z)–tilt(y) gimbal with
asymmetric payloads, everything spinning. The world-frame angular
momentum and the total kinetic energy — both computed INDEPENDENTLY in
this test from the state via the gimbal chain kinematics — must be
conserved. This pins the joint-space mass matrix A(q), the Hamel bias,
and the generalized momentum integrator all at once: an error in any
off-diagonal coupling term shows up as a secular drift.

All parts sit at zero offsets, so the craft COM is fixed and the
angular chain is the whole story (the hand-computed L has no orbital
terms to get wrong).
"""

import numpy as np

from manta import Craft, Sim, TargetNumpy, World
from manta.fields import GravityField
from manta.parts import Mass, RevoluteJoint


I_BODY = np.diag([0.8, 1.1, 0.9])
I_PAN  = np.diag([0.020, 0.060, 0.045])    # asymmetric plate on the pan
I_TILT = np.diag([0.010, 0.004, 0.013])    # asymmetric rod on the tilt


def _rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _ry(b):
    c, s = np.cos(b), np.sin(b)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _quat_to_R(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def _build():
    c = Craft("gim")
    c.add(Mass("body", mass=5.0, moi=tuple(np.diag(I_BODY))))
    pan = RevoluteJoint("pan", mode="passive", axis=(0.0, 0.0, 1.0))
    pan.add(Mass("plate", mass=0.4, moi=tuple(np.diag(I_PAN))))
    tilt = RevoluteJoint("tilt", mode="passive", axis=(0.0, 1.0, 0.0))
    tilt.add(Mass("rod", mass=0.2, moi=tuple(np.diag(I_TILT))))
    pan.add(tilt)
    c.add(pan)
    return c


def _L_and_KE(state):
    """World-frame angular momentum + kinetic energy from the state,
    via independent (numpy) gimbal-chain kinematics."""
    om = np.asarray(state["angular_velocity"], dtype=float).ravel()
    q  = np.asarray(state["orientation"], dtype=float).ravel()
    a, adot = float(state["pan.angle"]), float(state["pan.rate"])
    b, bdot = float(state["tilt.angle"]), float(state["tilt.rate"])

    R1 = _rz(a)                      # craft ← pan output
    R2 = R1 @ _ry(b)                 # craft ← tilt output
    w_pan  = om + adot * np.array([0.0, 0.0, 1.0])
    w_tilt = w_pan + bdot * (R1 @ np.array([0.0, 1.0, 0.0]))

    I1c = R1 @ I_PAN @ R1.T
    I2c = R2 @ I_TILT @ R2.T
    L_craft = I_BODY @ om + I1c @ w_pan + I2c @ w_tilt
    KE = 0.5 * (om @ I_BODY @ om + w_pan @ I1c @ w_pan
                + w_tilt @ I2c @ w_tilt)
    R_body = _quat_to_R(q)
    return R_body @ L_craft, KE


def test_gimbal_conserves_momentum_and_energy():
    c = _build()
    w = World().add_field(GravityField().add_uniform((0.0, 0.0, 0.0)))
    w.add_craft(c, angular_velocity=(0.4, -0.3, 0.6),
                **{"pan.rate": 2.5, "tilt.rate": -1.8,
                   "pan.angle": 0.3, "tilt.angle": -0.5})
    rt = TargetNumpy(Sim(w))

    L0, E0 = _L_and_KE(rt.state["gim"])
    dt = 1e-4
    worst_L = worst_E = 0.0
    for i in range(int(2.0 / dt)):
        rt.step(dt)
        if i % 1000 == 999:
            L, E = _L_and_KE(rt.state["gim"])
            worst_L = max(worst_L, np.linalg.norm(L - L0) / np.linalg.norm(L0))
            worst_E = max(worst_E, abs(E - E0) / E0)
    assert worst_L < 1e-4, f"angular-momentum drift {worst_L:.2e}"
    assert worst_E < 1e-3, f"energy drift {worst_E:.2e}"


def test_gimbal_momentum_drift_converges_with_dt():
    """Halving dt should at least halve the drifts (first-order
    integrator over exact dynamics — a dynamics error would leave a
    dt-independent floor)."""
    def drifts(dt, T=0.5):
        c = _build()
        w = World().add_field(GravityField().add_uniform((0.0, 0.0, 0.0)))
        w.add_craft(c, angular_velocity=(0.4, -0.3, 0.6),
                    **{"pan.rate": 2.5, "tilt.rate": -1.8})
        rt = TargetNumpy(Sim(w))
        L0, E0 = _L_and_KE(rt.state["gim"])
        for _ in range(int(T / dt)):
            rt.step(dt)
        L, E = _L_and_KE(rt.state["gim"])
        return (np.linalg.norm(L - L0) / np.linalg.norm(L0),
                abs(E - E0) / E0)

    dL2, dE2 = drifts(2e-4)
    dL1, dE1 = drifts(1e-4)
    assert dL1 < 0.6 * dL2 + 1e-12, f"L drift not converging: {dL2:.2e}→{dL1:.2e}"
    assert dE1 < 0.6 * dE2 + 1e-12, f"E drift not converging: {dE2:.2e}→{dE1:.2e}"

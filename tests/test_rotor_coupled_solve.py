"""Coupled body/rotor solve — regression for the α_mount energy pump.

The old fixed-base RevoluteJoint approximation (a) solved the body α with the
FULL inertia aggregate (a freely-spinning rotor's axial inertia cannot
react body torques), and (b) omitted the −â·α mount feedback on the
rotor's relative rate. On a fast rotor with a precessing mount the two
errors compound into exponential energy growth (|ω| ×~30 per 0.1 s in
the spinning-top configuration). The coupled solve fixes both, plus the
Coriolis joint torque now reacts on the mount (axial-momentum
consistency for asymmetric rotors).

Also pins the rigid-body state conventions these dynamics rely on:
`angular_velocity` is CraftFrame (body rates); `velocity`/`position`
are the craft origin in WorldFrame.
"""

import numpy as np

from manta import Craft, Sim, TargetNumpy, World
from manta.fields import GravityField
from manta.parts import RevoluteJoint, Mass


def _R(q):
    w, x, y, z = q
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-w*z),   2*(x*z+w*y)],
        [2*(x*y+w*z),   1-2*(x*x+z*z), 2*(y*z-w*x)],
        [2*(x*z-w*y),   2*(y*z+w*x),   1-2*(x*x+y*y)]])


def test_fast_rotor_on_precessing_mount_no_energy_pump():
    """Spinning-top rig, torque-free: light pole, heavy disk on a passive
    z-RevoluteJoint at the top, body tilted and tumbling slightly, rotor at
    150 rad/s. Total world-frame angular momentum must be conserved and
    the body rates bounded. (The old model blew |ω| up ×1000 within a
    second here.)"""
    c = Craft("top")
    c.add(Mass("pole", mass=0.02, moi=(2e-4, 2e-4, 1e-5),
               mount_offset=(0.0, 0.0, 0.15)))
    rotor = RevoluteJoint("rotor", mode="passive", axis=(0.0, 0.0, 1.0),
                  mount_offset=(0.0, 0.0, 0.3))
    rotor.add(Mass("disk", mass=0.4, moi=(0.002, 0.002, 0.004)))
    c.add(rotor)

    w = World().add_field(GravityField(g=(0.0, 0.0, 0.0)))
    tilt = 0.3
    w.add_craft(c, orientation=(np.cos(tilt/2), np.sin(tilt/2), 0.0, 0.0),
                angular_velocity=(0.3, 0.2, 0.1),
                **{"rotor.rate": 150.0})
    sim = TargetNumpy(Sim(w))

    # Aggregate inertia about the craft COM (pole + disk, diagonal).
    z_com = (0.02 * 0.15 + 0.4 * 0.3) / 0.42
    I_xx = (2e-4 + 0.02 * (0.15 - z_com) ** 2
            + 0.002 + 0.4 * (0.3 - z_com) ** 2)
    I_b = np.diag([I_xx, I_xx, 1e-5 + 0.004])
    I_ax = 0.004

    def L_world():
        st = sim.state["top"]
        q = np.asarray(st["orientation"]).ravel()
        om = np.asarray(st["angular_velocity"]).ravel()   # body frame
        s = float(st["rotor.rate"])
        L_body = I_b @ om + I_ax * s * np.array([0.0, 0.0, 1.0])
        return _R(q) @ L_body

    L0 = L_world()
    max_om = 0.0
    for _ in range(8000):                                  # 2 s
        sim.step(2.5e-4)
        max_om = max(max_om, float(np.linalg.norm(
            np.asarray(sim.state["top"]["angular_velocity"]).ravel())))

    np.testing.assert_allclose(L_world(), L0, atol=2e-4,
                               err_msg="angular momentum not conserved")
    assert max_om < 5.0, f"body rates blew up: |ω|max = {max_om:.1f}"


def test_reaction_wheel_momentum_exact():
    """Spin a wheel up with the actuator on a free body: the total axial
    angular momentum I_stator·ω + I_a·(ω + s) stays exactly zero."""
    c = Craft("rw")
    c.add(Mass("body", mass=1.0, moi=(0.1, 0.1, 0.1)))
    j = RevoluteJoint("wheel", mode="saturating", stall_torque=1.0)
    j.add(Mass("disk", mass=0.1, moi=(0.005, 0.005, 0.05)))
    c.add(j)
    w = World().add_field(GravityField(g=(0.0, 0.0, 0.0)))
    w.add_craft(c, **{"wheel.torque_cmd": 0.5})
    sim = TargetNumpy(Sim(w))
    for _ in range(1000):
        sim.step(0.001)
    st = sim.state["rw"]
    om_z = float(np.asarray(st["angular_velocity"]).ravel()[2])
    s = float(st["wheel.rate"])
    # I_total·ω + I_a·s — the wheel's relative spin vs the whole craft.
    L_z = (0.1 + 0.05) * om_z + 0.05 * s
    np.testing.assert_allclose(L_z, 0.0, atol=1e-9)


def test_angular_velocity_state_is_body_frame():
    """Convention pin: seed a 90°-about-x-tilted body with
    angular_velocity (0, 0, w). The physical rotation must be about the
    BODY z-axis (= world −y): the body x-axis swings toward world +z."""
    c = Craft("b")
    c.add(Mass("m", mass=1.0, moi=(0.2, 0.3, 0.4)))
    w = World()
    w.add_craft(c, orientation=(np.cos(np.pi/4), np.sin(np.pi/4), 0, 0),
                angular_velocity=(0.0, 0.0, 1.0))
    sim = TargetNumpy(Sim(w))
    for _ in range(500):
        sim.step(0.001)                       # 0.5 rad of rotation
    R = _R(np.asarray(sim.state["b"]["orientation"]).ravel())
    np.testing.assert_allclose(
        R[:, 0], [np.cos(0.5), 0.0, np.sin(0.5)], atol=1e-6)

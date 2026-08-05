"""Moving-COM correction — linear momentum of a free-floating articulated craft.

When a joint actuates, the craft COM translates within the body. The body
origin must recoil so the *system* COM obeys Newton's first law: with no
external force the system COM moves in a straight line at constant velocity
(here nonzero, because the spinning off-axis rim carries linear momentum
p = m·ω×r, so v_com = p/m_total). Without the moving-COM term in the origin
transfer, the origin doesn't recoil and the system COM wobbles off that line
by the rotor's circular amplitude (~m·d/(M+m)).
"""

import numpy as np

from manta import Craft, Sim, TargetNumpy, World
from manta.fields import GravityField
from manta.parts import RevoluteJoint, Mass


def _quat_to_R(q) -> np.ndarray:
    """Rotation matrix from a (w, x, y, z) quaternion."""
    w, x, y, z = (float(v) for v in np.asarray(q).ravel())
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y)],
        [2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y)],
    ])


def test_moving_com_keeps_free_floating_system_com_fixed():
    M, m, d = 100.0, 1.0, 1.0          # heavy bus, light off-axis flywheel

    c = Craft("sat")
    c.add(Mass("bus", mass=M, moi=(10.0, 10.0, 10.0)))
    wheel = RevoluteJoint("wheel", mode="passive", axis=(0.0, 0.0, 1.0))
    # Rim mass offset d along x → COM swings on a circle as the wheel turns.
    wheel.add(Mass("rim", mass=m, moi=(0.01, 0.01, 0.01), mount_offset=(d, 0.0, 0.0)))
    c.add(wheel)

    w = World().add_field(GravityField().add_uniform((0.0, 0.0, 0.0)))  # free
    w.add_craft(c, **{"wheel.rate": 10.0})                              # spin it
    sim = TargetNumpy(Sim(w))

    def system_com(st) -> np.ndarray:
        theta = float(np.asarray(st["sat"]["wheel.angle"]).ravel()[0])
        com_body = np.array([m * d * np.cos(theta),
                             m * d * np.sin(theta), 0.0]) / (M + m)
        R   = _quat_to_R(st["sat"]["orientation"])
        pos = np.asarray(st["sat"]["position"]).ravel()
        return pos + R @ com_body

    # System COM at rest in body but moving in world: the rim (offset d,
    # spinning at θ̇₀) carries p = m·(θ̇₀·ẑ)×(d,0,0) = (0, m·d·θ̇₀, 0), so the
    # system COM drifts at v_com = p / (M + m) along +y.
    theta_dot0 = 10.0
    com0  = system_com(sim.state)
    v_com = np.array([0.0, m * d * theta_dot0 / (M + m), 0.0])

    dt = 0.001
    drift = 0.0
    for step in range(1, 501):
        state = sim.step(dt=dt)
        predicted = com0 + v_com * (step * dt)        # straight line, const v
        drift = max(drift, float(np.linalg.norm(system_com(state) - predicted)))

    # Uncorrected, the COM wobbles off the line by ~m·d/(M+m) ≈ 0.0099 m.
    uncorrected = m * d / (M + m)
    assert drift < uncorrected / 20.0

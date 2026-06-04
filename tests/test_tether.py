"""Tether coupling — multi-craft component dynamics."""

import numpy as np

from manta import Craft, Sim, TargetNumpy, World
from manta.couplings import Tether
from manta.parts import DragSurface, Mass, TetherEndpoint, Thruster


def _make_craft(name: str, mass: float = 1.0) -> Craft:
    c = Craft(name)
    c.add(Mass("body", mass=mass, moi=(0.1, 0.1, 0.1)))
    c.add(TetherEndpoint("hook"))
    return c


def test_at_rest_length_zero_force():
    """Two crafts exactly L_rest apart with zero velocity: no force, no
    motion."""
    L = 5.0
    a = _make_craft("a")
    b = _make_craft("b")
    w = World()    # no gravity
    w.add_craft(a, position=(0, 0, 0))
    w.add_craft(b, position=(L, 0, 0))
    w.add_coupling(Tether(a, "hook", b, "hook",
                          stiffness=10.0, damping=1.0, rest_length=L))
    cw = TargetNumpy(Sim(w))

    for _ in range(500):
        cw.step(0.001)

    pa = np.array(cw.state["a"]["position"]).ravel()
    pb = np.array(cw.state["b"]["position"]).ravel()
    np.testing.assert_allclose(pa, np.zeros(3), atol=1e-9)
    np.testing.assert_allclose(pb, (L, 0, 0), atol=1e-9)


def test_stretched_spring_pulls_crafts_together():
    """Start crafts farther apart than rest length → spring pulls them in."""
    L_rest = 5.0
    L_init = 7.0
    a = _make_craft("a")
    b = _make_craft("b")
    w = World()
    w.add_craft(a, position=(0, 0, 0))
    w.add_craft(b, position=(L_init, 0, 0))
    w.add_coupling(Tether(a, "hook", b, "hook",
                          stiffness=10.0, damping=5.0, rest_length=L_rest))
    cw = TargetNumpy(Sim(w))

    for _ in range(2000):
        cw.step(0.001)

    pa = np.array(cw.state["a"]["position"]).ravel()
    pb = np.array(cw.state["b"]["position"]).ravel()
    dist = np.linalg.norm(pb - pa)
    # With damping the system settles to rest length.
    assert np.isclose(dist, L_rest, atol=0.05)


def test_compressed_spring_pushes_crafts_apart():
    """Start crafts closer than rest length → spring pushes them apart."""
    L_rest = 5.0
    L_init = 3.0
    a = _make_craft("a")
    b = _make_craft("b")
    w = World()
    w.add_craft(a, position=(0, 0, 0))
    w.add_craft(b, position=(L_init, 0, 0))
    w.add_coupling(Tether(a, "hook", b, "hook",
                          stiffness=10.0, damping=5.0, rest_length=L_rest))
    cw = TargetNumpy(Sim(w))

    for _ in range(2000):
        cw.step(0.001)

    pa = np.array(cw.state["a"]["position"]).ravel()
    pb = np.array(cw.state["b"]["position"]).ravel()
    dist = np.linalg.norm(pb - pa)
    assert np.isclose(dist, L_rest, atol=0.05)


def test_momentum_conservation_no_external_forces():
    """Two crafts coupled by a tether, no external forces → total
    momentum stays at its initial value forever."""
    a = _make_craft("a", mass=1.0)
    b = _make_craft("b", mass=2.0)
    w = World()
    w.add_craft(a, position=(0, 0, 0), velocity=(1.0, 0, 0))
    w.add_craft(b, position=(7.0, 0, 0))
    w.add_coupling(Tether(a, "hook", b, "hook",
                          stiffness=100.0, damping=0.0, rest_length=5.0))
    cw = TargetNumpy(Sim(w))

    # Initial momentum
    va0 = np.array(cw.state["a"]["velocity"]).ravel()
    vb0 = np.array(cw.state["b"]["velocity"]).ravel()
    p0 = 1.0 * va0 + 2.0 * vb0   # m_a·v_a + m_b·v_b

    for _ in range(2000):
        cw.step(0.001)

    va = np.array(cw.state["a"]["velocity"]).ravel()
    vb = np.array(cw.state["b"]["velocity"]).ravel()
    p_final = 1.0 * va + 2.0 * vb
    np.testing.assert_allclose(p_final, p0, atol=1e-3)


def test_damping_dissipates_relative_motion():
    """With damping, oscillation amplitude shrinks over time."""
    a = _make_craft("a")
    b = _make_craft("b")
    w = World()
    w.add_craft(a, position=(0, 0, 0))
    w.add_craft(b, position=(8.0, 0, 0))
    w.add_coupling(Tether(a, "hook", b, "hook",
                          stiffness=50.0, damping=10.0, rest_length=5.0))
    cw = TargetNumpy(Sim(w))

    # Run with damping.
    energies = []
    for i in range(3000):
        cw.step(0.001)
        if i % 200 == 0:
            va = np.array(cw.state["a"]["velocity"]).ravel()
            vb = np.array(cw.state["b"]["velocity"]).ravel()
            pa = np.array(cw.state["a"]["position"]).ravel()
            pb = np.array(cw.state["b"]["position"]).ravel()
            ke = 0.5 * (np.dot(va, va) + np.dot(vb, vb))
            pe = 0.5 * 50.0 * (np.linalg.norm(pb - pa) - 5.0) ** 2
            energies.append(ke + pe)

    # Energy should be monotonically decreasing (with some tolerance for
    # discretization noise).
    assert energies[-1] < 0.1 * energies[0]


def test_offset_endpoint_produces_torque():
    """A tether attached at a body-frame offset transmits torque to the
    craft when stretched."""
    a = Craft("a")
    a.add(Mass("body", mass=1.0, moi=(0.05, 0.05, 0.05)))
    # Endpoint at +y offset → stretch along +x produces torque about z.
    a.add(TetherEndpoint("hook", transform=(0.0, 0.5, 0.0)))

    b = Craft("b")
    b.add(Mass("body", mass=100.0, moi=(1.0, 1.0, 1.0)))   # heavy → stays put
    b.add(TetherEndpoint("hook"))

    w = World()
    w.add_craft(a, position=(0, 0, 0))
    w.add_craft(b, position=(5.0, 0, 0))
    w.add_coupling(Tether(a, "hook", b, "hook",
                          stiffness=10.0, damping=0.0, rest_length=2.0))
    cw = TargetNumpy(Sim(w))

    cw.step(0.001)

    omega = np.array(cw.state["a"]["angular_velocity"]).ravel()
    # The tether pulls A's hook (at +y, +x relative to body center) toward B
    # (which is at +x in anchor). The force pulls A in +x direction, applied
    # at +y offset → torque about -z. Sign depends on convention.
    assert abs(omega[2]) > 1e-6
    # x and y components small (no torque about those axes by symmetry).
    assert abs(omega[0]) < 1e-9
    assert abs(omega[1]) < 1e-9

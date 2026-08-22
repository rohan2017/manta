"""Tether coupling — multi-craft component dynamics."""

import numpy as np
import pytest

from manta import Craft, Sim, TargetNumpy, World
from manta.couplings import Tether
from manta.fields import GravityField
from manta.parts import Mass, TetherEndpoint


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
    w = World().add_field(GravityField.none())    # no gravity
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
    """Start crafts farther apart than rest length → the taut tether
    pulls them in; once inside rest length it goes slack, so the final
    separation is at most rest length (never pushed back out)."""
    L_rest = 5.0
    L_init = 7.0
    a = _make_craft("a")
    b = _make_craft("b")
    w = World().add_field(GravityField.none())
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
    # The pull happened, and the slack rope never pushed them back apart.
    assert dist < L_init - 0.5
    assert dist <= L_rest + 0.05


def test_slack_tether_exerts_no_force():
    """Start crafts closer than rest length → the tether is slack and
    exerts exactly zero force; nothing moves."""
    L_rest = 5.0
    L_init = 3.0
    a = _make_craft("a")
    b = _make_craft("b")
    w = World().add_field(GravityField.none())
    w.add_craft(a, position=(0, 0, 0))
    w.add_craft(b, position=(L_init, 0, 0))
    w.add_coupling(Tether(a, "hook", b, "hook",
                          stiffness=10.0, damping=5.0, rest_length=L_rest))
    cw = TargetNumpy(Sim(w))

    for _ in range(2000):
        cw.step(0.001)

    pa = np.array(cw.state["a"]["position"]).ravel()
    pb = np.array(cw.state["b"]["position"]).ravel()
    np.testing.assert_allclose(pa, np.zeros(3), atol=1e-12)
    np.testing.assert_allclose(pb, (L_init, 0, 0), atol=1e-12)


def test_damper_cannot_push_while_taut():
    """Barely taut with B closing fast: the raw spring+damper sum is
    strongly compressive, but a rope can't push — the tension clamp
    holds the force at ~zero instead of shoving A away from B."""
    a = _make_craft("a")
    b = _make_craft("b")
    w = World().add_field(GravityField.none())
    w.add_craft(a, position=(0, 0, 0))
    # Stretch = 0.1 m (spring +1 N), closing at 10 m/s (damper −50 N).
    w.add_craft(b, position=(5.1, 0, 0), velocity=(-10.0, 0, 0))
    w.add_coupling(Tether(a, "hook", b, "hook",
                          stiffness=10.0, damping=5.0, rest_length=5.0))
    cw = TargetNumpy(Sim(w))

    cw.step(0.001)

    va = np.array(cw.state["a"]["velocity"]).ravel()
    # A pushed away from B would move in −x. Old rigid-spring model:
    # Δv_a ≈ −49 N · 1 ms = −0.049 m/s. Clamped: essentially nothing.
    assert va[0] > -1e-6
    assert abs(va[0]) < 1e-3


def test_momentum_conservation_no_external_forces():
    """Two crafts coupled by a tether, no external forces → total
    momentum stays at its initial value forever."""
    a = _make_craft("a", mass=1.0)
    b = _make_craft("b", mass=2.0)
    w = World().add_field(GravityField.none())
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
    w = World().add_field(GravityField.none())
    w.add_craft(a, position=(0, 0, 0))
    w.add_craft(b, position=(8.0, 0, 0))
    w.add_coupling(Tether(a, "hook", b, "hook",
                          stiffness=50.0, damping=10.0, rest_length=5.0))
    cw = TargetNumpy(Sim(w))

    # Run with damping. Elastic PE only accrues while taut (slack rope
    # stores nothing).
    energies = []
    for i in range(3000):
        cw.step(0.001)
        if i % 200 == 0:
            va = np.array(cw.state["a"]["velocity"]).ravel()
            vb = np.array(cw.state["b"]["velocity"]).ravel()
            pa = np.array(cw.state["a"]["position"]).ravel()
            pb = np.array(cw.state["b"]["position"]).ravel()
            ke = 0.5 * (np.dot(va, va) + np.dot(vb, vb))
            stretch = max(0.0, np.linalg.norm(pb - pa) - 5.0)
            pe = 0.5 * 50.0 * stretch ** 2
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
    a.add(TetherEndpoint("hook", mount_offset=(0.0, 0.5, 0.0)))

    b = Craft("b")
    b.add(Mass("body", mass=100.0, moi=(1.0, 1.0, 1.0)))   # heavy → stays put
    b.add(TetherEndpoint("hook"))

    w = World().add_field(GravityField.none())
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


def test_tether_rejects_nested_endpoint():
    """An endpoint under a joint would get a silently wrong lever arm —
    the Tether constructor rejects it."""
    from manta.parts import RevoluteJoint

    a = Craft("a")
    a.add(Mass("body", mass=1.0, moi=(0.1, 0.1, 0.1)))
    j = a.add(RevoluteJoint("gim", axis=(0.0, 0.0, 1.0)))
    j.add(TetherEndpoint("hook"))
    b = _make_craft("b")
    import pytest
    with pytest.raises(ValueError, match="root"):
        Tether(a, "hook", b, "hook", stiffness=10.0)


def test_tether_endpoint_lookup_ignores_non_endpoints():
    """A same-named non-TetherEndpoint part must not satisfy the lookup."""
    a = Craft("a")
    a.add(Mass("hook", mass=1.0))      # decoy — not a TetherEndpoint
    b = _make_craft("b")
    with pytest.raises(ValueError, match="no TetherEndpoint"):
        Tether(a, "hook", b, "hook", stiffness=10.0)


@pytest.mark.parametrize(
    ("argument", "value"),
    (
        ("stiffness", -1.0),
        ("damping", -1.0),
        ("rest_length", -1.0),
        ("slack_smoothing", -1.0),
        ("stiffness", float("nan")),
        ("damping", float("inf")),
    ),
)
def test_tether_rejects_nonphysical_configuration(argument, value):
    a = _make_craft("a")
    b = _make_craft("b")
    kwargs = {"stiffness": 1.0, argument: value}
    with pytest.raises(ValueError, match=argument):
        Tether(a, "hook", b, "hook", **kwargs)

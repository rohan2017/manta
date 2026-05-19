"""World + Anchor + Coupling — top-level container tests."""

import numpy as np
import pytest

from manta_next import World, Anchor, Coupling, Craft, CompiledWorld
from manta_next.parts import Mass, SpinningRotor


# ---------------------------------------------------------------------------
# Anchor
# ---------------------------------------------------------------------------

def test_anchor_basic():
    a = Anchor("lab_floor")
    assert a.name == "lab_floor"
    assert a.gravity is None
    assert "lab_floor" in repr(a)


def test_anchor_gravity_override():
    moon = Anchor("moon_surface", gravity=(0.0, 0.0, -1.62))
    assert moon.gravity == (0.0, 0.0, -1.62)


# ---------------------------------------------------------------------------
# Single-craft world matches direct Craft.compile_tick
# ---------------------------------------------------------------------------

def test_single_craft_world_matches_direct_compile():
    """Compiling through World should produce identical numerics to
    directly compiling the craft."""
    c1 = Craft("solo_a")
    c1.add(Mass("body", mass=1.0))
    direct_tick = c1.compile_tick(gravity_anchor=(0.0, 0.0, -9.81))

    c2 = Craft("solo_b")
    c2.add(Mass("body", mass=1.0))
    w = World().set_gravity((0.0, 0.0, -9.81))
    w.add_craft(c2, position=(0.0, 0.0, 100.0))
    cw = w.compile()

    # Direct numerics.
    state_direct = c1.initial_state(position=(0.0, 0.0, 100.0))
    for _ in range(100):
        out = direct_tick(dt=0.01, **state_direct)
        state_direct = {k: out[k] for k in state_direct}

    # World numerics.
    state_world = cw.initial_state()
    for _ in range(100):
        state_world = cw.step(state_world, dt=0.01)

    assert np.allclose(state_world["solo_b"]["position"],
                       state_direct["position"], atol=1e-12)
    assert np.allclose(state_world["solo_b"]["velocity"],
                       state_direct["velocity"], atol=1e-12)


# ---------------------------------------------------------------------------
# Multi-craft world: independent free-falls
# ---------------------------------------------------------------------------

def test_two_crafts_fall_independently():
    """Two crafts at different starting positions with different masses,
    both fall under gravity. State stays separate per craft."""
    w = World().set_gravity((0.0, 0.0, -9.81))

    a = Craft("alice")
    a.add(Mass("body", mass=1.0))
    w.add_craft(a, position=(0.0, 0.0, 50.0))

    b = Craft("bob")
    b.add(Mass("body", mass=5.0))
    w.add_craft(b, position=(10.0, 0.0, 100.0))

    cw = w.compile()
    assert set(cw.components) == {"alice", "bob"}

    state = cw.initial_state()
    assert state["alice"]["position"][2] == 50.0
    assert state["bob"]["position"][2]   == 100.0

    for _ in range(100):
        state = cw.step(state, dt=0.01)

    # Both fall by 4.905 m in 1 s.
    assert np.isclose(state["alice"]["position"][2], 50.0  - 4.905, atol=1e-6)
    assert np.isclose(state["bob"]["position"][2],   100.0 - 4.905, atol=1e-6)
    # Alice's x stays at 0; Bob's at 10.
    assert np.allclose(state["alice"]["position"][:2], [0.0, 0.0])
    assert np.allclose(state["bob"]["position"][:2],   [10.0, 0.0])


# ---------------------------------------------------------------------------
# Per-anchor gravity override
# ---------------------------------------------------------------------------

def test_per_anchor_gravity_override():
    """A craft on a Moon-gravity anchor falls slower than one on the
    world-default anchor."""
    w = World().set_gravity((0.0, 0.0, -9.81))

    moon = Anchor("moon", gravity=(0.0, 0.0, -1.62))
    w.add_anchor(moon)

    earth_drone = Craft("earth_drone")
    earth_drone.add(Mass("body", mass=1.0))
    w.add_craft(earth_drone, position=(0.0, 0.0, 100.0))

    moon_drone = Craft("moon_drone")
    moon_drone.add(Mass("body", mass=1.0))
    w.add_craft(moon_drone, anchor="moon", position=(0.0, 0.0, 100.0))

    cw = w.compile()
    state = cw.initial_state()
    for _ in range(100):
        state = cw.step(state, dt=0.01)

    earth_dz = 100.0 - state["earth_drone"]["position"][2]
    moon_dz  = 100.0 - state["moon_drone"]["position"][2]
    assert np.isclose(earth_dz, 4.905,  atol=1e-6)
    assert np.isclose(moon_dz,  0.810,  atol=1e-6)
    # Moon drone falls ~6x slower (g_moon / g_earth = 0.165).
    assert moon_dz / earth_dz == pytest.approx(1.62 / 9.81, rel=1e-9)


# ---------------------------------------------------------------------------
# Per-craft state with extra slots
# ---------------------------------------------------------------------------

def test_world_carries_part_state_through_step():
    """Part-state slots (e.g. SpinningRotor.angle) propagate through
    World.step like any other state."""
    w = World().set_gravity((0.0, 0.0, 0.0))   # zero gravity for clarity
    c = Craft("with_rotor")
    c.add(Mass("body", mass=1.0))
    c.add(SpinningRotor("wheel", spin_rate=2.0))
    w.add_craft(c, **{"wheel.angle": 0.5})       # initial angle override

    cw = w.compile()
    state = cw.initial_state()
    assert state["with_rotor"]["wheel.angle"] == 0.5

    for _ in range(500):
        state = cw.step(state, dt=0.001)

    # 0.5 + 2.0 · 0.5 s = 1.5
    assert np.isclose(state["with_rotor"]["wheel.angle"], 1.5, atol=1e-9)


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------

def test_add_craft_duplicate_name_raises():
    w = World()
    a = Craft("dup")
    a.add(Mass("body", mass=1.0))
    b = Craft("dup")
    b.add(Mass("body", mass=1.0))
    w.add_craft(a)
    with pytest.raises(ValueError, match="collides"):
        w.add_craft(b)


def test_add_craft_same_instance_twice_raises():
    w = World()
    c = Craft("once")
    c.add(Mass("body", mass=1.0))
    w.add_craft(c)
    with pytest.raises(ValueError, match="already added"):
        w.add_craft(c)


def test_add_craft_unknown_anchor_name_raises():
    w = World()
    c = Craft("c")
    c.add(Mass("body", mass=1.0))
    with pytest.raises(KeyError, match="no anchor named"):
        w.add_craft(c, anchor="missing")


def test_add_coupling_not_yet_supported():
    w = World()
    a = Craft("a"); a.add(Mass("body", mass=1.0))
    b = Craft("b"); b.add(Mass("body", mass=1.0))
    w.add_craft(a); w.add_craft(b)

    class FakeCoupling(Coupling):
        def __init__(self, ca, cb):
            self._a = ca; self._b = cb
        @property
        def craft_a(self): return self._a
        @property
        def craft_b(self): return self._b

    with pytest.raises(NotImplementedError, match="Coupling"):
        w.add_coupling(FakeCoupling(a, b))


def test_compile_succeeds_with_zero_couplings():
    """Just sanity that the component algorithm works for the no-coupling
    case (degenerate connected-component graph)."""
    w = World()
    w.add_craft(_make_craft("a"))
    w.add_craft(_make_craft("b"))
    w.add_craft(_make_craft("c"))
    cw = w.compile()
    assert set(cw.components) == {"a", "b", "c"}


def _make_craft(name: str) -> Craft:
    c = Craft(name)
    c.add(Mass("body", mass=1.0))
    return c


# ---------------------------------------------------------------------------
# CompiledWorld accessors
# ---------------------------------------------------------------------------

def test_compiled_world_exposes_per_craft_tick():
    w = World()
    c = Craft("solo"); c.add(Mass("body", mass=1.0))
    w.add_craft(c)
    cw = w.compile()
    tick = cw.tick("solo")
    assert tick is not None
    # Direct tick call should work.
    state = cw.initial_state()["solo"]
    out = tick(dt=0.01, **state)
    assert "position" in out


def test_compiled_world_repr_lists_components():
    w = World()
    w.add_craft(_make_craft("a"))
    w.add_craft(_make_craft("b"))
    cw = w.compile()
    r = repr(cw)
    assert "a" in r and "b" in r

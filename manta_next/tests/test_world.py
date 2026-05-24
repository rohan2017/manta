"""World + Coupling — top-level container tests."""

import numpy as np
import pytest

from manta_next import World, Coupling, Craft, CompiledWorld
from manta_next.fields import GravityField
from manta_next.parts import Joint, Mass


# ---------------------------------------------------------------------------
# Single-craft world matches direct Craft.compile_tick
# ---------------------------------------------------------------------------

def test_single_craft_world_matches_direct_compile():
    """Compiling through World should produce identical numerics to
    directly compiling the craft."""
    c1 = Craft("solo_a")
    c1.add(Mass("body", mass=1.0))
    direct_tick = c1.compile_tick(gravity_field=GravityField(g=(0.0, 0.0, -9.81)))

    c2 = Craft("solo_b")
    c2.add(Mass("body", mass=1.0))
    w = World().add_field(GravityField().add_uniform((0.0, 0.0, -9.81)))
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
    w = World().add_field(GravityField().add_uniform((0.0, 0.0, -9.81)))

    a = Craft("alice")
    a.add(Mass("body", mass=1.0))
    w.add_craft(a, position=(0.0, 0.0, 50.0))

    b = Craft("bob")
    b.add(Mass("body", mass=5.0))
    w.add_craft(b, position=(10.0, 0.0, 100.0))

    cw = w.compile()
    assert {c.name for c in cw.crafts} == {"alice", "bob"}

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
# Per-craft state with extra slots
# ---------------------------------------------------------------------------

def test_world_carries_part_state_through_step():
    """Part-state slots (e.g. a passive Joint's angle/rate) propagate
    through World.step like any other state."""
    w = World().add_field(GravityField().add_uniform((0.0, 0.0, 0.0)))   # zero gravity for clarity
    c = Craft("with_rotor")
    c.add(Mass("body", mass=1.0))
    j = Joint("wheel", mode="passive")
    j.add(Mass("wheel_disk", mass=0.1, moi=(0.001, 0.001, 0.005)))
    c.add(j)
    w.add_craft(c, **{"wheel.angle": 0.5, "wheel.rate": 2.0})

    cw = w.compile()
    state = cw.initial_state()
    assert state["with_rotor"]["wheel.angle"] == 0.5
    assert state["with_rotor"]["wheel.rate"]  == 2.0

    for _ in range(500):
        state = cw.step(state, dt=0.001)

    # Free-spinning passive joint: angle = 0.5 + 2.0 · 0.5 s = 1.5; rate
    # stays at 2.0 (no friction in v1).
    assert np.isclose(state["with_rotor"]["wheel.angle"], 1.5, atol=1e-9)
    assert np.isclose(state["with_rotor"]["wheel.rate"],  2.0, atol=1e-9)


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


def test_add_coupling_rejects_unregistered_craft():
    """add_coupling requires both endpoint crafts to be registered."""
    w = World()
    a = Craft("a"); a.add(Mass("body", mass=1.0))
    b = Craft("b"); b.add(Mass("body", mass=1.0))
    w.add_craft(a)   # only `a` registered

    class FakeCoupling(Coupling):
        def __init__(self, ca, cb):
            self._a = ca; self._b = cb
        @property
        def craft_a(self): return self._a
        @property
        def craft_b(self): return self._b

    with pytest.raises(ValueError, match="not registered"):
        w.add_coupling(FakeCoupling(a, b))


def test_compile_handles_multiple_independent_crafts():
    """A world with multiple crafts and no couplings compiles to a
    single tick spanning all of them."""
    w = World()
    w.add_craft(_make_craft("a"))
    w.add_craft(_make_craft("b"))
    w.add_craft(_make_craft("c"))
    cw = w.compile()
    assert {c.name for c in cw.crafts} == {"a", "b", "c"}


def _make_craft(name: str) -> Craft:
    c = Craft(name)
    c.add(Mass("body", mass=1.0))
    return c


# ---------------------------------------------------------------------------
# CompiledWorld accessors
# ---------------------------------------------------------------------------

def test_compiled_world_exposes_world_tick():
    """`cw.tick` returns the single CompiledGraph driving the world.
    Direct invocation uses flat-prefixed casadi inputs (`<craft>.slot`)."""
    w = World()
    c = Craft("solo"); c.add(Mass("body", mass=1.0))
    w.add_craft(c)
    cw = w.compile()
    tick = cw.tick
    assert tick is not None
    # Direct tick call uses flat-prefixed names.
    state = cw.initial_state()["solo"]
    flat  = {f"solo.{k}": v for k, v in state.items()}
    out   = tick(dt=0.01, t=0.0, **flat)
    assert "solo.position" in out


def test_compiled_world_repr_lists_crafts():
    w = World()
    w.add_craft(_make_craft("a"))
    w.add_craft(_make_craft("b"))
    cw = w.compile()
    r = repr(cw)
    assert "a" in r and "b" in r

"""World + Coupling — top-level container tests."""

import numpy as np
import pytest

from manta import Coupling, Craft, Sim, TargetNumpy, World
from manta.fields import GravityField
from manta.parts import Mass, RevoluteJoint

# ---------------------------------------------------------------------------
# Part-tree name uniqueness
# ---------------------------------------------------------------------------

def test_world_requires_an_explicit_gravity_declaration():
    """Forgetting GravityField used to resolve as silent zero-g. Gravity is
    now declared — a real field, a planet, or GravityField.none()."""
    def world():
        craft = Craft("c")
        craft.add(Mass("body", mass=1.0, moi=(0.1, 0.1, 0.1)))
        w = World("gravity_contract")
        w.add_craft(craft)
        return w

    with pytest.raises(ValueError, match="declares no gravity"):
        Sim(world())
    weightless = world().add_field(GravityField.none())
    assert GravityField in {type(f) for f in weightless.fields}
    sim = TargetNumpy(Sim(weightless))
    for _ in range(5):
        sim.step(0.1)
    np.testing.assert_allclose(sim.state["c"]["velocity"], 0.0)


def test_duplicate_part_name_raises():
    """Duplicate names anywhere in one craft's tree are refused at add():
    every lookup (state keys, sensors, parameters) is by name and would
    silently take the first match."""
    c = Craft("d")
    c.add(Mass("body", mass=1.0))
    with pytest.raises(ValueError, match="unique"):
        c.add(Mass("body", mass=2.0))
    # ...including across branches: root "wheel_m" vs a joint's child.
    j = c.add(RevoluteJoint("wheel"))
    j.add(Mass("wheel_m", mass=0.1))
    with pytest.raises(ValueError, match="unique"):
        c.add(Mass("wheel_m", mass=0.2))
    # Same name on DIFFERENT crafts is fine (uniqueness is per tree).
    c2 = Craft("e")
    c2.add(Mass("body", mass=1.0))


def test_field_source_without_emits_field_raises():
    """A FieldSource subclass that forgets `emits_field` gets a named
    error during snapshot resolution — not an AttributeError from inside the error
    message that was trying to report it."""
    from manta.parts.field_source.base import FieldSource

    class Sloppy(FieldSource):
        def make_disturbance(self, craft, offset):  # pragma: no cover
            raise AssertionError("never reached")

    c = Craft("d")
    c.add(Mass("body", mass=1.0))
    c.add(Sloppy("src"))
    w = World().add_field(GravityField().add_uniform((0.0, 0.0, -9.81)))
    w.add_craft(c)
    with pytest.raises(TypeError, match="emits_field"):
        w.snapshot()


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

    sim = Sim(w)
    assert {c.name for c in sim.crafts} == {"alice", "bob"}
    cw = TargetNumpy(sim)

    assert cw.state["alice"]["position"][2] == 50.0
    assert cw.state["bob"]["position"][2]   == 100.0

    for _ in range(100):
        cw.step(0.01)

    # Both fall by 4.905 m in 1 s.
    assert np.isclose(cw.state["alice"]["position"][2], 50.0  - 4.905, atol=1e-6)
    assert np.isclose(cw.state["bob"]["position"][2],   100.0 - 4.905, atol=1e-6)
    # Alice's x stays at 0; Bob's at 10.
    assert np.allclose(cw.state["alice"]["position"][:2], [0.0, 0.0])
    assert np.allclose(cw.state["bob"]["position"][:2],   [10.0, 0.0])


def test_transform_resolves_private_snapshot_and_authoring_world_stays_editable():
    c = Craft("c")
    c.add(Mass("body", mass=1.0))
    w = World().add_field(GravityField.none())
    w.add_craft(c)
    first = Sim(w)

    from manta.parts import Thruster
    c.add(Thruster("late", force=(1.0, 0.0, 0.0)))
    second = Sim(w)

    removed = c.remove("late")
    from manta import EKF, UKF
    third = EKF(w, sensors=[])

    from manta.parts import PositionSensor
    c.add(PositionSensor("gps", position_noise_sigma=0.1))
    fourth = UKF(w)

    assert first.world is not w and second.world is not w
    assert first.world is not second.world
    assert removed.name == "late" and removed.parent is None
    assert not first.module().port("u").fields
    assert [f.name for f in second.module().port("u").fields] == [
        "c.late.throttle"]
    assert not third.module().port("u").fields
    assert fourth.model.sensor_names == ("c.gps.position",)
    assert first.model.artifact_id == third.model.artifact_id
    assert second.model.artifact_id != first.model.artifact_id
    assert fourth.model.artifact_id != third.model.artifact_id
    assert first.model.validation.valid
    assert (EKF(first.model, sensors=[]).model.artifact_id
            == first.model.artifact_id)
    assert (UKF(first.model, sensors=[]).model.artifact_id
            == first.model.artifact_id)
    reviewed = first.model.with_derivation("review", ("accepted", True))
    rebuilt = EKF(reviewed, sensors=[]).model
    assert rebuilt.model_id == first.model.model_id
    assert rebuilt.artifact_id == reviewed.artifact_id
    assert rebuilt.derivation == reviewed.derivation

    # An artifact can branch back into authoring without reusing its resolved
    # private World (where deferred hooks have already run).
    branch = first.model.world_copy()
    branch.crafts[0].add(PositionSensor("branch_gps", position_noise_sigma=0.1))
    branch_filter = EKF(branch)
    assert branch_filter.model.sensor_names == ("c.branch_gps.position",)


# ---------------------------------------------------------------------------
# Per-craft state with extra slots
# ---------------------------------------------------------------------------

def test_world_carries_part_state_through_step():
    """Part-state slots (e.g. a passive RevoluteJoint's angle/rate) propagate
    through World.step like any other state."""
    w = World().add_field(GravityField().add_uniform((0.0, 0.0, 0.0)))   # zero gravity for clarity
    c = Craft("with_rotor")
    c.add(Mass("body", mass=1.0))
    j = RevoluteJoint("wheel", mode="passive")
    j.add(Mass("wheel_disk", mass=0.1, moi=(0.001, 0.001, 0.005)))
    c.add(j)
    w.add_craft(c, **{"wheel.angle": 0.5, "wheel.rate": 2.0})

    cw = TargetNumpy(Sim(w))
    assert cw.state["with_rotor"]["wheel.angle"] == 0.5
    assert cw.state["with_rotor"]["wheel.rate"]  == 2.0

    for _ in range(500):
        cw.step(0.001)

    # Free-spinning passive joint: angle = 0.5 + 2.0 · 0.5 s = 1.5; rate
    # stays at 2.0 (no friction in v1).
    assert np.isclose(cw.state["with_rotor"]["wheel.angle"], 1.5, atol=1e-9)
    assert np.isclose(cw.state["with_rotor"]["wheel.rate"],  2.0, atol=1e-9)


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------

def test_add_craft_duplicate_name_raises():
    w = World().add_field(GravityField.none())
    a = Craft("dup")
    a.add(Mass("body", mass=1.0))
    b = Craft("dup")
    b.add(Mass("body", mass=1.0))
    w.add_craft(a)
    with pytest.raises(ValueError, match="collides"):
        w.add_craft(b)


def test_add_craft_same_instance_twice_raises():
    w = World().add_field(GravityField.none())
    c = Craft("once")
    c.add(Mass("body", mass=1.0))
    w.add_craft(c)
    with pytest.raises(ValueError, match="already added"):
        w.add_craft(c)


def test_add_coupling_rejects_unregistered_craft():
    """add_coupling requires both endpoint crafts to be registered."""
    w = World().add_field(GravityField.none())
    a = Craft("a"); a.add(Mass("body", mass=1.0))
    b = Craft("b"); b.add(Mass("body", mass=1.0))
    w.add_craft(a)   # only `a` registered

    class FakeCoupling(Coupling):
        def __init__(self, ca, cb):
            super().__init__("fake")
            self._a = ca; self._b = cb
        @property
        def craft_a(self): return self._a
        @property
        def craft_b(self): return self._b
        def compute_wrenches_sym(self, ctx_a, ctx_b):
            raise NotImplementedError   # add_coupling rejects before this

    with pytest.raises(ValueError, match="not registered"):
        w.add_coupling(FakeCoupling(a, b))


def test_coupling_identity_and_ownership_are_unique():
    from manta.couplings import Tether
    from manta.parts import TetherEndpoint

    def tether_craft(name):
        craft = Craft(name)
        craft.add(Mass("body", mass=1.0))
        craft.add(TetherEndpoint("hook"))
        return craft

    a, b = tether_craft("a"), tether_craft("b")
    first = World("first").add_field(GravityField.none())
    first.add_craft(a)
    first.add_craft(b)
    coupling = Tether(a, "hook", b, "hook", name="tow", stiffness=1.0)
    first.add_coupling(coupling)

    with pytest.raises(ValueError, match="already added"):
        first.add_coupling(coupling)
    with pytest.raises(ValueError, match="collides"):
        first.add_coupling(
            Tether(a, "hook", b, "hook", name="tow", stiffness=2.0))

    a2, b2 = tether_craft("a2"), tether_craft("b2")
    second = World("second").add_field(GravityField.none())
    second.add_craft(a2)
    second.add_craft(b2)
    foreign = Tether(a2, "hook", b2, "hook", stiffness=1.0)
    foreign._world = first
    with pytest.raises(ValueError, match="already belongs"):
        second.add_coupling(foreign)


def test_coupling_rejects_same_craft_endpoints():
    from manta.couplings import Tether
    from manta.parts import TetherEndpoint

    craft = Craft("loop")
    craft.add(Mass("body", mass=1.0))
    craft.add(TetherEndpoint("left"))
    craft.add(TetherEndpoint("right"))
    world = World().add_field(GravityField.none())
    world.add_craft(craft)
    with pytest.raises(ValueError, match="distinct crafts"):
        world.add_coupling(
            Tether(craft, "left", craft, "right", stiffness=1.0))


def test_compile_handles_multiple_independent_crafts():
    """A world with multiple crafts and no couplings compiles to a
    single tick spanning all of them."""
    w = World().add_field(GravityField.none())
    w.add_craft(_make_craft("a"))
    w.add_craft(_make_craft("b"))
    w.add_craft(_make_craft("c"))
    sim = Sim(w)
    assert {c.name for c in sim.crafts} == {"a", "b", "c"}
    TargetNumpy(sim)


def _make_craft(name: str) -> Craft:
    c = Craft(name)
    c.add(Mass("body", mass=1.0))
    return c


# ---------------------------------------------------------------------------
# Sim accessors
# ---------------------------------------------------------------------------

def test_compiled_world_exposes_world_tick():
    """`cw.tick` returns the single CompiledGraph driving the world.
    Direct invocation uses flat-prefixed casadi inputs (`<craft>.slot`)."""
    w = World().add_field(GravityField.none())
    c = Craft("solo"); c.add(Mass("body", mass=1.0))
    w.add_craft(c)
    sim = Sim(w)
    cw = TargetNumpy(sim)
    tick = sim.tick
    assert tick is not None
    # Direct tick call uses flat-prefixed names.
    state = cw.initial_state()["solo"]
    flat  = {f"solo.{k}": v for k, v in state.items()}
    out   = tick(dt=0.01, t=0.0, **flat)
    assert "solo.position" in out


def test_compiled_world_repr_lists_crafts():
    w = World().add_field(GravityField.none())
    w.add_craft(_make_craft("a"))
    w.add_craft(_make_craft("b"))
    r = repr(Sim(w))
    assert "a" in r and "b" in r

"""Per-Part R3 state slots — `State(manifold="R3", frame=...)`.

The R3 manifold for user-declared Part state is wired through
compile_world_tick, initial_state, and StateSpec.from_world.
Use case: a Part that carries its own 3-vector state internally
(e.g. an integrator accumulating a velocity error, a wind estimate
local to one Part, etc.).
"""

import casadi as ca
import numpy as np
import pytest

from manta import Craft, EKF, TargetNumpy, World
from manta.fields import GravityField
from manta.ir.frames import CraftFrame, WorldFrame
from manta.ir.types import Vec3
from manta.ir.wrench import Wrench
from manta.parts import Mass
from manta.parts.base import Output, Part, PartUpdate, State


class _R3StatePart(Part):
    """A no-op Part that carries a single R3 state and exposes it as
    an Output. Used to verify the R3-state plumbing end-to-end."""
    bias = State(init=(1.0, 2.0, 3.0), manifold="R3", frame=WorldFrame)
    bias_out = Output(shape="vec3")

    def update(self, ctx) -> PartUpdate:
        zero = Vec3[CraftFrame].constant((0.0, 0.0, 0.0))
        return PartUpdate(
            wrench=Wrench(force=zero, torque=zero),
            outputs={"bias_out": self.bias},
            # Passthrough — no state change tick-to-tick.
            new_state={"bias": self.bias},
        )


# ---------------------------------------------------------------------------
# Declaration validation
# ---------------------------------------------------------------------------

def test_state_r3_requires_length_3_init():
    with pytest.raises(ValueError, match="length-3"):
        State(init=(1.0, 2.0), manifold="R3")
    with pytest.raises(ValueError, match="3-element"):
        State(init=42, manifold="R3")


def test_state_so3_still_unsupported():
    with pytest.raises(NotImplementedError, match="SO3"):
        State(init=(1.0, 0.0, 0.0, 0.0), manifold="SO3")


# ---------------------------------------------------------------------------
# initial_state seeds the R3 slot as a 3-vector
# ---------------------------------------------------------------------------

def test_initial_state_seeds_r3_as_ndarray():
    c = Craft("c"); c.add(Mass("body", mass=1.0)); c.add(_R3StatePart("p"))
    init = c.initial_state()
    np.testing.assert_allclose(init["p.bias"], (1.0, 2.0, 3.0))


def test_initial_state_override_accepts_3_tuple():
    c = Craft("c"); c.add(Mass("body", mass=1.0)); c.add(_R3StatePart("p"))
    init = c.initial_state(**{"p.bias": (7.0, 8.0, 9.0)})
    np.testing.assert_allclose(init["p.bias"], (7.0, 8.0, 9.0))


# ---------------------------------------------------------------------------
# World + TargetNumpy: passthrough preserves the R3 state
# ---------------------------------------------------------------------------

def test_r3_state_passthrough_through_world_tick():
    c = Craft("c"); c.add(Mass("body", mass=1.0)); c.add(_R3StatePart("p"))
    w = World().add_field(GravityField(g=(0, 0, 0)))
    w.add_craft(c)
    sim = TargetNumpy(w.compile())
    state = sim.initial_state()
    state["c"]["p.bias"] = np.array([0.5, -0.5, 1.5])
    for _ in range(20):
        state = sim.step(state, dt=0.01)
    np.testing.assert_allclose(state["c"]["p.bias"],
                                (0.5, -0.5, 1.5), atol=1e-12)


def test_r3_state_appears_in_outputs():
    """The Part exposes `bias_out = bias`, so the tick's sensor output
    matches the state."""
    c = Craft("c"); c.add(Mass("body", mass=1.0)); c.add(_R3StatePart("p"))
    w = World().add_field(GravityField(g=(0, 0, 0)))
    w.add_craft(c)
    sim = TargetNumpy(w.compile())
    state = sim.initial_state()
    state["c"]["p.bias"] = np.array([4.0, 5.0, 6.0])
    state = sim.step(state, dt=0.01)
    np.testing.assert_allclose(state["c"]["p.bias_out"],
                                (4.0, 5.0, 6.0), atol=1e-12)


# ---------------------------------------------------------------------------
# StateSpec + EKF pick up the R3 slot
# ---------------------------------------------------------------------------

def test_state_spec_picks_up_r3_state_slot():
    from manta.estimation.state_spec import StateSpec
    c = Craft("c"); c.add(Mass("body", mass=1.0)); c.add(_R3StatePart("p"))
    w = World().add_field(GravityField(g=(0, 0, 0)))
    w.add_craft(c)
    spec = StateSpec.from_world(w)
    slot = spec.slot("c.p.bias")
    assert slot.dim         == 3
    assert slot.tangent_dim == 3
    assert slot.manifold    == "R3"


def test_ekf_estimates_r3_state_from_output_observation():
    """An oracle measurement of the R3 state should pull the EKF's
    estimate toward truth."""
    c = Craft("c"); c.add(Mass("body", mass=1.0)); c.add(_R3StatePart("p"))
    w = World().add_field(GravityField(g=(0, 0, 0)))
    w.add_craft(c)
    ekf = TargetNumpy(EKF(w))

    # Broad prior on the bias slot.
    n = ekf.spec.tangent_dim
    bias_slot = ekf.spec.slot("c.p.bias")
    P = np.eye(n) * 1e-6
    P[bias_slot.tangent_offset:bias_slot.tangent_offset + 3,
      bias_slot.tangent_offset:bias_slot.tangent_offset + 3] = np.eye(3) * 1.0
    ekf.reset(P=P)

    # Oracle measurement: observe bias_out directly. truth = (4, 5, 6).
    truth = np.array([4.0, 5.0, 6.0])
    from manta.estimation import measurement_slot
    h = measurement_slot(ekf.spec, "c.p.bias")
    R = np.eye(3) * 1e-4
    for _ in range(20):
        ekf.predict(dt=0.01)
        ekf.update(h, z=truth, R=R)

    est = ekf.state_dict()["c"]["p.bias"]
    np.testing.assert_allclose(est, truth, atol=1e-2)

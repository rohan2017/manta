"""State-carrying Disturbance — Phase 2 of the field-bus design.

A Disturbance can declare `State` / `Noise` channels at class scope,
just like a Part. The framework walks them at compile time and:

  * Adds State + active-RW-bias slots to `StateSpec.from_world`.
  * Rebinds the Disturbance's attributes to symbolic graph inputs
    inside `contribute_at_sym`.
  * Synthesizes a driver input for RW Noise channels and emits
    `bias_next = bias + sqrt(dt)·driver` as a state output.

These tests verify the architecture end-to-end with a small WindBias
disturbance that adds a globally-constant velocity to the FluidField.
"""

import casadi as ca
import numpy as np

from manta import Craft, EKF, Sim, TargetNumpy, World
from manta.fields import (
    Disturbance, FluidField, FluidState, GravityField,
)
from manta.ir.frames import WorldFrame
from manta.ir.types import Vec3
from manta.parts import Mass
from manta.parts.base import RandomWalkNoise


class WindBias(Disturbance):
    """A globally-constant wind velocity, evolved as a random walk."""
    field_value_shape = FluidState
    velocity = RandomWalkNoise("R3", sigma=1e-3)

    def contribute_at_sym(self, point, t):
        return FluidState(
            density=ca.MX(0.0),
            velocity=Vec3[WorldFrame].from_mx(self.velocity._mx),
        )


def _build_world():
    w = World().add_field(GravityField(g=(0.0, 0.0, -9.81)))
    ff = FluidField()
    ff.add(WindBias(name="wind"))
    w.add_field(ff)

    c = Craft("drone")
    c.add(Mass("body", mass=1.0))
    w.add_craft(c, position=(0.0, 0.0, 5.0))
    return w, c


# ---------------------------------------------------------------------------

def test_compiled_world_has_disturbance_state_slot():
    w, _ = _build_world()
    cw = TargetNumpy(Sim(w))
    init = cw.initial_state()
    assert "wind" in init
    assert "velocity"        in init["wind"]
    assert "velocity_driver" in init["wind"]
    # Bias state and driver both seeded at zero.
    np.testing.assert_allclose(init["wind"]["velocity"],        np.zeros(3))
    np.testing.assert_allclose(init["wind"]["velocity_driver"], np.zeros(3))


def test_bias_advances_by_sqrt_dt_times_driver():
    """With a constant driver each tick, the wind velocity evolves
    exactly as the documented model `next = bias + sqrt(dt)·driver`."""
    w, _ = _build_world()
    cw = TargetNumpy(Sim(w))
    state = cw.initial_state()
    drv = np.array([0.5, -0.25, 0.1])
    dt = 0.01
    expected = np.zeros(3)
    for _ in range(20):
        state["wind"]["velocity_driver"] = drv.copy()
        state = cw.step(state, dt=dt)
        expected = expected + np.sqrt(dt) * drv
    np.testing.assert_allclose(state["wind"]["velocity"], expected,
                               atol=1e-10)


def test_state_spec_picks_up_disturbance_bias():
    from manta.estimation.state_spec import StateSpec
    w, _ = _build_world()
    spec = StateSpec.from_world(w)
    names = [s.name for s in spec.slots]
    assert "wind.velocity" in names
    slot = spec.slot("wind.velocity")
    assert slot.ambient_dim              == 3
    assert slot.tangent_dim      == 3
    assert slot.manifold.kind    == "vec"


def test_ekf_has_disturbance_state_and_noise_channel():
    w, _ = _build_world()
    ekf = EKF(w)
    # 13 rigid + 3 wind.velocity = 16 ambient (12 + 3 = 15 tangent).
    assert ekf.spec.ambient_dim == 16
    assert ekf.spec.tangent_dim == 15
    # Auto-Q includes the wind driver as a process-noise channel.
    full_names = [s.full for s in ekf.sys.noise_specs]
    assert "wind.velocity_driver" in full_names


def test_ekf_state_dict_nests_disturbance_state():
    w, _ = _build_world()
    ekf = TargetNumpy(EKF(w))
    sd = ekf.state_dict()
    assert "drone" in sd and "wind" in sd
    assert "velocity" in sd["wind"]
    # Seeded at zero.
    np.testing.assert_allclose(sd["wind"]["velocity"], np.zeros(3),
                               atol=1e-12)


def test_unique_disturbance_names_required():
    """Two disturbances with the same `name` should raise at compile."""
    w = World().add_field(GravityField(g=(0, 0, -9.81)))
    ff = FluidField()
    ff.add(WindBias(name="wind"))
    ff.add(WindBias(name="wind"))   # collision
    w.add_field(ff)
    c = Craft("d"); c.add(Mass("body", mass=1.0))
    w.add_craft(c)
    import pytest
    with pytest.raises(ValueError, match="duplicate disturbance name"):
        Sim(w)

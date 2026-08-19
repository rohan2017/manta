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
import pytest

from manta import Craft, EKF, Sim, TargetNumpy, World
from manta.fields import (
    Disturbance, FluidField, FluidState, GravityField,
)
from manta.ir.frames import WorldFrame
from manta.ir.types import Vec3
from manta.parts import Mass, RandomWalkNoise, State


class WindBias(Disturbance):
    """A globally-constant wind velocity, evolved as a random walk."""
    field_value_shape = FluidState
    velocity = RandomWalkNoise("R3", sigma=1e-3)

    def contribute_at_sym(self, point, t):
        return FluidState(
            density=ca.MX(0.0),
            pressure=ca.MX(0.0),
            temperature=ca.MX(0.0),
            viscosity=ca.MX(0.0),
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
    drv = np.array([0.5, -0.25, 0.1])
    dt = 0.01
    expected = np.zeros(3)
    for _ in range(20):
        cw.state["wind"]["velocity_driver"] = drv.copy()
        cw.step(dt)
        expected = expected + np.sqrt(dt) * drv
    np.testing.assert_allclose(cw.state["wind"]["velocity"], expected,
                               atol=1e-10)


def test_state_spec_picks_up_disturbance_bias():
    w, _ = _build_world()
    from manta import state_spec_from_world
    spec = state_spec_from_world(w)
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


class SteadyWind(Disturbance):
    """A constant wind velocity held as a plain Vec3 State slot (no
    noise) — exercises non-scalar disturbance-declared State."""
    field_value_shape = FluidState
    velocity = State((1.0, -2.0, 0.5), manifold="R3")

    def contribute_at_sym(self, point, t):
        return FluidState(
            density=ca.MX(0.0),
            pressure=ca.MX(0.0),
            temperature=ca.MX(0.0),
            viscosity=ca.MX(0.0),
            velocity=Vec3[WorldFrame].from_mx(self.velocity._mx),
        )


def _build_vec_state_world():
    w = World().add_field(GravityField(g=(0.0, 0.0, -9.81)))
    ff = FluidField()
    ff.add(SteadyWind(name="steady"))
    w.add_field(ff)
    c = Craft("drone")
    c.add(Mass("body", mass=1.0))
    w.add_craft(c, position=(0.0, 0.0, 5.0))
    return w, c


def test_vec_disturbance_state_compiles_and_seeds():
    """A Vec3 State on a disturbance compiles; the initial state dict
    carries the declared init as a 3-vector."""
    w, _ = _build_vec_state_world()
    cw = TargetNumpy(Sim(w))
    init = cw.initial_state()
    assert "steady" in init
    np.testing.assert_allclose(init["steady"]["velocity"],
                               (1.0, -2.0, 0.5))


def test_vec_disturbance_state_is_identity_passthrough():
    """Plain (noiseless) vector State holds its value across steps, and
    a runtime override sticks."""
    w, _ = _build_vec_state_world()
    cw = TargetNumpy(Sim(w))
    for _ in range(10):
        cw.step(0.01)
    np.testing.assert_allclose(cw.state["steady"]["velocity"],
                               (1.0, -2.0, 0.5), atol=1e-12)
    cw.state["steady"]["velocity"] = np.array([3.0, 0.0, 0.0])
    cw.step(0.01)
    np.testing.assert_allclose(cw.state["steady"]["velocity"],
                               (3.0, 0.0, 0.0), atol=1e-12)


def test_vec_disturbance_state_spec_slot():
    w, _ = _build_vec_state_world()
    from manta import state_spec_from_world
    spec = state_spec_from_world(w)
    slot = spec.slot("steady.velocity")
    assert slot.ambient_dim == 3
    assert slot.tangent_dim == 3
    assert slot.manifold.kind == "vec"


def test_vec_disturbance_state_reaches_ekf():
    w, _ = _build_vec_state_world()
    ekf = TargetNumpy(EKF(w))
    sd = ekf.state_dict()
    assert "steady" in sd
    np.testing.assert_allclose(sd["steady"]["velocity"], (1.0, -2.0, 0.5),
                               atol=1e-12)


def test_unique_disturbance_names_required():
    """Duplicate names are rejected at the owning collection boundary."""
    ff = FluidField()
    ff.add(WindBias(name="wind"))
    with pytest.raises(ValueError, match="already exists"):
        ff.add(WindBias(name="wind"))

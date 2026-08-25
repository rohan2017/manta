"""ProjectedVelocityDamping's compact matrix-kernel contracts."""

import numpy as np
import pytest

from manta import Craft, Sim, TargetNumpy, World
from manta.fields import FluidField, GravityField
from manta.parts import Mass, ProjectedVelocityDamping


def _sim(part, *, velocity=(0.0, 0.0, 0.0), omega=(0.0, 0.0, 0.0)):
    craft = Craft("body")
    craft.add(Mass("mass", mass=10.0, moi=(1.0, 2.0, 3.0)))
    craft.add(part)
    world = (
        World()
        .add_field(GravityField().add_uniform((0.0, 0.0, 0.0)))
        .add_field(FluidField().add_uniform(density=1.0))
    )
    world.add_craft(craft, velocity=velocity, angular_velocity=omega)
    return TargetNumpy(Sim(world))


def test_projection_retains_mixed_translation_rate_term():
    # One scalar feature z = vx + wz, lifted into -Fx and -Tz.
    basis = np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
    lift = np.array([[-2.0], [0.0], [0.0], [0.0], [0.0], [-0.5]])
    sim = _sim(
        ProjectedVelocityDamping(
            "drag",
            linear_tensor=np.zeros((6, 6)),
            velocity_projection=basis,
            wrench_projection=lift,
        ),
        velocity=(1.0, 0.0, 0.0),
        omega=(0.0, 0.0, 2.0),
    )
    sim.step(1e-3)
    # z=3, z|z|=9 => Fx=-18 and Tz=-4.5 at the initial state.
    assert (1.0 - sim.state["body"]["velocity"][0]) / 1e-3 \
        == pytest.approx(1.8, rel=2e-3)
    assert (2.0 - sim.state["body"]["angular_velocity"][2]) / 1e-3 \
        == pytest.approx(1.5, rel=2e-3)


def test_constructor_rejects_inconsistent_shapes_and_nonfinite_values():
    valid = {
        "linear_tensor": np.zeros((6, 6)),
        "velocity_projection": np.zeros((2, 6)),
        "wrench_projection": np.zeros((6, 2)),
    }
    with pytest.raises(ValueError, match="linear_tensor"):
        ProjectedVelocityDamping("d", **{**valid, "linear_tensor": np.zeros((3, 3))})
    with pytest.raises(ValueError, match="velocity_projection"):
        ProjectedVelocityDamping(
            "d", **{**valid, "velocity_projection": np.zeros((2, 5))}
        )
    with pytest.raises(ValueError, match="wrench_projection"):
        ProjectedVelocityDamping(
            "d", **{**valid, "wrench_projection": np.zeros((6, 3))}
        )
    bad = np.zeros((6, 6))
    bad[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        ProjectedVelocityDamping("d", **{**valid, "linear_tensor": bad})

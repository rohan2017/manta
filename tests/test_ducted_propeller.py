"""The provisional ducted-propeller open-water contract."""

import numpy as np
import pytest

from manta import Craft, Sim, TargetNumpy, World
from manta.fields import FluidField, GravityField
from manta.parts import DuctedPropeller, Mass


def _run(*, velocity=(0.0, 0.0, 0.0), command=100.0,
         **overrides):
    prop = DuctedPropeller(
        "prop", max_static_thrust=100.0, max_static_torque=20.0,
        diameter=0.2, zero_thrust_advance_speed=2.0, **overrides)
    craft = Craft("boat")
    craft.add(Mass("mass", mass=100.0))
    craft.add(prop)
    world = (World()
             .add_field(GravityField(g=(0.0, 0.0, 0.0)))
             .add_field(FluidField().add_uniform(density=1025.0)))
    world.add_craft(craft, velocity=velocity)
    sim = TargetNumpy(Sim(world))
    sim.step(1e-4, u={"prop.thrust_command": command})
    return sim.outputs()["boat"]


def test_static_point_reproduces_bollard_thrust_and_torque():
    output = _run()
    assert output["prop.thrust"] == pytest.approx(100.0, rel=1e-5)
    assert output["prop.reaction_torque"] == pytest.approx(20.0, rel=1e-5)
    assert output["prop.advance_fraction"] == pytest.approx(0.0, abs=0.01)


def test_axial_advance_unloads_torque_faster_than_thrust():
    static = _run()
    advanced = _run(velocity=(1.0, 0.0, 0.0))
    thrust_fraction = advanced["prop.thrust"] / static["prop.thrust"]
    torque_fraction = (advanced["prop.reaction_torque"]
                       / static["prop.reaction_torque"])
    assert thrust_fraction == pytest.approx(0.75, abs=0.02)
    assert 0.0 < torque_fraction < thrust_fraction
    assert advanced["prop.advance_fraction"] == pytest.approx(0.5, abs=0.02)


def test_opposite_advance_adds_windmilling_braking_force():
    # A forward-moving hull commanded into reverse combines motor thrust with
    # through-disk drag. Torque remains at its static reverse calibration
    # because the inflow is not helpful advance along the commanded rotation.
    output = _run(velocity=(1.0, 0.0, 0.0), command=-100.0)
    assert output["prop.thrust"] == pytest.approx(-125.0, rel=0.01)
    assert output["prop.reaction_torque"] == pytest.approx(-20.0, rel=0.01)


def test_moving_propeller_has_no_flat_command_region():
    """Every finite command increment changes force at positive advance.

    This is a control contract, not merely curve aesthetics: an MPC
    linearization must retain a usable propeller column while the hull moves.
    """
    commands = np.linspace(-100.0, 100.0, 41)
    thrust = np.array([
        float(np.asarray(_run(velocity=(1.0, 0.0, 0.0),
                              command=float(command))["prop.thrust"]
                         ).reshape(-1)[0])
        for command in commands
    ])
    increments = np.diff(thrust) / np.diff(commands)
    assert np.all(increments > 0.95)
    assert _run(velocity=(1.0, 0.0, 0.0), command=0.0)[
        "prop.thrust"] < 0.0


def test_oblique_flow_reduces_axial_efficiency_without_side_force():
    aligned = _run(velocity=(0.0, 0.0, 0.0))
    oblique = _run(velocity=(0.0, 2.0, 0.0))
    assert 0.0 < oblique["prop.oblique_factor"] < 0.8
    assert abs(oblique["prop.thrust"]) < abs(aligned["prop.thrust"])


def test_default_advance_speed_is_the_ideal_static_far_wake():
    prop = DuctedPropeller(
        "p", max_static_thrust=100.0, max_static_torque=0.0,
        diameter=0.2, reference_density=1000.0)
    area = np.pi * 0.1 ** 2
    assert prop.zero_thrust_advance_speed == pytest.approx(
        np.sqrt(2.0 * 100.0 / (1000.0 * area)))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_static_thrust": 0.0},
        {"max_static_torque": -1.0},
        {"diameter": 0.0},
        {"reaction_sign": 0},
        {"torque_unload_exponent": 0.0},
    ],
)
def test_invalid_calibration_is_rejected(kwargs):
    base = {"max_static_thrust": 100.0, "max_static_torque": 20.0,
            "diameter": 0.2}
    base.update(kwargs)
    with pytest.raises(ValueError):
        DuctedPropeller("bad", **base)

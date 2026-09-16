import numpy as np

from manta import EKF, Craft, Sim, TargetNumpy, World
from manta.fields import GravityField
from manta.parts import ComponentPositionSensor, Mass


def _world() -> World:
    craft = Craft("probe")
    craft.add(Mass("body", mass=1.0, moi=(1.0, 1.0, 1.0)))
    craft.add(ComponentPositionSensor("gps", position_noise_sigma=0.5))
    world = World("component_position").add_field(
        GravityField(g=(0.0, 0.0, 0.0))
    )
    world.add_craft(craft, position=(1.0, 2.0, 3.0))
    return world


def test_component_position_sensor_shares_one_mount_but_splits_outputs():
    sim = TargetNumpy(Sim(_world()))
    sim.step(0.01)

    np.testing.assert_allclose(
        sim.reading("gps.horizontal_position"), (1.0, 2.0)
    )
    np.testing.assert_allclose(sim.reading("gps.vertical_position"), (3.0,))


def test_component_position_updates_gate_independently():
    transform = EKF(
        _world(),
        sensors=["gps.horizontal_position", "gps.vertical_position"],
        gates={"gps.horizontal_position": 9.0, "gps.vertical_position": 9.0},
    )
    runtime = TargetNumpy(transform)

    horizontal = runtime.update(
        "gps.horizontal_position", (1.1, 1.9), R=np.eye(2) * 0.25
    )
    vertical = runtime.update(
        "gps.vertical_position", (-20.0,), R=np.array(((0.25,),))
    )

    assert horizontal.accepted
    assert not vertical.accepted
    state = runtime.state_dict()["probe"]
    assert state["position"][0] != 1.0
    assert state["position"][1] != 2.0
    assert state["position"][2] == 3.0

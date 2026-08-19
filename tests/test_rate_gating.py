"""Rates are model metadata; downstream runtimes own scheduling policy."""

import numpy as np

from manta import Craft, Sim, TargetNumpy, World
from manta.fields import GravityField
from manta.ir.frames import PartFrame
from manta.ir.types import Vec3
from manta.ir.wrench import Wrench
from manta.parts import Input, Mass, Parameter, Part, PartUpdate, PositionSensor


class GatedThruster(Part):
    rate: float = Parameter(None)
    throttle: float = Input(default=0.0)

    def update(self, ctx):
        force = Vec3[PartFrame].constant((0.0, 0.0, 1.0)) * self.throttle
        return PartUpdate(
            wrench=Wrench(
                force=force,
                torque=Vec3[PartFrame].constant((0.0, 0.0, 0.0)),
            ),
            rates={"throttle": self.rate},
        )


def _world(sensor_rate=10.0, actuator_rate=50.0):
    craft = Craft("craft")
    craft.add(Mass("body", mass=1.0, moi=(0.1, 0.1, 0.1)))
    craft.add(PositionSensor("gps", rate=sensor_rate))
    craft.add(GatedThruster("thruster", rate=actuator_rate))
    world = World().add_field(GravityField(g=(0.0, 0.0, 0.0)))
    world.add_craft(craft)
    return world


def test_rates_are_preserved_in_tick_and_module_metadata():
    transform = Sim(_world())
    assert transform.tick.sample_rates == {
        "craft.gps.position": 10.0,
        "craft.thruster.throttle": 50.0,
    }
    assert transform.module().port("craft.gps.position").rate == 10.0
    throttle = transform.module().port("u").fields[0]
    assert throttle.name == "craft.thruster.throttle"
    assert throttle.rate == 50.0


def test_absent_rates_remain_absent_metadata():
    transform = Sim(_world(sensor_rate=None, actuator_rate=None))
    assert transform.tick.sample_rates == {}
    assert transform.module().port("craft.gps.position").rate is None
    assert transform.module().port("u").fields[0].rate is None


def test_readings_are_live_and_never_held_by_manta():
    runtime = TargetNumpy(Sim(_world()))
    runtime.step(0.02)
    first = np.asarray(runtime.reading("craft.gps.position")).copy()
    runtime.state["craft"]["position"] += 1.0
    runtime.step(0.02)
    moved = np.asarray(runtime.reading("craft.gps.position"))
    assert np.linalg.norm(moved - first) > 0.9

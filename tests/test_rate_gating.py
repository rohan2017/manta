"""Measurement rates partition simulator kernels and drive acquisition."""

import numpy as np

from manta import Craft, NoiseDriver, Sim, TargetNumpy, World
from manta.fields import GravityField
from manta.ir.frames import PartFrame
from manta.ir.types import Vec3
from manta.ir.wrench import Wrench
from manta.parts import IMU, Input, Mass, Parameter, Part, PartUpdate, PositionSensor


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
    profile = transform.module().metadata["transform_profile"]
    assert profile["sensor_scheduling"] == "dependency"
    assert profile["scheduled_measurement_groups"] == (
        ("sample_group_0", 10.0, ("craft.gps.position",)),
    )
    assert transform.module().entry("step").returns == ()
    assert transform.module().entry(
        "sample_group_0").returns == ("craft.gps.position",)


def test_absent_rates_remain_absent_metadata():
    transform = Sim(_world(sensor_rate=None, actuator_rate=None))
    assert transform.tick.sample_rates == {}
    assert transform.module().port("craft.gps.position").rate is None
    assert transform.module().port("u").fields[0].rate is None


def test_rate_limited_readings_are_sampled_when_due_and_held_between():
    runtime = TargetNumpy(Sim(_world()))
    runtime.step(0.02)
    first = np.asarray(runtime.reading("craft.gps.position")).copy()
    runtime.state["craft"]["position"] += 1.0
    runtime.step(0.02)
    held = np.asarray(runtime.reading("craft.gps.position"))
    np.testing.assert_array_equal(held, first)
    for _ in range(4):
        runtime.step(0.02)
    moved = np.asarray(runtime.reading("craft.gps.position"))
    assert np.linalg.norm(moved - first) > 0.9


def test_inline_oracle_is_explicit_and_evaluates_every_tick():
    transform = Sim(_world())
    module = transform.inline_module()
    profile = module.metadata["transform_profile"]
    assert profile["sensor_scheduling"] == "inline"
    assert profile["scheduled_measurement_groups"] == ()
    assert module.entry("step").returns == ("craft.gps.position",)
    runtime = TargetNumpy(module)
    runtime.step(0.02)
    first = np.asarray(runtime.reading("craft.gps.position")).copy()
    runtime.state["craft"]["position"] += 1.0
    runtime.step(0.02)
    assert np.linalg.norm(runtime.reading("craft.gps.position") - first) > 0.9


def test_checkpoint_restores_measurement_deadlines_exactly():
    runtime = TargetNumpy(Sim(_world()))
    runtime.step(0.06)
    checkpoint = runtime.checkpoint()
    runtime.state["craft"]["position"] += 1.0
    runtime.step(0.06)
    runtime.step(0.06)
    sampled = np.asarray(runtime.reading("craft.gps.position")).copy()
    runtime.restore(checkpoint)
    runtime.state["craft"]["position"] += 1.0
    runtime.step(0.06)
    runtime.step(0.06)
    np.testing.assert_array_equal(runtime.reading("craft.gps.position"), sampled)


def test_explicit_time_jump_resynchronizes_acquisition_phase():
    runtime = TargetNumpy(Sim(_world()))
    runtime.step(0.02)
    first = np.asarray(runtime.reading("craft.gps.position")).copy()
    runtime.state["craft"]["position"] += 1.0
    runtime.step(0.02, t=1000.0)
    assert np.linalg.norm(runtime.reading("craft.gps.position") - first) > 0.9


def test_measurement_noise_advances_on_acquisition_not_plant_ticks():
    def runtime(dt: float):
        craft = Craft("craft")
        craft.add(Mass("body", mass=1.0, moi=(0.1, 0.1, 0.1)))
        craft.add(PositionSensor(
            "gps", rate=10.0, position_noise_sigma=1.0))
        world = World().add_field(GravityField.none())
        world.add_craft(craft)
        sim = TargetNumpy(Sim(world))
        sim.attach_driver(NoiseDriver(42))
        for _ in range(round(0.1 / dt) + 1):
            sim.step(dt)
        return np.asarray(sim.reading("craft.gps.position")).copy()

    np.testing.assert_array_equal(runtime(0.01), runtime(0.02))


def test_acceleration_observation_stays_with_unactuated_plant_solve():
    """Compiler dependencies, not actuator presence, identify coupled outputs."""
    craft = Craft("craft")
    craft.add(Mass("body", mass=1.0, moi=(0.1, 0.1, 0.1)))
    craft.add(IMU("imu", rate=10.0))
    world = World().add_field(GravityField.none())
    world.add_craft(craft)

    profile = Sim(world).module().metadata["transform_profile"]
    assert profile["plant_coupled_measurements"] == ("craft.imu.accel",)
    assert profile["scheduled_measurement_groups"] == (
        ("sample_group_0", 10.0, ("craft.imu.gyro",)),
    )

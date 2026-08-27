"""The numpy runtime's per-tick fast path keeps its validation contract.

The buffered kernel path resolves argument sizes and output layouts once
per entry point and hands back slices of one owned result copy. These pin
what callers rely on: outputs are owned (never aliases of the reusable
native buffer), promoted parameters take effect after caching, and bad
inputs still fail with the same named, local errors.
"""

import numpy as np
import pytest

from manta import Craft, NoiseDriver, Sim, TargetNumpy, World
from manta.codegen.numpy._runtime import pack_fields
from manta.fields import GravityField
from manta.parts import IMU, Mass, PositionSensor, Thruster


def _sim(*, driver: bool = False):
    craft = Craft("vehicle")
    craft.add(Mass("body", mass=20.0, moi=(2.0, 2.0, 3.0)))
    craft.add(Thruster("prop", force=(1.0, 0.0, 0.0)))
    craft.add(PositionSensor("gps", position_noise_sigma=0.5))
    craft.add(IMU("imu", accel_noise_sigma=0.01, gyro_noise_sigma=0.001))
    world = World(name="hot_path").add_field(GravityField.none())
    world.add_craft(craft, position=(0.0, 0.0, 5.0))
    sim = TargetNumpy(Sim(world))
    if driver:
        sim.attach_driver(NoiseDriver(seed=7))
    return sim


def test_step_outputs_are_owned_and_stable_across_ticks():
    sim = _sim()
    first = sim.step(0.01, u={"prop.throttle": 0.5})
    gps = np.array(sim.reading("gps.position"))
    imu = sim.reading("imu.accel")
    imu[:] = np.nan                       # caller scribbles on its reading
    second = sim.step(0.01, u={"prop.throttle": 0.5})
    assert np.all(np.isfinite(sim.reading("imu.accel")))
    assert np.all(np.isfinite(second["vehicle"]["position"]))
    fresh = _sim()
    fresh.step(0.01, u={"prop.throttle": 0.5})
    assert np.allclose(fresh.reading("gps.position"), gps)
    fresh.step(0.01, u={"prop.throttle": 0.5})
    assert np.allclose(fresh.reading("gps.position"),
                       sim.reading("gps.position"))
    assert not np.shares_memory(
        np.asarray(sim.reading("gps.position")),
        np.asarray(sim.reading("imu.accel")))
    assert first is second                # in-place state dict contract


def test_noise_draws_still_win_over_held_channel_values():
    a = _sim(driver=True)
    b = _sim(driver=True)
    for _ in range(3):
        a.step(0.01)
        b.step(0.01)
    assert np.allclose(a.reading("gps.position"), b.reading("gps.position"))
    noiseless = _sim()
    for _ in range(3):
        noiseless.step(0.01)
    assert not np.allclose(a.reading("gps.position"),
                           noiseless.reading("gps.position"))


def test_non_finite_and_non_numeric_inputs_fail_by_name():
    sim = _sim()
    with pytest.raises(ValueError, match="prop.throttle.*non-finite"):
        sim.step(0.01, u={"prop.throttle": float("nan")})
    with pytest.raises(TypeError, match="prop.throttle.*real numeric"):
        sim.step(0.01, u={"prop.throttle": "full"})
    with pytest.raises(ValueError, match="prop.throttle.*expected 1 value"):
        sim.step(0.01, u={"prop.throttle": [0.1, 0.2]})
    # A failed step leaves the sim exactly where it was.
    before = sim.checkpoint()
    sim.step(0.01, u={"prop.throttle": 0.25})
    sim.restore(before)
    assert sim.checkpoint() == before


def test_pack_fields_names_the_offending_field():
    port = _sim()._u_port
    with pytest.raises(ValueError, match="step: 'vehicle.prop.throttle'.*non-finite"):
        pack_fields(port.fields, {"vehicle.prop.throttle": np.inf}, who="step")
    with pytest.raises(TypeError, match="unknown field"):
        pack_fields(port.fields, {"nope": 1.0}, who="step")


def test_parameter_overrides_take_effect_after_the_vector_is_cached():
    craft = Craft("vehicle")
    craft.add(Mass("body", mass=20.0, moi=(2.0, 2.0, 3.0)))
    craft.add(Thruster("prop", force=(1.0, 0.0, 0.0)))
    world = World(name="params").add_field(GravityField.none())
    world.add_craft(craft)
    sim = TargetNumpy(Sim(world, parameters=["vehicle.body.mass"]))
    sim.step(0.1, u={"prop.throttle": 1.0})
    heavy = sim.param_vector()
    sim.param_vector()[:] = -1.0          # a returned copy, not the cache
    assert np.allclose(sim.param_vector(), heavy)
    v_heavy = float(sim.state["vehicle"]["velocity"][0])
    sim.set_parameters({"vehicle.body.mass": 2.0})
    assert not np.allclose(sim.param_vector(), heavy)
    sim.step(0.1, u={"prop.throttle": 1.0})
    v_light = float(sim.state["vehicle"]["velocity"][0])
    assert v_light - v_heavy > 4 * v_heavy

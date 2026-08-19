"""Regression coverage for strict, transactional runtime/IR boundaries."""

import numpy as np
import pytest

from manta import (
    Craft, EKF, LQR, NoiseDriver, PID, Sim, SimCheckpoint, TargetNumpy, World,
)
from manta.fields import GravityField
from manta.ir.manifold import ScalarManifold
from manta.ir.state_spec import StateSpec
from manta.parts import Mass, PositionSensor, Thruster


def _world(name="audit", *, mass=1.0):
    craft = Craft("vehicle")
    craft.add(Mass("body", mass=mass))
    craft.add(PositionSensor("gps", position_noise_sigma=0.2))
    craft.add(Thruster("thruster_x", force=(1.0, 0.0, 0.0)))
    craft.add(Thruster("thruster_y", force=(0.0, 1.0, 0.0)))
    craft.add(Thruster("thruster", force=(0.0, 0.0, 1.0),
                       force_noise_sigma=0.1))
    world = World(name).add_field(GravityField(g=(0.0, 0.0, -9.81)))
    world.add_craft(craft, position=(0.0, 0.0, 5.0))
    return world


def test_state_spec_rejects_duplicate_and_unknown_slots():
    with pytest.raises(ValueError, match="duplicate slot"):
        StateSpec.from_layout((("x", ScalarManifold()),
                               ("x", ScalarManifold())))
    spec = StateSpec.from_layout((("position", ScalarManifold()),))
    with pytest.raises(ValueError, match="unknown keys"):
        spec.pack_any({"positon": 1.0}, base=np.array([2.0]))
    with pytest.raises(TypeError, match="real numeric"):
        spec.pack({"position": True})
    with pytest.raises(TypeError, match="real numeric"):
        spec.unpack(np.array(["2.0"]))


def test_filter_rejects_nonfinite_data_without_mutation():
    filt = TargetNumpy(EKF(_world()))
    before = filt.checkpoint()
    with pytest.raises(ValueError, match="non-finite"):
        filt.update("gps.position", [np.nan, 0.0, 5.0])
    with pytest.raises(ValueError, match="non-finite"):
        filt.predict(0.01, Q=np.full_like(filt.P, np.nan))
    with pytest.raises(ValueError, match="non-finite"):
        filt.reset(state={"vehicle": {"position": [np.nan, 0.0, 5.0]}})
    np.testing.assert_array_equal(filt.x, before.x)
    np.testing.assert_array_equal(filt.P, before.P)
    assert filt.time == before.time


@pytest.mark.parametrize("dt", [0.0, -0.1, np.nan, np.inf])
def test_recurrence_rejects_invalid_time(dt):
    pid = TargetNumpy(PID(1.0))
    with pytest.raises(ValueError):
        pid.step(dt, setpoint=1.0, measurement=0.0)


@pytest.mark.parametrize("kwargs", [
    {"kp": np.nan},
    {"kp": 1.0, "integral_limit": -1.0},
    {"kp": 1.0, "output_limit": -1.0},
])
def test_pid_rejects_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        PID(**kwargs)


def test_stochastic_sim_checkpoint_replays_exactly():
    sim = TargetNumpy(Sim(_world()))
    sim.attach_driver(NoiseDriver(42))
    checkpoint = sim.checkpoint()
    sim.step(0.01)
    state = sim.model_state()
    reading = sim.reading("gps.position").copy()
    sim.restore(checkpoint)
    sim.step(0.01)
    np.testing.assert_array_equal(sim.reading("gps.position"), reading)
    for owner, slots in state.items():
        for name, value in slots.items():
            np.testing.assert_array_equal(sim.model_state()[owner][name], value)


def test_sim_checkpoint_rejects_ambiguous_or_nonfinite_payloads():
    with pytest.raises(ValueError, match="duplicate name"):
        SimCheckpoint(
            (("vehicle.position", (0.0,)),
             ("vehicle.position", (1.0,))), (), 0.0, "artifact", None)
    with pytest.raises(ValueError, match="non-finite"):
        SimCheckpoint(
            (("vehicle.position", (np.nan,)),), (), 0.0, "artifact", None)


def test_runtime_rejects_overflowing_logical_time_before_mutation():
    pid = TargetNumpy(PID(1.0))
    before = pid.state
    with pytest.raises(ValueError, match="resulting time"):
        pid.step(1e308, t=1e308, setpoint=1.0, measurement=0.0)
    assert pid.state == before


def test_regulator_rejects_solution_from_another_artifact_atomically():
    kwargs = dict(x_ref={"vehicle": {"position": (0.0, 0.0, 5.0),
                                      "velocity": (0.0, 0.0, 0.0)}},
                  u_ref={"thruster.throttle": 9.81},
                  regulate=["vehicle.position", "vehicle.velocity"],
                  Q=np.eye(6), R=np.eye(3), dt=0.02)
    first = LQR(_world("first"), **kwargs)
    second = LQR(_world("second", mass=2.0), **kwargs)
    regulator = TargetNumpy(first)
    before = (regulator.gain, regulator.u_ff, regulator.x_ref)
    with pytest.raises(ValueError, match="different controller artifact"):
        regulator.reprogram(second.solution)
    np.testing.assert_array_equal(regulator.gain, before[0])
    np.testing.assert_array_equal(regulator.u_ff, before[1])
    np.testing.assert_array_equal(regulator.x_ref, before[2])


def test_world_registration_enforces_type_and_single_ownership():
    craft = Craft("c")
    first, second = World("first"), World("second")
    first.add_craft(craft)
    with pytest.raises(ValueError, match="already belongs"):
        second.add_craft(craft)
    with pytest.raises(TypeError, match="expected a Craft"):
        first.add_craft(object())

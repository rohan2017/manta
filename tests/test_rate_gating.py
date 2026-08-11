"""Rate gating — `PartUpdate.rates` (metadata only).

A part declares a rate inside `update()`; the value is returned
unchanged (the compiled kernel stays a pure function), and the rate is
recorded as metadata on the tick. Gating is the loop's job, uniform across
backends: `sim.rate_gate(name)` / `sim.command_latch()` build a `RateGate`
/ `CommandLatch` pre-configured from those declared rates, which the user
applies — fold a sensor once per 1/rate window, hold an actuator command
between windows (ZOH). The estimator/controller linearization never sees
the gate.
"""

import numpy as np
import pytest

from manta import (
    Craft, EKF, Sim, TargetNumpy, World,
)
from manta.fields import GravityField
from manta.ir.frames import PartFrame
from manta.ir.types import Vec3
from manta.ir.wrench import Wrench
from manta.parts import Mass, PositionSensor
from manta.parts import Input, Parameter, Part, PartUpdate


# ---------------------------------------------------------------------------
# A custom actuator that gates its command intake via PartUpdate.rates
# ---------------------------------------------------------------------------

class GatedThruster(Part):
    """+z thruster whose throttle is accepted at `rate` Hz (ZOH)."""

    rate: float = Parameter(None)
    gain: float = Parameter(1.0)
    throttle: float = Input(default=0.0)

    def update(self, ctx):
        f = Vec3[PartFrame].constant((0.0, 0.0, self.gain)) * self.throttle
        return PartUpdate(
            wrench=Wrench(force=f, torque=Vec3[PartFrame].constant((0, 0, 0))),
            rates={"throttle": self.rate})


# ---------------------------------------------------------------------------
# Metadata: sample_rates captured for outputs and inputs
# ---------------------------------------------------------------------------

def test_sample_rate_captured_on_tick():
    c = Craft("c")
    c.add(Mass("body", mass=1.0, moi=(0.1, 0.1, 0.1)))
    c.add(PositionSensor("gps", rate=10.0))
    c.add(GatedThruster("th", rate=50.0))
    w = World().add_field(GravityField(g=(0, 0, 0)))
    w.add_craft(c, position=(0, 0, 0))
    assert Sim(w).tick.sample_rates == {
        "c.gps.position": 10.0, "c.th.throttle": 50.0}


def test_no_rate_means_empty_map():
    c = Craft("c")
    c.add(Mass("body", mass=1.0, moi=(0.1, 0.1, 0.1)))
    c.add(PositionSensor("gps"))            # rate=None default
    w = World().add_field(GravityField(g=(0, 0, 0)))
    w.add_craft(c, position=(0, 0, 0))
    assert Sim(w).tick.sample_rates == {}


# ---------------------------------------------------------------------------
# Sensor output: the loop gates folding with a model-configured RateGate
# ---------------------------------------------------------------------------

def _gps_world(rate):
    c = Craft("c")
    c.add(Mass("body", mass=1.0, moi=(0.1, 0.1, 0.1)))
    c.add(PositionSensor("gps", rate=rate))
    w = World().add_field(GravityField(g=(0, 0, -9.81)))
    w.add_craft(c, position=(0, 0, 100))
    return w


def test_rate_gate_due_per_window():
    # 10 Hz sensor at a 50 Hz sim ⇒ the gate fires once every 5 ticks.
    sim = TargetNumpy(Sim(_gps_world(10.0)))
    gate = sim.rate_gate("c.gps.position")     # configured from the model
    due = []
    for i in range(15):
        sim.step(0.02)
        due.append(gate.due(i * 0.02))
    assert due == [True, False, False, False, False] * 3


@pytest.mark.parametrize("rate", [0.0, -1.0, float("nan"), float("inf")])
def test_rate_gate_rejects_invalid_rate(rate):
    from manta import RateGate
    with pytest.raises(ValueError):
        RateGate(rate)


@pytest.mark.parametrize("t", [float("nan"), float("inf")])
def test_rate_gate_rejects_non_finite_time(t):
    from manta import RateGate
    with pytest.raises(ValueError, match="must be finite"):
        RateGate(1.0).due(t)


def test_reading_is_live_not_held():
    # The kernel is pure: `reading()` tracks truth every step — holding
    # between samples is the loop's job (gate the *fold*, not the read).
    sim = TargetNumpy(Sim(_gps_world(10.0)))
    sim.step(0.02)
    first = np.asarray(sim.reading("c.gps.position")).copy()
    sim.state["c"]["position"] = sim.state["c"]["position"] + 1.0
    sim.step(0.02)
    moved = np.asarray(sim.reading("c.gps.position"))
    assert np.linalg.norm(moved - first) > 0.9


def test_default_rate_gate_fires_every_tick():
    sim = TargetNumpy(Sim(_gps_world(None)))
    gate = sim.rate_gate("c.gps.position")
    for i in range(5):
        sim.step(0.02)
        assert gate.due(i * 0.02)


def test_ekf_folds_once_per_window():
    c = Craft("c")
    c.add(Mass("body", mass=1.0, moi=(0.1, 0.1, 0.1)))
    c.add(PositionSensor("gps", rate=10.0, position_noise_sigma=0.1))
    w = World().add_field(GravityField(g=(0, 0, -9.81)))
    w.add_craft(c, position=(0, 0, 100))

    sim = TargetNumpy(Sim(w))
    ekf = TargetNumpy(EKF(w))
    ekf.reset(P=np.eye(ekf.spec.tangent_dim))
    ekf.Q = np.eye(ekf.spec.tangent_dim) * 1e-3
    gps = sim.rate_gate("c.gps.position")

    folds = 0
    for i in range(15):
        t = i * 0.02
        sim.step(0.02)
        before = np.trace(ekf.P)     # before the update+predict cycle
        if gps.due(t):               # fold only on a fresh window
            ekf.update("c.gps.position", sim.reading("c.gps.position"))
        ekf.predict(0.02)            # a fold's shrink beats predict's growth
        if np.trace(ekf.P) < before - 1e-9:
            folds += 1
    assert folds == 3                # one per 10 Hz window over 0.3 s


# ---------------------------------------------------------------------------
# Actuator: a CommandLatch (ZOH) applied in the loop holds the command
# ---------------------------------------------------------------------------

def test_command_latch_holds_between_intake_windows():
    """A 10 Hz actuator: the loop's `command_latch()` holds the command
    written at each window boundary; mid-window writes are ignored until
    the next boundary."""
    c = Craft("c")
    c.add(Mass("body", mass=1.0, moi=(0.1, 0.1, 0.1)))
    c.add(GatedThruster("th", rate=10.0, gain=1.0))
    w = World().add_field(GravityField(g=(0, 0, 0)))     # no gravity
    w.add_craft(c, position=(0, 0, 0))

    sim = TargetNumpy(Sim(w))
    latch = sim.command_latch()      # configured from the actuator's rate

    dt = 0.02
    # Window 0 [t=0..0.1): latch 1 N. Velocity gain = (1/1)·0.1 = 0.1 m/s.
    val = 1.0
    for i in range(5):
        u = latch.latch({"c.th.throttle": val}, i * dt)
        sim.step(dt, u=u)
        val = 100.0                  # mid-window write, ignored until t=0.1
    vz = np.asarray(sim.state["c"]["velocity"]).ravel()[2]
    # Only the 1 N latched at t=0 acted for the whole window: a=1, 0.1 s.
    assert abs(vz - 0.1) < 1e-6


def test_command_latch_default_rate_passes_through():
    c = Craft("c")
    c.add(Mass("body", mass=1.0, moi=(0.1, 0.1, 0.1)))
    c.add(GatedThruster("th", gain=1.0))     # rate=None
    w = World().add_field(GravityField(g=(0, 0, 0)))
    w.add_craft(c, position=(0, 0, 0))

    sim = TargetNumpy(Sim(w))
    latch = sim.command_latch()              # no gated inputs → identity
    for i in range(50):
        sim.step(0.02, u=latch.latch({"c.th.throttle": 2.0}, i * 0.02))
    vz = np.asarray(sim.state["c"]["velocity"]).ravel()[2]
    assert abs(vz - 2.0) < 1e-6     # 2 N · 1 kg⁻¹ · 1 s

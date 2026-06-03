"""Rate gating — `ctx.sample` / `ctx.hold` (Approach A).

A part declares a rate inside `update()`; the value is returned
unchanged (the compiled kernel stays a pure function), and the rate is
recorded as metadata on the tick. The Sim/EKF runtimes read it and gate
the matching *port*: a sensor output publishes a fresh sample once per
1/rate window (sample-and-hold), and an actuator command latches once
per window (zero-order-hold). The estimator/controller linearization
never sees the gate.
"""

import numpy as np

from manta import (
    Craft, EKF, NoiseDriver, Sim, TargetNumpy, World, wire,
)
from manta.fields import GravityField
from manta.ir.frames import PartFrame
from manta.ir.types import Vec3
from manta.ir.wrench import Wrench
from manta.parts import Mass, PositionSensor
from manta.parts.base import Input, Parameter, Part


# ---------------------------------------------------------------------------
# A custom actuator that gates its command intake with ctx.hold
# ---------------------------------------------------------------------------

class GatedThruster(Part):
    """+z thruster whose throttle is accepted at `rate` Hz (ctx.hold)."""

    rate: float = Parameter(None)
    gain: float = Parameter(1.0)
    throttle: float = Input(default=0.0)

    def update(self, ctx):
        thr = ctx.hold(self.throttle, rate=self.rate)
        f = Vec3[PartFrame].constant((0.0, 0.0, self.gain)) * thr
        return Wrench(force=f, torque=Vec3[PartFrame].constant((0, 0, 0)))


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
# Sensor output: sample-and-hold at the declared rate
# ---------------------------------------------------------------------------

def _gps_world(rate):
    c = Craft("c")
    c.add(Mass("body", mass=1.0, moi=(0.1, 0.1, 0.1)))
    c.add(PositionSensor("gps", rate=rate))
    w = World().add_field(GravityField(g=(0, 0, -9.81)))
    w.add_craft(c, position=(0, 0, 100))
    return w


def test_output_port_bumps_version_per_window():
    # 10 Hz sensor at a 50 Hz sim ⇒ a new sample every 5 ticks.
    sim = TargetNumpy(Sim(_gps_world(10.0)))
    gps = sim.out("c.gps.position")
    vers = []
    for _ in range(15):
        sim.step(0.02)
        vers.append(gps.version)
    assert vers == [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3]


def test_output_holds_value_between_samples():
    sim = TargetNumpy(Sim(_gps_world(10.0)))
    gps = sim.out("c.gps.position")
    sim.step(0.02)
    held = gps.read().copy()              # captured at the first sample
    # Move truth during the hold window; the port must not change.
    for _ in range(3):
        sim.state["c"]["position"] = sim.state["c"]["position"] + 1.0
        sim.step(0.02)
        np.testing.assert_array_equal(gps.read(), held)


def test_default_rate_publishes_every_tick():
    sim = TargetNumpy(Sim(_gps_world(None)))
    gps = sim.out("c.gps.position")
    for _ in range(5):
        sim.step(0.02)
    assert gps.version == 5


# ---------------------------------------------------------------------------
# EKF folds a slow measurement exactly once per window
# ---------------------------------------------------------------------------

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
    wire(sim.out("c.gps.position"), ekf.meas("c.gps.position"))

    folds = 0
    for _ in range(15):
        sim.step(0.02)
        before = np.trace(ekf.P)
        ekf.step(0.02)               # update shrinks P; predict-only grows it
        if np.trace(ekf.P) < before - 1e-9:
            folds += 1
    assert folds == 3                # one per 10 Hz window over 0.3 s


# ---------------------------------------------------------------------------
# Actuator: ctx.hold gates the command (ZOH) on both sim and EKF
# ---------------------------------------------------------------------------

def test_command_zoh_holds_between_intake_windows():
    """A 10 Hz actuator at a 50 Hz sim sees only the command latched at
    each window boundary; commands written mid-window are ignored until
    the next boundary."""
    c = Craft("c")
    c.add(Mass("body", mass=1.0, moi=(0.1, 0.1, 0.1)))
    c.add(GatedThruster("th", rate=10.0, gain=1.0))
    w = World().add_field(GravityField(g=(0, 0, 0)))     # no gravity
    w.add_craft(c, position=(0, 0, 0))

    sim = TargetNumpy(Sim(w))
    from manta import Signal
    src = Signal("u", dim=1, latched=True)
    wire(src, sim.command("c.th.throttle"))

    dt = 0.02
    # Window 0 [t=0..0.1): latch 1 N. Velocity gain = (1/1)·0.1 = 0.1 m/s.
    src.set(np.array([1.0]))
    for _ in range(5):
        sim.step(dt)
        # Change the source mid-window — must be ignored until next window.
        src.set(np.array([100.0]))
    vz = np.asarray(sim.state["c"]["velocity"]).ravel()[2]
    # Only the 1 N latched at t=0 acted for the whole window: a=1, 0.1 s.
    assert abs(vz - 0.1) < 1e-6


def test_command_default_rate_applies_every_tick():
    c = Craft("c")
    c.add(Mass("body", mass=1.0, moi=(0.1, 0.1, 0.1)))
    c.add(GatedThruster("th", gain=1.0))     # rate=None
    w = World().add_field(GravityField(g=(0, 0, 0)))
    w.add_craft(c, position=(0, 0, 0))

    sim = TargetNumpy(Sim(w))
    from manta import Signal
    src = Signal("u", dim=1, latched=True)
    wire(src, sim.command("c.th.throttle"))
    src.set(np.array([2.0]))
    for _ in range(50):
        sim.step(0.02)
    vz = np.asarray(sim.state["c"]["velocity"]).ravel()[2]
    assert abs(vz - 2.0) < 1e-6     # 2 N · 1 kg⁻¹ · 1 s

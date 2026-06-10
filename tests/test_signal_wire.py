"""Signal + wire() — typed value channels and the runtime ports.

A `Signal` is a value mailbox with a freshness flag, a timestamp, and a
monotonic version. `wire(producer, consumer)` makes the consumer pull
the producer on read; the version counter lets a consumer fold a sample
in exactly once per fresh emission (multi-rate safe, fan-out safe).
"""

import numpy as np
import pytest

from manta import (
    Craft, EKF, LQR, NoiseDriver, Sim, Signal, TargetNumpy, World, wire,
)
from manta.fields import GravityField
from manta.parts import Mass, PositionSensor, Thruster


# ---------------------------------------------------------------------------
# Signal primitive
# ---------------------------------------------------------------------------

def test_signal_set_bumps_version():
    s = Signal("x", dim=3)
    assert s.version == 0 and s.value is None
    s.set(np.array([1.0, 2.0, 3.0]), t=0.5)
    assert s.version == 1 and s.t == 0.5
    np.testing.assert_array_equal(s.read(), [1.0, 2.0, 3.0])


def test_wire_pull_through():
    p = Signal("p", dim=2)
    c = Signal("c", dim=2)
    wire(p, c)
    p.set(np.array([7.0, 8.0]), t=1.0)
    np.testing.assert_array_equal(c.read(), [7.0, 8.0])   # pulled from p
    assert c.cur_version == 1 and c.cur_t == 1.0


def test_wire_rejects_dim_mismatch():
    with pytest.raises(ValueError):
        wire(Signal("a", dim=3), Signal("b", dim=2))


def test_wire_rejects_double_wire():
    c = Signal("c", dim=1)
    wire(Signal("p1", dim=1), c)
    with pytest.raises(ValueError):
        wire(Signal("p2", dim=1), c)


def test_fan_out_one_producer_many_consumers():
    p = Signal("p", dim=1)
    c1, c2 = Signal("c1", dim=1), Signal("c2", dim=1)
    wire(p, c1)
    wire(p, c2)
    p.set(np.array([42.0]))
    assert c1.read()[0] == 42.0 and c2.read()[0] == 42.0
    assert c1.cur_version == c2.cur_version == 1


# ---------------------------------------------------------------------------
# Runtime ports
# ---------------------------------------------------------------------------

def _world():
    c = Craft("c")
    c.add(Mass("body", mass=2.0, moi=(0.1, 0.1, 0.1)))
    c.add(Thruster("tz", force=(0.0, 0.0, 1.0)))
    c.add(PositionSensor("gps", position_noise_sigma=0.05))
    w = World().add_field(GravityField(g=(0.0, 0.0, 0.0)))
    w.add_craft(c, position=(0.0, 0.0, 0.0))
    return w, c


def test_sim_publishes_output_port_each_bus_step():
    w, _ = _world()
    sim = TargetNumpy(Sim(w))
    gps = sim.out("c.gps.position")
    assert gps.version == 0
    sim.step(0.02)
    assert gps.version == 1
    sim.step(0.02)
    assert gps.version == 2


def test_sim_command_port_drives_truth():
    """A wired command feeds the actuator in the held state."""
    w, _ = _world()
    sim = TargetNumpy(Sim(w))
    src = Signal("u", dim=1, latched=True)
    wire(src, sim.command("c.tz.throttle"))
    src.set(np.array([4.0]))             # 4 N on 2 kg, no gravity
    for _ in range(50):
        sim.step(0.02)
    vz = np.asarray(sim.state["c"]["velocity"]).ravel()[2]
    assert vz == pytest.approx(2.0, rel=0.05)   # a=2 m/s² · 1 s


def test_sim_held_state_is_mutable():
    w, _ = _world()
    sim = TargetNumpy(Sim(w))
    gps = sim.out("c.gps.position")          # register the port first
    sim.state["c"]["position"] = np.array([1.0, 2.0, 3.0])
    sim.step(0.01)
    # gps reads start-of-tick position → the value we seeded.
    np.testing.assert_allclose(gps.read(), [1.0, 2.0, 3.0], atol=1e-9)


def test_ekf_meas_port_consumed_once_per_fresh_sample():
    w, _ = _world()
    ekf = TargetNumpy(EKF(w))
    ekf.reset(state={"c": {"position": [1.0, 0.0, 0.0]}},
              P=np.eye(ekf.spec.tangent_dim))
    gps = ekf.meas("c.gps.position")
    gps.set(np.array([0.0, 0.0, 0.0]))   # one fresh sample
    ekf.step(0.02)                       # folds it
    P_after = np.trace(ekf.P)
    ekf.step(0.02)                       # no new sample → predict only
    # Predict-only grows P (no information) → the sample was not reapplied.
    assert np.trace(ekf.P) > P_after


# ---------------------------------------------------------------------------
# Multi-rate: producer emits slower than the consumer steps
# ---------------------------------------------------------------------------

def test_multi_rate_wired_tracks_truth():
    tw, _ = _world()
    sim = TargetNumpy(Sim(tw))
    sim.attach_driver(NoiseDriver(seed=0))
    sim.state["c"]["position"] = np.array([5.0, 0.0, 0.0])

    ew, _ = _world()
    ekf = TargetNumpy(EKF(ew))
    ekf.reset(state={"c": {"position": [0.0, 0.0, 0.0]}},
              P=np.eye(ekf.spec.tangent_dim))
    ekf.Q = np.eye(ekf.spec.tangent_dim) * 1e-3

    gps_out = sim.out("c.gps.position")
    gps_in  = ekf.meas("c.gps.position")

    dt = 0.02
    for i in range(200):
        sim.step(dt)
        # Only forward the fix every 4th step → consumer sees a slower
        # producer; version-gating must avoid reapplying stale samples.
        if i % 4 == 0:
            gps_in.set(gps_out.read(), t=gps_out.cur_t)
        ekf.step(dt)

    est = np.asarray(ekf.state_dict()["c"]["position"]).ravel()
    truth = np.asarray(sim.state["c"]["position"]).ravel()
    assert np.linalg.norm(est - truth) < 0.5


# ---------------------------------------------------------------------------
# Full closed loop: sim ↔ ekf ↔ lqr, wired
# ---------------------------------------------------------------------------

def test_wired_closed_loop_regulates_to_setpoint():
    M, G = 2.0, 9.81
    target = np.array([0.0, 0.0, 10.0])
    dt = 0.02

    def build():
        c = Craft("c")
        c.add(Mass("body", mass=M))
        c.add(Thruster("tx", force=(1, 0, 0)))
        c.add(Thruster("ty", force=(0, 1, 0)))
        c.add(Thruster("tz", force=(0, 0, 1)))
        c.add(PositionSensor("gps", position_noise_sigma=0.05))
        w = World().add_field(GravityField(g=(0, 0, -G)))
        w.add_craft(c, position=(0, 0, 10))
        return w, c

    w, c = build()
    sim = TargetNumpy(Sim(w))
    sim.attach_driver(NoiseDriver(seed=7))
    sim.state["c"]["position"] = np.array([3.0, -2.0, 7.0])

    ekf = TargetNumpy(EKF(w))
    ekf.reset(state={"c": c.initial_state(position=(0, 0, 10))},
              P=np.eye(ekf.spec.tangent_dim))
    ekf.Q = np.eye(ekf.spec.tangent_dim) * 1e-4

    lqr = TargetNumpy(LQR(
        w, x_ref={"c": {"position": tuple(target), "velocity": (0, 0, 0)}},
        u_ref={"tz.throttle": M * G}, regulate=["c.position", "c.velocity"],
        Q=np.diag([10, 10, 10, 1, 1, 1]), R=np.eye(3) * 0.1, dt=dt))

    wire(sim.out("c.gps.position"), ekf.meas("c.gps.position"))
    for nm in lqr.input_names:
        wire(lqr.command(nm), sim.command(nm))
        wire(lqr.command(nm), ekf.command(nm))
    wire(ekf.estimate, lqr.estimate_in)

    for _ in range(500):
        lqr.compute()
        sim.step(dt)
        ekf.step(dt)

    final = np.asarray(sim.state["c"]["position"]).ravel()
    assert np.linalg.norm(final - target) < 0.1

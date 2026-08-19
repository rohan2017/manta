"""Value-based plumbing — step(u=) / reading / update / predict.

There is no wire/Signal bus and no runtime-owned loop: the user moves
values between the pure kernels in their own loop, the same shape in every
backend. The sim realizes raw sensor readings each step; the user folds
them into the EKF with `update` and then `predict`s (the user owns the
order). These tests pin that explicit dataflow, including the multi-rate /
closed-loop cases the wired bus used to cover.
"""

import numpy as np
import pytest

from manta import Craft, EKF, LQR, NoiseDriver, Sim, TargetNumpy, World
from manta.fields import GravityField
from manta.parts import Mass, PositionSensor, Thruster


def _world():
    c = Craft("c")
    c.add(Mass("body", mass=2.0, moi=(0.1, 0.1, 0.1)))
    c.add(Thruster("tz", force=(0.0, 0.0, 1.0)))
    c.add(PositionSensor("gps", position_noise_sigma=0.05))
    w = World().add_field(GravityField(g=(0.0, 0.0, 0.0)))
    w.add_craft(c, position=(0.0, 0.0, 0.0))
    return w, c


# ---------------------------------------------------------------------------
# Sim: readings + inputs
# ---------------------------------------------------------------------------

def test_step_u_drives_truth():
    w, _ = _world()
    sim = TargetNumpy(Sim(w))
    for _ in range(50):
        sim.step(0.02, u={"c.tz.throttle": 4.0})  # 4 N on 2 kg, no gravity
    vz = np.asarray(sim.state["c"]["velocity"]).ravel()[2]
    assert vz == pytest.approx(2.0, rel=0.05)    # a = 2 m/s² · 1 s


def test_step_u_is_a_plain_dict_to_perturb_or_drop():
    """`u` is an ordinary dict the user owns: scale it to perturb, omit a key
    to drop an input back to its declared default (0 ⇒ no thrust here)."""
    w, _ = _world()
    perturbed = TargetNumpy(Sim(w))
    for _ in range(50):
        u = {"c.tz.throttle": 4.0}
        u["c.tz.throttle"] *= 0.5                 # perturb before applying
        perturbed.step(0.02, u=u)
    assert np.asarray(perturbed.state["c"]["velocity"]).ravel()[2] == \
        pytest.approx(1.0, rel=0.05)              # half the thrust → half vz

    dropped = TargetNumpy(Sim(w))
    for _ in range(50):
        dropped.step(0.02, u={})                  # omit it → default 0
    assert np.asarray(dropped.state["c"]["velocity"]).ravel()[2] == \
        pytest.approx(0.0, abs=1e-9)


def test_reading_is_start_of_tick_state():
    w, _ = _world()
    sim = TargetNumpy(Sim(w))
    sim.state["c"]["position"] = np.array([1.0, 2.0, 3.0])
    sim.step(0.01)
    # The GPS reads the start-of-tick position → the value we seeded.
    np.testing.assert_allclose(
        sim.reading("c.gps.position"), [1.0, 2.0, 3.0], atol=1e-9)


# ---------------------------------------------------------------------------
# Multi-rate: the downstream loop folds the filter slower than the plant
# ---------------------------------------------------------------------------

def test_multi_rate_update_tracks_truth():
    tw, _ = _world()
    sim = TargetNumpy(Sim(tw))
    sim.attach_driver(NoiseDriver(seed=0))
    sim.state["c"]["position"] = np.array([5.0, 0.0, 0.0])

    ew, _ = _world()
    ekf = TargetNumpy(EKF(ew))
    ekf.reset(state={"c": {"position": [0.0, 0.0, 0.0]}},
              P=np.eye(ekf.spec.tangent_dim))
    ekf.Q = np.eye(ekf.spec.tangent_dim) * 1e-3

    dt = 0.02
    for i in range(200):
        sim.step(dt)
        if i % 4 == 0:                       # fold a slower stream...
            ekf.update("c.gps.position", sim.reading("c.gps.position"))
        ekf.predict(dt)                      # ...predicting between fixes

    est = np.asarray(ekf.state_dict()["c"]["position"]).ravel()
    truth = np.asarray(sim.state["c"]["position"]).ravel()
    assert np.linalg.norm(est - truth) < 0.5


# ---------------------------------------------------------------------------
# Full closed loop: sim ↔ ekf ↔ lqr, plumbed by hand
# ---------------------------------------------------------------------------

def test_closed_loop_regulates_to_setpoint():
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
    ekf.reset_from_model_record(
        {"c": c.initial_state(position=(0, 0, 10))},
        P=np.eye(ekf.spec.tangent_dim))
    ekf.Q = np.eye(ekf.spec.tangent_dim) * 1e-4

    lqr = TargetNumpy(LQR(
        w, x_ref={"c": {"position": tuple(target), "velocity": (0, 0, 0)}},
        u_ref={"tz.throttle": M * G}, regulate=["c.position", "c.velocity"],
        Q=np.diag([10, 10, 10, 1, 1, 1]), R=np.eye(3) * 0.1, dt=dt))

    for _ in range(500):
        u = lqr.control(ekf.state_dict())          # estimate → commands
        sim.step(dt, u=u)
        ekf.update("c.gps.position",               # fold the reading...
                   sim.reading("c.gps.position"), u=u)
        ekf.predict(dt, u=u)                        # ...then predict
    final = np.asarray(sim.state["c"]["position"]).ravel()
    assert np.linalg.norm(final - target) < 0.1

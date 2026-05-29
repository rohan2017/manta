"""EKF measurement bus + step() — freshness-gated, multi-rate updates.

`feed()` drops a reading into a mailbox; `step()` predicts then folds in
every fresh measurement (clearing the flag) and drops stale ones. Inputs
latch (zero-order hold) and drive the predict.
"""

import numpy as np

from manta import Sim, TargetNumpy, World
from manta.craft import Craft
from manta.fields import GravityField
from manta.estimation import EKF
from manta.parts.structure.mass import Mass
from manta.parts.sensor.position_sensor import PositionSensor


def _gps_craft(name="drone", sigma=0.1):
    c = Craft(name)
    c.add(Mass("body", mass=1.0))
    c.add(PositionSensor("gps", position_noise_sigma=sigma))
    return c


def _truth_sim(z0=100.0):
    tw = World().add_field(GravityField(g=(0, 0, -9.81)))
    tw.add_craft(_gps_craft(), position=(0, 0, z0))
    sim = TargetNumpy(Sim(tw))
    return sim, sim.initial_state()


def _est_ekf():
    ew = World().add_field(GravityField(g=(0, 0, -9.81)))
    ew.add_craft(_gps_craft())
    return TargetNumpy(EKF(ew))


# ---------------------------------------------------------------------------
# Multi-rate: GPS slower than the filter; step() predicts between fixes
# ---------------------------------------------------------------------------

def test_step_multi_rate_tracks_truth():
    sim, st = _truth_sim()
    rt = _est_ekf()
    rt.reset(state={"drone": {"position": [5.0, 0.0, 105.0]}}, P=np.eye(12))

    rng = np.random.default_rng(0)
    dt = 0.02
    for i in range(200):
        st = sim.step(st, dt=dt, t=i * dt)
        truth = np.asarray(st["drone"]["position"])
        # GPS arrives at 1/4 the filter rate.
        if i % 4 == 0:
            rt.feed("gps.position", truth + rng.normal(0, 0.1, 3), t=i * dt)
        rt.step(dt, t=i * dt)

    est = np.asarray(rt.state_dict()["drone"]["position"])
    truth = np.asarray(st["drone"]["position"])
    assert np.linalg.norm(est - truth) < 0.5


# ---------------------------------------------------------------------------
# A fed measurement is consumed exactly once
# ---------------------------------------------------------------------------

def test_step_consumes_measurement_once():
    rt = _est_ekf()
    rt.reset(state={"drone": {"position": [10.0, 0.0, 100.0]}}, P=np.eye(12))
    rt.feed("gps.position", np.array([0.0, 0.0, 100.0]))
    assert rt._meas["drone.gps.position"]["fresh"] is True

    rt.step(0.02)                                   # applies the fix
    assert rt._meas["drone.gps.position"]["fresh"] is False
    P_after_update = np.trace(rt.P)

    rt.step(0.02)                                   # no feed → predict only
    # A predict-only step GROWS covariance (no information added) — proof
    # the consumed measurement was not silently reapplied (an update would
    # have shrunk trace(P) instead).
    assert np.trace(rt.P) > P_after_update


# ---------------------------------------------------------------------------
# Stale (old-timestamp) measurements are dropped
# ---------------------------------------------------------------------------

def test_step_drops_stale_measurement():
    rt = _est_ekf()
    rt.reset(state={"drone": {"position": [0.0, 0.0, 100.0]}}, P=np.eye(12))
    # Advance the filter clock to t=1.0.
    for i in range(50):
        rt.step(0.02, t=i * 0.02)
    pos_before = rt.x[:3].copy()
    # A measurement timestamped far in the past must be ignored.
    rt.feed("gps.position", np.array([999.0, 999.0, 999.0]), t=-100.0)
    rt.step(0.02, t=1.0)
    # Position must not have jumped toward the bogus reading.
    assert np.linalg.norm(rt.x[:2] - pos_before[:2]) < 1.0
    assert rt._meas["drone.gps.position"]["fresh"] is False


# ---------------------------------------------------------------------------
# Latched inputs drive the predict
# ---------------------------------------------------------------------------

def test_step_latched_inputs_drive_predict():
    from manta.parts.actuation.thruster import Thruster

    def craft():
        c = Craft("rocket")
        c.add(Mass("body", mass=1.0))
        c.add(Thruster("t", force=(0.0, 0.0, 1.0)))
        return c

    w = World().add_field(GravityField(g=(0, 0, 0.0)))   # no gravity
    w.add_craft(craft())
    rt = TargetNumpy(EKF(w))

    rt.inputs["t.throttle"] = 10.0      # latched control
    for _ in range(10):
        rt.step(0.05)
    vz = rt.state_dict()["rocket"]["velocity"][2]
    # 10 N for 0.5 s on 1 kg ⇒ ~5 m/s upward.
    assert vz > 1.0

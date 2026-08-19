"""EKF predict / update — the uniform kernel surface, driven by hand.

The filter exposes only the pure kernels: `update(name, z)` folds one
measurement at the current state, `predict(dt)` propagates. The loop owns
the order (fold each fresh measurement, then predict) and the rate gating
— identically to a C++ driving loop. `u` (latched commands) drives predict.
"""

import numpy as np
import pytest

from manta import Sim, TargetNumpy, World
from manta.craft import Craft
from manta.estimation import EKF
from manta.fields import GravityField
from manta.parts.sensor.position_sensor import PositionSensor
from manta.parts.structure.mass import Mass


def _gps_craft(name="drone", sigma=0.1):
    c = Craft(name)
    c.add(Mass("body", mass=1.0))
    c.add(PositionSensor("gps", position_noise_sigma=sigma))
    return c


def _truth_sim(z0=100.0):
    tw = World().add_field(GravityField(g=(0, 0, -9.81)))
    tw.add_craft(_gps_craft(), position=(0, 0, z0))
    return TargetNumpy(Sim(tw))


def _est_ekf():
    ew = World().add_field(GravityField(g=(0, 0, -9.81)))
    ew.add_craft(_gps_craft())
    return TargetNumpy(EKF(ew))


# ---------------------------------------------------------------------------
# Multi-rate: GPS slower than the filter; predict between fixes
# ---------------------------------------------------------------------------

def test_multi_rate_tracks_truth():
    sim = _truth_sim()
    rt = _est_ekf()
    rt.reset(state={"drone": {"position": [5.0, 0.0, 105.0]}}, P=np.eye(12))

    rng = np.random.default_rng(0)
    dt = 0.02
    for i in range(200):
        # A measurement sampled at i·dt is the truth at the *start* of the
        # interval: fold it (update-then-predict) before propagating over
        # [i·dt, (i+1)·dt], so update before advancing the truth.
        truth = np.asarray(sim.state["drone"]["position"])
        if i % 4 == 0:                              # GPS at 1/4 the filter rate
            rt.update("gps.position", truth + rng.normal(0, 0.1, 3), t=i * dt)
        rt.predict(dt, t=i * dt)
        sim.step(dt, t=i * dt)

    est = np.asarray(rt.state_dict()["drone"]["position"])
    truth = np.asarray(sim.state["drone"]["position"])
    assert np.linalg.norm(est - truth) < 0.5


# ---------------------------------------------------------------------------
# update folds information (shrinks P); predict-only grows it
# ---------------------------------------------------------------------------

def test_update_folds_predict_grows():
    rt = _est_ekf()
    rt.reset(state={"drone": {"position": [10.0, 0.0, 100.0]}}, P=np.eye(12))

    rt.update("gps.position", np.array([0.0, 0.0, 100.0]))
    P_after_update = np.trace(rt.P)

    rt.predict(0.02)                # no information added → covariance grows
    assert np.trace(rt.P) > P_after_update


# ---------------------------------------------------------------------------
# Latched inputs drive the predict
# ---------------------------------------------------------------------------

def test_inputs_drive_predict():
    from manta.parts.actuation.thruster import Thruster

    def craft():
        c = Craft("rocket")
        c.add(Mass("body", mass=1.0))
        c.add(Thruster("t", force=(0.0, 0.0, 1.0)))
        return c

    w = World().add_field(GravityField(g=(0, 0, 0.0)))   # no gravity
    w.add_craft(craft())
    rt = TargetNumpy(EKF(w))

    # Omitted inputs fall back to their declared default (zero here), while
    # misspelled inputs fail at the public runtime boundary.
    rt.predict(0.05)
    assert rt.state_dict()["rocket"]["velocity"][2] == pytest.approx(0.0)
    with pytest.raises(KeyError, match="unknown input"):
        rt.predict(0.05, u={"nope.bad": 1.0})

    for _ in range(10):
        rt.predict(0.05, u={"t.throttle": 10.0})

    vz = rt.state_dict()["rocket"]["velocity"][2]
    # 10 N for 0.5 s on 1 kg ⇒ ~5 m/s upward.
    assert vz > 1.0

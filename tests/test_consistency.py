"""NEES consistency check — does the EKF's covariance match its error?

Validates the Monte-Carlo NEES tool by driving the *same* well-modeled
filter with deliberately wrong process noise and confirming the verdict
tracks: too-small Q ⇒ overconfident (ANEES ≫ dof), too-large Q ⇒
conservative (ANEES ≪ dof), and a tuned Q lands in the χ² band. Results
are deterministic at a fixed seed.
"""

import numpy as np
import pytest

from manta import Craft, World
from manta.estimation import nees
from manta.fields import GravityField
from manta.parts import IMU, Mass, PositionSensor, Thruster

M, G = 1.0, 9.81


def _hover_world():
    c = Craft("c")
    c.add(Mass("body", mass=M, moi=(0.05, 0.05, 0.08)))
    c.add(Thruster("t", force=(0, 0, 1), force_noise_sigma=0.3))
    c.add(IMU("imu", gyro_noise_sigma=0.01))
    c.add(PositionSensor("gps", position_noise_sigma=0.05))
    w = World().add_field(GravityField(g=(0, 0, -G)))
    w.add_craft(c, position=(0, 0, 5))
    return w


_KW = dict(dt=0.01, steps=250, control={"t.throttle": M * G}, runs=12, seed=0)


def test_nees_report_structure():
    rep = nees(_hover_world(), **_KW)
    assert rep.dof == 12
    assert rep.anees > 0
    assert rep.lower < rep.upper
    assert rep.samples == rep.runs * (250 - 250 // 5)
    assert "ANEES" in rep.summary()


def test_nees_unknown_sensor_raises():
    """A typo'd sensor name must raise, not silently run without it."""
    with pytest.raises(KeyError, match="unknown sensor"):
        nees(_hover_world(), sensors=["gps.positio"], **_KW)


def test_zero_process_noise_is_overconfident():
    rep = nees(_hover_world(), Q=np.zeros((12, 12)), **_KW)
    assert not rep.consistent
    assert rep.verdict == "overconfident"
    assert rep.anees > rep.upper


def test_inflated_process_noise_is_conservative():
    rep = nees(_hover_world(), Q=np.eye(12), **_KW)
    assert not rep.consistent
    assert rep.verdict == "conservative"
    assert rep.anees < rep.lower


def test_tuned_Q_is_consistent():
    rep = nees(_hover_world(), Q=np.eye(12) * 1e-6, **_KW)
    assert rep.consistent
    assert rep.lower <= rep.anees <= rep.upper


def test_anees_decreases_with_Q():
    """More assumed process noise ⇒ bigger covariance ⇒ smaller NEES."""
    tiny = nees(_hover_world(), Q=np.eye(12) * 1e-8, **_KW).anees
    big = nees(_hover_world(), Q=np.eye(12) * 1e-2, **_KW).anees
    assert tiny > big


def test_observable_subspace_isolates_modeling_from_observability():
    """The full-state NEES is overconfident because attitude (yaw) is
    unobservable from GPS+gyro — not because the noise model is wrong.
    Restricting to the observable subspace shows the filter IS consistent
    there, i.e. the auto-Q is correct; the overconfidence is the EKF
    shrinking covariance on an unobservable direction."""
    from manta import EKF, TargetNumpy
    w = _hover_world()
    basis = EKF(w).observability().basis
    kw = dict(dt=0.01, steps=300, control={"t.throttle": M * G},
              runs=20, seed=0)
    full = nees(w, **kw)
    sub = nees(w, observable_basis=basis, **kw)
    assert full.verdict == "overconfident"
    assert sub.dof == basis.shape[1] < 12
    assert sub.consistent

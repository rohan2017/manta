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


_KW = {"dt": 0.01, "steps": 250, "control": {"t.throttle": M * G}, "runs": 12, "seed": 0}


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


def test_nees_accepts_ukf():
    """`estimator=UKF` audits the sigma-point filter's covariance — the
    filter whose moments differ from the EKF's is exactly the one whose
    consistency needs checking. A shorter run than the EKF cases: this
    pins the plumbing (UKF accepted, report sane), not a verdict."""
    from manta import UKF
    rep = nees(_hover_world(), estimator=UKF,
               dt=0.01, steps=120, control={"t.throttle": M * G},
               runs=4, seed=0)
    assert rep.dof == 12
    assert np.isfinite(rep.anees) and rep.anees > 0


def test_nees_bad_P0_raises():
    """A non-PD P0 fails with a named error at the door, not a bare
    LinAlgError from inside the ensemble loop."""
    with pytest.raises(ValueError, match="positive-definite"):
        nees(_hover_world(), P0=np.diag([-1.0] + [0.1] * 11), **_KW)


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
    from manta import EKF
    w = _hover_world()
    basis = EKF(w).observability().basis
    kw = {"dt": 0.01, "steps": 300, "control": {"t.throttle": M * G},
              "runs": 20, "seed": 0}
    full = nees(w, **kw)
    sub = nees(w, observable_basis=basis, **kw)
    assert full.verdict == "overconfident"
    assert sub.dof == basis.shape[1] < 12
    assert sub.consistent


def test_chi2_gate_matches_tabulated_quantiles_without_scipy():
    """The exported gate helper is exact (not Wilson–Hilferty) at the small
    dofs real sensors have, and validates its inputs."""
    from manta.estimation import chi2_gate, chi2_quantile
    from manta.estimation.consistency import chi2_cdf

    tabulated = {
        (1, 0.95): 3.841459, (2, 0.95): 5.991465, (3, 0.95): 7.814728,
        (3, 0.99): 11.344867, (3, 0.999): 16.266236, (6, 0.99): 16.811894,
        (10, 0.95): 18.307038, (100, 0.975): 129.561197,
        (1, 0.5): 0.454936, (2, 0.05): 0.102587,
    }
    for (dof, confidence), expected in tabulated.items():
        assert chi2_gate(dof, confidence) == pytest.approx(expected, abs=2e-6)
        assert chi2_cdf(chi2_gate(dof, confidence), dof) == pytest.approx(
            confidence, abs=1e-12)
    assert chi2_quantile(260.0, 0.975) == pytest.approx(306.6, abs=0.2)
    for bad in ((0, 0.95), (-1, 0.95), (2.0, 0.95), (True, 0.95)):
        with pytest.raises(ValueError, match="dof"):
            chi2_gate(*bad)
    for bad in ((3, 0.0), (3, 1.0), (3, 1.5), (3, "0.9")):
        with pytest.raises(ValueError, match="confidence"):
            chi2_gate(*bad)

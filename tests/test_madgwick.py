"""Madgwick AHRS recurrence block — numpy behavior + codegen smoke.

Madgwick adds zero backend code: it reuses the generic `lower_recurrence`
path PID establishes and Mahony already roundtrips through compiled C++
(quaternion output included). So there's no separate Madgwick C++ roundtrip
— these tests pin the convention (gyro integration, accelerometer
convergence, quaternion stays unit) and that the C++ target emits its files.
"""

from pathlib import Path

import numpy as np
import pytest

from manta import Madgwick, TargetCpp, TargetNumpy


def test_madgwick_is_recurrence_block():
    f = Madgwick(beta=0.1)
    from manta.ir.module import Hosting
    assert f.module().hosting is Hosting.HELD
    assert [p.name for p in f.inputs] == ["gyro", "accel"]
    assert [p.name for p in f.outputs] == ["orientation"]


def test_madgwick_gyro_only_integration():
    """beta = 0 ⇒ pure gyro integration: a constant body-z rate yields a
    rotation about z by ω·t."""
    r = TargetNumpy(Madgwick(beta=0.0))
    dt, wz, T = 0.001, 2.0, 1.0
    for _ in range(int(T / dt)):
        q = r.step(dt, gyro=[0.0, 0.0, wz], accel=[0.0, 0.0, 1.0])["orientation"]
    theta = wz * T
    np.testing.assert_allclose(
        q, [np.cos(theta / 2), 0.0, 0.0, np.sin(theta / 2)], atol=1e-3)


def test_madgwick_accel_convergence():
    """With gyro at rest, the estimate's predicted gravity converges to the
    measured accelerometer direction."""
    r = TargetNumpy(Madgwick(beta=0.5))
    acc = np.array([0.3, 0.0, 0.95]); acc /= np.linalg.norm(acc)
    for _ in range(20000):
        q = r.step(0.005, gyro=[0.0, 0.0, 0.0], accel=acc)["orientation"]
    q0, q1, q2, q3 = q
    g_pred = np.array([2 * (q1 * q3 - q0 * q2),
                       2 * (q0 * q1 + q2 * q3),
                       1 - 2 * q1 * q1 - 2 * q2 * q2])
    angle = np.degrees(np.arccos(np.clip(g_pred @ acc, -1, 1)))
    assert angle < 0.5
    assert np.linalg.norm(q) == pytest.approx(1.0, abs=1e-9)


def test_madgwick_quaternion_stays_unit():
    r = TargetNumpy(Madgwick(beta=0.2))
    rng = np.random.default_rng(0)
    for _ in range(500):
        q = r.step(0.01, gyro=rng.normal(0, 1, 3),
                   accel=rng.normal(0, 1, 3) + [0, 0, 9.81])["orientation"]
        assert np.linalg.norm(q) == pytest.approx(1.0, abs=1e-9)


# --- codegen smoke ----------------------------------------------------------


def test_madgwick_emits_cpp_files(tmp_path: Path):
    result = TargetCpp(Madgwick(beta=0.1), tmp_path, class_name="Madgwick")
    for p in (result.kernels_c, result.kernels_h, result.wrapper_hpp,
              result.wrapper_cpp, result.cmakelists):
        assert p.exists(), p
